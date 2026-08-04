// Copyright 2026-present the zvec-go project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package zvec

import (
	"encoding/json"
	"io"
	"math"
	"slices"
	"strings"
)

const (
	DefaultHNSWM              = 50
	DefaultHNSWEFConstruction = 500
	DefaultHNSWEFSearch       = 300
	MaxHNSWM                  = 32767
	MaxGraphEFSearch          = 2048
	DefaultPrefetchOffset     = 8
	DefaultPrefetchLines      = 0

	DefaultIVFNList       = 1024
	DefaultIVFNIterations = 10

	DefaultRaBitQTotalBits   = 7
	DefaultRaBitQNumClusters = 16

	DefaultDiskANNMaxDegree = 100
	DefaultDiskANNListSize  = 50
	DefaultDiskANNPQChunks  = 0

	DefaultVamanaMaxDegree      = 64
	DefaultVamanaSearchListSize = 100
	DefaultVamanaEFSearch       = 200
)

const DefaultVamanaAlpha float32 = 1.2

// IndexParams is the common, sealed interface for field index parameters.
// Concrete values have value semantics and may be stored directly in a
// FieldSchema.
type IndexParams interface {
	IndexType() IndexType
	Validate() error
	cloneIndexParams() IndexParams
}

// QuantizerParams configures preprocessing shared by quantized vector indexes.
type QuantizerParams struct {
	// EnableRotate rotates vectors before INT8 or INT4 quantization.
	EnableRotate bool
}

type vectorIndexConfig struct {
	metric    MetricType
	quantize  QuantizeType
	quantizer QuantizerParams
}

// InvertIndexParams configures a scalar inverted index.
type InvertIndexParams struct {
	EnableRangeOptimization bool
	EnableExtendedWildcard  bool
}

// NewInvertIndexParams returns baseline-compatible inverted-index defaults.
func NewInvertIndexParams() InvertIndexParams {
	return InvertIndexParams{EnableRangeOptimization: true}
}

func (InvertIndexParams) IndexType() IndexType { return IndexTypeInvert }
func (InvertIndexParams) Validate() error      { return nil }
func (p InvertIndexParams) cloneIndexParams() IndexParams {
	return p
}

// FlatIndexParams configures exact vector search.
type FlatIndexParams struct {
	Metric    MetricType
	Quantize  QuantizeType
	Quantizer QuantizerParams
}

func NewFlatIndexParams(metric MetricType) FlatIndexParams {
	return FlatIndexParams{Metric: metric}
}

func (FlatIndexParams) IndexType() IndexType { return IndexTypeFlat }
func (p FlatIndexParams) Validate() error {
	return validateVectorIndexParams(p.IndexType(), p.vectorConfig())
}
func (p FlatIndexParams) cloneIndexParams() IndexParams { return p }
func (p FlatIndexParams) vectorConfig() vectorIndexConfig {
	return vectorIndexConfig{p.Metric, p.Quantize, p.Quantizer}
}

// HNSWIndexParams configures a dense or sparse HNSW index.
type HNSWIndexParams struct {
	Metric              MetricType
	M                   int
	EFConstruction      int
	Quantize            QuantizeType
	UseContiguousMemory bool
	Quantizer           QuantizerParams
}

func NewHNSWIndexParams(metric MetricType) HNSWIndexParams {
	return HNSWIndexParams{
		Metric:         metric,
		M:              DefaultHNSWM,
		EFConstruction: DefaultHNSWEFConstruction,
	}
}

func (HNSWIndexParams) IndexType() IndexType { return IndexTypeHNSW }
func (p HNSWIndexParams) Validate() error {
	if err := validateVectorIndexParams(p.IndexType(), p.vectorConfig()); err != nil {
		return err
	}
	if p.M <= 0 || p.M > MaxHNSWM {
		return invalidArgument("validate HNSW index params", "M must be in [1, %d]", MaxHNSWM)
	}
	if p.EFConstruction < p.M {
		return invalidArgument("validate HNSW index params", "EFConstruction must be at least M")
	}
	return nil
}
func (p HNSWIndexParams) cloneIndexParams() IndexParams { return p }
func (p HNSWIndexParams) vectorConfig() vectorIndexConfig {
	return vectorIndexConfig{p.Metric, p.Quantize, p.Quantizer}
}

// HNSWRaBitQIndexParams configures an HNSW index backed by RaBitQ codes.
type HNSWRaBitQIndexParams struct {
	Metric         MetricType
	TotalBits      int
	NumClusters    int
	SampleCount    int
	M              int
	EFConstruction int
}

func NewHNSWRaBitQIndexParams(metric MetricType) HNSWRaBitQIndexParams {
	return HNSWRaBitQIndexParams{
		Metric:         metric,
		TotalBits:      DefaultRaBitQTotalBits,
		NumClusters:    DefaultRaBitQNumClusters,
		M:              DefaultHNSWM,
		EFConstruction: DefaultHNSWEFConstruction,
	}
}

func (HNSWRaBitQIndexParams) IndexType() IndexType { return IndexTypeHNSWRaBitQ }
func (p HNSWRaBitQIndexParams) Validate() error {
	if err := validateVectorIndexParams(
		p.IndexType(),
		vectorIndexConfig{metric: p.Metric, quantize: QuantizeTypeRaBitQ},
	); err != nil {
		return err
	}
	if p.M <= 0 || p.M > MaxHNSWM || p.EFConstruction < p.M {
		return invalidArgument("validate HNSW RaBitQ index params", "M must be in [1, %d] and EFConstruction must be at least M", MaxHNSWM)
	}
	if p.TotalBits <= 0 {
		return invalidArgument("validate HNSW RaBitQ index params", "TotalBits must be positive")
	}
	if p.NumClusters <= 0 {
		return invalidArgument("validate HNSW RaBitQ index params", "NumClusters must be positive")
	}
	if p.SampleCount < 0 {
		return invalidArgument("validate HNSW RaBitQ index params", "SampleCount cannot be negative")
	}
	return nil
}
func (p HNSWRaBitQIndexParams) cloneIndexParams() IndexParams { return p }
func (p HNSWRaBitQIndexParams) vectorConfig() vectorIndexConfig {
	return vectorIndexConfig{metric: p.Metric, quantize: QuantizeTypeRaBitQ}
}

// IVFIndexParams configures an inverted-file vector index.
type IVFIndexParams struct {
	Metric      MetricType
	NList       int
	NIterations int
	UseSOAR     bool
	Quantize    QuantizeType
	Quantizer   QuantizerParams
}

func NewIVFIndexParams(metric MetricType) IVFIndexParams {
	return IVFIndexParams{
		Metric:      metric,
		NList:       DefaultIVFNList,
		NIterations: DefaultIVFNIterations,
	}
}

func (IVFIndexParams) IndexType() IndexType { return IndexTypeIVF }
func (p IVFIndexParams) Validate() error {
	if err := validateVectorIndexParams(p.IndexType(), p.vectorConfig()); err != nil {
		return err
	}
	if p.NList <= 0 {
		return invalidArgument("validate IVF index params", "NList must be positive")
	}
	if p.NIterations <= 0 {
		return invalidArgument("validate IVF index params", "NIterations must be positive")
	}
	return nil
}
func (p IVFIndexParams) cloneIndexParams() IndexParams { return p }
func (p IVFIndexParams) vectorConfig() vectorIndexConfig {
	return vectorIndexConfig{p.Metric, p.Quantize, p.Quantizer}
}

// DiskANNIndexParams configures a disk-backed graph index.
type DiskANNIndexParams struct {
	Metric    MetricType
	MaxDegree int
	ListSize  int
	PQChunks  int
	Quantize  QuantizeType
	Quantizer QuantizerParams
}

func NewDiskANNIndexParams(metric MetricType) DiskANNIndexParams {
	return DiskANNIndexParams{
		Metric:    metric,
		MaxDegree: DefaultDiskANNMaxDegree,
		ListSize:  DefaultDiskANNListSize,
		PQChunks:  DefaultDiskANNPQChunks,
	}
}

func (DiskANNIndexParams) IndexType() IndexType { return IndexTypeDiskANN }
func (p DiskANNIndexParams) Validate() error {
	if err := validateVectorIndexParams(p.IndexType(), p.vectorConfig()); err != nil {
		return err
	}
	if p.MaxDegree <= 0 || p.ListSize <= 0 {
		return invalidArgument("validate DiskANN index params", "MaxDegree and ListSize must be positive")
	}
	if p.PQChunks < 0 {
		return invalidArgument("validate DiskANN index params", "PQChunks cannot be negative")
	}
	return nil
}
func (p DiskANNIndexParams) cloneIndexParams() IndexParams { return p }
func (p DiskANNIndexParams) vectorConfig() vectorIndexConfig {
	return vectorIndexConfig{p.Metric, p.Quantize, p.Quantizer}
}

// VamanaIndexParams configures an in-memory Vamana graph.
type VamanaIndexParams struct {
	Metric              MetricType
	MaxDegree           int
	SearchListSize      int
	Alpha               float32
	SaturateGraph       bool
	UseContiguousMemory bool
	UseIDMap            bool
	Quantize            QuantizeType
	Quantizer           QuantizerParams
}

func NewVamanaIndexParams(metric MetricType) VamanaIndexParams {
	return VamanaIndexParams{
		Metric:         metric,
		MaxDegree:      DefaultVamanaMaxDegree,
		SearchListSize: DefaultVamanaSearchListSize,
		Alpha:          DefaultVamanaAlpha,
	}
}

func (VamanaIndexParams) IndexType() IndexType { return IndexTypeVamana }
func (p VamanaIndexParams) Validate() error {
	if err := validateVectorIndexParams(p.IndexType(), p.vectorConfig()); err != nil {
		return err
	}
	if p.MaxDegree <= 0 {
		return invalidArgument("validate Vamana index params", "MaxDegree must be positive")
	}
	if p.SearchListSize < p.MaxDegree {
		return invalidArgument("validate Vamana index params", "SearchListSize must be at least MaxDegree")
	}
	if math.IsNaN(float64(p.Alpha)) || math.IsInf(float64(p.Alpha), 0) || p.Alpha < 1 {
		return invalidArgument("validate Vamana index params", "Alpha must be finite and at least 1")
	}
	return nil
}
func (p VamanaIndexParams) cloneIndexParams() IndexParams { return p }
func (p VamanaIndexParams) vectorConfig() vectorIndexConfig {
	return vectorIndexConfig{p.Metric, p.Quantize, p.Quantizer}
}

// FTSIndexParams configures full-text tokenization and token filters.
type FTSIndexParams struct {
	Tokenizer   string
	Filters     []string
	ExtraParams string
}

// NewFTSIndexParams returns the baseline defaults: standard tokenization and
// a lowercase filter.
func NewFTSIndexParams() FTSIndexParams {
	return FTSIndexParams{Tokenizer: "standard", Filters: []string{"lowercase"}}
}

func (FTSIndexParams) IndexType() IndexType { return IndexTypeFTS }
func (p FTSIndexParams) Validate() error {
	tokenizer := p.Tokenizer
	if tokenizer == "" {
		tokenizer = "standard"
	}
	switch tokenizer {
	case "standard", "ngram", "jieba", "whitespace":
	default:
		return invalidArgument("validate FTS index params", "unknown tokenizer %q", p.Tokenizer)
	}
	for _, filter := range p.Filters {
		switch filter {
		case "lowercase", "ascii_folding", "stemmer":
		default:
			return invalidArgument("validate FTS index params", "unknown filter %q", filter)
		}
	}

	extra, err := decodeFTSExtraParams(p.ExtraParams)
	if err != nil {
		return err
	}
	if err := validateTokenizerExtraParams(tokenizer, extra); err != nil {
		return err
	}
	if slices.Contains(p.Filters, "stemmer") {
		if err := validateStemmerExtraParams(extra); err != nil {
			return err
		}
	}
	return nil
}
func (p FTSIndexParams) cloneIndexParams() IndexParams {
	p.Filters = slices.Clone(p.Filters)
	return p
}

func validateVectorIndexParams(indexType IndexType, config vectorIndexConfig) error {
	if !config.metric.Valid() || config.metric == MetricTypeUndefined {
		return invalidArgument("validate vector index params", "invalid metric %s", config.metric)
	}
	if !config.quantize.Valid() {
		return invalidArgument("validate vector index params", "invalid quantization %s", config.quantize)
	}
	if config.quantize == QuantizeTypeRaBitQ && indexType != IndexTypeHNSWRaBitQ {
		return invalidArgument("validate vector index params", "RaBitQ quantization requires an HNSW_RABITQ index")
	}
	if config.quantizer.EnableRotate && config.quantize != QuantizeTypeInt8 && config.quantize != QuantizeTypeInt4 {
		return invalidArgument("validate vector index params", "rotation is only valid with INT8 or INT4 quantization")
	}
	return nil
}

func decodeFTSExtraParams(raw string) (map[string]any, error) {
	if raw == "" {
		return map[string]any{}, nil
	}
	decoder := json.NewDecoder(strings.NewReader(raw))
	var extra map[string]any
	if err := decoder.Decode(&extra); err != nil {
		return nil, invalidArgument("validate FTS index params", "invalid ExtraParams JSON: %v", err)
	}
	if extra == nil {
		return nil, invalidArgument("validate FTS index params", "ExtraParams must be a JSON object")
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		return nil, invalidArgument("validate FTS index params", "ExtraParams contains trailing JSON")
	}
	return extra, nil
}

func validateTokenizerExtraParams(tokenizer string, extra map[string]any) error {
	switch tokenizer {
	case "standard":
		if value, ok := extra["max_token_length"]; ok {
			length, ok := jsonPositiveInteger(value)
			if !ok || length > 1_048_576 {
				return invalidArgument("validate FTS index params", "max_token_length must be an integer in [1, 1048576]")
			}
		}
	case "ngram":
		minimum, err := ngramSize(extra, "ngram_min", 2)
		if err != nil {
			return err
		}
		maximum, err := ngramSize(extra, "ngram_max", 2)
		if err != nil {
			return err
		}
		if minimum > maximum || maximum-minimum > 1 {
			return invalidArgument("validate FTS index params", "ngram_min must be <= ngram_max and differ by at most 1")
		}
		if _, exists := extra["custom_token_chars"]; exists {
			return invalidArgument("validate FTS index params", "custom_token_chars is not supported")
		}
		if value, exists := extra["token_chars"]; exists {
			values, ok := value.([]any)
			if !ok {
				return invalidArgument("validate FTS index params", "token_chars must be an array")
			}
			for _, value := range values {
				name, ok := value.(string)
				if !ok || !validTokenChar(name) {
					return invalidArgument("validate FTS index params", "invalid token_chars entry %v", value)
				}
			}
		}
	case "jieba":
		for _, key := range []string{"jieba_dict_dir", "user_dict_path"} {
			if value, exists := extra[key]; exists {
				if _, ok := value.(string); !ok {
					return invalidArgument("validate FTS index params", "%s must be a string", key)
				}
			}
		}
		if value, exists := extra["cut_mode"]; exists {
			mode, ok := value.(string)
			if !ok || (mode != "search" && mode != "mix" && mode != "full" && mode != "hmm") {
				return invalidArgument("validate FTS index params", "cut_mode must be search, mix, full, or hmm")
			}
		}
	}
	return nil
}

func ngramSize(extra map[string]any, key string, fallback int64) (int64, error) {
	value, exists := extra[key]
	if !exists {
		return fallback, nil
	}
	size, ok := jsonPositiveInteger(value)
	if !ok || size > math.MaxUint32 {
		return 0, invalidArgument("validate FTS index params", "%s must be a positive uint32", key)
	}
	return size, nil
}

func jsonPositiveInteger(value any) (int64, bool) {
	number, ok := value.(float64)
	if !ok || number < 1 || number > math.MaxInt64 || math.Trunc(number) != number {
		return 0, false
	}
	return int64(number), true
}

func validTokenChar(name string) bool {
	switch name {
	case "letter", "digit", "whitespace", "punctuation", "symbol":
		return true
	default:
		return false
	}
}

func validateStemmerExtraParams(extra map[string]any) error {
	value, exists := extra["stemmer_lang"]
	if !exists {
		return nil
	}
	language, ok := value.(string)
	if !ok || !supportedStemmerLanguages[language] {
		return invalidArgument("validate FTS index params", "unsupported stemmer language %v", value)
	}
	return nil
}

var supportedStemmerLanguages = map[string]bool{
	"arabic": true, "armenian": true, "basque": true, "catalan": true,
	"danish": true, "dutch": true, "english": true, "finnish": true,
	"french": true, "german": true, "greek": true, "hindi": true,
	"hungarian": true, "indonesian": true, "irish": true, "italian": true,
	"lithuanian": true, "lovins": true, "nepali": true, "norwegian": true,
	"porter": true, "portuguese": true, "romanian": true, "russian": true,
	"serbian": true, "spanish": true, "swedish": true, "tamil": true,
	"turkish": true, "yiddish": true,
}
