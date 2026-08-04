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
	"errors"
	"testing"
)

func TestIndexParamsDefaultsValidate(t *testing.T) {
	params := []IndexParams{
		NewInvertIndexParams(),
		NewFlatIndexParams(MetricTypeIP),
		NewHNSWIndexParams(MetricTypeL2),
		NewHNSWRaBitQIndexParams(MetricTypeCosine),
		NewIVFIndexParams(MetricTypeIP),
		NewDiskANNIndexParams(MetricTypeL2),
		NewVamanaIndexParams(MetricTypeCosine),
		NewFTSIndexParams(),
	}
	for _, params := range params {
		if err := params.Validate(); err != nil {
			t.Errorf("%T defaults: %v", params, err)
		}
	}
}

func TestIndexParamsRejectInvalidValues(t *testing.T) {
	hnsw := NewHNSWIndexParams(MetricTypeL2)
	hnsw.EFConstruction = hnsw.M - 1
	hnswLarge := NewHNSWIndexParams(MetricTypeL2)
	hnswLarge.M = MaxHNSWM + 1
	hnswLarge.EFConstruction = hnswLarge.M
	ivf := NewIVFIndexParams(MetricTypeL2)
	ivf.NList = 0
	diskANN := NewDiskANNIndexParams(MetricTypeL2)
	diskANN.PQChunks = -1
	vamana := NewVamanaIndexParams(MetricTypeL2)
	vamana.SearchListSize = vamana.MaxDegree - 1
	rotated := NewFlatIndexParams(MetricTypeL2)
	rotated.Quantizer.EnableRotate = true
	rabitq := NewFlatIndexParams(MetricTypeL2)
	rabitq.Quantize = QuantizeTypeRaBitQ

	params := []IndexParams{
		FlatIndexParams{}, hnsw, hnswLarge, ivf, diskANN, vamana, rotated, rabitq,
		HNSWRaBitQIndexParams{Metric: MetricTypeL2, TotalBits: 7, NumClusters: 16, M: 10, EFConstruction: 9},
	}
	for _, params := range params {
		if err := params.Validate(); !errors.Is(err, ErrInvalidArgument) {
			t.Errorf("%T invalid error = %v", params, err)
		}
	}
}

func TestRotationAcceptsIntegerQuantization(t *testing.T) {
	for _, quantize := range []QuantizeType{QuantizeTypeInt8, QuantizeTypeInt4} {
		params := NewFlatIndexParams(MetricTypeL2)
		params.Quantize = quantize
		params.Quantizer.EnableRotate = true
		if err := params.Validate(); err != nil {
			t.Errorf("rotated %s params: %v", quantize, err)
		}
	}
}

func TestFTSIndexParamsValidation(t *testing.T) {
	valid := []FTSIndexParams{
		NewFTSIndexParams(),
		{Tokenizer: "whitespace"},
		{Tokenizer: "standard", ExtraParams: "{\"max_token_length\":1024}"},
		{Tokenizer: "ngram", ExtraParams: "{\"ngram_min\":2,\"ngram_max\":3,\"token_chars\":[\"letter\",\"digit\"]}"},
		{Tokenizer: "jieba", ExtraParams: "{\"cut_mode\":\"hmm\"}"},
		{Tokenizer: "standard", Filters: []string{"stemmer"}, ExtraParams: "{\"stemmer_lang\":\"porter\"}"},
	}
	for _, params := range valid {
		if err := params.Validate(); err != nil {
			t.Errorf("valid FTS params %#v: %v", params, err)
		}
	}

	invalid := []FTSIndexParams{
		{Tokenizer: "unknown"},
		{Tokenizer: "standard", Filters: []string{"unknown"}},
		{Tokenizer: "standard", ExtraParams: "[]"},
		{Tokenizer: "standard", ExtraParams: "{\"max_token_length\":0}"},
		{Tokenizer: "ngram", ExtraParams: "{\"ngram_min\":1,\"ngram_max\":3}"},
		{Tokenizer: "ngram", ExtraParams: "{\"token_chars\":[\"emoji\"]}"},
		{Tokenizer: "jieba", ExtraParams: "{\"cut_mode\":\"invalid\"}"},
		{Tokenizer: "standard", Filters: []string{"stemmer"}, ExtraParams: "{\"stemmer_lang\":\"nonexistent_lang\"}"},
	}
	for _, params := range invalid {
		if err := params.Validate(); !errors.Is(err, ErrInvalidArgument) {
			t.Errorf("invalid FTS params %#v error = %v", params, err)
		}
	}
}
