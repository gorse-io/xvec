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
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math"

	"github.com/gorse-io/zvec/internal/ailego"
	"github.com/gorse-io/zvec/internal/core"
)

type collectionVectorIndex struct {
	indexType IndexType
	metric    core.Metric
	quantize  QuantizeType
	rotate    bool
	flat      FlatIndexParams
	hnsw      HNSWIndexParams
	ivf       IVFIndexParams
}

type collectionQueryConfig struct {
	options        QueryOptions
	scaleFactor    float32
	ef             int
	nprobe         int
	prefetchOffset uint32
	prefetchLines  uint32
}

func resolveCollectionVectorIndex(field FieldSchema, op, path string) (collectionVectorIndex, error) {
	index := field.EffectiveIndex()
	if indexParamsNil(index) || !field.DataType.IsVector() {
		return collectionVectorIndex{}, invalidArgument(op, "field %q does not have a vector index", field.Name)
	}
	if err := index.Validate(); err != nil {
		return collectionVectorIndex{}, err
	}
	spec := collectionVectorIndex{indexType: index.IndexType()}
	switch value := index.(type) {
	case FlatIndexParams:
		spec.flat = value
	case *FlatIndexParams:
		if value == nil {
			return collectionVectorIndex{}, invalidArgument(op, "field %q has nil Flat index parameters", field.Name)
		}
		spec.flat = *value
	case HNSWIndexParams:
		spec.hnsw = value
	case *HNSWIndexParams:
		if value == nil {
			return collectionVectorIndex{}, invalidArgument(op, "field %q has nil HNSW index parameters", field.Name)
		}
		spec.hnsw = *value
	case IVFIndexParams:
		spec.ivf = value
	case *IVFIndexParams:
		if value == nil {
			return collectionVectorIndex{}, invalidArgument(op, "field %q has nil IVF index parameters", field.Name)
		}
		spec.ivf = *value
	default:
		return collectionVectorIndex{}, notSupported(op, path, fmt.Sprintf("index %s on field %q is not implemented", index.IndexType(), field.Name))
	}

	var metric MetricType
	switch spec.indexType {
	case IndexTypeFlat:
		metric, spec.quantize, spec.rotate = spec.flat.Metric, spec.flat.Quantize, spec.flat.Quantizer.EnableRotate
	case IndexTypeHNSW:
		metric, spec.quantize, spec.rotate = spec.hnsw.Metric, spec.hnsw.Quantize, spec.hnsw.Quantizer.EnableRotate
	case IndexTypeIVF:
		if field.DataType.IsSparseVector() {
			return collectionVectorIndex{}, invalidArgument(op, "sparse field %q cannot use IVF", field.Name)
		}
		if spec.ivf.UseSOAR {
			return collectionVectorIndex{}, notSupported(op, path, fmt.Sprintf("IVF SOAR layout on field %q is not implemented", field.Name))
		}
		metric, spec.quantize, spec.rotate = spec.ivf.Metric, spec.ivf.Quantize, spec.ivf.Quantizer.EnableRotate
	default:
		return collectionVectorIndex{}, notSupported(op, path, fmt.Sprintf("index %s on field %q is not implemented", spec.indexType, field.Name))
	}
	converted, err := toCoreMetric(metric)
	if err != nil {
		return collectionVectorIndex{}, err
	}
	spec.metric = converted
	if spec.quantize == QuantizeTypeRaBitQ {
		return collectionVectorIndex{}, notSupported(op, path, fmt.Sprintf("RaBitQ index on field %q is not implemented", field.Name))
	}
	if field.DataType.IsSparseVector() && spec.quantize != QuantizeTypeUndefined && spec.quantize != QuantizeTypeFP16 {
		return collectionVectorIndex{}, notSupported(op, path, fmt.Sprintf("%s sparse quantization on field %q is not implemented", spec.quantize, field.Name))
	}
	return spec, nil
}

func collectionQueryParams(params QueryParams, spec collectionVectorIndex) (collectionQueryConfig, error) {
	if params == nil || isNilInterface(params) {
		switch spec.indexType {
		case IndexTypeFlat:
			value := NewFlatQueryParams()
			params = value
		case IndexTypeHNSW:
			value := NewHNSWQueryParams()
			params = value
		case IndexTypeIVF:
			value := NewIVFQueryParams()
			params = value
		}
	}
	if params.IndexType() != spec.indexType {
		return collectionQueryConfig{}, invalidArgument(
			"query", "query parameters for %s cannot be used with %s", params.IndexType(), spec.indexType,
		)
	}
	if err := params.Validate(); err != nil {
		return collectionQueryConfig{}, err
	}
	switch value := params.(type) {
	case FlatQueryParams:
		return collectionQueryConfig{options: value.QueryOptions, scaleFactor: value.ScaleFactor}, nil
	case *FlatQueryParams:
		if value != nil {
			return collectionQueryConfig{options: value.QueryOptions, scaleFactor: value.ScaleFactor}, nil
		}
	case HNSWQueryParams:
		return collectionQueryConfig{
			options: value.QueryOptions, scaleFactor: 1, ef: value.EF,
			prefetchOffset: value.PrefetchOffset, prefetchLines: value.PrefetchLines,
		}, nil
	case *HNSWQueryParams:
		if value != nil {
			return collectionQueryConfig{
				options: value.QueryOptions, scaleFactor: 1, ef: value.EF,
				prefetchOffset: value.PrefetchOffset, prefetchLines: value.PrefetchLines,
			}, nil
		}
	case IVFQueryParams:
		return collectionQueryConfig{
			options: value.QueryOptions, scaleFactor: value.ScaleFactor, nprobe: value.NProbe,
		}, nil
	case *IVFQueryParams:
		if value != nil {
			return collectionQueryConfig{
				options: value.QueryOptions, scaleFactor: value.ScaleFactor, nprobe: value.NProbe,
			}, nil
		}
	}
	return collectionQueryConfig{}, invalidArgument("query", "invalid %s query parameter value", spec.indexType)
}

type collectionDenseIndex interface {
	core.DenseProvider
	core.DenseQuerySearcher
}

type collectionHNSWIndex interface {
	collectionDenseIndex
	SearchHNSW(ctx context.Context, query []float32, options core.HNSWSearchOptions) ([]core.Result, error)
}

type collectionIVFIndex interface {
	collectionDenseIndex
	SearchIVF(ctx context.Context, query []float32, options core.IVFSearchOptions) ([]core.Result, error)
}

func searchCollectionDense(
	ctx context.Context,
	schemaName string,
	field FieldSchema,
	documents []Document,
	query []float32,
	topK int,
	filter core.CandidateFilter,
	spec collectionVectorIndex,
	config collectionQueryConfig,
) ([]core.Result, error) {
	final := core.SearchOptions{TopK: topK, Radius: config.options.Radius, Filter: filter}
	if config.options.Linear || spec.indexType == IndexTypeFlat {
		index, err := buildCollectionDenseFlat(ctx, schemaName, field, documents, spec)
		if err != nil {
			return nil, err
		}
		return executeCollectionDenseSearch(ctx, index, query, final, config.options.UseRefiner, config.scaleFactor,
			func(options core.SearchOptions) ([]core.Result, error) {
				return index.SearchWithOptions(ctx, query, options)
			})
	}
	switch spec.indexType {
	case IndexTypeHNSW:
		index, err := buildCollectionDenseHNSW(ctx, schemaName, field, documents, spec)
		if err != nil {
			return nil, err
		}
		return executeCollectionDenseSearch(ctx, index, query, final, config.options.UseRefiner, config.scaleFactor,
			func(options core.SearchOptions) ([]core.Result, error) {
				return index.SearchHNSW(ctx, query, core.HNSWSearchOptions{
					SearchOptions: options, EF: config.ef,
					PrefetchOffset: config.prefetchOffset, PrefetchLines: config.prefetchLines,
				})
			})
	case IndexTypeIVF:
		index, err := buildCollectionDenseIVF(ctx, schemaName, field, documents, spec, 0)
		if err != nil {
			return nil, err
		}
		return executeCollectionDenseSearch(ctx, index, query, final, config.options.UseRefiner, config.scaleFactor,
			func(options core.SearchOptions) ([]core.Result, error) {
				return index.SearchIVF(ctx, query, core.IVFSearchOptions{SearchOptions: options, NProbe: config.nprobe})
			})
	default:
		return nil, fmt.Errorf("unsupported dense collection index %s", spec.indexType)
	}
}

func executeCollectionDenseSearch(
	ctx context.Context,
	index collectionDenseIndex,
	query []float32,
	options core.SearchOptions,
	useRefiner bool,
	scaleFactor float32,
	baseSearch func(core.SearchOptions) ([]core.Result, error),
) ([]core.Result, error) {
	if !useRefiner {
		return baseSearch(options)
	}
	candidateCount, err := core.RefinementCandidateCount(options.TopK, scaleFactor)
	if err != nil {
		return nil, err
	}
	candidates, err := baseSearch(core.SearchOptions{TopK: candidateCount, Filter: options.Filter})
	if err != nil {
		return nil, fmt.Errorf("base candidate search: %w", err)
	}
	refiner, err := core.NewOriginalVectorRefiner(index, index.Metric())
	if err != nil {
		return nil, err
	}
	return refiner.Refine(ctx, query, candidates, options)
}

func buildCollectionDenseFlat(
	ctx context.Context,
	schemaName string,
	field FieldSchema,
	documents []Document,
	spec collectionVectorIndex,
) (collectionDenseIndex, error) {
	candidates, err := collectionDenseCandidates(ctx, field, documents)
	if err != nil {
		return nil, err
	}
	if spec.quantize == QuantizeTypeUndefined {
		index, err := core.NewDenseFlatIndex(int(field.Dimension), spec.metric)
		if err != nil {
			return nil, err
		}
		for _, candidate := range candidates {
			if err := index.Add(ctx, candidate.Key, candidate.Vector); err != nil {
				return nil, err
			}
		}
		return index, nil
	}
	kind, err := toCoreQuantization(spec.quantize)
	if err != nil {
		return nil, err
	}
	reformer, err := collectionReformer(schemaName, field, spec)
	if err != nil {
		return nil, err
	}
	return core.NewScalarQuantizedFlatIndex(ctx, int(field.Dimension), spec.metric, kind, reformer, candidates)
}

func buildCollectionDenseHNSW(
	ctx context.Context,
	schemaName string,
	field FieldSchema,
	documents []Document,
	spec collectionVectorIndex,
) (collectionHNSWIndex, error) {
	candidates, err := collectionDenseCandidates(ctx, field, documents)
	if err != nil {
		return nil, err
	}
	options := core.DefaultHNSWBuildOptions(spec.metric)
	options.M = spec.hnsw.M
	options.EFConstruction = spec.hnsw.EFConstruction
	builder, err := core.NewHNSWBuilder(int(field.Dimension), options)
	if err != nil {
		return nil, err
	}
	for _, candidate := range candidates {
		if err := builder.Add(ctx, candidate.Key, candidate.Vector); err != nil {
			return nil, err
		}
	}
	base, err := builder.Build(ctx)
	if err != nil {
		return nil, err
	}
	if spec.quantize == QuantizeTypeUndefined {
		return base, nil
	}
	kind, err := toCoreQuantization(spec.quantize)
	if err != nil {
		return nil, err
	}
	reformer, err := collectionReformer(schemaName, field, spec)
	if err != nil {
		return nil, err
	}
	return core.NewScalarQuantizedHNSWIndex(ctx, base, kind, reformer)
}

func buildCollectionDenseIVF(
	ctx context.Context,
	schemaName string,
	field FieldSchema,
	documents []Document,
	spec collectionVectorIndex,
	workers int,
) (collectionIVFIndex, error) {
	candidates, err := collectionDenseCandidates(ctx, field, documents)
	if err != nil {
		return nil, err
	}
	options := core.DefaultIVFBuildOptions(spec.metric)
	options.NList = spec.ivf.NList
	options.NIterations = spec.ivf.NIterations
	options.Workers = workers
	builder, err := core.NewIVFBuilder(int(field.Dimension), options)
	if err != nil {
		return nil, err
	}
	for _, candidate := range candidates {
		if err := builder.Add(ctx, candidate.Key, candidate.Vector); err != nil {
			return nil, err
		}
	}
	base, err := builder.Build(ctx)
	if err != nil {
		return nil, err
	}
	if spec.quantize == QuantizeTypeUndefined {
		return base, nil
	}
	kind, err := toCoreQuantization(spec.quantize)
	if err != nil {
		return nil, err
	}
	reformer, err := collectionReformer(schemaName, field, spec)
	if err != nil {
		return nil, err
	}
	return core.NewScalarQuantizedIVFIndex(ctx, base, kind, reformer)
}

func buildCollectionSparseIndex(
	ctx context.Context,
	field FieldSchema,
	documents []Document,
	spec collectionVectorIndex,
	linear bool,
) (core.SparseQuerySearcher, error) {
	if linear || spec.indexType == IndexTypeFlat {
		index, err := core.NewSparseFlatIndex(spec.metric)
		if err != nil {
			return nil, err
		}
		if err := addCollectionSparseDocuments(ctx, index, field, documents, spec.quantize); err != nil {
			return nil, err
		}
		return index, nil
	}
	if spec.indexType != IndexTypeHNSW {
		return nil, fmt.Errorf("unsupported sparse collection index %s", spec.indexType)
	}
	options := core.DefaultSparseHNSWBuildOptions()
	options.M = spec.hnsw.M
	options.EFConstruction = spec.hnsw.EFConstruction
	builder, err := core.NewSparseHNSWBuilder(options)
	if err != nil {
		return nil, err
	}
	if err := addCollectionSparseDocuments(ctx, builder, field, documents, spec.quantize); err != nil {
		return nil, err
	}
	return builder.Build(ctx)
}

type sparseCollectionBuilder interface {
	AddSparse(ctx context.Context, key uint64, vector core.SparseVector) error
}

func addCollectionSparseDocuments(
	ctx context.Context,
	builder sparseCollectionBuilder,
	field FieldSchema,
	documents []Document,
	quantize QuantizeType,
) error {
	for _, document := range documents {
		if err := ctx.Err(); err != nil {
			return err
		}
		value, found := document.Fields[field.Name]
		if !found || value == nil {
			continue
		}
		vector, err := sparseValueToCore(value)
		if err != nil {
			return fmt.Errorf("document %d field %q: %w", document.DocID, field.Name, err)
		}
		if quantize == QuantizeTypeFP16 {
			vector, err = sparseFP16Vector(vector)
			if err != nil {
				return fmt.Errorf("document %d field %q: %w", document.DocID, field.Name, err)
			}
		}
		if err := builder.AddSparse(ctx, document.DocID, vector); err != nil {
			return err
		}
	}
	return nil
}

func collectionDenseCandidates(ctx context.Context, field FieldSchema, documents []Document) ([]core.Candidate, error) {
	candidates := make([]core.Candidate, 0, len(documents))
	for _, document := range documents {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		value, found := document.Fields[field.Name]
		if !found || value == nil {
			continue
		}
		vector, err := denseValueToFloat32(value)
		if err != nil {
			return nil, fmt.Errorf("document %d field %q: %w", document.DocID, field.Name, err)
		}
		candidates = append(candidates, core.Candidate{Key: document.DocID, Vector: vector})
	}
	return candidates, nil
}

func toCoreQuantization(value QuantizeType) (core.Quantization, error) {
	switch value {
	case QuantizeTypeFP16:
		return core.QuantizationFP16, nil
	case QuantizeTypeInt8:
		return core.QuantizationInt8, nil
	case QuantizeTypeInt4:
		return core.QuantizationInt4, nil
	default:
		return 0, notSupported("query", "", fmt.Sprintf("quantization %s is not implemented", value))
	}
}

func collectionReformer(schemaName string, field FieldSchema, spec collectionVectorIndex) (core.DenseReformer, error) {
	if !spec.rotate {
		return nil, nil
	}
	dimension := int(field.Dimension)
	signs := make([]byte, 4*((dimension+7)/8))
	seed := []byte(fmt.Sprintf("zvec-go-rotation-v1\x00%s\x00%s\x00%d\x00%d\x00%d", schemaName, field.Name, dimension, spec.indexType, spec.quantize))
	for offset, counter := 0, uint64(0); offset < len(signs); counter++ {
		input := make([]byte, len(seed)+8)
		copy(input, seed)
		binary.LittleEndian.PutUint64(input[len(seed):], counter)
		digest := sha256.Sum256(input)
		offset += copy(signs[offset:], digest[:])
	}
	rotator, err := core.NewFHTRotatorFromSigns(dimension, signs)
	if err != nil {
		return nil, err
	}
	return core.NewRotationReformer(rotator)
}

func sparseFP16Vector(vector core.SparseVector) (core.SparseVector, error) {
	result := core.SparseVector{
		Indices: append([]uint32(nil), vector.Indices...),
		Values:  make([]float32, len(vector.Values)),
	}
	for index, value := range vector.Values {
		converted := ailego.Float16BitsToFloat32(ailego.Float32ToFloat16Bits(value))
		if math.IsInf(float64(converted), 0) || math.IsNaN(float64(converted)) {
			return core.SparseVector{}, core.ErrQuantizationOverflow
		}
		result.Values[index] = converted
	}
	return result, nil
}

func validateSparseRefiner(config collectionQueryConfig, field FieldSchema) error {
	if config.options.UseRefiner {
		return notSupported("query", "", fmt.Sprintf("original-vector refinement is not implemented for sparse field %q", field.Name))
	}
	return nil
}

func validateCollectionVectorRepresentations(ctx context.Context, schema CollectionSchema, document Document) error {
	for _, field := range schema.Fields {
		if err := ctx.Err(); err != nil {
			return err
		}
		if !field.DataType.IsVector() {
			continue
		}
		value, found := document.Fields[field.Name]
		if !found || value == nil {
			continue
		}
		spec, err := resolveCollectionVectorIndex(field, "validate document", "")
		if err != nil {
			if errors.Is(err, ErrNotSupported) {
				continue
			}
			return err
		}
		if spec.quantize == QuantizeTypeUndefined {
			continue
		}
		if field.DataType.IsSparseVector() {
			vector, err := sparseValueToCore(value)
			if err == nil && spec.quantize == QuantizeTypeFP16 {
				_, err = sparseFP16Vector(vector)
			}
			if err != nil {
				return invalidArgument("validate document", "field %q cannot be represented by %s: %v", field.Name, spec.quantize, err)
			}
			continue
		}
		vector, err := denseValueToFloat32(value)
		if err != nil {
			return invalidArgument("validate document", "field %q: %v", field.Name, err)
		}
		kind, err := toCoreQuantization(spec.quantize)
		if err != nil {
			return err
		}
		reformer, err := collectionReformer(schema.Name, field, spec)
		if err != nil {
			return invalidArgument("validate document", "field %q reformer: %v", field.Name, err)
		}
		_, err = core.NewScalarQuantizedFlatIndex(ctx, int(field.Dimension), spec.metric, kind, reformer, []core.Candidate{{Vector: vector}})
		if err != nil {
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				return err
			}
			return invalidArgument("validate document", "field %q cannot be represented by %s: %v", field.Name, spec.quantize, err)
		}
	}
	return nil
}
