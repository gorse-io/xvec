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
	"reflect"
	"regexp"
)

const (
	MaxDenseDimensions       = 20_000
	MaxSparseDimensions      = 16_384
	MaxScalarFields          = 1_024
	MaxVectorFields          = 5
	DefaultMaxDocsPerSegment = 10_000_000
	MinMaxDocsPerSegment     = 1_000
	MinRaBitQDimensions      = 64
	MaxRaBitQDimensions      = 4_095
)

var (
	collectionNamePattern = regexp.MustCompile(`^[a-zA-Z0-9_-]{3,64}$`)
	fieldNamePattern      = regexp.MustCompile(`^[a-zA-Z0-9_-]{1,32}$`)
)

// FieldSchema describes one scalar, array, dense-vector, or sparse-vector
// field. Dimension is required only for dense vectors. A nil Index is valid;
// collection creation treats it as an exact Flat/IP index for vector fields.
type FieldSchema struct {
	Name      string
	DataType  DataType
	Nullable  bool
	Dimension uint32
	Index     IndexParams
}

// NewField returns a scalar or array field schema.
func NewField(name string, dataType DataType) FieldSchema {
	return FieldSchema{Name: name, DataType: dataType}
}

// NewVectorField returns a dense-vector field schema.
func NewVectorField(name string, dataType DataType, dimension uint32) FieldSchema {
	return FieldSchema{Name: name, DataType: dataType, Dimension: dimension}
}

// IndexType reports the configured index type, or IndexTypeUndefined when no
// index is configured.
func (f FieldSchema) IndexType() IndexType {
	if indexParamsNil(f.Index) {
		return IndexTypeUndefined
	}
	return f.Index.IndexType()
}

// EffectiveIndex returns an independent configured index. Vector fields with
// no explicit index receive the baseline Flat/IP default. Scalar fields with
// no index return nil.
func (f FieldSchema) EffectiveIndex() IndexParams {
	if !indexParamsNil(f.Index) {
		return f.Index.cloneIndexParams()
	}
	if f.DataType.IsVector() {
		params := NewFlatIndexParams(MetricTypeIP)
		return params
	}
	return nil
}

// Clone returns a deep copy of f.
func (f FieldSchema) Clone() FieldSchema {
	if !indexParamsNil(f.Index) {
		f.Index = f.Index.cloneIndexParams()
	} else {
		f.Index = nil
	}
	return f
}

// Validate checks the field name, type, dimension, index parameters, and all
// supported type/index/metric/quantization combinations.
func (f FieldSchema) Validate() error {
	if !f.DataType.Valid() || f.DataType == DataTypeUndefined {
		return invalidArgument("validate field schema", "field %q has an invalid data type", f.Name)
	}
	if !fieldNamePattern.MatchString(f.Name) {
		return invalidArgument("validate field schema", "field name %q must match %s", f.Name, fieldNamePattern)
	}

	if f.DataType.IsVector() {
		if err := f.validateVectorField(); err != nil {
			return err
		}
	} else if !indexParamsNil(f.Index) {
		if err := validateIndexParams(f.Index); err != nil {
			return err
		}
		if f.Index.IndexType().IsVector() {
			return invalidArgument("validate field schema", "scalar field %q cannot use vector index %s", f.Name, f.Index.IndexType())
		}
		if f.Index.IndexType() == IndexTypeFTS && f.DataType != DataTypeString {
			return invalidArgument("validate field schema", "FTS index requires STRING, got %s", f.DataType)
		}
		if f.Index.IndexType() != IndexTypeInvert && f.Index.IndexType() != IndexTypeFTS {
			return invalidArgument("validate field schema", "unsupported scalar index %s", f.Index.IndexType())
		}
	}
	return nil
}

func (f FieldSchema) validateVectorField() error {
	isSparse := f.DataType.IsSparseVector()
	if !isSparse && (f.Dimension == 0 || f.Dimension > MaxDenseDimensions) {
		return invalidArgument("validate field schema", "dense vector field %q dimension must be in [1, %d]", f.Name, MaxDenseDimensions)
	}
	if isSparse && f.Dimension != 0 {
		return invalidArgument("validate field schema", "sparse vector field %q must not set Dimension", f.Name)
	}
	if !supportedVectorDataType(f.DataType) {
		return invalidArgument("validate field schema", "vector data type %s is not currently supported", f.DataType)
	}
	if indexParamsNil(f.Index) {
		return nil
	}
	if err := validateIndexParams(f.Index); err != nil {
		return err
	}
	config, ok := f.Index.(interface{ vectorConfig() vectorIndexConfig })
	if !ok {
		return invalidArgument("validate field schema", "vector field %q requires vector index params", f.Name)
	}
	indexType := f.Index.IndexType()
	vectorConfig := config.vectorConfig()

	if isSparse {
		if indexType != IndexTypeFlat && indexType != IndexTypeHNSW {
			return invalidArgument("validate field schema", "sparse vectors support only FLAT and HNSW indexes")
		}
		if vectorConfig.metric != MetricTypeIP {
			return invalidArgument("validate field schema", "sparse vectors support only the IP metric")
		}
	} else if !indexType.IsVector() {
		return invalidArgument("validate field schema", "dense vector field %q cannot use index %s", f.Name, indexType)
	}

	if indexType == IndexTypeHNSWRaBitQ {
		if f.Dimension < MinRaBitQDimensions || f.Dimension > MaxRaBitQDimensions {
			return invalidArgument("validate field schema", "HNSW_RABITQ dimension must be in [%d, %d]", MinRaBitQDimensions, MaxRaBitQDimensions)
		}
		if f.DataType != DataTypeVectorFP32 {
			return invalidArgument("validate field schema", "HNSW_RABITQ requires VECTOR_FP32")
		}
		if vectorConfig.metric != MetricTypeL2 && vectorConfig.metric != MetricTypeIP && vectorConfig.metric != MetricTypeCosine {
			return invalidArgument("validate field schema", "HNSW_RABITQ supports only L2, IP, and COSINE")
		}
	}

	if !quantizationSupported(f.DataType, vectorConfig.quantize) {
		return invalidArgument(
			"validate field schema",
			"data type %s does not support %s quantization",
			f.DataType,
			vectorConfig.quantize,
		)
	}
	if !isSparse && vectorConfig.quantize == QuantizeTypeInt4 && f.Dimension%2 != 0 {
		return invalidArgument("validate field schema", "INT4 quantization requires an even vector dimension")
	}
	if indexType == IndexTypeIVF && vectorConfig.metric == MetricTypeIP &&
		f.DataType != DataTypeVectorFP16 && f.DataType != DataTypeVectorFP32 {
		return invalidArgument("validate field schema", "IVF with IP requires VECTOR_FP16 or VECTOR_FP32")
	}
	if vectorConfig.metric == MetricTypeCosine &&
		f.DataType != DataTypeVectorFP16 && f.DataType != DataTypeVectorFP32 {
		return invalidArgument("validate field schema", "COSINE requires VECTOR_FP16 or VECTOR_FP32")
	}
	if chunks, ok := diskANNPQChunks(f.Index); ok && chunks > int(f.Dimension) {
		return invalidArgument("validate field schema", "DiskANN PQChunks cannot exceed vector dimension")
	}
	return nil
}

// CollectionSchema describes a collection and preserves field order.
type CollectionSchema struct {
	Name              string
	Fields            []FieldSchema
	MaxDocsPerSegment uint64
}

// NewCollectionSchema returns a deep-copied schema with the baseline segment
// size default.
func NewCollectionSchema(name string, fields ...FieldSchema) CollectionSchema {
	schema := CollectionSchema{
		Name:              name,
		MaxDocsPerSegment: DefaultMaxDocsPerSegment,
		Fields:            make([]FieldSchema, len(fields)),
	}
	for index := range fields {
		schema.Fields[index] = fields[index].Clone()
	}
	return schema
}

// Clone returns a deep copy of s.
func (s CollectionSchema) Clone() CollectionSchema {
	clone := s
	clone.Fields = make([]FieldSchema, len(s.Fields))
	for index := range s.Fields {
		clone.Fields[index] = s.Fields[index].Clone()
	}
	return clone
}

// Field returns an independent copy of the named field.
func (s CollectionSchema) Field(name string) (FieldSchema, bool) {
	for _, field := range s.Fields {
		if field.Name == name {
			return field.Clone(), true
		}
	}
	return FieldSchema{}, false
}

// Validate checks collection-level limits, duplicate names, and every field.
func (s CollectionSchema) Validate() error {
	if !collectionNamePattern.MatchString(s.Name) {
		return invalidArgument("validate collection schema", "collection name %q must match %s", s.Name, collectionNamePattern)
	}
	if len(s.Fields) == 0 {
		return invalidArgument("validate collection schema", "collection %q must have at least one field", s.Name)
	}
	if s.MaxDocsPerSegment < MinMaxDocsPerSegment {
		return invalidArgument("validate collection schema", "MaxDocsPerSegment must be at least %d", MinMaxDocsPerSegment)
	}

	names := make(map[string]struct{}, len(s.Fields))
	scalarCount := 0
	vectorCount := 0
	for _, field := range s.Fields {
		if _, exists := names[field.Name]; exists {
			return invalidArgument("validate collection schema", "duplicate field %q", field.Name)
		}
		names[field.Name] = struct{}{}
		if field.DataType.IsVector() {
			vectorCount++
		} else {
			scalarCount++
		}
		if err := field.Validate(); err != nil {
			return err
		}
	}
	if scalarCount > MaxScalarFields {
		return invalidArgument("validate collection schema", "scalar field count %d exceeds %d", scalarCount, MaxScalarFields)
	}
	if vectorCount > MaxVectorFields {
		return invalidArgument("validate collection schema", "vector field count %d exceeds %d", vectorCount, MaxVectorFields)
	}
	return nil
}

// ElementType returns the scalar element type of an array and returns t for
// non-array data types.
func (t DataType) ElementType() DataType {
	switch t {
	case DataTypeArrayBinary:
		return DataTypeBinary
	case DataTypeArrayString:
		return DataTypeString
	case DataTypeArrayBool:
		return DataTypeBool
	case DataTypeArrayInt32:
		return DataTypeInt32
	case DataTypeArrayInt64:
		return DataTypeInt64
	case DataTypeArrayUint32:
		return DataTypeUint32
	case DataTypeArrayUint64:
		return DataTypeUint64
	case DataTypeArrayFloat:
		return DataTypeFloat
	case DataTypeArrayDouble:
		return DataTypeDouble
	default:
		return t
	}
}

func validateIndexParams(params IndexParams) error {
	if indexParamsNil(params) {
		return invalidArgument("validate index params", "index params are nil")
	}
	if !params.IndexType().Valid() || params.IndexType() == IndexTypeUndefined {
		return invalidArgument("validate index params", "invalid index type %s", params.IndexType())
	}
	return params.Validate()
}

func indexParamsNil(params IndexParams) bool {
	if params == nil {
		return true
	}
	value := reflect.ValueOf(params)
	return value.Kind() == reflect.Pointer && value.IsNil()
}

func supportedVectorDataType(dataType DataType) bool {
	switch dataType {
	case DataTypeVectorFP16, DataTypeVectorFP32, DataTypeVectorInt8,
		DataTypeSparseVectorFP16, DataTypeSparseVectorFP32:
		return true
	default:
		return false
	}
}

func quantizationSupported(dataType DataType, quantize QuantizeType) bool {
	if quantize == QuantizeTypeUndefined {
		return true
	}
	switch dataType {
	case DataTypeVectorFP32:
		return quantize == QuantizeTypeFP16 || quantize == QuantizeTypeInt8 ||
			quantize == QuantizeTypeInt4 || quantize == QuantizeTypeRaBitQ
	case DataTypeSparseVectorFP32:
		return quantize == QuantizeTypeFP16
	default:
		return false
	}
}

func diskANNPQChunks(params IndexParams) (int, bool) {
	switch params := params.(type) {
	case DiskANNIndexParams:
		return params.PQChunks, true
	case *DiskANNIndexParams:
		if params != nil {
			return params.PQChunks, true
		}
	}
	return 0, false
}
