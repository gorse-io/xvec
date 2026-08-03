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
	"fmt"
	"testing"
)

func TestValidCollectionSchema(t *testing.T) {
	hnsw := NewHNSWIndexParams(MetricTypeL2)
	sparse := NewFlatIndexParams(MetricTypeIP)
	sparse.Quantize = QuantizeTypeFP16
	fts := NewFTSIndexParams()
	schema := NewCollectionSchema(
		"books",
		NewField("title", DataTypeString),
		FieldSchema{Name: "body", DataType: DataTypeString, Nullable: true, Index: fts},
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 128, Index: hnsw},
		FieldSchema{Name: "terms", DataType: DataTypeSparseVectorFP32, Index: sparse},
	)
	if err := schema.Validate(); err != nil {
		t.Fatal(err)
	}
	if got := schema.Fields[0].DataType.ElementType(); got != DataTypeString {
		t.Fatalf("ElementType = %s", got)
	}
	if got := DataTypeArrayInt32.ElementType(); got != DataTypeInt32 {
		t.Fatalf("array ElementType = %s", got)
	}
	if schema.Fields[0].EffectiveIndex() != nil {
		t.Fatal("scalar field unexpectedly received a default index")
	}
	if got := schema.Fields[2].IndexType(); got != IndexTypeHNSW {
		t.Fatalf("index type = %s", got)
	}
}

func TestVectorFieldSchemaValidation(t *testing.T) {
	flatIP := NewFlatIndexParams(MetricTypeIP)
	hnswL2 := NewHNSWIndexParams(MetricTypeL2)
	ivfIP := NewIVFIndexParams(MetricTypeIP)
	rabitq := NewHNSWRaBitQIndexParams(MetricTypeMIPSL2)
	diskANN := NewDiskANNIndexParams(MetricTypeL2)
	diskANN.PQChunks = 129

	tests := []FieldSchema{
		{Name: "zero_dim", DataType: DataTypeVectorFP32},
		{Name: "too_wide", DataType: DataTypeVectorFP32, Dimension: MaxDenseDimensions + 1},
		{Name: "unsupported", DataType: DataTypeVectorFP64, Dimension: 128},
		{Name: "sparse_dim", DataType: DataTypeSparseVectorFP32, Dimension: 1},
		{Name: "sparse_ivf", DataType: DataTypeSparseVectorFP32, Index: ivfIP},
		{Name: "sparse_l2", DataType: DataTypeSparseVectorFP32, Index: hnswL2},
		{Name: "dense_invert", DataType: DataTypeVectorFP32, Dimension: 128, Index: NewInvertIndexParams()},
		{Name: "cosine_int8", DataType: DataTypeVectorInt8, Dimension: 128, Index: NewFlatIndexParams(MetricTypeCosine)},
		{Name: "ivf_ip_int8", DataType: DataTypeVectorInt8, Dimension: 128, Index: ivfIP},
		{Name: "rabitq_dim", DataType: DataTypeVectorFP32, Dimension: 32, Index: NewHNSWRaBitQIndexParams(MetricTypeL2)},
		{Name: "rabitq_metric", DataType: DataTypeVectorFP32, Dimension: 128, Index: rabitq},
		{Name: "quant_fp16", DataType: DataTypeVectorFP16, Dimension: 128, Index: FlatIndexParams{Metric: MetricTypeL2, Quantize: QuantizeTypeFP16}},
		{Name: "disk_chunks", DataType: DataTypeVectorFP32, Dimension: 128, Index: diskANN},
	}
	for _, field := range tests {
		if err := field.Validate(); !errors.Is(err, ErrInvalidArgument) {
			t.Errorf("field %s error = %v", field.Name, err)
		}
	}

	valid := []FieldSchema{
		{Name: "dense", DataType: DataTypeVectorFP32, Dimension: 128, Index: hnswL2},
		{Name: "sparse", DataType: DataTypeSparseVectorFP32, Index: flatIP},
		{Name: "rabitq", DataType: DataTypeVectorFP32, Dimension: 128, Index: NewHNSWRaBitQIndexParams(MetricTypeCosine)},
	}
	for _, field := range valid {
		if err := field.Validate(); err != nil {
			t.Errorf("valid field %s: %v", field.Name, err)
		}
	}
}

func TestScalarFieldSchemaValidation(t *testing.T) {
	fts := NewFTSIndexParams()
	invalid := []FieldSchema{
		{Name: "bad name", DataType: DataTypeString},
		{Name: "undefined", DataType: DataTypeUndefined},
		{Name: "fts_number", DataType: DataTypeInt32, Index: fts},
		{Name: "scalar_hnsw", DataType: DataTypeString, Index: NewHNSWIndexParams(MetricTypeL2)},
	}
	for _, field := range invalid {
		if err := field.Validate(); !errors.Is(err, ErrInvalidArgument) {
			t.Errorf("field %s error = %v", field.Name, err)
		}
	}
	var typedNil *HNSWIndexParams
	field := FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 8, Index: typedNil}
	if err := field.Validate(); err != nil {
		t.Fatalf("typed nil index should be treated as absent: %v", err)
	}
	if field.IndexType() != IndexTypeUndefined {
		t.Fatalf("typed nil index type = %s", field.IndexType())
	}
}

func TestCollectionSchemaValidation(t *testing.T) {
	field := NewField("title", DataTypeString)
	tests := []CollectionSchema{
		NewCollectionSchema("ab", field),
		{Name: "books", Fields: []FieldSchema{field}},
		NewCollectionSchema("books"),
		NewCollectionSchema("books", field, field),
	}
	for _, schema := range tests {
		if err := schema.Validate(); !errors.Is(err, ErrInvalidArgument) {
			t.Errorf("schema %#v error = %v", schema, err)
		}
	}

	fields := make([]FieldSchema, MaxVectorFields+1)
	for index := range fields {
		fields[index] = NewVectorField(fmt.Sprintf("v%d", index), DataTypeVectorFP32, 8)
	}
	if err := NewCollectionSchema("vectors", fields...).Validate(); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("vector count error = %v", err)
	}
	scalarFields := make([]FieldSchema, MaxScalarFields+1)
	for index := range scalarFields {
		scalarFields[index] = NewField(fmt.Sprintf("s%d", index), DataTypeInt32)
	}
	if err := NewCollectionSchema("scalars", scalarFields...).Validate(); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("scalar count error = %v", err)
	}
}

func TestSchemaCloneIsDeep(t *testing.T) {
	fts := NewFTSIndexParams()
	schema := NewCollectionSchema("books", FieldSchema{Name: "body", DataType: DataTypeString, Index: fts})
	clone := schema.Clone()
	params := clone.Fields[0].Index.(FTSIndexParams)
	params.Filters[0] = "stemmer"
	clone.Fields[0].Index = params
	original := schema.Fields[0].Index.(FTSIndexParams)
	if original.Filters[0] != "lowercase" {
		t.Fatal("Clone shares FTS filter storage")
	}
}
