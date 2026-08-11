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

package xvec

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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
	{
		err := schema.Validate()
		require.NoError(t, err)
	}
	{
		got := schema.Fields[0].DataType.ElementType()
		require.Equal(t, DataTypeString, got)
	}
	{
		got := DataTypeArrayInt32.ElementType()
		require.Equal(t, DataTypeInt32, got)
	}
	require.Nil(t, schema.Fields[0].EffectiveIndex(),
		"scalar field unexpectedly received a default index")
	{
		got := schema.Fields[2].IndexType()
		require.Equal(t, IndexTypeHNSW, got)
	}
}

func TestVectorFieldSchemaValidation(t *testing.T) {
	flatIP := NewFlatIndexParams(MetricTypeIP)
	hnswL2 := NewHNSWIndexParams(MetricTypeL2)
	ivfIP := NewIVFIndexParams(MetricTypeIP)
	rabitq := NewHNSWRaBitQIndexParams(MetricTypeMIPSL2)
	diskANN := NewDiskANNIndexParams(MetricTypeL2)
	diskANN.PQChunks = 129
	diskANNAuto := NewDiskANNIndexParams(MetricTypeL2)

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
		{Name: "quant_int4_odd", DataType: DataTypeVectorFP32, Dimension: 127, Index: FlatIndexParams{Metric: MetricTypeL2, Quantize: QuantizeTypeInt4}},
		{Name: "disk_chunks", DataType: DataTypeVectorFP32, Dimension: 128, Index: diskANN},
		{Name: "disk_auto_chunks", DataType: DataTypeVectorFP32, Dimension: 1, Index: diskANNAuto},
		{Name: "disk_int8", DataType: DataTypeVectorInt8, Dimension: 128, Index: NewDiskANNIndexParams(MetricTypeL2)},
	}
	for _, field := range tests {
		{
			err := field.Validate()
			assert.ErrorIs(t, err, ErrInvalidArgument)
		}
	}

	valid := []FieldSchema{
		{Name: "dense", DataType: DataTypeVectorFP32, Dimension: 128, Index: hnswL2},
		{Name: "quant_int4_even", DataType: DataTypeVectorFP32, Dimension: 128, Index: FlatIndexParams{Metric: MetricTypeL2, Quantize: QuantizeTypeInt4}},
		{Name: "sparse", DataType: DataTypeSparseVectorFP32, Index: flatIP},
		{Name: "rabitq", DataType: DataTypeVectorFP32, Dimension: 128, Index: NewHNSWRaBitQIndexParams(MetricTypeCosine)},
		{Name: "disk_fp16", DataType: DataTypeVectorFP16, Dimension: 128, Index: NewDiskANNIndexParams(MetricTypeL2)},
	}
	for _, field := range valid {
		{
			err := field.Validate()
			assert.NoError(t, err)
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
		{
			err := field.Validate()
			assert.ErrorIs(t, err, ErrInvalidArgument)
		}
	}
	var typedNil *HNSWIndexParams
	field := FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 8, Index: typedNil}
	{
		err := field.Validate()
		require.NoError(t, err)
	}
	require.Equal(t, IndexTypeUndefined, field.IndexType())
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
		{
			err := schema.Validate()
			assert.ErrorIs(t, err, ErrInvalidArgument)
		}
	}

	fields := make([]FieldSchema, MaxVectorFields+1)
	for index := range fields {
		fields[index] = NewVectorField(fmt.Sprintf("v%d", index), DataTypeVectorFP32, 8)
	}
	{
		err := NewCollectionSchema("vectors", fields...).Validate()
		require.ErrorIs(t, err, ErrInvalidArgument)
	}

	scalarFields := make([]FieldSchema, MaxScalarFields+1)
	for index := range scalarFields {
		scalarFields[index] = NewField(fmt.Sprintf("s%d", index), DataTypeInt32)
	}
	{
		err := NewCollectionSchema("scalars", scalarFields...).Validate()
		require.ErrorIs(t, err, ErrInvalidArgument)
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
	require.True(t, original.Filters[0] == "lowercase",
		"Clone shares FTS filter storage")
}

func TestCollectionSchemaCodecRoundTripIndexParameters(t *testing.T) {
	indexParameters := []IndexParams{
		NewFlatIndexParams(MetricTypeIP),
		NewHNSWIndexParams(MetricTypeL2),
		NewHNSWRaBitQIndexParams(MetricTypeCosine),
		NewIVFIndexParams(MetricTypeIP),
		NewDiskANNIndexParams(MetricTypeL2),
		NewVamanaIndexParams(MetricTypeCosine),
	}
	for _, parameters := range indexParameters {
		t.Run(parameters.IndexType().String(), func(t *testing.T) {
			dimension := uint32(128)
			schema := NewCollectionSchema("codec", FieldSchema{
				Name: "vector", DataType: DataTypeVectorFP32,
				Dimension: dimension, Index: parameters,
			})
			encoded, err := marshalCollectionSchema(schema)
			require.NoError(t, err)

			decoded, err := unmarshalCollectionSchema(encoded)
			require.NoError(t, err)
			require.Equal(t, schema, decoded)
		})
	}

	scalar := NewCollectionSchema("scalar",
		FieldSchema{Name: "tag", DataType: DataTypeString, Index: NewInvertIndexParams()},
		FieldSchema{Name: "text", DataType: DataTypeString, Index: NewFTSIndexParams()},
	)
	encoded, err := marshalCollectionSchema(scalar)
	require.NoError(t, err)

	decoded, err := unmarshalCollectionSchema(encoded)
	require.NoError(t, err)
	require.Equal(t, scalar, decoded)
}

func TestCollectionSchemaCodecLocksDiskANNAndVamanaValues(t *testing.T) {
	schema := NewCollectionSchema("mapping",
		FieldSchema{Name: "disk", DataType: DataTypeVectorFP32, Dimension: 8, Index: NewDiskANNIndexParams(MetricTypeL2)},
		FieldSchema{Name: "memory", DataType: DataTypeVectorFP32, Dimension: 8, Index: NewVamanaIndexParams(MetricTypeL2)},
	)
	encoded, err := marshalCollectionSchema(schema)
	require.NoError(t, err)
	require.True(t, bytes.Contains(encoded, []byte(`"name":"disk","data_type":23,"nullable":false,"dimension":8,"index_type":5`)))
	require.True(t, bytes.Contains(encoded, []byte(`"name":"memory","data_type":23,"nullable":false,"dimension":8,"index_type":6`)))
}

func TestCollectionSchemaCodecRejectsUnknownOrDamagedData(t *testing.T) {
	tests := [][]byte{
		[]byte(`{"codec_version":2,"name":"bad","fields":[],"max_docs_per_segment":1000}`),
		[]byte(`{"codec_version":1,"name":"bad","fields":[],"max_docs_per_segment":1000,"unknown":true}`),
		[]byte(`{"codec_version":1,"name":"bad","fields":[{"name":"v","data_type":23,"dimension":2,"index_type":99,"index":{}}],"max_docs_per_segment":1000}`),
		[]byte(`{"codec_version":1,"name":"bad","fields":[{"name":"v","data_type":23,"dimension":2,"index_type":3}],"max_docs_per_segment":1000}`),
	}
	for _, encoded := range tests {
		{
			_, err := unmarshalCollectionSchema(encoded)
			require.Error(t, err)
		}
	}
}

func FuzzCollectionSchemaCodec(f *testing.F) {
	schema := NewCollectionSchema("fuzzed", FieldSchema{
		Name: "vector", DataType: DataTypeVectorFP32, Dimension: 4,
		Index: NewFlatIndexParams(MetricTypeCosine),
	})
	encoded, err := marshalCollectionSchema(schema)
	require.NoError(f, err)

	f.Add(encoded)
	f.Add([]byte(`{"codec_version":1}`))
	f.Fuzz(func(t *testing.T, input []byte) {
		decoded, err := unmarshalCollectionSchema(input)
		if err != nil {
			return
		}
		{
			err := decoded.Validate()
			require.NoError(t, err)
		}

		reencoded, err := marshalCollectionSchema(decoded)
		require.NoError(t, err)
		{
			_, err := unmarshalCollectionSchema(reencoded)
			require.NoError(t, err)
		}
	})
}
