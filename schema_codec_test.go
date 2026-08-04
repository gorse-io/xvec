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
	"bytes"
	"testing"

	"github.com/stretchr/testify/require"
)

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
