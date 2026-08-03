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
	"reflect"
	"testing"
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
			if err != nil {
				t.Fatal(err)
			}
			decoded, err := unmarshalCollectionSchema(encoded)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(decoded, schema) {
				t.Fatalf("decoded schema = %#v, want %#v", decoded, schema)
			}
		})
	}

	scalar := NewCollectionSchema("scalar",
		FieldSchema{Name: "tag", DataType: DataTypeString, Index: NewInvertIndexParams()},
		FieldSchema{Name: "text", DataType: DataTypeString, Index: NewFTSIndexParams()},
	)
	encoded, err := marshalCollectionSchema(scalar)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := unmarshalCollectionSchema(encoded)
	if err != nil || !reflect.DeepEqual(decoded, scalar) {
		t.Fatalf("scalar round trip = %#v, %v", decoded, err)
	}
}

func TestCollectionSchemaCodecLocksDiskANNAndVamanaValues(t *testing.T) {
	schema := NewCollectionSchema("mapping",
		FieldSchema{Name: "disk", DataType: DataTypeVectorFP32, Dimension: 8, Index: NewDiskANNIndexParams(MetricTypeL2)},
		FieldSchema{Name: "memory", DataType: DataTypeVectorFP32, Dimension: 8, Index: NewVamanaIndexParams(MetricTypeL2)},
	)
	encoded, err := marshalCollectionSchema(schema)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(encoded, []byte(`"name":"disk","data_type":23,"nullable":false,"dimension":8,"index_type":5`)) {
		t.Fatalf("DiskANN mapping missing from %s", encoded)
	}
	if !bytes.Contains(encoded, []byte(`"name":"memory","data_type":23,"nullable":false,"dimension":8,"index_type":6`)) {
		t.Fatalf("Vamana mapping missing from %s", encoded)
	}
}

func TestCollectionSchemaCodecRejectsUnknownOrDamagedData(t *testing.T) {
	tests := [][]byte{
		[]byte(`{"codec_version":2,"name":"bad","fields":[],"max_docs_per_segment":1000}`),
		[]byte(`{"codec_version":1,"name":"bad","fields":[],"max_docs_per_segment":1000,"unknown":true}`),
		[]byte(`{"codec_version":1,"name":"bad","fields":[{"name":"v","data_type":23,"dimension":2,"index_type":99,"index":{}}],"max_docs_per_segment":1000}`),
		[]byte(`{"codec_version":1,"name":"bad","fields":[{"name":"v","data_type":23,"dimension":2,"index_type":3}],"max_docs_per_segment":1000}`),
	}
	for _, encoded := range tests {
		if _, err := unmarshalCollectionSchema(encoded); err == nil {
			t.Fatalf("damaged schema succeeded: %s", encoded)
		}
	}
}

func FuzzCollectionSchemaCodec(f *testing.F) {
	schema := NewCollectionSchema("fuzzed", FieldSchema{
		Name: "vector", DataType: DataTypeVectorFP32, Dimension: 4,
		Index: NewFlatIndexParams(MetricTypeCosine),
	})
	encoded, err := marshalCollectionSchema(schema)
	if err != nil {
		f.Fatal(err)
	}
	f.Add(encoded)
	f.Add([]byte(`{"codec_version":1}`))
	f.Fuzz(func(t *testing.T, input []byte) {
		decoded, err := unmarshalCollectionSchema(input)
		if err != nil {
			return
		}
		if err := decoded.Validate(); err != nil {
			t.Fatalf("successful decode produced invalid schema: %v", err)
		}
		reencoded, err := marshalCollectionSchema(decoded)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := unmarshalCollectionSchema(reencoded); err != nil {
			t.Fatalf("re-encoded schema failed: %v", err)
		}
	})
}
