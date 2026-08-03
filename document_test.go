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
	"encoding/binary"
	"errors"
	"math"
	"reflect"
	"slices"
	"testing"

	"github.com/gorse-io/zvec/internal/ailego"
)

func TestDocumentCloneAndFieldOwnership(t *testing.T) {
	binaryValue := Binary{1, 2}
	dense := VectorFP32{1, 2}
	sparse := SparseVectorFP32{Indices: []uint32{9, 3}, Values: []float32{1, 2}}
	document, err := NewDocument("book-1", map[string]any{
		"binary": binaryValue,
		"dense":  dense,
		"sparse": sparse,
		"title":  "Go",
	})
	if err != nil {
		t.Fatal(err)
	}
	binaryValue[0], dense[0], sparse.Indices[0] = 9, 9, 99
	if got, _ := document.Field("binary"); !reflect.DeepEqual(got, Binary{1, 2}) {
		t.Fatalf("binary field = %#v", got)
	}
	if got, _ := document.Field("dense"); !reflect.DeepEqual(got, VectorFP32{1, 2}) {
		t.Fatalf("dense field = %#v", got)
	}
	if got, _ := document.Field("sparse"); !reflect.DeepEqual(got, SparseVectorFP32{Indices: []uint32{3, 9}, Values: []float32{2, 1}}) {
		t.Fatalf("canonical sparse field = %#v", got)
	}
	value, found := document.Field("dense")
	if !found {
		t.Fatal("dense field missing")
	}
	value.(VectorFP32)[0] = 77
	again, _ := document.Field("dense")
	if again.(VectorFP32)[0] != 1 {
		t.Fatal("Field returned shared storage")
	}
	if _, found := document.Field("missing"); found {
		t.Fatal("missing field was found")
	}
	if got := document.FieldNames(); !reflect.DeepEqual(got, []string{"binary", "dense", "sparse", "title"}) {
		t.Fatalf("field names = %#v", got)
	}
}

func TestDocumentRejectsAmbiguousOrInvalidValues(t *testing.T) {
	tests := []struct {
		name   string
		key    string
		fields map[string]any
	}{
		{name: "empty key", fields: map[string]any{"x": int32(1)}},
		{name: "invalid key", key: string([]byte{0xff}), fields: map[string]any{"x": int32(1)}},
		{name: "plain int", key: "key", fields: map[string]any{"x": 1}},
		{name: "ambiguous bytes", key: "key", fields: map[string]any{"x": []byte{1}}},
		{name: "plain float slice", key: "key", fields: map[string]any{"x": []float32{1}}},
		{name: "non-finite", key: "key", fields: map[string]any{"x": float32(math.NaN())}},
		{name: "invalid int4", key: "key", fields: map[string]any{"x": VectorInt4{8}}},
		{name: "duplicate sparse", key: "key", fields: map[string]any{"x": SparseVectorFP32{Indices: []uint32{1, 1}, Values: []float32{1, 2}}}},
		{name: "invalid field name", key: "key", fields: map[string]any{"bad name": int32(1)}},
	}
	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			if _, err := NewDocument(testCase.key, testCase.fields); !errors.Is(err, ErrInvalidArgument) {
				t.Fatalf("error = %v", err)
			}
		})
	}
}

func TestProjectDocumentSemantics(t *testing.T) {
	schema := NewCollectionSchema("books",
		NewField("title", DataTypeString),
		NewField("year", DataTypeInt32),
		NewVectorField("dense", DataTypeVectorFP32, 2),
		FieldSchema{Name: "sparse", DataType: DataTypeSparseVectorFP32},
	)
	document, err := NewDocument("book-1", map[string]any{
		"title": "Go", "year": int32(2026),
		"dense":  VectorFP32{1, 2},
		"sparse": SparseVectorFP32{Indices: []uint32{1}, Values: []float32{3}},
	})
	if err != nil {
		t.Fatal(err)
	}
	document.Score, document.DocID = 0.25, 42

	allScalar, err := ProjectDocument(document, schema, Projection{})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(allScalar.FieldNames(), []string{"title", "year"}) {
		t.Fatalf("all scalar = %#v", allScalar.Fields)
	}
	empty, err := ProjectDocument(document, schema, Projection{OutputFields: []string{}})
	if err != nil || len(empty.Fields) != 0 {
		t.Fatalf("empty projection = %#v, %v", empty, err)
	}
	selected, err := ProjectDocument(document, schema, Projection{OutputFields: []string{"year"}, IncludeVectors: true})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(selected.FieldNames(), []string{"dense", "sparse", "year"}) {
		t.Fatalf("selected fields = %#v", selected.FieldNames())
	}
	if selected.PrimaryKey != document.PrimaryKey || selected.Score != document.Score || selected.DocID != document.DocID {
		t.Fatalf("metadata changed: %#v", selected)
	}
	wildcard, err := ProjectDocument(document, schema, Projection{OutputFields: []string{"*"}})
	if err != nil || !reflect.DeepEqual(wildcard.FieldNames(), []string{"title", "year"}) {
		t.Fatalf("wildcard = %#v, %v", wildcard, err)
	}
	selected.Fields["dense"].(VectorFP32)[0] = 99
	if document.Fields["dense"].(VectorFP32)[0] != 1 {
		t.Fatal("projection shares vector storage")
	}
}

func TestProjectionValidation(t *testing.T) {
	schema := NewCollectionSchema("books", NewField("title", DataTypeString), NewVectorField("dense", DataTypeVectorFP32, 2))
	tests := []Projection{
		{OutputFields: []string{"missing"}},
		{OutputFields: []string{"title", "title"}},
		{OutputFields: []string{"dense"}},
		{OutputFields: []string{"*", "title"}},
	}
	for _, projection := range tests {
		if err := projection.Validate(schema); !errors.Is(err, ErrInvalidArgument) {
			t.Fatalf("projection %#v error = %v", projection, err)
		}
	}
	nilProjection := Projection{}
	clone := nilProjection.Clone()
	if clone.OutputFields != nil {
		t.Fatal("clone changed nil output fields")
	}
	emptyProjection := Projection{OutputFields: []string{}}
	clone = emptyProjection.Clone()
	if clone.OutputFields == nil || len(clone.OutputFields) != 0 {
		t.Fatal("clone changed empty output fields")
	}
}

func TestDocumentSchemaValidation(t *testing.T) {
	schema := NewCollectionSchema("books",
		NewField("title", DataTypeString),
		FieldSchema{Name: "year", DataType: DataTypeInt32, Nullable: true},
		NewVectorField("dense", DataTypeVectorFP32, 2),
	)
	valid, err := NewDocument("book-1", map[string]any{
		"title": "Go", "year": nil, "dense": VectorFP32{1, 2},
	})
	if err != nil || valid.Validate(schema) != nil {
		t.Fatalf("valid document = %#v, %v, %v", valid, err, valid.Validate(schema))
	}
	tests := []map[string]any{
		{"dense": VectorFP32{1, 2}},
		{"title": nil, "dense": VectorFP32{1, 2}},
		{"title": "Go", "dense": VectorFP32{1}},
		{"title": "Go", "dense": VectorFP64{1, 2}},
		{"title": "Go", "dense": VectorFP32{1, 2}, "missing": int32(1)},
	}
	for _, fields := range tests {
		document, err := NewDocument("book-1", fields)
		if err != nil {
			t.Fatal(err)
		}
		if err := document.Validate(schema); !errors.Is(err, ErrInvalidArgument) {
			t.Fatalf("fields %#v error = %v", fields, err)
		}
	}
}

func TestDocumentPayloadRoundTripEveryType(t *testing.T) {
	fields := map[string]any{
		"a_binary": Binary{0, 1, 2},
		"a_string": "世界",
		"a_bool":   true,
		"a_i32":    int32(-32),
		"a_i64":    int64(-64),
		"a_u32":    uint32(32),
		"a_u64":    uint64(64),
		"a_f32":    float32(1.5),
		"a_f64":    float64(2.5),
		"b_binary": BinaryArray{{1}, nil, {2, 3}},
		"b_string": StringArray{"a", "世界"},
		"b_bool":   BoolArray{true, false},
		"b_i32":    Int32Array{-1, 2},
		"b_i64":    Int64Array{-3, 4},
		"b_u32":    Uint32Array{5, 6},
		"b_u64":    Uint64Array{7, 8},
		"b_f32":    Float32Array{1.25, 2.5},
		"b_f64":    Float64Array{3.25, 4.5},
		"c_vb32":   VectorBinary32{1, 2},
		"c_vb64":   VectorBinary64{3, 4},
		"c_vf16":   VectorFP16{Float16FromFloat32(1), Float16FromFloat32(2)},
		"c_vf32":   VectorFP32{1, 2},
		"c_vf64":   VectorFP64{3, 4},
		"c_vi4":    VectorInt4{-8, 7},
		"c_vi8":    VectorInt8{-9, 10},
		"c_vi16":   VectorInt16{-11, 12},
		"d_sf16":   SparseVectorFP16{Indices: []uint32{9, 1}, Values: []Float16{Float16FromFloat32(9), Float16FromFloat32(1)}},
		"d_sf32":   SparseVectorFP32{Indices: []uint32{8, 2}, Values: []float32{8, 2}},
		"z_null":   nil,
	}
	document, err := NewDocument("key", fields)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := marshalDocumentPayload(document.Fields)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := unmarshalDocumentPayload(encoded)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(decoded, document.Fields) {
		t.Fatalf("decoded = %#v\nwant = %#v", decoded, document.Fields)
	}
	reencoded, err := marshalDocumentPayload(decoded)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(encoded, reencoded) {
		t.Fatal("document codec is not deterministic")
	}
}

func TestDocumentPayloadRejectsCorruption(t *testing.T) {
	encoded, err := marshalDocumentPayload(map[string]any{"value": VectorFP32{1, 2}})
	if err != nil {
		t.Fatal(err)
	}
	for length := 0; length < len(encoded); length++ {
		if _, err := unmarshalDocumentPayload(encoded[:length]); !errors.Is(err, errDocumentPayloadCorrupt) {
			t.Fatalf("truncation %d error = %v", length, err)
		}
	}
	tests := []func([]byte) []byte{
		func(data []byte) []byte { data[0] ^= 1; return data },
		func(data []byte) []byte { data[8] = 2; fixDocumentHeaderCRC(data); return data },
		func(data []byte) []byte { data[10] = 31; fixDocumentHeaderCRC(data); return data },
		func(data []byte) []byte { data[24] ^= 1; fixDocumentHeaderCRC(data); return data },
		func(data []byte) []byte { return append(data, 0) },
	}
	for index, mutate := range tests {
		corrupt := mutate(slices.Clone(encoded))
		if _, err := unmarshalDocumentPayload(corrupt); !errors.Is(err, errDocumentPayloadCorrupt) {
			t.Fatalf("corruption %d error = %v", index, err)
		}
	}
}

func TestDocumentPayloadRejectsNonCanonicalSparse(t *testing.T) {
	encoded, err := marshalDocumentPayload(map[string]any{
		"sparse": SparseVectorFP32{Indices: []uint32{1, 2}, Values: []float32{1, 2}},
	})
	if err != nil {
		t.Fatal(err)
	}
	dataStart := documentHeaderSize + documentFieldHeaderSize + len("sparse")
	binary.LittleEndian.PutUint32(encoded[dataStart:dataStart+4], 2)
	binary.LittleEndian.PutUint32(encoded[dataStart+4:dataStart+8], 1)
	fixDocumentPayloadCRC(encoded)
	if _, err := unmarshalDocumentPayload(encoded); !errors.Is(err, errDocumentPayloadCorrupt) {
		t.Fatalf("unsorted sparse error = %v", err)
	}
}

func FuzzDocumentPayload(f *testing.F) {
	seed, err := marshalDocumentPayload(map[string]any{
		"title": "seed", "vector": VectorFP32{1, 2},
	})
	if err != nil {
		f.Fatal(err)
	}
	f.Add(seed)
	f.Add(seed[:documentHeaderSize])
	f.Add([]byte("not a document"))
	f.Fuzz(func(t *testing.T, data []byte) {
		fields, err := unmarshalDocumentPayload(data)
		if err != nil {
			return
		}
		encoded, err := marshalDocumentPayload(fields)
		if err != nil {
			t.Fatalf("decoded fields cannot be encoded: %v", err)
		}
		if _, err := unmarshalDocumentPayload(encoded); err != nil {
			t.Fatalf("reencoded payload cannot be decoded: %v", err)
		}
	})
}

func fixDocumentPayloadCRC(encoded []byte) {
	binary.LittleEndian.PutUint32(encoded[24:28], ailego.CRC32C(encoded[documentHeaderSize:]))
	fixDocumentHeaderCRC(encoded)
}

func fixDocumentHeaderCRC(encoded []byte) {
	binary.LittleEndian.PutUint32(encoded[28:32], ailego.CRC32C(encoded[:28]))
}
