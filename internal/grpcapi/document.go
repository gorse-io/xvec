// Copyright 2026-present the xvec project
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

package grpcapi

import (
	"sort"

	"github.com/gorse-io/xvec"
	xvecv1 "github.com/gorse-io/xvec/internal/proto/xvec/v1"
)

func SchemaToProto(schema xvec.CollectionSchema) (*xvecv1.CollectionSchema, error) {
	message := &xvecv1.CollectionSchema{Name: schema.Name, MaxDocsPerSegment: schema.MaxDocsPerSegment, Fields: make([]*xvecv1.FieldSchema, len(schema.Fields))}
	for i, field := range schema.Fields {
		index, err := IndexParamsToProto(field.Index)
		if err != nil {
			return nil, err
		}
		message.Fields[i] = &xvecv1.FieldSchema{Name: field.Name, DataType: xvecv1.DataType(field.DataType), Nullable: field.Nullable, Dimension: field.Dimension, Index: index}
	}
	return message, nil
}

func SchemaFromProto(message *xvecv1.CollectionSchema) (xvec.CollectionSchema, error) {
	if message == nil {
		return xvec.CollectionSchema{}, invalid("collection schema is nil")
	}
	schema := xvec.CollectionSchema{Name: message.Name, MaxDocsPerSegment: message.MaxDocsPerSegment, Fields: make([]xvec.FieldSchema, len(message.Fields))}
	for i, field := range message.Fields {
		if field == nil {
			return xvec.CollectionSchema{}, invalid("field schema %d is nil", i)
		}
		index, err := IndexParamsFromProto(field.Index)
		if err != nil {
			return xvec.CollectionSchema{}, err
		}
		schema.Fields[i] = xvec.FieldSchema{Name: field.Name, DataType: xvec.DataType(field.DataType), Nullable: field.Nullable, Dimension: field.Dimension, Index: index}
	}
	return schema, nil
}

func ProjectionToProto(projection xvec.Projection) *xvecv1.Projection {
	return &xvecv1.Projection{OutputFields: append([]string(nil), projection.OutputFields...), OutputFieldsSet: projection.OutputFields != nil, IncludeVectors: projection.IncludeVectors}
}

func ProjectionFromProto(message *xvecv1.Projection) (xvec.Projection, error) {
	if message == nil {
		return xvec.Projection{}, nil
	}
	var fields []string
	if message.OutputFieldsSet {
		fields = append([]string{}, message.OutputFields...)
	}
	return xvec.Projection{OutputFields: fields, IncludeVectors: message.IncludeVectors}, nil
}

func DocumentToProto(document xvec.Document) (*xvecv1.Document, error) {
	names := make([]string, 0, len(document.Fields))
	for name := range document.Fields {
		names = append(names, name)
	}
	sort.Strings(names)
	message := &xvecv1.Document{PrimaryKey: document.PrimaryKey, Score: document.Score, DocId: document.DocID, Fields: make([]*xvecv1.FieldEntry, 0, len(names))}
	for _, name := range names {
		value, err := valueToProto(document.Fields[name])
		if err != nil {
			return nil, err
		}
		message.Fields = append(message.Fields, &xvecv1.FieldEntry{Name: name, Value: value})
	}
	return message, nil
}

func DocumentFromProto(message *xvecv1.Document) (xvec.Document, error) {
	if message == nil {
		return xvec.Document{}, invalid("document is nil")
	}
	document := xvec.Document{PrimaryKey: message.PrimaryKey, Score: message.Score, DocID: message.DocId, Fields: make(map[string]any, len(message.Fields))}
	for i, field := range message.Fields {
		if field == nil {
			return xvec.Document{}, invalid("field entry %d is nil", i)
		}
		if _, exists := document.Fields[field.Name]; exists {
			return xvec.Document{}, invalid("duplicate field entry %q", field.Name)
		}
		value, err := valueFromProto(field.Value)
		if err != nil {
			return xvec.Document{}, err
		}
		document.Fields[field.Name] = value
	}
	return document, nil
}

func valueToProto(value any) (*xvecv1.Value, error) {
	switch v := value.(type) {
	case nil:
		return &xvecv1.Value{Kind: &xvecv1.Value_NullValue{NullValue: &xvecv1.NullValue{}}}, nil
	case xvec.Binary:
		return &xvecv1.Value{Kind: &xvecv1.Value_BinaryValue{BinaryValue: append([]byte(nil), v...)}}, nil
	case string:
		return &xvecv1.Value{Kind: &xvecv1.Value_StringValue{StringValue: v}}, nil
	case bool:
		return &xvecv1.Value{Kind: &xvecv1.Value_BoolValue{BoolValue: v}}, nil
	case int32:
		return &xvecv1.Value{Kind: &xvecv1.Value_Int32Value{Int32Value: v}}, nil
	case int64:
		return &xvecv1.Value{Kind: &xvecv1.Value_Int64Value{Int64Value: v}}, nil
	case uint32:
		return &xvecv1.Value{Kind: &xvecv1.Value_Uint32Value{Uint32Value: v}}, nil
	case uint64:
		return &xvecv1.Value{Kind: &xvecv1.Value_Uint64Value{Uint64Value: v}}, nil
	case float32:
		return &xvecv1.Value{Kind: &xvecv1.Value_FloatValue{FloatValue: v}}, nil
	case float64:
		return &xvecv1.Value{Kind: &xvecv1.Value_DoubleValue{DoubleValue: v}}, nil
	case xvec.BinaryArray:
		values := make([][]byte, len(v))
		for i := range v {
			values[i] = append([]byte(nil), v[i]...)
		}
		return &xvecv1.Value{Kind: &xvecv1.Value_BinaryArray{BinaryArray: &xvecv1.BytesArray{Values: values}}}, nil
	case xvec.StringArray:
		return &xvecv1.Value{Kind: &xvecv1.Value_StringArray{StringArray: &xvecv1.StringArray{Values: append([]string(nil), v...)}}}, nil
	case xvec.BoolArray:
		return &xvecv1.Value{Kind: &xvecv1.Value_BoolArray{BoolArray: &xvecv1.BoolArray{Values: append([]bool(nil), v...)}}}, nil
	case xvec.Int32Array:
		return &xvecv1.Value{Kind: &xvecv1.Value_Int32Array{Int32Array: &xvecv1.Int32Array{Values: append([]int32(nil), v...)}}}, nil
	case xvec.Int64Array:
		return &xvecv1.Value{Kind: &xvecv1.Value_Int64Array{Int64Array: &xvecv1.Int64Array{Values: append([]int64(nil), v...)}}}, nil
	case xvec.Uint32Array:
		return &xvecv1.Value{Kind: &xvecv1.Value_Uint32Array{Uint32Array: &xvecv1.Uint32Array{Values: append([]uint32(nil), v...)}}}, nil
	case xvec.Uint64Array:
		return &xvecv1.Value{Kind: &xvecv1.Value_Uint64Array{Uint64Array: &xvecv1.Uint64Array{Values: append([]uint64(nil), v...)}}}, nil
	case xvec.Float32Array:
		return &xvecv1.Value{Kind: &xvecv1.Value_FloatArray{FloatArray: &xvecv1.FloatArray{Values: append([]float32(nil), v...)}}}, nil
	case xvec.Float64Array:
		return &xvecv1.Value{Kind: &xvecv1.Value_DoubleArray{DoubleArray: &xvecv1.DoubleArray{Values: append([]float64(nil), v...)}}}, nil
	case xvec.VectorBinary32:
		return &xvecv1.Value{Kind: &xvecv1.Value_VectorBinary32{VectorBinary32: &xvecv1.Uint32Array{Values: append([]uint32(nil), v...)}}}, nil
	case xvec.VectorBinary64:
		return &xvecv1.Value{Kind: &xvecv1.Value_VectorBinary64{VectorBinary64: &xvecv1.Uint64Array{Values: append([]uint64(nil), v...)}}}, nil
	case xvec.VectorFP16:
		values := make([]uint32, len(v))
		for i := range v {
			values[i] = uint32(v[i])
		}
		return &xvecv1.Value{Kind: &xvecv1.Value_VectorFp16{VectorFp16: &xvecv1.Uint32Array{Values: values}}}, nil
	case xvec.VectorFP32:
		return &xvecv1.Value{Kind: &xvecv1.Value_VectorFp32{VectorFp32: &xvecv1.FloatArray{Values: append([]float32(nil), v...)}}}, nil
	case xvec.VectorFP64:
		return &xvecv1.Value{Kind: &xvecv1.Value_VectorFp64{VectorFp64: &xvecv1.DoubleArray{Values: append([]float64(nil), v...)}}}, nil
	case xvec.VectorInt4:
		return signedVectorToProto(v, 4), nil
	case xvec.VectorInt8:
		return signedVectorToProto(v, 8), nil
	case xvec.VectorInt16:
		values := make([]int32, len(v))
		for i := range v {
			values[i] = int32(v[i])
		}
		return &xvecv1.Value{Kind: &xvecv1.Value_VectorInt16{VectorInt16: &xvecv1.Int32Vector{Values: values}}}, nil
	case xvec.SparseVectorFP16:
		values := make([]uint32, len(v.Values))
		for i := range v.Values {
			values[i] = uint32(v.Values[i])
		}
		return &xvecv1.Value{Kind: &xvecv1.Value_SparseVectorFp16{SparseVectorFp16: &xvecv1.SparseFP16Vector{Indices: append([]uint32(nil), v.Indices...), Values: values}}}, nil
	case xvec.SparseVectorFP32:
		return &xvecv1.Value{Kind: &xvecv1.Value_SparseVectorFp32{SparseVectorFp32: &xvecv1.SparseFP32Vector{Indices: append([]uint32(nil), v.Indices...), Values: append([]float32(nil), v.Values...)}}}, nil
	default:
		return nil, invalid("unsupported document value %T", value)
	}
}

func signedVectorToProto[T ~int8](v []T, bits int) *xvecv1.Value {
	values := make([]int32, len(v))
	for i := range v {
		values[i] = int32(v[i])
	}
	vector := &xvecv1.Int32Vector{Values: values}
	if bits == 4 {
		return &xvecv1.Value{Kind: &xvecv1.Value_VectorInt4{VectorInt4: vector}}
	}
	return &xvecv1.Value{Kind: &xvecv1.Value_VectorInt8{VectorInt8: vector}}
}

func valueFromProto(message *xvecv1.Value) (any, error) {
	if message == nil {
		return nil, invalid("field value is nil")
	}
	switch v := message.Kind.(type) {
	case *xvecv1.Value_NullValue:
		return nil, nil
	case *xvecv1.Value_BinaryValue:
		return xvec.Binary(append([]byte(nil), v.BinaryValue...)), nil
	case *xvecv1.Value_StringValue:
		return v.StringValue, nil
	case *xvecv1.Value_BoolValue:
		return v.BoolValue, nil
	case *xvecv1.Value_Int32Value:
		return v.Int32Value, nil
	case *xvecv1.Value_Int64Value:
		return v.Int64Value, nil
	case *xvecv1.Value_Uint32Value:
		return v.Uint32Value, nil
	case *xvecv1.Value_Uint64Value:
		return v.Uint64Value, nil
	case *xvecv1.Value_FloatValue:
		return v.FloatValue, nil
	case *xvecv1.Value_DoubleValue:
		return v.DoubleValue, nil
	case *xvecv1.Value_BinaryArray:
		if v.BinaryArray == nil {
			return nil, invalid("binary array is nil")
		}
		out := make(xvec.BinaryArray, len(v.BinaryArray.Values))
		for i := range out {
			out[i] = append(xvec.Binary{}, v.BinaryArray.Values[i]...)
		}
		return out, nil
	case *xvecv1.Value_StringArray:
		if v.StringArray != nil {
			return xvec.StringArray(append([]string(nil), v.StringArray.Values...)), nil
		}
	case *xvecv1.Value_BoolArray:
		if v.BoolArray != nil {
			return xvec.BoolArray(append([]bool(nil), v.BoolArray.Values...)), nil
		}
	case *xvecv1.Value_Int32Array:
		if v.Int32Array != nil {
			return xvec.Int32Array(append([]int32(nil), v.Int32Array.Values...)), nil
		}
	case *xvecv1.Value_Int64Array:
		if v.Int64Array != nil {
			return xvec.Int64Array(append([]int64(nil), v.Int64Array.Values...)), nil
		}
	case *xvecv1.Value_Uint32Array:
		if v.Uint32Array != nil {
			return xvec.Uint32Array(append([]uint32(nil), v.Uint32Array.Values...)), nil
		}
	case *xvecv1.Value_Uint64Array:
		if v.Uint64Array != nil {
			return xvec.Uint64Array(append([]uint64(nil), v.Uint64Array.Values...)), nil
		}
	case *xvecv1.Value_FloatArray:
		if v.FloatArray != nil {
			return xvec.Float32Array(append([]float32(nil), v.FloatArray.Values...)), nil
		}
	case *xvecv1.Value_DoubleArray:
		if v.DoubleArray != nil {
			return xvec.Float64Array(append([]float64(nil), v.DoubleArray.Values...)), nil
		}
	case *xvecv1.Value_VectorBinary32:
		if v.VectorBinary32 != nil {
			return xvec.VectorBinary32(append([]uint32(nil), v.VectorBinary32.Values...)), nil
		}
	case *xvecv1.Value_VectorBinary64:
		if v.VectorBinary64 != nil {
			return xvec.VectorBinary64(append([]uint64(nil), v.VectorBinary64.Values...)), nil
		}
	case *xvecv1.Value_VectorFp16:
		if v.VectorFp16 != nil {
			out := make(xvec.VectorFP16, len(v.VectorFp16.Values))
			for i, n := range v.VectorFp16.Values {
				if n > 65535 {
					return nil, invalid("FP16 bits out of range")
				}
				out[i] = xvec.Float16(n)
			}
			return out, nil
		}
	case *xvecv1.Value_VectorFp32:
		if v.VectorFp32 != nil {
			return xvec.VectorFP32(append([]float32(nil), v.VectorFp32.Values...)), nil
		}
	case *xvecv1.Value_VectorFp64:
		if v.VectorFp64 != nil {
			return xvec.VectorFP64(append([]float64(nil), v.VectorFp64.Values...)), nil
		}
	case *xvecv1.Value_VectorInt4:
		if v.VectorInt4 != nil {
			out := make(xvec.VectorInt4, len(v.VectorInt4.Values))
			for i, n := range v.VectorInt4.Values {
				if n < -8 || n > 7 {
					return nil, invalid("INT4 value out of range")
				}
				out[i] = int8(n)
			}
			return out, nil
		}
	case *xvecv1.Value_VectorInt8:
		if v.VectorInt8 != nil {
			out := make(xvec.VectorInt8, len(v.VectorInt8.Values))
			for i, n := range v.VectorInt8.Values {
				if n < -128 || n > 127 {
					return nil, invalid("INT8 value out of range")
				}
				out[i] = int8(n)
			}
			return out, nil
		}
	case *xvecv1.Value_VectorInt16:
		if v.VectorInt16 != nil {
			out := make(xvec.VectorInt16, len(v.VectorInt16.Values))
			for i, n := range v.VectorInt16.Values {
				if n < -32768 || n > 32767 {
					return nil, invalid("INT16 value out of range")
				}
				out[i] = int16(n)
			}
			return out, nil
		}
	case *xvecv1.Value_SparseVectorFp16:
		if v.SparseVectorFp16 != nil {
			out := xvec.SparseVectorFP16{Indices: append([]uint32(nil), v.SparseVectorFp16.Indices...), Values: make([]xvec.Float16, len(v.SparseVectorFp16.Values))}
			for i, n := range v.SparseVectorFp16.Values {
				if n > 65535 {
					return nil, invalid("sparse FP16 bits out of range")
				}
				out.Values[i] = xvec.Float16(n)
			}
			return out, nil
		}
	case *xvecv1.Value_SparseVectorFp32:
		if v.SparseVectorFp32 != nil {
			return xvec.SparseVectorFP32{Indices: append([]uint32(nil), v.SparseVectorFp32.Indices...), Values: append([]float32(nil), v.SparseVectorFp32.Values...)}, nil
		}
	}
	return nil, invalid("value kind is missing")
}
