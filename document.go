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

package xvec

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"slices"
	"sort"
	"unicode/utf8"

	"github.com/gorse-io/xvec/internal/ailego/hash"
)

// MaxPrimaryKeyBytes is the maximum UTF-8 primary-key size accepted by the
// native collection format.
const MaxPrimaryKeyBytes = 64 << 10

// Document is the public document and query-result model. Fields contains
// scalar, array, dense-vector, and sparse-vector values using the explicit Go
// types declared by this package. Score and DocID are populated by queries;
// writes ignore them.
type Document struct {
	PrimaryKey string
	Fields     map[string]any
	Score      float32
	DocID      uint64
}

// NewDocument clones fields and returns a document ready for validation or a
// write operation.
func NewDocument(primaryKey string, fields map[string]any) (Document, error) {
	document := Document{PrimaryKey: primaryKey, Fields: fields}
	return document.Clone()
}

// Clone returns a deep, independently mutable document.
func (d Document) Clone() (Document, error) {
	if d.PrimaryKey == "" || len(d.PrimaryKey) > MaxPrimaryKeyBytes || !utf8.ValidString(d.PrimaryKey) {
		return Document{}, invalidArgument("clone document", "primary key must be valid UTF-8 with 1 to %d bytes", MaxPrimaryKeyBytes)
	}
	clone := d
	clone.Fields = make(map[string]any, len(d.Fields))
	for name, value := range d.Fields {
		if !fieldNamePattern.MatchString(name) {
			return Document{}, invalidArgument("clone document", "field name %q must match %s", name, fieldNamePattern)
		}
		cloned, _, err := cloneDocumentValue(value)
		if err != nil {
			return Document{}, invalidArgument("clone document", "field %q: %v", name, err)
		}
		clone.Fields[name] = cloned
	}
	return clone, nil
}

// Field returns a cloned field value.
func (d Document) Field(name string) (any, bool) {
	value, found := d.Fields[name]
	if !found {
		return nil, false
	}
	clone, _, err := cloneDocumentValue(value)
	if err != nil {
		return nil, false
	}
	return clone, true
}

// FieldNames returns field names in bytewise ascending order.
func (d Document) FieldNames() []string {
	names := make([]string, 0, len(d.Fields))
	for name := range d.Fields {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// Validate checks primary-key, field presence, nullability, exact Go value
// types, and dense-vector dimensions against schema. It applies full insert and
// upsert semantics: every non-nullable schema field must be present.
func (d Document) Validate(schema CollectionSchema) error {
	return validateDocumentAgainstSchema(d, schema, false)
}

func validateDocumentAgainstSchema(document Document, schema CollectionSchema, partial bool) error {
	if document.PrimaryKey == "" || len(document.PrimaryKey) > MaxPrimaryKeyBytes || !utf8.ValidString(document.PrimaryKey) {
		return invalidArgument("clone document", "primary key must be valid UTF-8 with 1 to %d bytes", MaxPrimaryKeyBytes)
	}
	for name := range document.Fields {
		if !fieldNamePattern.MatchString(name) {
			return invalidArgument("clone document", "field name %q must match %s", name, fieldNamePattern)
		}
	}
	if err := schema.Validate(); err != nil {
		return err
	}
	for name := range document.Fields {
		if _, found := schema.Field(name); !found {
			return invalidArgument("validate document", "field %q does not exist in schema", name)
		}
	}
	for _, field := range schema.Fields {
		value, found := document.Fields[field.Name]
		if !found {
			if !partial && !field.Nullable {
				return invalidArgument("validate document", "required field %q is missing", field.Name)
			}
			continue
		}
		if value == nil {
			if !field.Nullable {
				return invalidArgument("validate document", "field %q is not nullable", field.Name)
			}
			continue
		}
		dataType, err := validateDocumentValue(value)
		if err != nil {
			return invalidArgument("validate document", "field %q: %v", field.Name, err)
		}
		if dataType != field.DataType {
			return invalidArgument("validate document", "field %q has type %s, schema requires %s", field.Name, dataType, field.DataType)
		}
		if field.DataType.IsDenseVector() {
			vector, ok := value.(DenseVector)
			if !ok || vector.Dimension() != int(field.Dimension) {
				got := 0
				if ok {
					got = vector.Dimension()
				}
				return invalidArgument("validate document", "vector field %q has dimension %d, want %d", field.Name, got, field.Dimension)
			}
		}
	}
	return nil
}

// validateDocumentValue checks common immutable/scalar and FP32 ingestion
// values without cloning them. Less common composite representations retain
// cloneDocumentValue's canonicalization and validation behavior.
func validateDocumentValue(value any) (DataType, error) {
	switch value := value.(type) {
	case nil:
		return DataTypeUndefined, nil
	case string:
		if !utf8.ValidString(value) {
			return 0, invalidArgument("clone value", "string is not valid UTF-8")
		}
		return DataTypeString, nil
	case bool:
		return DataTypeBool, nil
	case int32:
		return DataTypeInt32, nil
	case int64:
		return DataTypeInt64, nil
	case uint32:
		return DataTypeUint32, nil
	case uint64:
		return DataTypeUint64, nil
	case float32:
		if !finiteDocumentFloat(float64(value)) {
			return 0, invalidArgument("clone value", "FLOAT is not finite")
		}
		return DataTypeFloat, nil
	case float64:
		if !finiteDocumentFloat(value) {
			return 0, invalidArgument("clone value", "DOUBLE is not finite")
		}
		return DataTypeDouble, nil
	case VectorFP32:
		if err := validateFiniteFloat32s(value); err != nil {
			return 0, err
		}
		return DataTypeVectorFP32, nil
	default:
		_, dataType, err := cloneDocumentValue(value)
		return dataType, err
	}
}

// Projection describes result shaping. A nil OutputFields slice selects every
// scalar/array field, a non-nil empty slice selects none, and a non-empty slice
// selects the named scalar/array fields. IncludeVectors independently includes
// every vector field.
type Projection struct {
	OutputFields   []string
	IncludeVectors bool
}

// Clone returns an independent projection while preserving nil-versus-empty
// field selection semantics.
func (p Projection) Clone() Projection {
	p.OutputFields = slices.Clone(p.OutputFields)
	return p
}

// Validate checks output field count, duplicates, existence, and scalar type.
// The special field "*" selects every scalar field and must appear alone.
func (p Projection) Validate(schema CollectionSchema) error {
	if len(p.OutputFields) > MaxScalarFields {
		return invalidArgument("validate projection", "output field count %d exceeds %d", len(p.OutputFields), MaxScalarFields)
	}
	seen := make(map[string]struct{}, len(p.OutputFields))
	for _, name := range p.OutputFields {
		if name == "*" {
			if len(p.OutputFields) != 1 {
				return invalidArgument("validate projection", "wildcard output field must appear alone")
			}
			continue
		}
		if _, exists := seen[name]; exists {
			return invalidArgument("validate projection", "duplicate output field %q", name)
		}
		seen[name] = struct{}{}
		field, found := schema.Field(name)
		if !found {
			return invalidArgument("validate projection", "output field %q does not exist", name)
		}
		if field.DataType.IsVector() {
			return invalidArgument("validate projection", "output field %q is a vector; use IncludeVectors", name)
		}
	}
	return nil
}

// ProjectDocument applies scalar selection and vector inclusion, cloning every
// retained value and preserving primary key, score, and internal document ID.
func ProjectDocument(document Document, schema CollectionSchema, projection Projection) (Document, error) {
	if err := projection.Validate(schema); err != nil {
		return Document{}, err
	}
	selected := make(map[string]struct{}, len(projection.OutputFields))
	selectAllScalar := projection.OutputFields == nil || len(projection.OutputFields) == 1 && projection.OutputFields[0] == "*"
	if !selectAllScalar {
		for _, name := range projection.OutputFields {
			selected[name] = struct{}{}
		}
	}
	result := Document{
		PrimaryKey: document.PrimaryKey,
		Fields:     make(map[string]any),
		Score:      document.Score,
		DocID:      document.DocID,
	}
	for _, field := range schema.Fields {
		include := false
		if field.DataType.IsVector() {
			include = projection.IncludeVectors
		} else if selectAllScalar {
			include = true
		} else {
			_, include = selected[field.Name]
		}
		if !include {
			continue
		}
		value, found := document.Fields[field.Name]
		if !found {
			continue
		}
		clone, dataType, err := cloneDocumentValue(value)
		if err != nil {
			return Document{}, invalidArgument("project document", "field %q: %v", field.Name, err)
		}
		if value != nil && dataType != field.DataType {
			return Document{}, invalidArgument("project document", "field %q has type %s, schema requires %s", field.Name, dataType, field.DataType)
		}
		result.Fields[field.Name] = clone
	}
	return result, nil
}

func cloneDocumentValue(value any) (any, DataType, error) {
	switch value := value.(type) {
	case nil:
		return nil, DataTypeUndefined, nil
	case Binary:
		return Binary(slices.Clone(value)), DataTypeBinary, nil
	case string:
		if !utf8.ValidString(value) {
			return nil, 0, invalidArgument("clone value", "string is not valid UTF-8")
		}
		return value, DataTypeString, nil
	case bool:
		return value, DataTypeBool, nil
	case int32:
		return value, DataTypeInt32, nil
	case int64:
		return value, DataTypeInt64, nil
	case uint32:
		return value, DataTypeUint32, nil
	case uint64:
		return value, DataTypeUint64, nil
	case float32:
		if !finiteDocumentFloat(float64(value)) {
			return nil, 0, invalidArgument("clone value", "FLOAT is not finite")
		}
		return value, DataTypeFloat, nil
	case float64:
		if !finiteDocumentFloat(value) {
			return nil, 0, invalidArgument("clone value", "DOUBLE is not finite")
		}
		return value, DataTypeDouble, nil
	case BinaryArray:
		clone := make(BinaryArray, len(value))
		for index := range value {
			clone[index] = Binary(slices.Clone(value[index]))
		}
		return clone, DataTypeArrayBinary, nil
	case StringArray:
		for _, element := range value {
			if !utf8.ValidString(element) {
				return nil, 0, invalidArgument("clone value", "STRING array contains invalid UTF-8")
			}
		}
		return slices.Clone(value), DataTypeArrayString, nil
	case BoolArray:
		return slices.Clone(value), DataTypeArrayBool, nil
	case Int32Array:
		return slices.Clone(value), DataTypeArrayInt32, nil
	case Int64Array:
		return slices.Clone(value), DataTypeArrayInt64, nil
	case Uint32Array:
		return slices.Clone(value), DataTypeArrayUint32, nil
	case Uint64Array:
		return slices.Clone(value), DataTypeArrayUint64, nil
	case Float32Array:
		if err := validateFiniteFloat32s(value); err != nil {
			return nil, 0, err
		}
		return slices.Clone(value), DataTypeArrayFloat, nil
	case Float64Array:
		for _, element := range value {
			if !finiteDocumentFloat(element) {
				return nil, 0, invalidArgument("clone value", "DOUBLE array contains a non-finite value")
			}
		}
		return slices.Clone(value), DataTypeArrayDouble, nil
	case VectorBinary32:
		return slices.Clone(value), DataTypeVectorBinary32, nil
	case VectorBinary64:
		return slices.Clone(value), DataTypeVectorBinary64, nil
	case VectorFP16:
		for _, element := range value {
			if !finiteDocumentFloat(float64(element.Float32())) {
				return nil, 0, invalidArgument("clone value", "FP16 vector contains a non-finite value")
			}
		}
		return slices.Clone(value), DataTypeVectorFP16, nil
	case VectorFP32:
		if err := validateFiniteFloat32s(value); err != nil {
			return nil, 0, err
		}
		return slices.Clone(value), DataTypeVectorFP32, nil
	case VectorFP64:
		for _, element := range value {
			if !finiteDocumentFloat(element) {
				return nil, 0, invalidArgument("clone value", "FP64 vector contains a non-finite value")
			}
		}
		return slices.Clone(value), DataTypeVectorFP64, nil
	case VectorInt4:
		if err := value.Validate(); err != nil {
			return nil, 0, err
		}
		return slices.Clone(value), DataTypeVectorInt4, nil
	case VectorInt8:
		return slices.Clone(value), DataTypeVectorInt8, nil
	case VectorInt16:
		return slices.Clone(value), DataTypeVectorInt16, nil
	case SparseVectorFP16:
		canonical, err := value.Canonical()
		if err != nil {
			return nil, 0, err
		}
		for _, element := range canonical.Values {
			if !finiteDocumentFloat(float64(element.Float32())) {
				return nil, 0, invalidArgument("clone value", "sparse FP16 vector contains a non-finite value")
			}
		}
		return canonical, DataTypeSparseVectorFP16, nil
	case SparseVectorFP32:
		canonical, err := value.Canonical()
		if err != nil {
			return nil, 0, err
		}
		if err := validateFiniteFloat32s(canonical.Values); err != nil {
			return nil, 0, err
		}
		return canonical, DataTypeSparseVectorFP32, nil
	default:
		return nil, 0, invalidArgument("clone value", "unsupported Go type %T", value)
	}
}

func validateFiniteFloat32s(values []float32) error {
	for _, value := range values {
		if !finiteDocumentFloat(float64(value)) {
			return invalidArgument("clone value", "FLOAT value is not finite")
		}
	}
	return nil
}

func finiteDocumentFloat(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

const (
	documentCodecVersion    uint16 = 1
	documentHeaderSize             = 32
	documentFieldHeaderSize        = 16
	maxDocumentPayload             = 64 << 20
)

var (
	documentMagic             = [8]byte{'Z', 'V', 'E', 'C', 'D', 'O', 'C', 0}
	errDocumentPayloadCorrupt = errors.New("xvec: corrupt document payload")
)

func marshalDocumentPayload(fields map[string]any) ([]byte, error) {
	if len(fields) > MaxScalarFields+MaxVectorFields {
		return nil, invalidArgument("encode document", "field count %d exceeds schema limits", len(fields))
	}
	names := make([]string, 0, len(fields))
	for name := range fields {
		if !fieldNamePattern.MatchString(name) {
			return nil, invalidArgument("encode document", "field name %q must match %s", name, fieldNamePattern)
		}
		names = append(names, name)
	}
	sort.Strings(names)
	payload := make([]byte, 0)
	for _, name := range names {
		value, dataType, err := documentValueForEncoding(fields[name])
		if err != nil {
			return nil, invalidArgument("encode document", "field %q: %v", name, err)
		}
		data, count, err := encodeDocumentValue(value, dataType)
		if err != nil {
			return nil, invalidArgument("encode document", "field %q: %v", name, err)
		}
		if len(payload) > maxDocumentPayload-documentFieldHeaderSize-len(name)-len(data) {
			return nil, invalidArgument("encode document", "payload exceeds %d bytes", maxDocumentPayload)
		}
		fieldHeader := make([]byte, documentFieldHeaderSize)
		binary.LittleEndian.PutUint16(fieldHeader[:2], uint16(len(name)))
		binary.LittleEndian.PutUint16(fieldHeader[2:4], uint16(dataType))
		binary.LittleEndian.PutUint32(fieldHeader[4:8], count)
		binary.LittleEndian.PutUint64(fieldHeader[8:16], uint64(len(data)))
		payload = append(payload, fieldHeader...)
		payload = append(payload, name...)
		payload = append(payload, data...)
	}
	encoded := make([]byte, documentHeaderSize+len(payload))
	copy(encoded[:8], documentMagic[:])
	binary.LittleEndian.PutUint16(encoded[8:10], documentCodecVersion)
	binary.LittleEndian.PutUint16(encoded[10:12], documentHeaderSize)
	binary.LittleEndian.PutUint32(encoded[12:16], uint32(len(names)))
	binary.LittleEndian.PutUint64(encoded[16:24], uint64(len(payload)))
	binary.LittleEndian.PutUint32(encoded[24:28], hashutil.CRC32C(payload))
	binary.LittleEndian.PutUint32(encoded[28:32], hashutil.CRC32C(encoded[:28]))
	copy(encoded[documentHeaderSize:], payload)
	return encoded, nil
}

// documentValueForEncoding avoids an intermediate vector clone for the most
// common ingestion representation. encodeDocumentValue synchronously copies
// FP32 values into the schema payload, so retaining a second private vector is
// unnecessary. Representations that need canonicalization keep the existing
// clone path.
func documentValueForEncoding(value any) (any, DataType, error) {
	if vector, ok := value.(VectorFP32); ok {
		if err := validateFiniteFloat32s(vector); err != nil {
			return nil, 0, err
		}
		return vector, DataTypeVectorFP32, nil
	}
	return cloneDocumentValue(value)
}

func unmarshalDocumentPayload(encoded []byte) (map[string]any, error) {
	if len(encoded) < documentHeaderSize {
		return nil, fmt.Errorf("%w: shorter than header", errDocumentPayloadCorrupt)
	}
	if !bytes.Equal(encoded[:8], documentMagic[:]) {
		return nil, fmt.Errorf("%w: invalid magic", errDocumentPayloadCorrupt)
	}
	if version := binary.LittleEndian.Uint16(encoded[8:10]); version != documentCodecVersion {
		return nil, fmt.Errorf("%w: unsupported version %d", errDocumentPayloadCorrupt, version)
	}
	if size := binary.LittleEndian.Uint16(encoded[10:12]); size != documentHeaderSize {
		return nil, fmt.Errorf("%w: invalid header size %d", errDocumentPayloadCorrupt, size)
	}
	if got, want := hashutil.CRC32C(encoded[:28]), binary.LittleEndian.Uint32(encoded[28:32]); got != want {
		return nil, fmt.Errorf("%w: header checksum", errDocumentPayloadCorrupt)
	}
	fieldCount := binary.LittleEndian.Uint32(encoded[12:16])
	if fieldCount > MaxScalarFields+MaxVectorFields {
		return nil, fmt.Errorf("%w: impossible field count %d", errDocumentPayloadCorrupt, fieldCount)
	}
	payloadLength := binary.LittleEndian.Uint64(encoded[16:24])
	if payloadLength > maxDocumentPayload || payloadLength != uint64(len(encoded)-documentHeaderSize) {
		return nil, fmt.Errorf("%w: invalid payload length %d", errDocumentPayloadCorrupt, payloadLength)
	}
	payload := encoded[documentHeaderSize:]
	if got, want := hashutil.CRC32C(payload), binary.LittleEndian.Uint32(encoded[24:28]); got != want {
		return nil, fmt.Errorf("%w: payload checksum", errDocumentPayloadCorrupt)
	}
	fields := make(map[string]any, int(fieldCount))
	offset := 0
	previousName := ""
	for index := uint32(0); index < fieldCount; index++ {
		if len(payload)-offset < documentFieldHeaderSize {
			return nil, fmt.Errorf("%w: truncated field %d header", errDocumentPayloadCorrupt, index)
		}
		header := payload[offset : offset+documentFieldHeaderSize]
		nameLength := int(binary.LittleEndian.Uint16(header[:2]))
		dataType := DataType(binary.LittleEndian.Uint16(header[2:4]))
		count := binary.LittleEndian.Uint32(header[4:8])
		dataLength := binary.LittleEndian.Uint64(header[8:16])
		offset += documentFieldHeaderSize
		if nameLength == 0 || nameLength > len(payload)-offset || dataLength > uint64(len(payload)-offset-nameLength) {
			return nil, fmt.Errorf("%w: invalid field %d lengths", errDocumentPayloadCorrupt, index)
		}
		name := string(payload[offset : offset+nameLength])
		offset += nameLength
		if !fieldNamePattern.MatchString(name) || index > 0 && name <= previousName {
			return nil, fmt.Errorf("%w: invalid or unsorted field name %q", errDocumentPayloadCorrupt, name)
		}
		data := payload[offset : offset+int(dataLength)]
		offset += int(dataLength)
		value, err := decodeDocumentValue(dataType, count, data)
		if err != nil {
			return nil, fmt.Errorf("%w: field %q: %v", errDocumentPayloadCorrupt, name, err)
		}
		fields[name] = value
		previousName = name
	}
	if offset != len(payload) {
		return nil, fmt.Errorf("%w: trailing field data", errDocumentPayloadCorrupt)
	}
	return fields, nil
}

func encodeDocumentValue(value any, dataType DataType) ([]byte, uint32, error) {
	if value == nil {
		return nil, 0, nil
	}
	switch value := value.(type) {
	case Binary:
		return slices.Clone(value), 1, nil
	case string:
		return []byte(value), 1, nil
	case bool:
		if value {
			return []byte{1}, 1, nil
		}
		return []byte{0}, 1, nil
	case int32:
		return encodeUint32(uint32(value)), 1, nil
	case int64:
		return encodeUint64(uint64(value)), 1, nil
	case uint32:
		return encodeUint32(value), 1, nil
	case uint64:
		return encodeUint64(value), 1, nil
	case float32:
		return encodeUint32(math.Float32bits(value)), 1, nil
	case float64:
		return encodeUint64(math.Float64bits(value)), 1, nil
	case BinaryArray:
		return encodeByteSlices(value), uint32(len(value)), nil
	case StringArray:
		items := make([]Binary, len(value))
		for index := range value {
			items[index] = Binary(value[index])
		}
		return encodeByteSlices(items), uint32(len(value)), nil
	case BoolArray:
		data := make([]byte, len(value))
		for index, element := range value {
			if element {
				data[index] = 1
			}
		}
		return data, uint32(len(value)), nil
	case Int32Array:
		return encodeFixed32(len(value), func(index int) uint32 { return uint32(value[index]) }), uint32(len(value)), nil
	case Int64Array:
		return encodeFixed64(len(value), func(index int) uint64 { return uint64(value[index]) }), uint32(len(value)), nil
	case Uint32Array:
		return encodeFixed32(len(value), func(index int) uint32 { return value[index] }), uint32(len(value)), nil
	case Uint64Array:
		return encodeFixed64(len(value), func(index int) uint64 { return value[index] }), uint32(len(value)), nil
	case Float32Array:
		return encodeFixed32(len(value), func(index int) uint32 { return math.Float32bits(value[index]) }), uint32(len(value)), nil
	case Float64Array:
		return encodeFixed64(len(value), func(index int) uint64 { return math.Float64bits(value[index]) }), uint32(len(value)), nil
	case VectorBinary32:
		return encodeFixed32(len(value), func(index int) uint32 { return value[index] }), uint32(len(value)), nil
	case VectorBinary64:
		return encodeFixed64(len(value), func(index int) uint64 { return value[index] }), uint32(len(value)), nil
	case VectorFP16:
		return encodeFixed16(len(value), func(index int) uint16 { return uint16(value[index]) }), uint32(len(value)), nil
	case VectorFP32:
		return encodeFixed32(len(value), func(index int) uint32 { return math.Float32bits(value[index]) }), uint32(len(value)), nil
	case VectorFP64:
		return encodeFixed64(len(value), func(index int) uint64 { return math.Float64bits(value[index]) }), uint32(len(value)), nil
	case VectorInt4:
		return int8Bytes(value), uint32(len(value)), nil
	case VectorInt8:
		return int8Bytes(value), uint32(len(value)), nil
	case VectorInt16:
		return encodeFixed16(len(value), func(index int) uint16 { return uint16(value[index]) }), uint32(len(value)), nil
	case SparseVectorFP16:
		data := encodeFixed32(len(value.Indices), func(index int) uint32 { return value.Indices[index] })
		data = append(data, encodeFixed16(len(value.Values), func(index int) uint16 { return uint16(value.Values[index]) })...)
		return data, uint32(len(value.Indices)), nil
	case SparseVectorFP32:
		data := encodeFixed32(len(value.Indices), func(index int) uint32 { return value.Indices[index] })
		data = append(data, encodeFixed32(len(value.Values), func(index int) uint32 { return math.Float32bits(value.Values[index]) })...)
		return data, uint32(len(value.Indices)), nil
	default:
		return nil, 0, fmt.Errorf("type %s has unexpected Go value %T", dataType, value)
	}
}

func decodeDocumentValue(dataType DataType, count uint32, data []byte) (any, error) {
	if dataType == DataTypeUndefined {
		if count != 0 || len(data) != 0 {
			return nil, errors.New("NULL has data")
		}
		return nil, nil
	}
	if !dataType.Valid() {
		return nil, fmt.Errorf("unknown data type %d", dataType)
	}
	if !dataType.IsArray() && !dataType.IsVector() && count != 1 {
		return nil, fmt.Errorf("scalar count is %d", count)
	}
	switch dataType {
	case DataTypeBinary:
		return Binary(slices.Clone(data)), nil
	case DataTypeString:
		if !utf8.Valid(data) {
			return nil, errors.New("STRING is not valid UTF-8")
		}
		return string(data), nil
	case DataTypeBool:
		if len(data) != 1 || data[0] > 1 {
			return nil, errors.New("invalid BOOL")
		}
		return data[0] == 1, nil
	case DataTypeInt32:
		value, err := decodeOne32(data)
		return int32(value), err
	case DataTypeInt64:
		value, err := decodeOne64(data)
		return int64(value), err
	case DataTypeUint32:
		return decodeOne32(data)
	case DataTypeUint64:
		return decodeOne64(data)
	case DataTypeFloat:
		bits, err := decodeOne32(data)
		value := math.Float32frombits(bits)
		if err == nil && !finiteDocumentFloat(float64(value)) {
			err = errors.New("non-finite FLOAT")
		}
		return value, err
	case DataTypeDouble:
		bits, err := decodeOne64(data)
		value := math.Float64frombits(bits)
		if err == nil && !finiteDocumentFloat(value) {
			err = errors.New("non-finite DOUBLE")
		}
		return value, err
	case DataTypeArrayBinary:
		items, err := decodeByteSlices(count, data, false)
		return BinaryArray(items), err
	case DataTypeArrayString:
		items, err := decodeByteSlices(count, data, true)
		if err != nil {
			return nil, err
		}
		result := make(StringArray, len(items))
		for index := range items {
			result[index] = string(items[index])
		}
		return result, nil
	case DataTypeArrayBool:
		if uint64(count) != uint64(len(data)) {
			return nil, errors.New("BOOL array length mismatch")
		}
		result := make(BoolArray, len(data))
		for index, element := range data {
			if element > 1 {
				return nil, errors.New("invalid BOOL array element")
			}
			result[index] = element == 1
		}
		return result, nil
	case DataTypeArrayInt32:
		values, err := decodeFixed32(count, data)
		return Int32Array(map32(values, func(value uint32) int32 { return int32(value) })), err
	case DataTypeArrayInt64:
		values, err := decodeFixed64(count, data)
		return Int64Array(map64(values, func(value uint64) int64 { return int64(value) })), err
	case DataTypeArrayUint32:
		values, err := decodeFixed32(count, data)
		return Uint32Array(values), err
	case DataTypeArrayUint64:
		values, err := decodeFixed64(count, data)
		return Uint64Array(values), err
	case DataTypeArrayFloat:
		values, err := decodeFixed32(count, data)
		result := map32(values, math.Float32frombits)
		if err == nil {
			err = validateFiniteFloat32s(result)
		}
		return Float32Array(result), err
	case DataTypeArrayDouble:
		values, err := decodeFixed64(count, data)
		result := map64(values, math.Float64frombits)
		if err == nil {
			for _, value := range result {
				if !finiteDocumentFloat(value) {
					err = errors.New("non-finite DOUBLE array")
					break
				}
			}
		}
		return Float64Array(result), err
	case DataTypeVectorBinary32:
		values, err := decodeFixed32(count, data)
		return VectorBinary32(values), err
	case DataTypeVectorBinary64:
		values, err := decodeFixed64(count, data)
		return VectorBinary64(values), err
	case DataTypeVectorFP16:
		values, err := decodeFixed16(count, data)
		result := map16(values, func(value uint16) Float16 { return Float16(value) })
		if err == nil {
			for _, value := range result {
				if !finiteDocumentFloat(float64(value.Float32())) {
					err = errors.New("non-finite FP16 vector")
					break
				}
			}
		}
		return VectorFP16(result), err
	case DataTypeVectorFP32:
		values, err := decodeFixed32(count, data)
		result := map32(values, math.Float32frombits)
		if err == nil {
			err = validateFiniteFloat32s(result)
		}
		return VectorFP32(result), err
	case DataTypeVectorFP64:
		values, err := decodeFixed64(count, data)
		result := map64(values, math.Float64frombits)
		if err == nil {
			for _, value := range result {
				if !finiteDocumentFloat(value) {
					err = errors.New("non-finite FP64 vector")
					break
				}
			}
		}
		return VectorFP64(result), err
	case DataTypeVectorInt4, DataTypeVectorInt8:
		if uint64(count) != uint64(len(data)) {
			return nil, errors.New("integer vector length mismatch")
		}
		values := make([]int8, len(data))
		for index := range data {
			values[index] = int8(data[index])
		}
		if dataType == DataTypeVectorInt4 {
			result := VectorInt4(values)
			return result, result.Validate()
		}
		return VectorInt8(values), nil
	case DataTypeVectorInt16:
		values, err := decodeFixed16(count, data)
		return VectorInt16(map16(values, func(value uint16) int16 { return int16(value) })), err
	case DataTypeSparseVectorFP16:
		if uint64(count)*6 != uint64(len(data)) {
			return nil, errors.New("sparse FP16 length mismatch")
		}
		indices, err := decodeFixed32(count, data[:int(count)*4])
		values, valuesErr := decodeFixed16(count, data[int(count)*4:])
		if err != nil || valuesErr != nil {
			return nil, errors.Join(err, valuesErr)
		}
		if !strictlyIncreasing(indices) {
			return nil, errors.New("sparse FP16 indices are not strictly increasing")
		}
		result := SparseVectorFP16{Indices: indices, Values: map16(values, func(value uint16) Float16 { return Float16(value) })}
		clone, _, err := cloneDocumentValue(result)
		return clone, err
	case DataTypeSparseVectorFP32:
		if uint64(count)*8 != uint64(len(data)) {
			return nil, errors.New("sparse FP32 length mismatch")
		}
		indices, err := decodeFixed32(count, data[:int(count)*4])
		values, valuesErr := decodeFixed32(count, data[int(count)*4:])
		if err != nil || valuesErr != nil {
			return nil, errors.Join(err, valuesErr)
		}
		if !strictlyIncreasing(indices) {
			return nil, errors.New("sparse FP32 indices are not strictly increasing")
		}
		result := SparseVectorFP32{Indices: indices, Values: map32(values, math.Float32frombits)}
		clone, _, err := cloneDocumentValue(result)
		return clone, err
	default:
		return nil, fmt.Errorf("unsupported data type %s", dataType)
	}
}

func strictlyIncreasing(values []uint32) bool {
	for index := 1; index < len(values); index++ {
		if values[index] <= values[index-1] {
			return false
		}
	}
	return true
}

func encodeByteSlices(values BinaryArray) []byte {
	size := 0
	for _, value := range values {
		size += 4 + len(value)
	}
	data := make([]byte, 0, size)
	for _, value := range values {
		var length [4]byte
		if value == nil {
			binary.LittleEndian.PutUint32(length[:], math.MaxUint32)
			data = append(data, length[:]...)
			continue
		}
		binary.LittleEndian.PutUint32(length[:], uint32(len(value)))
		data = append(data, length[:]...)
		data = append(data, value...)
	}
	return data
}

func decodeByteSlices(count uint32, data []byte, requireUTF8 bool) ([]Binary, error) {
	if uint64(count) > uint64(len(data)/4) {
		return nil, errors.New("impossible variable array count")
	}
	result := make([]Binary, 0, int(count))
	offset := 0
	for range count {
		if len(data)-offset < 4 {
			return nil, errors.New("truncated variable array length")
		}
		encodedLength := binary.LittleEndian.Uint32(data[offset : offset+4])
		offset += 4
		if encodedLength == math.MaxUint32 {
			if requireUTF8 {
				return nil, errors.New("STRING array contains a nil element")
			}
			result = append(result, nil)
			continue
		}
		length := uint64(encodedLength)
		if length > uint64(len(data)-offset) {
			return nil, errors.New("truncated variable array value")
		}
		value := slices.Clone(data[offset : offset+int(length)])
		if requireUTF8 && !utf8.Valid(value) {
			return nil, errors.New("STRING array contains invalid UTF-8")
		}
		result = append(result, Binary(value))
		offset += int(length)
	}
	if offset != len(data) {
		return nil, errors.New("trailing variable array data")
	}
	return result, nil
}

func encodeUint32(value uint32) []byte {
	data := make([]byte, 4)
	binary.LittleEndian.PutUint32(data, value)
	return data
}

func encodeUint64(value uint64) []byte {
	data := make([]byte, 8)
	binary.LittleEndian.PutUint64(data, value)
	return data
}

func encodeFixed16(count int, value func(int) uint16) []byte {
	data := make([]byte, count*2)
	for index := range count {
		binary.LittleEndian.PutUint16(data[index*2:index*2+2], value(index))
	}
	return data
}

func encodeFixed32(count int, value func(int) uint32) []byte {
	data := make([]byte, count*4)
	for index := range count {
		binary.LittleEndian.PutUint32(data[index*4:index*4+4], value(index))
	}
	return data
}

func encodeFixed64(count int, value func(int) uint64) []byte {
	data := make([]byte, count*8)
	for index := range count {
		binary.LittleEndian.PutUint64(data[index*8:index*8+8], value(index))
	}
	return data
}

func decodeOne32(data []byte) (uint32, error) {
	if len(data) != 4 {
		return 0, errors.New("32-bit scalar length mismatch")
	}
	return binary.LittleEndian.Uint32(data), nil
}

func decodeOne64(data []byte) (uint64, error) {
	if len(data) != 8 {
		return 0, errors.New("64-bit scalar length mismatch")
	}
	return binary.LittleEndian.Uint64(data), nil
}

func decodeFixed16(count uint32, data []byte) ([]uint16, error) {
	if uint64(count)*2 != uint64(len(data)) {
		return nil, errors.New("16-bit value length mismatch")
	}
	result := make([]uint16, int(count))
	for index := range result {
		result[index] = binary.LittleEndian.Uint16(data[index*2 : index*2+2])
	}
	return result, nil
}

func decodeFixed32(count uint32, data []byte) ([]uint32, error) {
	if uint64(count)*4 != uint64(len(data)) {
		return nil, errors.New("32-bit value length mismatch")
	}
	result := make([]uint32, int(count))
	for index := range result {
		result[index] = binary.LittleEndian.Uint32(data[index*4 : index*4+4])
	}
	return result, nil
}

func decodeFixed64(count uint32, data []byte) ([]uint64, error) {
	if uint64(count)*8 != uint64(len(data)) {
		return nil, errors.New("64-bit value length mismatch")
	}
	result := make([]uint64, int(count))
	for index := range result {
		result[index] = binary.LittleEndian.Uint64(data[index*8 : index*8+8])
	}
	return result, nil
}

func int8Bytes[T ~[]int8](values T) []byte {
	data := make([]byte, len(values))
	for index := range values {
		data[index] = byte(values[index])
	}
	return data
}

func map16[T any](values []uint16, convert func(uint16) T) []T {
	result := make([]T, len(values))
	for index, value := range values {
		result[index] = convert(value)
	}
	return result
}

func map32[T any](values []uint32, convert func(uint32) T) []T {
	result := make([]T, len(values))
	for index, value := range values {
		result[index] = convert(value)
	}
	return result
}

func map64[T any](values []uint64, convert func(uint64) T) []T {
	result := make([]T, len(values))
	for index, value := range values {
		result[index] = convert(value)
	}
	return result
}
