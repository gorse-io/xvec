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
	"math"
	"slices"
	"sort"
	"unicode/utf8"
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
	if _, err := document.Clone(); err != nil {
		return err
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
		_, dataType, err := cloneDocumentValue(value)
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
