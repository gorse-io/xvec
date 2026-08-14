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

package sqlengine

import (
	"bytes"
	"fmt"
	"math"
	"slices"
)

// ValueKind identifies an exact scalar storage type. Width remains part of
// the kind so schema binding cannot accidentally narrow or widen a value.
type ValueKind uint8

const (
	ValueBinary ValueKind = iota + 1
	ValueString
	ValueBool
	ValueInt32
	ValueInt64
	ValueUint32
	ValueUint64
	ValueFloat32
	ValueFloat64
)

func (k ValueKind) String() string {
	switch k {
	case ValueBinary:
		return "BINARY"
	case ValueString:
		return "STRING"
	case ValueBool:
		return "BOOL"
	case ValueInt32:
		return "INT32"
	case ValueInt64:
		return "INT64"
	case ValueUint32:
		return "UINT32"
	case ValueUint64:
		return "UINT64"
	case ValueFloat32:
		return "FLOAT"
	case ValueFloat64:
		return "DOUBLE"
	default:
		return fmt.Sprintf("ValueKind(%d)", k)
	}
}

func (k ValueKind) valid() bool { return k >= ValueBinary && k <= ValueFloat64 }

// Value is an immutable typed scalar, homogeneous array, or typed NULL. Its
// constructors clone byte and array data so a bound predicate remains safe to
// share between concurrent readers.
type Value struct {
	kind     ValueKind
	null     bool
	array    bool
	binary   []byte
	text     string
	boolean  bool
	signed   int64
	unsigned uint64
	number   float64
	elements []Value
}

// NullValue returns a typed scalar or array NULL.
func NullValue(kind ValueKind, array bool) (Value, error) {
	if !kind.valid() {
		return Value{}, fmt.Errorf("sql: invalid NULL value kind %d", kind)
	}
	return Value{kind: kind, null: true, array: array}, nil
}

func BinaryValue(value []byte) Value {
	return Value{kind: ValueBinary, binary: slices.Clone(value)}
}

func StringValue(value string) Value { return Value{kind: ValueString, text: value} }
func BoolValue(value bool) Value     { return Value{kind: ValueBool, boolean: value} }
func Int32Value(value int32) Value   { return Value{kind: ValueInt32, signed: int64(value)} }
func Int64Value(value int64) Value   { return Value{kind: ValueInt64, signed: value} }
func Uint32Value(value uint32) Value { return Value{kind: ValueUint32, unsigned: uint64(value)} }
func Uint64Value(value uint64) Value { return Value{kind: ValueUint64, unsigned: value} }

// Float32Value stores the exact value represented by value.
func Float32Value(value float32) (Value, error) {
	if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
		return Value{}, fmt.Errorf("sql: FLOAT value must be finite")
	}
	return Value{kind: ValueFloat32, number: float64(value)}, nil
}

// Float64Value rejects non-finite values, which are not valid document data.
func Float64Value(value float64) (Value, error) {
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return Value{}, fmt.Errorf("sql: DOUBLE value must be finite")
	}
	return Value{kind: ValueFloat64, number: value}, nil
}

// ArrayValue builds a homogeneous non-null array. Empty arrays retain their
// element kind. Nested arrays and null elements are deliberately unsupported.
func ArrayValue(kind ValueKind, elements ...Value) (Value, error) {
	if !kind.valid() {
		return Value{}, fmt.Errorf("sql: invalid array element kind %d", kind)
	}
	clone := make([]Value, len(elements))
	for index, element := range elements {
		if element.null || element.array || element.kind != kind {
			return Value{}, fmt.Errorf("sql: array element %d is %s, want non-null scalar %s", index, element.describe(), kind)
		}
		clone[index] = element.clone()
	}
	return Value{kind: kind, array: true, elements: clone}, nil
}

func (v Value) Kind() ValueKind { return v.kind }
func (v Value) IsNull() bool    { return v.null }
func (v Value) IsArray() bool   { return v.array }

// Len returns an array length and false for scalar or NULL values.
func (v Value) Len() (int, bool) {
	if v.null || !v.array {
		return 0, false
	}
	return len(v.elements), true
}

func (v Value) clone() Value {
	v.binary = slices.Clone(v.binary)
	if v.elements != nil {
		v.elements = make([]Value, len(v.elements))
		for index := range v.elements {
			v.elements[index] = v.elements[index].clone()
		}
	}
	return v
}

func (v Value) describe() string {
	if !v.kind.valid() {
		return "invalid"
	}
	description := v.kind.String()
	if v.array {
		description = "ARRAY_" + description
	}
	if v.null {
		description += " NULL"
	}
	return description
}

func compareValues(left, right Value) (int, error) {
	if left.null || right.null {
		return 0, fmt.Errorf("sql: cannot compare NULL values directly")
	}
	if left.array || right.array {
		return 0, fmt.Errorf("sql: comparison requires scalar values")
	}
	if left.kind != right.kind || !left.kind.valid() {
		return 0, fmt.Errorf("sql: cannot compare %s with %s", left.describe(), right.describe())
	}
	switch left.kind {
	case ValueBinary:
		return bytes.Compare(left.binary, right.binary), nil
	case ValueString:
		if left.text < right.text {
			return -1, nil
		}
		if left.text > right.text {
			return 1, nil
		}
	case ValueBool:
		if left.boolean != right.boolean {
			if !left.boolean {
				return -1, nil
			}
			return 1, nil
		}
	case ValueInt32, ValueInt64:
		if left.signed < right.signed {
			return -1, nil
		}
		if left.signed > right.signed {
			return 1, nil
		}
	case ValueUint32, ValueUint64:
		if left.unsigned < right.unsigned {
			return -1, nil
		}
		if left.unsigned > right.unsigned {
			return 1, nil
		}
	case ValueFloat32, ValueFloat64:
		if left.number < right.number {
			return -1, nil
		}
		if left.number > right.number {
			return 1, nil
		}
	default:
		return 0, fmt.Errorf("sql: invalid scalar kind %d", left.kind)
	}
	return 0, nil
}
