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

package sql

import (
	"math"
	"testing"
)

func TestValueConstructorsPreserveExactKindsAndOwnership(t *testing.T) {
	binary := []byte{1, 2}
	value := BinaryValue(binary)
	binary[0] = 9
	if value.binary[0] != 1 || value.Kind() != ValueBinary || value.IsArray() || value.IsNull() {
		t.Fatalf("binary value = %#v", value)
	}

	element := BinaryValue([]byte{3, 4})
	array, err := ArrayValue(ValueBinary, element)
	if err != nil {
		t.Fatal(err)
	}
	element.binary[0] = 8
	if length, ok := array.Len(); !ok || length != 1 || array.elements[0].binary[0] != 3 {
		t.Fatalf("array = %#v, length=%d valid=%t", array, length, ok)
	}

	null, err := NullValue(ValueString, true)
	if err != nil || !null.IsNull() || !null.IsArray() || null.Kind() != ValueString {
		t.Fatalf("null = %#v, error=%v", null, err)
	}
}

func TestValueRejectsInvalidFloatAndArrayForms(t *testing.T) {
	if _, err := Float32Value(float32(math.Inf(1))); err == nil {
		t.Fatal("infinite FLOAT succeeded")
	}
	if _, err := Float64Value(math.NaN()); err == nil {
		t.Fatal("NaN DOUBLE succeeded")
	}
	if _, err := NullValue(0, false); err == nil {
		t.Fatal("invalid null kind succeeded")
	}
	if _, err := ArrayValue(ValueInt32, Int64Value(1)); err == nil {
		t.Fatal("mixed-width array succeeded")
	}
	null, _ := NullValue(ValueInt32, false)
	if _, err := ArrayValue(ValueInt32, null); err == nil {
		t.Fatal("null array element succeeded")
	}
	array, _ := ArrayValue(ValueInt32, Int32Value(1))
	if _, err := ArrayValue(ValueInt32, array); err == nil {
		t.Fatal("nested array succeeded")
	}
}
