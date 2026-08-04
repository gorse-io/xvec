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

	"github.com/stretchr/testify/require"
)

func TestValueConstructorsPreserveExactKindsAndOwnership(t *testing.T) {
	binary := []byte{1, 2}
	value := BinaryValue(binary)
	binary[0] = 9
	require.True(t, value.binary[0] == 1)
	require.Equal(t, ValueBinary, value.Kind())
	require.False(t, value.IsArray())
	require.False(t, value.IsNull())

	element := BinaryValue([]byte{3, 4})
	array, err := ArrayValue(ValueBinary, element)
	require.NoError(t, err)

	element.binary[0] = 8
	{
		length, ok := array.Len()
		require.True(t, ok)
		require.True(t, length == 1)
		require.True(t, array.elements[0].binary[0] == 3)
	}

	null, err := NullValue(ValueString, true)
	require.NoError(t, err)
	require.True(t, null.IsNull())
	require.True(t, null.IsArray())
	require.Equal(t, ValueString, null.Kind())
}

func TestValueRejectsInvalidFloatAndArrayForms(t *testing.T) {
	{
		_, err := Float32Value(float32(math.Inf(1)))
		require.Error(t, err,
			"infinite FLOAT succeeded")
	}
	{
		_, err := Float64Value(math.NaN())
		require.Error(t, err,
			"NaN DOUBLE succeeded")
	}
	{
		_, err := NullValue(0, false)
		require.Error(t, err,
			"invalid null kind succeeded")
	}
	{
		_, err := ArrayValue(ValueInt32, Int64Value(1))
		require.Error(t, err,
			"mixed-width array succeeded")
	}

	null, _ := NullValue(ValueInt32, false)
	{
		_, err := ArrayValue(ValueInt32, null)
		require.Error(t, err,
			"null array element succeeded")
	}

	array, _ := ArrayValue(ValueInt32, Int32Value(1))
	{
		_, err := ArrayValue(ValueInt32, array)
		require.Error(t, err,
			"nested array succeeded")
	}
}
