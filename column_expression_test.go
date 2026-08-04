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
	"testing"

	"github.com/stretchr/testify/require"
)

func TestColumnExpressionArithmeticAndCasts(t *testing.T) {
	schema := NewCollectionSchema("expressions",
		FieldSchema{Name: "i", DataType: DataTypeInt32},
		FieldSchema{Name: "u", DataType: DataTypeUint64},
		FieldSchema{Name: "f", DataType: DataTypeFloat},
		FieldSchema{Name: "nullable", DataType: DataTypeDouble, Nullable: true},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	fields := map[string]any{"i": int32(7), "u": uint64(math.MaxUint64), "f": float32(2.5), "nullable": nil}
	tests := []struct {
		expression string
		target     DataType
		want       any
	}{
		{"i + 2 * 3", DataTypeInt64, int64(13)},
		{"(i + 2) * 3", DataTypeInt32, int32(27)},
		{"-i / 2", DataTypeInt32, int32(-3)},
		{"f * 2 + 0.75", DataTypeFloat, float32(5.75)},
		{"9.9", DataTypeInt32, int32(9)},
		{"u + 1", DataTypeUint64, uint64(0)},
		{"nullable + 1", DataTypeDouble, nil},
		{"1e2 + i", DataTypeDouble, float64(107)},
	}
	for _, test := range tests {
		t.Run(test.expression, func(t *testing.T) {
			expression, err := parseColumnExpression(test.expression, schema)
			require.NoError(t, err)

			got, err := expression.evaluate(fields, test.target)
			require.NoError(t, err)
			require.Equal(t, test.want, got)
		})
	}
}

func TestColumnExpressionRejectsInvalidInput(t *testing.T) {
	schema := NewCollectionSchema("expression_errors",
		FieldSchema{Name: "number", DataType: DataTypeInt32},
		FieldSchema{Name: "text", DataType: DataTypeString},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	for _, input := range []string{"", "missing", "text", "number +", "(number + 1", "number $ 1", "1e"} {
		{
			_, err := parseColumnExpression(input, schema)
			require.Error(t, err)
		}
	}
	for _, input := range []string{"number / 0", "1 / (number - number)"} {
		expression, err := parseColumnExpression(input, schema)
		require.NoError(t, err)
		{
			_, err := expression.evaluate(map[string]any{"number": int32(3)}, DataTypeInt32)
			require.Error(t, err)
		}
	}
}

func FuzzColumnExpression(f *testing.F) {
	f.Add("number + 1")
	f.Add("-(number * 2.5)")
	f.Add("nullable / 0")
	f.Add("CASE WHEN number > 0 THEN 1 END")
	schema := NewCollectionSchema("fuzz_expressions",
		FieldSchema{Name: "number", DataType: DataTypeInt32},
		FieldSchema{Name: "nullable", DataType: DataTypeDouble, Nullable: true},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	fields := map[string]any{"number": int32(7), "nullable": nil}
	f.Fuzz(func(t *testing.T, input string) {
		expression, err := parseColumnExpression(input, schema)
		if err != nil {
			return
		}
		for _, target := range []DataType{DataTypeInt32, DataTypeInt64, DataTypeUint32, DataTypeUint64, DataTypeFloat, DataTypeDouble} {
			_, _ = expression.evaluate(fields, target)
		}
	})
}
