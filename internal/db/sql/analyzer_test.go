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
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func testFilterSchema(t *testing.T) Schema {
	t.Helper()
	schema, err := NewSchema([]Field{
		{Name: "i32", Kind: ValueInt32, Filterable: true},
		{Name: "i64", Kind: ValueInt64, Filterable: true},
		{Name: "u32", Kind: ValueUint32, Filterable: true},
		{Name: "u64", Kind: ValueUint64, Filterable: true},
		{Name: "f32", Kind: ValueFloat32, Filterable: true},
		{Name: "f64", Kind: ValueFloat64, Filterable: true},
		{Name: "name", Kind: ValueString, Nullable: true, Filterable: true},
		{Name: "data", Kind: ValueBinary, Filterable: true},
		{Name: "active", Kind: ValueBool, Filterable: true},
		{Name: "tags", Kind: ValueString, Array: true, Nullable: true, Filterable: true},
		{Name: "numbers", Kind: ValueInt32, Array: true, Filterable: true},
		{Name: "blobs", Kind: ValueBinary, Array: true, Filterable: true},
		{Name: "embedding", Filterable: false},
	})
	require.NoError(t, err)

	return schema
}

func rowResolver(values map[string]Value) Resolver {
	return func(field Field) (Value, error) {
		if value, found := values[field.Name]; found {
			return value, nil
		}
		return NullValue(field.Kind, field.Array)
	}
}

func TestBuildPlanRewritesEqualitySetsAndEvaluatesLogic(t *testing.T) {
	plan, err := BuildPlan("i32=1 OR name='x' OR i32 IN (2, 3) OR i32=4", testFilterSchema(t))
	require.NoError(t, err)
	{
		got, want := plan.Normalized(), "(i32 IN (1, 2, 3, 4) OR name = 'x')"
		require.Equal(t, want, got)
	}
	require.True(t, plan.Stats().EqualitySets == 1)

	fields := plan.Fields()
	require.Len(t, fields, 2)
	require.True(t, fields[0].Name == "i32")
	require.True(t, fields[1].Name == "name")

	for _, testCase := range []struct {
		values map[string]Value
		want   Truth
	}{
		{map[string]Value{"i32": Int32Value(3), "name": StringValue("no")}, TruthTrue},
		{map[string]Value{"i32": Int32Value(8), "name": StringValue("x")}, TruthTrue},
		{map[string]Value{"i32": Int32Value(8), "name": StringValue("no")}, TruthFalse},
	} {
		got, evalErr := plan.Evaluate(rowResolver(testCase.values))
		require.NoError(t, evalErr)
		require.Equal(t, testCase.want, got)
	}
}

func TestRewriteFilterDoesNotChangeInequalityOR(t *testing.T) {
	expression, err := ParseFilter("i32 != 1 OR i32 != 2")
	require.NoError(t, err)

	rewritten, stats := RewriteFilter(expression)
	require.True(t, stats.EqualitySets == 0)
	require.True(t, Format(rewritten) == "(i32 != 1 OR i32 != 2)")

	plan, err := BuildPlan("i32 != 1 OR i32 != 2", testFilterSchema(t))
	require.NoError(t, err)

	matched, err := plan.Match(rowResolver(map[string]Value{"i32": Int32Value(1)}))
	require.NoError(t, err)
	require.True(t, matched)
}

func TestBuildPlanContainEmptyRewritesAndConstantFolding(t *testing.T) {
	tests := []struct {
		filter  string
		explain string
		stats   PlanStats
		values  map[string]Value
		want    Truth
	}{
		{"tags CONTAIN_ALL ()", "tags IS NOT NULL", PlanStats{EmptyContainRewrites: 1}, map[string]Value{"tags": mustArray(t, ValueString)}, TruthTrue},
		{"tags NOT CONTAIN_ANY ()", "tags IS NOT NULL", PlanStats{EmptyContainRewrites: 1}, nil, TruthFalse},
		{"tags CONTAIN_ANY ()", "FALSE", PlanStats{EmptyContainRewrites: 1}, nil, TruthFalse},
		{"tags NOT CONTAIN_ALL () AND i32=1", "FALSE", PlanStats{EmptyContainRewrites: 1, ConstantFolds: 1}, map[string]Value{"i32": Int32Value(1)}, TruthFalse},
		{"tags CONTAIN_ANY () OR i32=1", "i32 = 1", PlanStats{EmptyContainRewrites: 1, ConstantFolds: 1}, map[string]Value{"i32": Int32Value(1)}, TruthTrue},
	}
	for _, testCase := range tests {
		plan, err := BuildPlan(testCase.filter, testFilterSchema(t))
		require.NoError(t, err)
		assert.Equal(t, testCase.explain, plan.Explain())
		assert.Equal(t, testCase.stats, plan.Stats())

		got, evalErr := plan.Evaluate(rowResolver(testCase.values))
		assert.False(t, evalErr != nil || got != testCase.want)
	}
}

func mustArray(t *testing.T, kind ValueKind, values ...Value) Value {
	t.Helper()
	value, err := ArrayValue(kind, values...)
	require.NoError(t, err)

	return value
}

func TestBuildPlanArrayLengthAndNull(t *testing.T) {
	plan, err := BuildPlan("array_length(numbers) >= 2 AND name IS NULL", testFilterSchema(t))
	require.NoError(t, err)
	require.True(t, plan.Explain() == "(array_length(numbers) >= 2 AND name IS NULL)")

	values := map[string]Value{
		"numbers": mustArray(t, ValueInt32, Int32Value(1), Int32Value(2)),
	}
	got, err := plan.Evaluate(rowResolver(values))
	require.NoError(t, err)
	require.Equal(t, TruthTrue, got)

	nullArray, _ := NullValue(ValueInt32, true)
	values["numbers"] = nullArray
	got, err = plan.Evaluate(rowResolver(values))
	require.NoError(t, err)
	require.Equal(t, TruthUnknown, got)
}

func TestBuildPlanBindsNumericBoundaries(t *testing.T) {
	filter := "i32=-2147483648 AND i64=-9223372036854775808 AND u32=4294967295 AND u64=18446744073709551615 AND f32=1.5F AND f64=2D"
	plan, err := BuildPlan(filter, testFilterSchema(t))
	require.NoError(t, err)

	f32, _ := Float32Value(1.5)
	f64, _ := Float64Value(2)
	values := map[string]Value{
		"i32": Int32Value(-2147483648), "i64": Int64Value(-9223372036854775808),
		"u32": Uint32Value(4294967295), "u64": Uint64Value(18446744073709551615),
		"f32": f32, "f64": f64,
	}
	matched, err := plan.Match(rowResolver(values))
	require.NoError(t, err)
	require.True(t, matched)
}

func TestBuildPlanRejectsSemanticErrorsWithPosition(t *testing.T) {
	tests := []struct {
		filter  string
		message string
	}{
		{"missing=1", "does not exist"},
		{"embedding=1", "cannot be used"},
		{"i32='x'", "integer literal"},
		{"i32=2147483648", "invalid INT32 literal"},
		{"u32=-1", "invalid UINT32 literal"},
		{"active < true", "support"},
		{"i32 LIKE '1%'", "requires a STRING"},
		{"i32 CONTAIN_ANY (1)", "requires an array"},
		{"numbers = 1", "supports only"},
		{"array_length(i32) > 1", "requires an array"},
		{"array_length(numbers, i32) > 1", "exactly one"},
		{"array_length(numbers) > '1'", "integer literal"},
		{"ARRAY_LENGTH(numbers) > 1", "not supported"},
	}
	for _, testCase := range tests {
		t.Run(testCase.filter, func(t *testing.T) {
			_, err := BuildPlan(testCase.filter, testFilterSchema(t))
			var analysisErr *AnalysisError
			require.ErrorAs(t, err, &analysisErr)
			require.True(t, analysisErr.Position.Line >= 1)
			require.True(t, analysisErr.Position.Column >= 1)
			require.Contains(t, analysisErr.Message, testCase.message)
		})
	}
}

func TestBuildPlanBinarySetsAndContain(t *testing.T) {
	plan, err := BuildPlan("data IN ('x', 'y') AND blobs CONTAIN_ALL ('x', 'z')", testFilterSchema(t))
	require.NoError(t, err)

	values := map[string]Value{
		"data":  BinaryValue([]byte("y")),
		"blobs": mustArray(t, ValueBinary, BinaryValue([]byte("x")), BinaryValue([]byte("z"))),
	}
	matched, err := plan.Match(rowResolver(values))
	require.NoError(t, err)
	require.True(t, matched)
}

func TestPlanIsSafeForConcurrentEvaluation(t *testing.T) {
	plan, err := BuildPlan("name LIKE 'user-_%' AND i32 IN (1, 2, 3)", testFilterSchema(t))
	require.NoError(t, err)

	resolver := rowResolver(map[string]Value{"name": StringValue("user-22"), "i32": Int32Value(2)})
	var wait sync.WaitGroup
	for index := 0; index < 32; index++ {
		wait.Add(1)
		go func() {
			defer wait.Done()
			for iteration := 0; iteration < 100; iteration++ {
				matched, evalErr := plan.Match(resolver)
				if !assert.NoError(t, evalErr) {
					return
				}
				if !assert.True(t, matched) {
					return
				}
			}
		}()
	}
	wait.Wait()
}

func TestNewSchemaValidationAndPlanDefensiveFields(t *testing.T) {
	{
		_, err := NewSchema([]Field{{Name: "", Kind: ValueString, Filterable: true}})
		require.Error(t, err,
			"empty field name succeeded")
	}
	{
		_, err := NewSchema([]Field{{Name: "a"}, {Name: "a"}})
		require.Error(t, err,
			"duplicate field succeeded")
	}
	{
		_, err := NewSchema([]Field{{Name: "a", Filterable: true}})
		require.Error(t, err,
			"filterable invalid kind succeeded")
	}
	{
		_, err := NewSchema([]Field{{Name: "a", Kind: ValueString, Indexed: true}})
		require.Error(t, err,
			"unfilterable indexed field succeeded")
	}

	plan, err := BuildPlan("i32=1", testFilterSchema(t))
	require.NoError(t, err)

	fields := plan.Fields()
	fields[0].Name = "changed"
	require.True(t, plan.Fields()[0].Name == "i32",
		"plan fields were not defensively copied")
}

func FuzzBuildPlan(f *testing.F) {
	schema, err := NewSchema([]Field{
		{Name: "i32", Kind: ValueInt32, Filterable: true},
		{Name: "name", Kind: ValueString, Nullable: true, Filterable: true},
		{Name: "tags", Kind: ValueString, Array: true, Nullable: true, Filterable: true},
	})
	require.NoError(f, err)

	for _, seed := range []string{
		"i32=1 OR i32=2", "name LIKE 'user-_%'", "tags CONTAIN_ALL ('a')",
		"array_length(tags) >= 1", "tags CONTAIN_ANY () OR name IS NULL",
	} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, input string) {
		plan, err := BuildPlan(input, schema)
		if err != nil {
			return
		}
		tags, err := ArrayValue(ValueString, StringValue("a"))
		require.NoError(t, err)

		values := map[string]Value{"i32": Int32Value(1), "name": StringValue("user-a"), "tags": tags}
		{
			_, err := plan.Evaluate(rowResolver(values))
			require.NoError(t, err)
		}
		require.False(t, plan.Explain() == "",
			"successful plan has empty explanation")
	})
}
