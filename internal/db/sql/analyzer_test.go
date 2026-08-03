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
	"errors"
	"strings"
	"sync"
	"testing"
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
	if err != nil {
		t.Fatal(err)
	}
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
	if err != nil {
		t.Fatal(err)
	}
	if got, want := plan.Normalized(), "(i32 IN (1, 2, 3, 4) OR name = 'x')"; got != want {
		t.Fatalf("normalized = %q, want %q", got, want)
	}
	if plan.Stats().EqualitySets != 1 {
		t.Fatalf("stats = %#v", plan.Stats())
	}
	fields := plan.Fields()
	if len(fields) != 2 || fields[0].Name != "i32" || fields[1].Name != "name" {
		t.Fatalf("fields = %#v", fields)
	}
	for _, testCase := range []struct {
		values map[string]Value
		want   Truth
	}{
		{map[string]Value{"i32": Int32Value(3), "name": StringValue("no")}, TruthTrue},
		{map[string]Value{"i32": Int32Value(8), "name": StringValue("x")}, TruthTrue},
		{map[string]Value{"i32": Int32Value(8), "name": StringValue("no")}, TruthFalse},
	} {
		got, evalErr := plan.Evaluate(rowResolver(testCase.values))
		if evalErr != nil || got != testCase.want {
			t.Fatalf("Evaluate(%v) = %s, %v; want %s", testCase.values, got, evalErr, testCase.want)
		}
	}
}

func TestRewriteFilterDoesNotChangeInequalityOR(t *testing.T) {
	expression, err := ParseFilter("i32 != 1 OR i32 != 2")
	if err != nil {
		t.Fatal(err)
	}
	rewritten, stats := RewriteFilter(expression)
	if stats.EqualitySets != 0 || Format(rewritten) != "(i32 != 1 OR i32 != 2)" {
		t.Fatalf("rewrite = %q stats=%#v", Format(rewritten), stats)
	}
	plan, err := BuildPlan("i32 != 1 OR i32 != 2", testFilterSchema(t))
	if err != nil {
		t.Fatal(err)
	}
	matched, err := plan.Match(rowResolver(map[string]Value{"i32": Int32Value(1)}))
	if err != nil || !matched {
		t.Fatalf("semantics-changing inequality rewrite: matched=%t error=%v", matched, err)
	}
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
		if err != nil {
			t.Fatalf("BuildPlan(%q): %v", testCase.filter, err)
		}
		if plan.Explain() != testCase.explain {
			t.Errorf("Explain(%q) = %q, want %q", testCase.filter, plan.Explain(), testCase.explain)
		}
		if plan.Stats() != testCase.stats {
			t.Errorf("Stats(%q) = %#v, want %#v", testCase.filter, plan.Stats(), testCase.stats)
		}
		got, evalErr := plan.Evaluate(rowResolver(testCase.values))
		if evalErr != nil || got != testCase.want {
			t.Errorf("Evaluate(%q) = %s, %v; want %s", testCase.filter, got, evalErr, testCase.want)
		}
	}
}

func mustArray(t *testing.T, kind ValueKind, values ...Value) Value {
	t.Helper()
	value, err := ArrayValue(kind, values...)
	if err != nil {
		t.Fatal(err)
	}
	return value
}

func TestBuildPlanArrayLengthAndNull(t *testing.T) {
	plan, err := BuildPlan("array_length(numbers) >= 2 AND name IS NULL", testFilterSchema(t))
	if err != nil {
		t.Fatal(err)
	}
	if plan.Explain() != "(array_length(numbers) >= 2 AND name IS NULL)" {
		t.Fatalf("explain = %q", plan.Explain())
	}
	values := map[string]Value{
		"numbers": mustArray(t, ValueInt32, Int32Value(1), Int32Value(2)),
	}
	got, err := plan.Evaluate(rowResolver(values))
	if err != nil || got != TruthTrue {
		t.Fatalf("array_length plan = %s, %v", got, err)
	}
	nullArray, _ := NullValue(ValueInt32, true)
	values["numbers"] = nullArray
	got, err = plan.Evaluate(rowResolver(values))
	if err != nil || got != TruthUnknown {
		t.Fatalf("NULL array_length plan = %s, %v", got, err)
	}
}

func TestBuildPlanBindsNumericBoundaries(t *testing.T) {
	filter := "i32=-2147483648 AND i64=-9223372036854775808 AND u32=4294967295 AND u64=18446744073709551615 AND f32=1.5F AND f64=2D"
	plan, err := BuildPlan(filter, testFilterSchema(t))
	if err != nil {
		t.Fatal(err)
	}
	f32, _ := Float32Value(1.5)
	f64, _ := Float64Value(2)
	values := map[string]Value{
		"i32": Int32Value(-2147483648), "i64": Int64Value(-9223372036854775808),
		"u32": Uint32Value(4294967295), "u64": Uint64Value(18446744073709551615),
		"f32": f32, "f64": f64,
	}
	matched, err := plan.Match(rowResolver(values))
	if err != nil || !matched {
		t.Fatalf("boundary plan matched=%t error=%v", matched, err)
	}
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
		{"data IN ('x')", "does not support BINARY"},
		{"i32 CONTAIN_ANY (1)", "requires an array"},
		{"numbers = 1", "supports only"},
		{"blobs CONTAIN_ALL ('x')", "does not support ARRAY_BINARY"},
		{"array_length(i32) > 1", "requires an array"},
		{"array_length(numbers, i32) > 1", "exactly one"},
		{"array_length(numbers) > '1'", "integer literal"},
		{"ARRAY_LENGTH(numbers) > 1", "not supported"},
	}
	for _, testCase := range tests {
		t.Run(testCase.filter, func(t *testing.T) {
			_, err := BuildPlan(testCase.filter, testFilterSchema(t))
			var analysisErr *AnalysisError
			if !errors.As(err, &analysisErr) {
				t.Fatalf("error = %T %v", err, err)
			}
			if analysisErr.Position.Line < 1 || analysisErr.Position.Column < 1 || !strings.Contains(analysisErr.Message, testCase.message) {
				t.Fatalf("analysis error = %#v", analysisErr)
			}
		})
	}
}

func TestPlanIsSafeForConcurrentEvaluation(t *testing.T) {
	plan, err := BuildPlan("name LIKE 'user-_%' AND i32 IN (1, 2, 3)", testFilterSchema(t))
	if err != nil {
		t.Fatal(err)
	}
	resolver := rowResolver(map[string]Value{"name": StringValue("user-22"), "i32": Int32Value(2)})
	var wait sync.WaitGroup
	for index := 0; index < 32; index++ {
		wait.Add(1)
		go func() {
			defer wait.Done()
			for iteration := 0; iteration < 100; iteration++ {
				matched, evalErr := plan.Match(resolver)
				if evalErr != nil || !matched {
					t.Errorf("concurrent evaluation matched=%t error=%v", matched, evalErr)
					return
				}
			}
		}()
	}
	wait.Wait()
}

func TestNewSchemaValidationAndPlanDefensiveFields(t *testing.T) {
	if _, err := NewSchema([]Field{{Name: "", Kind: ValueString, Filterable: true}}); err == nil {
		t.Fatal("empty field name succeeded")
	}
	if _, err := NewSchema([]Field{{Name: "a"}, {Name: "a"}}); err == nil {
		t.Fatal("duplicate field succeeded")
	}
	if _, err := NewSchema([]Field{{Name: "a", Filterable: true}}); err == nil {
		t.Fatal("filterable invalid kind succeeded")
	}
	if _, err := NewSchema([]Field{{Name: "a", Kind: ValueString, Indexed: true}}); err == nil {
		t.Fatal("unfilterable indexed field succeeded")
	}
	plan, err := BuildPlan("i32=1", testFilterSchema(t))
	if err != nil {
		t.Fatal(err)
	}
	fields := plan.Fields()
	fields[0].Name = "changed"
	if plan.Fields()[0].Name != "i32" {
		t.Fatal("plan fields were not defensively copied")
	}
}

func FuzzBuildPlan(f *testing.F) {
	schema, err := NewSchema([]Field{
		{Name: "i32", Kind: ValueInt32, Filterable: true},
		{Name: "name", Kind: ValueString, Nullable: true, Filterable: true},
		{Name: "tags", Kind: ValueString, Array: true, Nullable: true, Filterable: true},
	})
	if err != nil {
		f.Fatal(err)
	}
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
		if err != nil {
			t.Fatal(err)
		}
		values := map[string]Value{"i32": Int32Value(1), "name": StringValue("user-a"), "tags": tags}
		if _, err := plan.Evaluate(rowResolver(values)); err != nil {
			t.Fatalf("successful plan evaluation failed: input=%q plan=%q error=%v", input, plan.Explain(), err)
		}
		if plan.Explain() == "" {
			t.Fatal("successful plan has empty explanation")
		}
	})
}
