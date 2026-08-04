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

package core

import (
	"context"
	"errors"
	"math"
	"reflect"
	"strings"
	"sync"
	"testing"
)

func TestSimplifyFTSQueryParsedExpressions(t *testing.T) {
	pipeline := newFTSStandardTestPipeline(t)
	tests := []struct {
		query string
		want  string
	}{
		{"apple apple", "apple^2.000000"},
		{"apple AND apple", "apple^2.000000"},
		{"+apple -apple", "<empty>"},
		{"+apple apple", "apple^2.000000"},
		{"a OR (b OR c)", "OR(a b c)"},
		{"a AND (b AND c)", "AND(a b c)"},
		{"a AND (b NOT c)", "AND(a AND(b -c))"},
		{"+a b", "AND(a ?b)"},
		{"foo +bar baz +bay", "AND(bar bay ?OR(foo baz))"},
		{"a -b", "AND(a -b)"},
		{"-a -b", "<empty>"},
		{"a OR b OR a", "OR(a^2.000000 b)"},
		{`"a b" OR "a b"`, `"a b"^2.000000`},
		{"a NOT (b OR c)", "AND(a -OR(b c))"},
	}
	for _, test := range tests {
		t.Run(test.query, func(t *testing.T) {
			parsed, err := ParseFTSQuery(context.Background(), test.query, pipeline, FTSDefaultOperatorOR)
			if err != nil {
				t.Fatal(err)
			}
			before := parsed.String()
			simplified, err := SimplifyFTSQuery(context.Background(), parsed)
			if err != nil {
				t.Fatal(err)
			}
			if got := simplified.String(); got != test.want {
				t.Fatalf("simplified = %q, want %q (raw %q)", got, test.want, before)
			}
			if parsed.String() != before {
				t.Fatalf("simplification mutated input: %q -> %q", before, parsed.String())
			}
			twice, err := SimplifyFTSQuery(context.Background(), simplified)
			if err != nil || twice.String() != simplified.String() {
				t.Fatalf("second simplify = %#v, %v", twice, err)
			}
		})
	}
}

func TestSimplifyFTSQueryEmptyAndModifierRules(t *testing.T) {
	term := func(text string, flags FTSQueryModifier) FTSQueryNode {
		if flags.Boost == 0 {
			flags.Boost = 1
		}
		return &FTSTermQueryNode{Flags: flags, Term: text}
	}
	empty := func(flags FTSQueryModifier) FTSQueryNode {
		if flags.Boost == 0 {
			flags.Boost = 1
		}
		return &FTSEmptyQueryNode{Flags: flags}
	}
	tests := []struct {
		name string
		node FTSQueryNode
		want string
	}{
		{"and positive empty", &FTSAndQueryNode{Flags: defaultFTSQueryModifier(), Children: []FTSQueryNode{term("a", FTSQueryModifier{}), empty(FTSQueryModifier{})}}, "<empty>"},
		{"and negative empty", &FTSAndQueryNode{Flags: defaultFTSQueryModifier(), Children: []FTSQueryNode{term("a", FTSQueryModifier{}), empty(FTSQueryModifier{MustNot: true})}}, "a"},
		{"or drops empty", &FTSOrQueryNode{Flags: defaultFTSQueryModifier(), Children: []FTSQueryNode{empty(FTSQueryModifier{}), term("a", FTSQueryModifier{})}}, "a"},
		{"or all empty", &FTSOrQueryNode{Flags: defaultFTSQueryModifier(), Children: []FTSQueryNode{empty(FTSQueryModifier{})}}, "<empty>"},
		{"and all negative", &FTSAndQueryNode{Flags: defaultFTSQueryModifier(), Children: []FTSQueryNode{term("a", FTSQueryModifier{MustNot: true})}}, "<empty>"},
		{"must and should", &FTSOrQueryNode{Flags: defaultFTSQueryModifier(), Children: []FTSQueryNode{term("a", FTSQueryModifier{Must: true}), term("b", FTSQueryModifier{})}}, "AND(a ?b)"},
		{"must and negative", &FTSOrQueryNode{Flags: defaultFTSQueryModifier(), Children: []FTSQueryNode{term("a", FTSQueryModifier{Must: true}), term("b", FTSQueryModifier{MustNot: true})}}, "AND(a -b)"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			simplified, err := SimplifyFTSQuery(context.Background(), test.node)
			if err != nil {
				t.Fatal(err)
			}
			if got := simplified.String(); got != test.want {
				t.Fatalf("simplified = %q, want %q", got, test.want)
			}
		})
	}
}

func TestSimplifyFTSQueryPhraseDedupAndOwnership(t *testing.T) {
	phraseTerms := []string{"to", "be"}
	node := &FTSOrQueryNode{
		Flags: defaultFTSQueryModifier(),
		Children: []FTSQueryNode{
			&FTSPhraseQueryNode{Flags: FTSQueryModifier{Boost: 2}, Terms: phraseTerms},
			&FTSPhraseQueryNode{Flags: FTSQueryModifier{Boost: 3}, Terms: []string{"to", "be"}},
			&FTSPhraseQueryNode{Flags: defaultFTSQueryModifier(), Terms: []string{"be", "to"}},
		},
	}
	simplified, err := SimplifyFTSQuery(context.Background(), node)
	if err != nil {
		t.Fatal(err)
	}
	if got := simplified.String(); got != `OR("to be"^5.000000 "be to")` {
		t.Fatalf("simplified = %q", got)
	}
	phraseTerms[0] = "changed"
	node.Children[1].(*FTSPhraseQueryNode).Terms[0] = "also changed"
	if simplified.String() != `OR("to be"^5.000000 "be to")` {
		t.Fatal("simplified AST aliases input storage")
	}
}

func TestSimplifyFTSQueryInvalidAndCancellation(t *testing.T) {
	if node, err := SimplifyFTSQuery(nil, &FTSEmptyQueryNode{}); node != nil || !errors.Is(err, ErrInvalidFTSQueryAST) {
		t.Fatalf("nil context = %#v, %v", node, err)
	}
	if node, err := SimplifyFTSQuery(context.Background(), nil); node != nil || !errors.Is(err, ErrInvalidFTSQueryAST) {
		t.Fatalf("nil root = %#v, %v", node, err)
	}
	var typedNil *FTSTermQueryNode
	if node, err := SimplifyFTSQuery(context.Background(), typedNil); node != nil || !errors.Is(err, ErrInvalidFTSQueryAST) {
		t.Fatalf("typed nil root = %#v, %v", node, err)
	}
	cyclic := &FTSAndQueryNode{Flags: defaultFTSQueryModifier()}
	cyclic.Children = []FTSQueryNode{cyclic}
	if node, err := SimplifyFTSQuery(context.Background(), cyclic); node != nil || !errors.Is(err, ErrInvalidFTSQueryAST) {
		t.Fatalf("cycle = %#v, %v", node, err)
	}
	notFinite := &FTSTermQueryNode{Flags: FTSQueryModifier{Boost: float32(math.NaN())}, Term: "a"}
	if node, err := SimplifyFTSQuery(context.Background(), notFinite); node != nil || !errors.Is(err, ErrInvalidFTSQueryAST) {
		t.Fatalf("nonfinite boost = %#v, %v", node, err)
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := SimplifyFTSQuery(canceled, &FTSTermQueryNode{Flags: defaultFTSQueryModifier(), Term: "a"}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled = %v", err)
	}
	children := make([]FTSQueryNode, 10_000)
	for index := range children {
		children[index] = &FTSTermQueryNode{Flags: defaultFTSQueryModifier(), Term: strings.Repeat("x", index%3+1)}
	}
	if _, err := SimplifyFTSQuery(newCancelAfterChecks(3), &FTSOrQueryNode{Flags: defaultFTSQueryModifier(), Children: children}); !errors.Is(err, context.Canceled) {
		t.Fatalf("midway cancellation = %v", err)
	}
}

func TestSimplifyFTSQueryConcurrentUse(t *testing.T) {
	pipeline := newFTSStandardTestPipeline(t)
	parsed, err := ParseFTSQuery(context.Background(), "+apple banana -pear apple", pipeline, FTSDefaultOperatorOR)
	if err != nil {
		t.Fatal(err)
	}
	want, err := SimplifyFTSQuery(context.Background(), parsed)
	if err != nil {
		t.Fatal(err)
	}
	var wait sync.WaitGroup
	errorsChannel := make(chan error, 32)
	for worker := 0; worker < 32; worker++ {
		wait.Add(1)
		go func() {
			defer wait.Done()
			for iteration := 0; iteration < 50; iteration++ {
				got, err := SimplifyFTSQuery(context.Background(), parsed)
				if err != nil || got.String() != want.String() {
					errorsChannel <- errors.New("simplification differs")
					return
				}
			}
		}()
	}
	wait.Wait()
	close(errorsChannel)
	for err := range errorsChannel {
		t.Fatal(err)
	}
}

func FuzzSimplifyFTSQuery(f *testing.F) {
	for _, seed := range []string{"a", "a OR b", "+a b -c", `"a b" AND c`, "a NOT (b OR c)"} {
		f.Add(seed)
	}
	pipeline := newFTSStandardTestPipeline(f)
	f.Fuzz(func(t *testing.T, query string) {
		parsed, err := ParseFTSQuery(context.Background(), query, pipeline, FTSDefaultOperatorOR)
		if err != nil {
			return
		}
		first, err := SimplifyFTSQuery(context.Background(), parsed)
		if err != nil {
			t.Fatal(err)
		}
		second, err := SimplifyFTSQuery(context.Background(), first)
		if err != nil {
			t.Fatal(err)
		}
		if first.String() != second.String() || reflect.TypeOf(first) != reflect.TypeOf(second) {
			t.Fatalf("simplifier is not idempotent: %q -> %q", first, second)
		}
	})
}
