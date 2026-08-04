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
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/gorse-io/zvec/internal/ailego"
)

var ftsQueryTestDocuments = []string{
	"apple banana quick brown fox",
	"apple quick fox brown",
	"banana quick brown",
	"apple banana banana quick brown fox",
	"grape slow fox",
	"",
	"apple quick quick brown fox",
	"apple quick brown fox quick brown",
}

func TestFTSQueryIteratorTermPhraseAndBoolean(t *testing.T) {
	dictionary := buildFTSQueryTestDictionary(t)
	pipeline := newFTSStandardTestPipeline(t)
	tests := []struct {
		query string
		want  []uint32
	}{
		{"apple", []uint32{0, 1, 3, 6, 7}},
		{"apple OR grape", []uint32{0, 1, 3, 4, 6, 7}},
		{"apple AND banana", []uint32{0, 3}},
		{"apple NOT banana", []uint32{1, 6, 7}},
		{"apple -banana", []uint32{1, 6, 7}},
		{"+apple grape", []uint32{0, 1, 3, 6, 7}},
		{"-apple", []uint32{}},
		{`"quick brown"`, []uint32{0, 2, 3, 6, 7}},
		{`"quick fox"`, []uint32{1}},
		{`"banana banana"`, []uint32{3}},
		{`"quick quick brown"`, []uint32{6}},
		{`"quick brown" OR grape`, []uint32{0, 2, 3, 4, 6, 7}},
		{`"quick brown" AND apple`, []uint32{0, 3, 6, 7}},
		{`apple NOT "quick brown"`, []uint32{1}},
		{`"!!!"`, []uint32{}},
		{"(apple OR grape) AND fox", []uint32{0, 1, 3, 4, 6, 7}},
		{"apple OR missing", []uint32{0, 1, 3, 6, 7}},
		{"apple AND missing", []uint32{}},
		{"apple NOT missing", []uint32{0, 1, 3, 6, 7}},
		{"missing NOT apple", []uint32{}},
		{"apple apple", []uint32{0, 1, 3, 6, 7}},
	}
	for _, test := range tests {
		t.Run(test.query, func(t *testing.T) {
			node, err := ParseFTSQuery(context.Background(), test.query, pipeline, FTSDefaultOperatorOR)
			if err != nil {
				t.Fatal(err)
			}
			iterator, err := NewFTSQueryIterator(context.Background(), dictionary, node, FTSQueryExecutionOptions{})
			if err != nil {
				t.Fatal(err)
			}
			if got := collectFTSQueryDocuments(t, iterator); !reflect.DeepEqual(got, test.want) {
				t.Fatalf("documents = %#v, want %#v", got, test.want)
			}
		})
	}
}

func TestFTSQueryIteratorAdvanceAndCost(t *testing.T) {
	dictionary := buildFTSQueryTestDictionary(t)
	pipeline := newFTSStandardTestPipeline(t)
	node, err := ParseFTSQuery(context.Background(), "apple OR grape", pipeline, FTSDefaultOperatorOR)
	if err != nil {
		t.Fatal(err)
	}
	iterator, err := NewFTSQueryIterator(context.Background(), dictionary, node, FTSQueryExecutionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if iterator.Cost() != 6 || iterator.Valid() || iterator.DocumentID() != 0 {
		t.Fatalf("initial iterator = cost %d valid %t doc %d", iterator.Cost(), iterator.Valid(), iterator.DocumentID())
	}
	if !iterator.Advance(context.Background(), 3) || iterator.DocumentID() != 3 {
		t.Fatalf("Advance(3) = %t doc %d err %v", iterator.Valid(), iterator.DocumentID(), iterator.Err())
	}
	if !iterator.Advance(context.Background(), 3) || iterator.DocumentID() != 3 {
		t.Fatal("Advance did not retain current match")
	}
	if !iterator.Next(context.Background()) || iterator.DocumentID() != 4 {
		t.Fatalf("Next = doc %d err %v", iterator.DocumentID(), iterator.Err())
	}
	if !iterator.Advance(context.Background(), 7) || iterator.DocumentID() != 7 {
		t.Fatalf("Advance(7) = doc %d err %v", iterator.DocumentID(), iterator.Err())
	}
	if iterator.Next(context.Background()) || iterator.Valid() || iterator.Err() != nil {
		t.Fatalf("exhausted iterator = valid %t err %v", iterator.Valid(), iterator.Err())
	}
	if iterator.Advance(context.Background(), 0) {
		t.Fatal("exhausted iterator restarted")
	}
}

func TestFTSQueryIteratorDeletionSnapshot(t *testing.T) {
	dictionary := buildFTSQueryTestDictionary(t)
	pipeline := newFTSStandardTestPipeline(t)
	node, err := ParseFTSQuery(context.Background(), `"quick brown"`, pipeline, FTSDefaultOperatorOR)
	if err != nil {
		t.Fatal(err)
	}
	deleted := ailego.NewBitmap(uint64(len(ftsQueryTestDocuments)))
	deleted.Set(0)
	deleted.Set(6)
	iterator, err := NewFTSQueryIterator(context.Background(), dictionary, node, FTSQueryExecutionOptions{DeletedDocuments: deleted})
	if err != nil {
		t.Fatal(err)
	}
	deleted.Clear(0)
	deleted.Set(2)
	if got, want := collectFTSQueryDocuments(t, iterator), []uint32{2, 3, 7}; !reflect.DeepEqual(got, want) {
		t.Fatalf("snapshot documents = %#v, want %#v", got, want)
	}

	invalid := ailego.NewBitmap(0)
	invalid.Set(uint64(len(ftsQueryTestDocuments)))
	if got, err := NewFTSQueryIterator(context.Background(), dictionary, node, FTSQueryExecutionOptions{DeletedDocuments: invalid}); got != nil || !errors.Is(err, ErrInvalidFTSQueryExecution) {
		t.Fatalf("invalid deletion = %#v, %v", got, err)
	}
}

func TestFTSQueryIteratorInvalidAndCancellation(t *testing.T) {
	dictionary := buildFTSQueryTestDictionary(t)
	pipeline := newFTSStandardTestPipeline(t)
	node, err := ParseFTSQuery(context.Background(), "apple", pipeline, FTSDefaultOperatorOR)
	if err != nil {
		t.Fatal(err)
	}
	if iterator, err := NewFTSQueryIterator(nil, dictionary, node, FTSQueryExecutionOptions{}); iterator != nil || !errors.Is(err, ErrInvalidFTSQueryExecution) {
		t.Fatalf("nil context = %#v, %v", iterator, err)
	}
	if iterator, err := NewFTSQueryIterator(context.Background(), nil, node, FTSQueryExecutionOptions{}); iterator != nil || !errors.Is(err, ErrInvalidFTSQueryExecution) {
		t.Fatalf("nil dictionary = %#v, %v", iterator, err)
	}
	if iterator, err := NewFTSQueryIterator(context.Background(), dictionary, nil, FTSQueryExecutionOptions{}); iterator != nil || !errors.Is(err, ErrInvalidFTSQueryAST) {
		t.Fatalf("nil AST = %#v, %v", iterator, err)
	}
	if (*FTSQueryIterator)(nil).Next(context.Background()) || (*FTSQueryIterator)(nil).Advance(context.Background(), 0) || !errors.Is((*FTSQueryIterator)(nil).Err(), ErrInvalidFTSQueryExecution) {
		t.Fatal("nil iterator behavior differs")
	}

	iterator, err := NewFTSQueryIterator(context.Background(), dictionary, node, FTSQueryExecutionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if iterator.Next(nil) || !errors.Is(iterator.Err(), ErrInvalidFTSQueryExecution) {
		t.Fatalf("nil iteration context = %v", iterator.Err())
	}

	iterator, err = NewFTSQueryIterator(context.Background(), dictionary, node, FTSQueryExecutionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if iterator.Next(canceled) || !errors.Is(iterator.Err(), context.Canceled) {
		t.Fatalf("canceled iteration = %v", iterator.Err())
	}

	iterator, err = NewFTSQueryIterator(context.Background(), dictionary, node, FTSQueryExecutionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	active, stop := context.WithCancel(context.Background())
	if !iterator.Next(active) {
		t.Fatal(iterator.Err())
	}
	stop()
	if iterator.Next(active) || !errors.Is(iterator.Err(), context.Canceled) || iterator.Valid() {
		t.Fatalf("mid-stream cancellation = valid %t err %v", iterator.Valid(), iterator.Err())
	}
}

func TestFTSQueryIteratorASTAndDictionaryOwnership(t *testing.T) {
	dictionary := buildFTSQueryTestDictionary(t)
	node := &FTSTermQueryNode{Flags: defaultFTSQueryModifier(), Term: "apple"}
	iterator, err := NewFTSQueryIterator(context.Background(), dictionary, node, FTSQueryExecutionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	node.Term = "missing"
	if got, want := collectFTSQueryDocuments(t, iterator), []uint32{0, 1, 3, 6, 7}; !reflect.DeepEqual(got, want) {
		t.Fatalf("documents after AST mutation = %#v", got)
	}

	encoded, err := dictionary.Encode(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	reopened, err := OpenFTSTermDictionary(context.Background(), encoded)
	if err != nil {
		t.Fatal(err)
	}
	query := &FTSPhraseQueryNode{Flags: defaultFTSQueryModifier(), Terms: []string{"banana", "banana"}}
	iterator, err = NewFTSQueryIterator(context.Background(), reopened, query, FTSQueryExecutionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	clear(encoded)
	if got, want := collectFTSQueryDocuments(t, iterator), []uint32{3}; !reflect.DeepEqual(got, want) {
		t.Fatalf("reopened documents = %#v", got)
	}
}

func TestFTSQueryIteratorConcurrentReaders(t *testing.T) {
	dictionary := buildFTSQueryTestDictionary(t)
	pipeline := newFTSStandardTestPipeline(t)
	node, err := ParseFTSQuery(context.Background(), `(apple OR grape) AND fox NOT banana`, pipeline, FTSDefaultOperatorOR)
	if err != nil {
		t.Fatal(err)
	}
	want := []uint32{1, 4, 6, 7}
	var wait sync.WaitGroup
	errorsChannel := make(chan error, 32)
	for worker := 0; worker < 32; worker++ {
		wait.Add(1)
		go func() {
			defer wait.Done()
			for iteration := 0; iteration < 50; iteration++ {
				iterator, err := NewFTSQueryIterator(context.Background(), dictionary, node, FTSQueryExecutionOptions{})
				if err != nil {
					errorsChannel <- err
					return
				}
				got := collectFTSQueryDocumentsError(iterator)
				if !reflect.DeepEqual(got.documents, want) || got.err != nil {
					errorsChannel <- fmt.Errorf("documents %#v: %v", got.documents, got.err)
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

func FuzzFTSQueryIterator(f *testing.F) {
	for _, seed := range []string{"apple", "apple OR grape", `"quick brown"`, "apple NOT banana", "+apple grape"} {
		f.Add(seed)
	}
	dictionary := buildFTSQueryTestDictionary(f)
	pipeline := newFTSStandardTestPipeline(f)
	f.Fuzz(func(t *testing.T, query string) {
		node, err := ParseFTSQuery(context.Background(), query, pipeline, FTSDefaultOperatorOR)
		if err != nil {
			return
		}
		iterator, err := NewFTSQueryIterator(context.Background(), dictionary, node, FTSQueryExecutionOptions{})
		if err != nil {
			t.Fatal(err)
		}
		var previous uint32
		first := true
		for iterator.Next(context.Background()) {
			documentID := iterator.DocumentID()
			if int(documentID) >= len(ftsQueryTestDocuments) || !first && documentID <= previous {
				t.Fatalf("invalid document sequence: %d after %d", documentID, previous)
			}
			first, previous = false, documentID
		}
		if err := iterator.Err(); err != nil {
			t.Fatal(err)
		}
	})
}

func BenchmarkFTSQueryIterator(b *testing.B) {
	documents := make([]string, 10_000)
	for index := range documents {
		documents[index] = fmt.Sprintf("common term-%03d phrase match", index%500)
	}
	dictionary := buildFTSQueryDictionaryFromDocuments(b, documents)
	pipeline := newFTSStandardTestPipeline(b)
	node, err := ParseFTSQuery(context.Background(), `common AND "phrase match"`, pipeline, FTSDefaultOperatorOR)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for b.Loop() {
		iterator, err := NewFTSQueryIterator(context.Background(), dictionary, node, FTSQueryExecutionOptions{})
		if err != nil {
			b.Fatal(err)
		}
		for iterator.Next(context.Background()) {
		}
		if err := iterator.Err(); err != nil {
			b.Fatal(err)
		}
	}
}

func buildFTSQueryTestDictionary(t testing.TB) *FTSTermDictionary {
	t.Helper()
	return buildFTSQueryDictionaryFromDocuments(t, ftsQueryTestDocuments)
}

func buildFTSQueryDictionaryFromDocuments(t testing.TB, documents []string) *FTSTermDictionary {
	t.Helper()
	builder := NewFTSFieldBuilder()
	for documentID, document := range documents {
		words := strings.Fields(document)
		tokens := make([]Token, len(words))
		for position, word := range words {
			tokens[position] = Token{Text: word, Position: uint32(position)}
		}
		if err := builder.AddDocument(context.Background(), uint32(documentID), tokens); err != nil {
			t.Fatal(err)
		}
	}
	dictionary, err := builder.Build(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	return dictionary
}

func collectFTSQueryDocuments(t testing.TB, iterator *FTSQueryIterator) []uint32 {
	t.Helper()
	result := collectFTSQueryDocumentsError(iterator)
	if result.err != nil {
		t.Fatal(result.err)
	}
	return result.documents
}

type ftsQueryCollectionResult struct {
	documents []uint32
	err       error
}

func collectFTSQueryDocumentsError(iterator *FTSQueryIterator) ftsQueryCollectionResult {
	documents := make([]uint32, 0)
	for iterator.Next(context.Background()) {
		documents = append(documents, iterator.DocumentID())
	}
	return ftsQueryCollectionResult{documents: documents, err: iterator.Err()}
}
