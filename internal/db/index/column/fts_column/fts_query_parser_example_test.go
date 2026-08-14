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

package ftscolumn_test

import (
	"context"
	"fmt"

	"github.com/gorse-io/xvec/internal/db/index/column/fts_column"
	"github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer"
)

func ExampleParseFTSQuery() {
	standardTokenizer, err := tokenizer.NewStandardTokenizer(tokenizer.DefaultStandardTokenizerOptions())
	if err != nil {
		panic(err)
	}
	analyzer, err := tokenizer.NewFTSTokenizerPipeline(standardTokenizer, tokenizer.NewLowercaseTokenFilter())
	if err != nil {
		panic(err)
	}

	query, err := ftscolumn.ParseFTSQuery(
		context.Background(),
		`+Vector -slow "exact phrase"`,
		analyzer,
		ftscolumn.FTSDefaultOperatorOR,
	)
	if err != nil {
		panic(err)
	}
	fmt.Println(query)
	// Output: OR(+vector -slow "exact phrase")
}

func ExampleFTSQueryIterator() {
	builder := ftscolumn.NewFTSFieldBuilder()
	documents := [][]tokenizer.Token{
		{{Text: "quick", Position: 0}, {Text: "brown", Position: 1}},
		{{Text: "quick", Position: 0}, {Text: "fox", Position: 1}, {Text: "brown", Position: 2}},
		{{Text: "quick", Position: 0}, {Text: "brown", Position: 1}, {Text: "fox", Position: 2}},
	}
	for documentID, tokens := range documents {
		if err := builder.AddDocument(context.Background(), uint32(documentID), tokens); err != nil {
			panic(err)
		}
	}
	dictionary, err := builder.Build(context.Background())
	if err != nil {
		panic(err)
	}
	analyzer, err := tokenizer.NewFTSTokenizerPipeline(tokenizer.NewWhitespaceTokenizer())
	if err != nil {
		panic(err)
	}
	query, err := ftscolumn.ParseFTSQuery(context.Background(), `"quick brown"`, analyzer, ftscolumn.FTSDefaultOperatorOR)
	if err != nil {
		panic(err)
	}
	iterator, err := ftscolumn.NewFTSQueryIterator(context.Background(), dictionary, query, ftscolumn.FTSQueryExecutionOptions{})
	if err != nil {
		panic(err)
	}
	for iterator.Next(context.Background()) {
		fmt.Println(iterator.DocumentID())
	}
	if err := iterator.Err(); err != nil {
		panic(err)
	}
	// Output:
	// 0
	// 2
}
