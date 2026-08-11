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

package core_test

import (
	"context"
	"fmt"

	"github.com/gorse-io/xvec/internal/core"
)

func ExampleParseFTSQuery() {
	tokenizer, err := core.NewStandardTokenizer(core.DefaultStandardTokenizerOptions())
	if err != nil {
		panic(err)
	}
	analyzer, err := core.NewFTSTokenizerPipeline(tokenizer, core.NewLowercaseTokenFilter())
	if err != nil {
		panic(err)
	}

	query, err := core.ParseFTSQuery(
		context.Background(),
		`+Vector -slow "exact phrase"`,
		analyzer,
		core.FTSDefaultOperatorOR,
	)
	if err != nil {
		panic(err)
	}
	fmt.Println(query)
	// Output: OR(+vector -slow "exact phrase")
}

func ExampleFTSQueryIterator() {
	builder := core.NewFTSFieldBuilder()
	documents := [][]core.Token{
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
	analyzer, err := core.NewFTSTokenizerPipeline(core.NewWhitespaceTokenizer())
	if err != nil {
		panic(err)
	}
	query, err := core.ParseFTSQuery(context.Background(), `"quick brown"`, analyzer, core.FTSDefaultOperatorOR)
	if err != nil {
		panic(err)
	}
	iterator, err := core.NewFTSQueryIterator(context.Background(), dictionary, query, core.FTSQueryExecutionOptions{})
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
