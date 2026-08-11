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

	"github.com/gorse-io/xvec/internal/ailego"
	"github.com/gorse-io/xvec/internal/core"
)

func ExampleSearchFTS() {
	ctx := context.Background()
	builder := core.NewFTSFieldBuilder()
	_ = builder.AddDocument(ctx, 0, []core.Token{
		{Text: "go", Position: 0}, {Text: "go", Position: 1},
	})
	_ = builder.AddDocument(ctx, 1, []core.Token{{Text: "go", Position: 0}})
	dictionary, err := builder.Build(ctx)
	if err != nil {
		panic(err)
	}
	stats, err := core.AggregateFTSCorpusStats(ctx, []core.FTSSegmentView{{Dictionary: dictionary}})
	if err != nil {
		panic(err)
	}
	scorer, err := core.NewBM25Scorer(core.DefaultBM25Params(), stats)
	if err != nil {
		panic(err)
	}
	query := &core.FTSTermQueryNode{
		Flags: core.FTSQueryModifier{Boost: 1},
		Term:  "go",
	}
	results, err := core.SearchFTS(ctx, dictionary, query, scorer, core.FTSSearchOptions{TopK: 10})
	if err != nil {
		panic(err)
	}
	fmt.Println(results[0].DocumentID, len(results))
	// Output: 0 2
}

func ExampleMergeFTSTermDictionaries() {
	ctx := context.Background()
	leftBuilder := core.NewFTSFieldBuilder()
	_ = leftBuilder.AddDocument(ctx, 0, []core.Token{{Text: "x", Position: 0}})
	_ = leftBuilder.AddDocument(ctx, 1, nil)
	left, _ := leftBuilder.Build(ctx)
	rightBuilder := core.NewFTSFieldBuilder()
	_ = rightBuilder.AddDocument(ctx, 0, []core.Token{{Text: "y", Position: 0}})
	right, _ := rightBuilder.Build(ctx)

	deleted := ailego.NewBitmap(2)
	deleted.Set(1)
	merged, err := core.MergeFTSTermDictionaries(ctx, []core.FTSSegmentView{
		{Dictionary: left, DeletedDocuments: deleted},
		{Dictionary: right},
	})
	if err != nil {
		panic(err)
	}
	fmt.Println(merged.Stats().TotalDocuments, merged.Terms())
	// Output: 2 [x y]
}
