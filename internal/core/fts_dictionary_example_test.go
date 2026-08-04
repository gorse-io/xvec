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

package core_test

import (
	"context"
	"fmt"

	"github.com/gorse-io/zvec/internal/core"
)

func ExampleFTSFieldBuilder() {
	ctx := context.Background()
	builder := core.NewFTSFieldBuilder()
	_ = builder.AddDocument(ctx, 0, []core.Token{
		{Text: "go", Position: 0},
		{Text: "vector", Position: 1},
		{Text: "go", Position: 2},
	})
	_ = builder.AddDocument(ctx, 1, []core.Token{{Text: "vector", Position: 0}})

	dictionary, err := builder.Build(ctx)
	if err != nil {
		panic(err)
	}
	info, postings, found := dictionary.Lookup("go")
	if !found {
		panic("missing term")
	}
	fmt.Printf("df=%d max_tf=%d\n", info.DocumentFrequency, info.MaximumTermFrequency)
	iterator := postings.Iterator()
	for iterator.Next() {
		fmt.Printf("doc=%d positions=%v\n", iterator.DocumentID(), iterator.Positions())
	}
	// Output:
	// df=1 max_tf=2
	// doc=0 positions=[0 2]
}
