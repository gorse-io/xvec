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

func ExampleHNSWRaBitQBuilder() {
	options := core.DefaultHNSWRaBitQBuildOptions(core.MetricL2)
	options.TotalBits = 4
	options.Clusters = 4
	options.MaxIterations = 4
	options.M = 4
	options.EFConstruction = 16
	options.Seed = 42
	builder, err := core.NewHNSWRaBitQBuilder(64, options)
	if err != nil {
		panic(err)
	}
	for key := uint64(1); key <= 16; key++ {
		vector := make([]float32, 64)
		for dimension := range vector {
			vector[dimension] = float32(int(key)+dimension%5) / 8
		}
		if err := builder.Add(context.Background(), key, vector); err != nil {
			panic(err)
		}
	}
	index, err := builder.Build(context.Background())
	if err != nil {
		panic(err)
	}
	query, _ := index.Vector(5)
	results, err := index.SearchHNSWRaBitQ(context.Background(), query, core.HNSWRaBitQSearchOptions{
		SearchOptions: core.SearchOptions{TopK: 3}, EF: 16, Refine: true,
	})
	if err != nil {
		panic(err)
	}
	fmt.Println(index.Dimension(), index.Len(), index.BuildOptions().TotalBits)
	fmt.Println(results[0].Key, results[0].Score)
	// Output:
	// 64 16 4
	// 5 0
}
