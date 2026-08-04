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

func ExampleVamanaBuilder() {
	options := core.DefaultVamanaBuildOptions(core.MetricL2)
	builder, err := core.NewVamanaBuilder(2, options)
	if err != nil {
		panic(err)
	}
	for _, candidate := range []core.Candidate{
		{Key: 10, Vector: []float32{1, 0}},
		{Key: 20, Vector: []float32{0, 1}},
		{Key: 30, Vector: []float32{-1, 0}},
	} {
		if err := builder.Add(context.Background(), candidate.Key, candidate.Vector); err != nil {
			panic(err)
		}
	}
	index, err := builder.Build(context.Background())
	if err != nil {
		panic(err)
	}
	results, err := index.SearchVamana(context.Background(), []float32{.9, .1}, core.VamanaSearchOptions{
		SearchOptions: core.SearchOptions{TopK: 2}, EFSearch: 20,
	})
	if err != nil {
		panic(err)
	}
	keys := make([]uint64, len(results))
	for position, result := range results {
		keys[position] = result.Key
	}
	fmt.Println(keys)
	// Output: [10 20]
}
