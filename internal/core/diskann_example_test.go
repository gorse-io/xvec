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

	"github.com/gorse-io/xvec/internal/core"
)

func ExampleDiskANNIndex() {
	options := core.DefaultDiskANNBuildOptions(core.MetricL2)
	options.MaxDegree = 2
	options.ListSize = 4
	options.PQChunks = 1
	builder, err := core.NewDiskANNBuilder(2, options)
	if err != nil {
		panic(err)
	}
	for key, vector := range map[uint64][]float32{
		10: {0, 0}, 20: {1, 0}, 30: {0, 2}, 40: {3, 3},
	} {
		if err := builder.Add(context.Background(), key, vector); err != nil {
			panic(err)
		}
	}
	index, err := builder.Build(context.Background())
	if err != nil {
		panic(err)
	}
	results, err := index.SearchDiskANN(context.Background(), []float32{0.9, 0}, core.DiskANNSearchOptions{
		SearchOptions: core.SearchOptions{TopK: 2}, ListSize: 4,
	})
	if err != nil {
		panic(err)
	}
	for _, result := range results {
		fmt.Printf("%d %.2f\n", result.Key, result.Score)
	}
	// Output:
	// 20 0.01
	// 10 0.81
}
