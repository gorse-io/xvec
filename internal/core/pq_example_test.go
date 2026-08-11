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

func ExampleTrainPQ() {
	vectors := [][]float32{
		{0, 0, 10, 10},
		{0, 1, 10, 11},
		{5, 5, 20, 20},
		{5, 6, 20, 21},
	}
	options := core.DefaultPQOptions(core.MetricL2)
	options.Chunks = 2
	model, err := core.TrainPQ(context.Background(), vectors, options)
	if err != nil {
		panic(err)
	}
	code, err := model.Encode(vectors[0])
	if err != nil {
		panic(err)
	}
	table, err := model.DistanceTable(vectors[0])
	if err != nil {
		panic(err)
	}
	score, err := table.Lookup(code)
	if err != nil {
		panic(err)
	}
	fmt.Println(model.Chunks(), len(code.Bytes()), score)
	// Output: 2 2 0
}
