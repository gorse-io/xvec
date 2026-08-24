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

	"github.com/gorse-io/xvec/internal/core/algorithm"
)

func ExampleTrainRaBitQ() {
	vectors := make([][]float32, 8)
	for row := range vectors {
		vectors[row] = make([]float32, 64)
		for column := range vectors[row] {
			vectors[row][column] = float32((row+1)*(column%7-3)) / 8
		}
	}

	options := core.DefaultRaBitQOptions(core.MetricL2)
	options.TotalBits = 3
	options.Clusters = 1
	options.MaxIterations = 2
	options.Seed = 42
	model, err := core.TrainRaBitQ(context.Background(), vectors, options)
	if err != nil {
		panic(err)
	}
	code, err := model.Encode(vectors[0])
	if err != nil {
		panic(err)
	}
	query, err := model.PrepareQuery(vectors[1])
	if err != nil {
		panic(err)
	}
	estimate, err := query.Estimate(code)
	if err != nil {
		panic(err)
	}

	fmt.Println(model.Dimension(), model.PaddedDimension(), model.TotalBits())
	fmt.Println(code.Cluster(), len(code.BinaryCode()), len(code.ExtraCode()))
	fmt.Println(estimate.LowerBound <= estimate.UpperBound)
	// Output:
	// 64 64 3
	// 0 8 16
	// true
}
