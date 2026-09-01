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

package mathbatch

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/gorse-io/xvec/internal/ailego/math"
	"github.com/stretchr/testify/require"
)

func TestInnerProducts2MatchFloat32Oracle(t *testing.T) {
	t.Parallel()
	for _, dimension := range []int{1, 7, 8, 15, 16, 17, 127, 128, 129, 768, 1536} {
		t.Run(fmt.Sprintf("dimension_%d", dimension), func(t *testing.T) {
			random := rand.New(rand.NewSource(int64(dimension * 3)))
			query := make([]float32, dimension+1)
			first := make([]float32, dimension+1)
			second := make([]float32, dimension+1)
			for index := 1; index <= dimension; index++ {
				query[index] = random.Float32()*2 - 1
				first[index] = random.Float32()*2 - 1
				second[index] = random.Float32()*2 - 1
			}
			query, first, second = query[1:], first[1:], second[1:]
			gotFirst, gotSecond := innerProducts2(query, first, second)
			requireFloat32Close(t, innerProductOracle(query, first), gotFirst)
			requireFloat32Close(t, innerProductOracle(query, second), gotSecond)
		})
	}
}

func TestInnerProducts4MatchFloat32Oracle(t *testing.T) {
	t.Parallel()
	for _, dimension := range []int{1, 7, 8, 15, 16, 17, 127, 128, 129, 768, 1536} {
		t.Run(fmt.Sprintf("dimension_%d", dimension), func(t *testing.T) {
			random := rand.New(rand.NewSource(int64(dimension * 5)))
			vectors := make([][]float32, 5)
			for vector := range vectors {
				vectors[vector] = make([]float32, dimension+1)
				for index := 1; index <= dimension; index++ {
					vectors[vector][index] = random.Float32()*2 - 1
				}
				vectors[vector] = vectors[vector][1:]
			}
			first, second, third, fourth := innerProducts4(vectors[0], vectors[1], vectors[2], vectors[3], vectors[4])
			for index, actual := range []float32{first, second, third, fourth} {
				requireFloat32Close(t, innerProductOracle(vectors[0], vectors[index+1]), actual)
			}
		})
	}
}

func TestInnerProductsDoNotAllocateOrMutate(t *testing.T) {
	query := []float32{0.2, 0.9, -0.4, 0.7}
	candidate := []float32{0.3, 0.5, 0.8, -0.1}
	queryCopy := append([]float32(nil), query...)
	candidateCopy := append([]float32(nil), candidate...)

	require.Zero(t, testing.AllocsPerRun(100, func() {
		benchmarkFirst, benchmarkSecond = innerProducts2(query, candidate, candidate)
		benchmarkFirst, benchmarkSecond, benchmarkThird, benchmarkFourth = innerProducts4(query, candidate, candidate, candidate, candidate)
	}))
	require.Equal(t, queryCopy, query)
	require.Equal(t, candidateCopy, candidate)
}

func BenchmarkInnerProducts(b *testing.B) {
	for _, dimension := range []int{128, 768, 1536} {
		query := make([]float32, dimension)
		candidate := make([]float32, dimension)
		for index := range query {
			query[index] = float32(index%17)/17 - 0.5
			candidate[index] = float32(index%23)/23 - 0.5
		}
		b.Run(fmt.Sprintf("Sequential2/%d", dimension), func(b *testing.B) {
			for b.Loop() {
				benchmarkFirst = mathutil.InnerProduct(query, candidate)
				benchmarkSecond = mathutil.InnerProduct(query, candidate)
			}
		})
		b.Run(fmt.Sprintf("Batch2/%d", dimension), func(b *testing.B) {
			for b.Loop() {
				benchmarkFirst, benchmarkSecond = innerProducts2(query, candidate, candidate)
			}
		})
		b.Run(fmt.Sprintf("Sequential4/%d", dimension), func(b *testing.B) {
			for b.Loop() {
				benchmarkFirst = mathutil.InnerProduct(query, candidate)
				benchmarkSecond = mathutil.InnerProduct(query, candidate)
				benchmarkThird = mathutil.InnerProduct(query, candidate)
				benchmarkFourth = mathutil.InnerProduct(query, candidate)
			}
		})
		b.Run(fmt.Sprintf("Batch4/%d", dimension), func(b *testing.B) {
			for b.Loop() {
				benchmarkFirst, benchmarkSecond, benchmarkThird, benchmarkFourth = innerProducts4(query, candidate, candidate, candidate, candidate)
			}
		})
	}
}

func requireFloat32Close(t *testing.T, expected, actual float32) {
	t.Helper()
	tolerance := float32(1e-5) * max(1, float32(math.Abs(float64(expected))))
	require.InDelta(t, expected, actual, float64(tolerance))
}

var (
	benchmarkFirst  float32
	benchmarkSecond float32
	benchmarkThird  float32
	benchmarkFourth float32
)
