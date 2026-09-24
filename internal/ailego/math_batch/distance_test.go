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

func TestCosineDistancesWithMagnitudes(t *testing.T) {
	t.Parallel()

	query := []float32{1, 2, 3}
	candidates := [][]float32{
		{4, 5, 6},
		{-1, 0, 1},
		{1, 2, 3},
		{0, 0, 0},
	}
	queryMagnitude := magnitude(query)
	candidateMagnitudes := []float32{
		magnitude(candidates[0]),
		magnitude(candidates[1]),
		magnitude(candidates[2]),
		magnitude(candidates[3]),
	}

	first, second := CosineDistances2WithMagnitudes(
		query, candidates[0], candidates[1],
		queryMagnitude, candidateMagnitudes[0], candidateMagnitudes[1],
	)
	require.InDelta(t, cosineDistance(query, candidates[0]), first, 1e-6)
	require.InDelta(t, cosineDistance(query, candidates[1]), second, 1e-6)

	first, second, third, fourth := CosineDistances4WithMagnitudes(
		query, candidates[0], candidates[1], candidates[2], candidates[3],
		queryMagnitude, candidateMagnitudes[0], candidateMagnitudes[1], candidateMagnitudes[2], candidateMagnitudes[3],
	)
	for index, actual := range []float32{first, second, third, fourth} {
		require.InDelta(t, cosineDistance(query, candidates[index]), actual, 1e-6)
	}
}

func TestInnerProductsOneToMany(t *testing.T) {
	t.Parallel()
	testOneToMany(t, func(query []float32, candidates [][]float32, output []float32) {
		InnerProducts(query, candidates, output)
	}, innerProductOracle)
}

func TestSquaredEuclideanDistancesOneToMany(t *testing.T) {
	t.Parallel()
	testOneToMany(t, func(query []float32, candidates [][]float32, output []float32) {
		SquaredEuclideanDistances(query, candidates, output)
	}, squaredEuclideanOracle)
}

func TestEuclideanDistancesOneToMany(t *testing.T) {
	t.Parallel()
	testOneToMany(t, func(query []float32, candidates [][]float32, output []float32) {
		EuclideanDistances(query, candidates, output)
	}, func(left, right []float32) float32 {
		return float32(math.Sqrt(float64(squaredEuclideanOracle(left, right))))
	})
}

func TestCosineDistancesOneToManyWithMagnitudes(t *testing.T) {
	t.Parallel()
	query := []float32{1, 2, 3}
	candidates := [][]float32{{4, 5, 6}, {-1, 0, 1}, {1, 2, 3}, {0, 0, 0}, {3, 2, 1}}
	candidateMagnitudes := make([]float32, len(candidates))
	for index := range candidates {
		candidateMagnitudes[index] = magnitude(candidates[index])
	}
	output := make([]float32, len(candidates))

	CosineDistancesWithMagnitudes(query, candidates, magnitude(query), candidateMagnitudes, output)

	for index := range candidates {
		require.InDelta(t, cosineDistance(query, candidates[index]), output[index], 1e-6)
	}
}

func TestOneToManyDistancesDoNotAllocate(t *testing.T) {
	query := []float32{0.2, 0.9, -0.4, 0.7}
	candidates := [][]float32{
		{0.3, 0.5, 0.8, -0.1},
		{0.1, -0.3, 0.4, 0.9},
		{-0.2, 0.6, 0.5, 0.2},
		{0.7, 0.1, -0.8, 0.4},
		{0.5, -0.4, 0.2, 0.3},
	}
	output := make([]float32, len(candidates))

	require.Zero(t, testing.AllocsPerRun(100, func() {
		InnerProducts(query, candidates, output)
		SquaredEuclideanDistances(query, candidates, output)
	}))
}

func testOneToMany(
	t *testing.T,
	distance func(query []float32, candidates [][]float32, output []float32),
	oracle func(left, right []float32) float32,
) {
	t.Helper()
	for _, dimension := range []int{1, 3, 7, 8, 17, 127} {
		for _, count := range []int{0, 1, 2, 3, 4, 5, 7, 8, 9} {
			t.Run(fmt.Sprintf("dimension_%d/count_%d", dimension, count), func(t *testing.T) {
				random := rand.New(rand.NewSource(int64(dimension*31 + count)))
				query := make([]float32, dimension)
				for index := range query {
					query[index] = random.Float32()*2 - 1
				}
				candidates := make([][]float32, count)
				for candidate := range candidates {
					candidates[candidate] = make([]float32, dimension)
					for index := range candidates[candidate] {
						candidates[candidate][index] = random.Float32()*2 - 1
					}
				}
				output := make([]float32, count)

				distance(query, candidates, output)

				for index := range candidates {
					requireFloat32Close(t, oracle(query, candidates[index]), output[index])
				}
			})
		}
	}
}

func magnitude(vector []float32) float32 {
	return float32(math.Sqrt(float64(innerProductOracle(vector, vector))))
}

func cosineDistance(left, right []float32) float32 {
	leftMagnitude := magnitude(left)
	rightMagnitude := magnitude(right)
	if leftMagnitude == 0 && rightMagnitude == 0 {
		return 0
	}
	if leftMagnitude == 0 || rightMagnitude == 0 {
		return 1
	}
	return 1 - innerProductOracle(left, right)/(leftMagnitude*rightMagnitude)
}

func innerProductOracle(left, right []float32) (sum float32) {
	for index, value := range left {
		sum += value * right[index]
	}
	return
}

func TestSquaredEuclideanDistancesMatchFloat32Oracle(t *testing.T) {
	t.Parallel()
	for _, dimension := range []int{1, 3, 4, 7, 8, 15, 16, 17, 127, 128, 129, 768, 1536} {
		t.Run(fmt.Sprintf("dimension_%d", dimension), func(t *testing.T) {
			random := rand.New(rand.NewSource(int64(dimension * 7)))
			vectors := make([][]float32, 5)
			for vector := range vectors {
				vectors[vector] = make([]float32, dimension+1)
				for index := 1; index <= dimension; index++ {
					vectors[vector][index] = random.Float32()*2 - 1
				}
				vectors[vector] = vectors[vector][1:]
			}

			first, second := SquaredEuclideanDistances2(vectors[0], vectors[1], vectors[2])
			requireFloat32Close(t, squaredEuclideanOracle(vectors[0], vectors[1]), first)
			requireFloat32Close(t, squaredEuclideanOracle(vectors[0], vectors[2]), second)
			firstEuclidean, secondEuclidean := EuclideanDistances2(vectors[0], vectors[1], vectors[2])
			requireFloat32Close(t, float32(math.Sqrt(float64(squaredEuclideanOracle(vectors[0], vectors[1])))), firstEuclidean)
			requireFloat32Close(t, float32(math.Sqrt(float64(squaredEuclideanOracle(vectors[0], vectors[2])))), secondEuclidean)

			first, second, third, fourth := SquaredEuclideanDistances4(
				vectors[0], vectors[1], vectors[2], vectors[3], vectors[4],
			)
			for index, actual := range []float32{first, second, third, fourth} {
				requireFloat32Close(t, squaredEuclideanOracle(vectors[0], vectors[index+1]), actual)
			}
			firstEuclidean, secondEuclidean, thirdEuclidean, fourthEuclidean := EuclideanDistances4(
				vectors[0], vectors[1], vectors[2], vectors[3], vectors[4],
			)
			for index, actual := range []float32{firstEuclidean, secondEuclidean, thirdEuclidean, fourthEuclidean} {
				requireFloat32Close(t, float32(math.Sqrt(float64(squaredEuclideanOracle(vectors[0], vectors[index+1])))), actual)
			}
		})
	}
}

func squaredEuclideanOracle(left, right []float32) (sum float32) {
	for index, value := range left {
		difference := value - right[index]
		sum += difference * difference
	}
	return
}

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
		benchmarkBatch2First, benchmarkBatch2Second = innerProducts2(query, candidate, candidate)
		benchmarkBatch4First, benchmarkBatch4Second, benchmarkBatch4Third, benchmarkBatch4Fourth = innerProducts4(query, candidate, candidate, candidate, candidate)
	}))
	require.Equal(t, queryCopy, query)
	require.Equal(t, candidateCopy, candidate)
}

func BenchmarkSquaredEuclideanDistancesOneToMany(b *testing.B) {
	const candidatesCount = 32
	for _, dimension := range []int{128, 768} {
		query := make([]float32, dimension)
		candidates := make([][]float32, candidatesCount)
		for candidate := range candidates {
			candidates[candidate] = make([]float32, dimension)
			for index := range query {
				query[index] = float32(index%17)/17 - 0.5
				candidates[candidate][index] = float32((index+candidate)%23)/23 - 0.5
			}
		}
		output := make([]float32, candidatesCount)
		b.Run(fmt.Sprintf("Sequential/%d", dimension), func(b *testing.B) {
			for b.Loop() {
				for index := range candidates {
					output[index] = mathutil.L2Squared(query, candidates[index])
				}
			}
			benchmarkOutput = output
		})
		b.Run(fmt.Sprintf("OneToMany/%d", dimension), func(b *testing.B) {
			for b.Loop() {
				SquaredEuclideanDistances(query, candidates, output)
			}
			benchmarkOutput = output
		})
	}
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

func testBatchKernels(t *testing.T, dot2 batch2Kernel, dot4 batch4Kernel, l2Squared2 batch2Kernel, l2Squared4 batch4Kernel) {
	t.Helper()
	for _, dimension := range []int{1, 3, 4, 7, 8, 15, 16, 17, 127, 128, 129, 768, 1536} {
		t.Run(fmt.Sprintf("dimension_%d", dimension), func(t *testing.T) {
			random := rand.New(rand.NewSource(int64(dimension * 11)))
			vectors := make([][]float32, 5)
			for vector := range vectors {
				vectors[vector] = make([]float32, dimension+1)
				for index := 1; index <= dimension; index++ {
					vectors[vector][index] = random.Float32()*2 - 1
				}
				vectors[vector] = vectors[vector][1:]
			}

			first, second := dot2(vectors[0], vectors[1], vectors[2])
			requireFloat32Close(t, innerProductOracle(vectors[0], vectors[1]), first)
			requireFloat32Close(t, innerProductOracle(vectors[0], vectors[2]), second)
			first, second, third, fourth := dot4(vectors[0], vectors[1], vectors[2], vectors[3], vectors[4])
			for index, actual := range []float32{first, second, third, fourth} {
				requireFloat32Close(t, innerProductOracle(vectors[0], vectors[index+1]), actual)
			}

			first, second = l2Squared2(vectors[0], vectors[1], vectors[2])
			requireFloat32Close(t, squaredEuclideanOracle(vectors[0], vectors[1]), first)
			requireFloat32Close(t, squaredEuclideanOracle(vectors[0], vectors[2]), second)
			first, second, third, fourth = l2Squared4(vectors[0], vectors[1], vectors[2], vectors[3], vectors[4])
			for index, actual := range []float32{first, second, third, fourth} {
				requireFloat32Close(t, squaredEuclideanOracle(vectors[0], vectors[index+1]), actual)
			}
		})
	}
}

func requireFloat32Close(t *testing.T, expected, actual float32) {
	t.Helper()
	tolerance := float32(1e-5) * max(1, float32(math.Abs(float64(expected))))
	require.InDelta(t, expected, actual, float64(tolerance))
}

func TestInnerProductsInt8(t *testing.T) {
	testInnerProductsInt8(t, InnerProductsInt8)
}

func TestInnerProductsInt4(t *testing.T) {
	query := []byte{0x87, 0xf1, 0x20}
	candidates := [][]byte{
		{0x78, 0x1f, 0x02},
		{0x87, 0xf1, 0x20},
		{0x00, 0x00, 0x00},
		{0xff, 0xff, 0xff},
		{0x12, 0x34, 0x56},
	}
	output := make([]int64, len(candidates))
	InnerProductsInt4(query, candidates, output)
	for i := range candidates {
		require.Equal(t, mathutil.InnerProductInt4(query, candidates[i]), output[i])
	}
}

func TestInnerProductsInt8Scalar4(t *testing.T) {
	testInnerProductsInt8(t, int8BatchWithKernel(innerProductsInt8Scalar4))
}

func int8BatchWithKernel(kernel func([]byte, []byte, []byte, []byte, []byte, []int64)) func([]byte, [][]byte, []int64) {
	return func(query []byte, candidates [][]byte, output []int64) {
		i := 0
		for ; i+4 <= len(candidates); i += 4 {
			kernel(query, candidates[i], candidates[i+1], candidates[i+2], candidates[i+3], output[i:i+4])
		}
		for ; i < len(candidates); i++ {
			output[i] = mathutil.InnerProductInt8(query, candidates[i])
		}
	}
}

func testInnerProductsInt8(t *testing.T, batch func([]byte, [][]byte, []int64)) {
	t.Helper()
	for _, dimension := range []int{0, 1, 15, 31, 32, 33, 63, 64, 65, 767, 768, 769, 65535, 65536, 65537, 131073} {
		for _, count := range []int{0, 1, 3, 4, 5, 7, 8, 9} {
			t.Run(fmt.Sprintf("dim=%d/count=%d", dimension, count), func(t *testing.T) {
				query := make([]byte, dimension+1)[1:]
				candidates := make([][]byte, count)
				for j := range candidates {
					offset := (j*7 + 3) % 32
					candidates[j] = make([]byte, dimension+offset)[offset:]
				}
				for _, extreme := range []bool{false, true} {
					for i := range query {
						query[i] = byte(i*13 + 5)
						if extreme {
							query[i] = 128
						}
					}
					want := make([]int64, count)
					for j := range candidates {
						for i := range query {
							candidates[j][i] = byte(i*29 + j*7)
							if extreme {
								if j%2 == 0 {
									candidates[j][i] = 128
								} else {
									candidates[j][i] = 127
								}
							}
							want[j] += int64(int8(query[i])) * int64(int8(candidates[j][i]))
						}
					}
					output := make([]int64, count+2)
					output[count], output[count+1] = 71, 73
					batch(query, candidates, output[:count])
					require.Equal(t, want, output[:count])
					require.Equal(t, []int64{71, 73}, output[count:])
				}
			})
		}
	}
}

func BenchmarkInnerProductsInt8(b *testing.B) {
	query := make([]byte, 768)
	candidates := make([][]byte, 32)
	for j := range candidates {
		candidates[j] = make([]byte, len(query))
		for i := range query {
			query[i], candidates[j][i] = byte(i), byte(i+j)
		}
	}
	output := make([]int64, len(candidates))
	for _, name := range []string{"single", "batch"} {
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if name == "batch" {
					InnerProductsInt8(query, candidates, output)
				} else {
					for i := range candidates {
						output[i] = mathutil.InnerProductInt8(query, candidates[i])
					}
				}
			}
		})
	}
}

var (
	benchmarkOutput       []float32
	benchmarkFirst        float32
	benchmarkSecond       float32
	benchmarkThird        float32
	benchmarkFourth       float32
	benchmarkBatch2First  float32
	benchmarkBatch2Second float32
	benchmarkBatch4First  float32
	benchmarkBatch4Second float32
	benchmarkBatch4Third  float32
	benchmarkBatch4Fourth float32
)
