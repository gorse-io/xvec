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

package core

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
	"unsafe"

	"github.com/gorse-io/xvec/internal/ailego/utility"
	"github.com/stretchr/testify/require"
)

func fp16DistanceTestVectors(t testing.TB, dimension int) (QuantizedVector, QuantizedVector) {
	t.Helper()
	left, right := make([]float32, dimension), make([]float32, dimension)
	for i := range left {
		left[i] = float32(math.Sin(float64(i)*.31)) * 8
		right[i] = float32(math.Cos(float64(i)*.17)) * 3
	}
	left[0] = float32(math.Copysign(0, -1))
	if dimension > 1 {
		left[1] = float32(math.Ldexp(1, -24))
		right[1] = -left[1]
	}
	l, err := QuantizeVector(QuantizationFP16, left)
	require.NoError(t, err)
	r, err := QuantizeVector(QuantizationFP16, right)
	require.NoError(t, err)
	return l, r
}

func unalignedFP16Codes(codes []byte) []byte {
	buffer := make([]byte, len(codes)+1)
	offset := 0
	if uintptr(unsafe.Pointer(unsafe.SliceData(buffer)))%2 == 0 {
		offset = 1
	}
	copy(buffer[offset:], codes)
	return buffer[offset : offset+len(codes)]
}

func TestFP16CodeDistancesMatchDecoded(t *testing.T) {
	t.Parallel()
	for _, dimension := range []int{1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 768, 769} {
		t.Run(fmt.Sprint(dimension), func(t *testing.T) {
			left, right := fp16DistanceTestVectors(t, dimension)
			ld, err := left.Decode()
			require.NoError(t, err)
			rd, err := right.Decode()
			require.NoError(t, err)
			unaligned := left
			unaligned.codes = unalignedFP16Codes(left.codes)
			_, ok := fp16CodeWords(unaligned.codes)
			require.False(t, ok)
			for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine, MetricMIPSL2} {
				distance, err := metric.Distance()
				require.NoError(t, err)
				want := distance(ld, rd)
				for _, l := range []QuantizedVector{left, unaligned} {
					got, err := QuantizedDistance(metric, l, right)
					require.NoError(t, err)
					require.InDelta(t, want, got, 2e-5*max(1, math.Abs(float64(want))))
				}
				// Exercise the portable byte-wise path even on little-endian SIMD hosts.
				got := fp16CodeDistanceScalar(metric, left.codes, right.codes)
				require.InDelta(t, want, got, 2e-5*max(1, math.Abs(float64(want))))
			}
		})
	}
	for _, values := range [][]float32{{0, 0}, {65504, -65504}, {0, float32(math.Ldexp(1, -24))}} {
		v, err := QuantizeVector(QuantizationFP16, values)
		require.NoError(t, err)
		zero, err := QuantizeVector(QuantizationFP16, []float32{0, 0})
		require.NoError(t, err)
		for _, r := range []QuantizedVector{v, zero} {
			for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine, MetricMIPSL2} {
				ld, err := v.Decode()
				require.NoError(t, err)
				rd, err := r.Decode()
				require.NoError(t, err)
				distance, err := metric.Distance()
				require.NoError(t, err)
				want := distance(ld, rd)
				got, err := QuantizedDistance(metric, v, r)
				require.NoError(t, err)
				require.InDelta(t, want, got, 2e-5*max(1, math.Abs(float64(want))))
				require.InDelta(t, want, fp16CodeDistanceScalar(metric, v.codes, r.codes), 2e-5*max(1, math.Abs(float64(want))))
			}
		}
	}
}

func TestFP16CodeValidationAllEncodings(t *testing.T) {
	t.Parallel()
	v := QuantizedVector{kind: QuantizationFP16, dimension: 1, codes: make([]byte, 2)}
	for bits := 0; bits <= math.MaxUint16; bits++ {
		binary.LittleEndian.PutUint16(v.codes, uint16(bits))
		decoded := float64(utility.Float16BitsToFloat32(uint16(bits)))
		err := v.validate()
		if math.IsNaN(decoded) || math.IsInf(decoded, 0) {
			require.ErrorIs(t, err, ErrInvalidQuantizedVector, "bits=%04x", bits)
		} else {
			require.NoError(t, err, "bits=%04x", bits)
		}
	}
}

func TestFP16CodeDistanceAllocations(t *testing.T) {
	left, right := fp16DistanceTestVectors(t, 768)
	storage := scalarQuantizedVectors{kind: QuantizationFP16, metric: MetricCosine, codes: []QuantizedVector{left}}
	var score float32
	for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine, MetricMIPSL2} {
		storage.metric = metric
		for _, trusted := range []bool{false, true} {
			allocs := testing.AllocsPerRun(100, func() {
				var err error
				if trusted {
					score, err = storage.distanceToCode(0, right)
				} else {
					score, err = QuantizedDistance(metric, left, right)
				}
				if err != nil {
					panic(err)
				}
			})
			require.Zero(t, allocs, "metric=%v trusted=%t", metric, trusted)
			require.False(t, math.IsNaN(float64(score)))
		}
	}
}

var fp16DistanceBenchmarkScore float32

func BenchmarkQuantizedDistanceFP16(b *testing.B) {
	left, right := fp16DistanceTestVectors(b, 768)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		var err error
		fp16DistanceBenchmarkScore, err = QuantizedDistance(MetricCosine, left, right)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFP16StorageDistance(b *testing.B) {
	left, right := fp16DistanceTestVectors(b, 768)
	storage := scalarQuantizedVectors{kind: QuantizationFP16, metric: MetricCosine, codes: []QuantizedVector{left}}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		var err error
		fp16DistanceBenchmarkScore, err = storage.distanceToCode(0, right)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func TestFP16HNSWTraversalAndReopen(t *testing.T) {
	ctx := context.Background()
	candidates := quantizedIndexCandidates(DefaultHNSWBruteForceThreshold + 101)
	index, err := NewScalarQuantizedHNSWIndex(ctx, buildDenseHNSWFromCandidates(t, candidates), QuantizationFP16, nil)
	require.NoError(t, err)
	query := candidates[333].Vector
	options := HNSWSearchOptions{SearchOptions: SearchOptions{TopK: 20, Filter: func(key uint64) bool { return key%3 != 0 }}, EF: len(candidates)}
	want := exactQuantizedResults(t, QuantizationFP16, MetricL2, nil, query, candidates, options.SearchOptions)
	require.NotEmpty(t, want)
	options.Radius = want[len(want)-1].Score + 1
	want = exactQuantizedResults(t, QuantizationFP16, MetricL2, nil, query, candidates, options.SearchOptions)
	path := filepath.Join(t.TempDir(), "graph")
	require.NoError(t, index.Save(ctx, path))
	reopened, err := OpenScalarQuantizedHNSWIndex(ctx, path, QuantizationFP16, nil)
	require.NoError(t, err)
	for _, source := range []*ScalarQuantizedHNSWIndex{index, reopened} {
		got, err := source.SearchHNSW(ctx, query, options)
		require.NoError(t, err)
		require.Equal(t, want, got)
		got, err = source.FlatIndex().SearchWithOptions(ctx, query, options.SearchOptions)
		require.NoError(t, err)
		require.Equal(t, want, got)
	}
	resaved := filepath.Join(t.TempDir(), "resaved")
	require.NoError(t, reopened.Save(ctx, resaved))
	original, err := os.ReadFile(path)
	require.NoError(t, err)
	restored, err := os.ReadFile(resaved)
	require.NoError(t, err)
	require.Equal(t, original, restored)
}
