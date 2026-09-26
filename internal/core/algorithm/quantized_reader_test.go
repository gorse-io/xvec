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
	"errors"
	"fmt"
	"slices"
	"testing"

	"github.com/stretchr/testify/require"
)

type testDenseReader struct {
	rows []Candidate
	fail bool
}

func (r *testDenseReader) ReadVector(position int, destination []float32) error {
	if r.fail {
		return errors.New("read failed")
	}
	copy(destination, r.rows[position].Vector)
	return nil
}

func TestQuantizedFlatVectorReader(t *testing.T) {
	ctx := context.Background()
	for _, kind := range []Quantization{QuantizationFP16, QuantizationInt8, QuantizationInt4} {
		for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine} {
			t.Run(fmt.Sprintf("%v/%v", kind, metric), func(t *testing.T) {
				candidates := quantizedIndexCandidates(40)
				keys := make([]uint64, len(candidates))
				for i, c := range candidates {
					keys[i] = c.Key
				}
				rotator, err := NewFHTRotatorFromSigns(4, []byte{1, 3, 5, 7})
				require.NoError(t, err)
				reformer, err := NewRotationReformer(rotator)
				require.NoError(t, err)
				reader := &testDenseReader{rows: candidates}
				got, err := NewScalarQuantizedFlatIndexWithVectorReader(ctx, 4, metric, kind, reformer, keys, reader)
				require.NoError(t, err)
				want, err := NewScalarQuantizedFlatIndex(ctx, 4, metric, kind, reformer, candidates)
				require.NoError(t, err)
				require.Nil(t, got.vectors.originals)
				require.Nil(t, got.vectors.originalRows)
				require.Same(t, reader, got.vectors.reader)
				key := keys[0]
				keys[0] = 9999
				original := slices.Clone(candidates[0].Vector)
				vector, found := got.Vector(key)
				require.True(t, found)
				require.Equal(t, original, vector)
				vector[0] += 100
				again, _ := got.Vector(key)
				require.Equal(t, original, again)
				query := candidates[7].Vector
				for _, filter := range []CandidateFilter{nil, func(key uint64) bool { return key%3 != 0 }} {
					options := SearchOptions{TopK: 12, Filter: filter}
					expected, err := want.SearchWithOptions(ctx, query, options)
					require.NoError(t, err)
					actual, err := got.SearchWithOptions(ctx, query, options)
					require.NoError(t, err)
					require.Equal(t, expected, actual)
					wantRefiner, err := NewOriginalVectorRefiner(want, metric)
					require.NoError(t, err)
					gotRefiner, err := NewOriginalVectorRefiner(got, metric)
					require.NoError(t, err)
					expected, err = RefinedSearch(ctx, want, wantRefiner, query, options, 100)
					require.NoError(t, err)
					actual, err = RefinedSearch(ctx, got, gotRefiner, query, options, 100)
					require.NoError(t, err)
					require.Equal(t, expected, actual)
				}
				reader.fail = true
				_, found = got.Vector(key)
				require.False(t, found)
				_, err = NewScalarQuantizedFlatIndexWithVectorReader(ctx, 4, metric, kind, reformer, []uint64{key}, reader)
				require.ErrorContains(t, err, "read failed")
			})
		}
	}
	_, err := NewScalarQuantizedFlatIndexWithVectorReader(ctx, 4, MetricL2, QuantizationInt4, nil, nil, nil)
	require.Error(t, err)
}

func TestFHTRotationReusableDestination(t *testing.T) {
	for _, dimension := range []int{1, 3, 4, 768, 1024} {
		t.Run(fmt.Sprint(dimension), func(t *testing.T) {
			signs := make([]byte, 4*((dimension+7)/8))
			for i := range signs {
				signs[i] = byte(i * 37)
			}
			rotator, err := NewFHTRotatorFromSigns(dimension, signs)
			require.NoError(t, err)
			vector := make([]float32, dimension)
			for i := range vector {
				vector[i] = float32(i%17) - 8
			}
			original := slices.Clone(vector)
			expected, err := rotator.Rotate(vector)
			require.NoError(t, err)
			destination := make([]float32, dimension)
			require.NoError(t, rotator.RotateInto(vector, destination))
			require.Equal(t, expected, destination)
			require.Equal(t, original, vector)
			allocs := testing.AllocsPerRun(20, func() {
				if err := rotator.RotateInto(vector, destination); err != nil {
					panic(err)
				}
			})
			require.Zero(t, allocs)
			require.NoError(t, rotator.RotateInto(vector, vector))
			require.Equal(t, expected, vector)
			require.Error(t, rotator.RotateInto(original, destination[:dimension-1]))
		})
	}
}
