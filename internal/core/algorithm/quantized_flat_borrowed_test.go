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
	"fmt"
	"slices"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestQuantizedFlatBorrowedVectors(t *testing.T) {
	ctx := context.Background()
	for _, kind := range []Quantization{QuantizationFP16, QuantizationInt8, QuantizationInt4} {
		t.Run(fmt.Sprint(kind), func(t *testing.T) {
			candidates := quantizedIndexCandidates(40)
			var reformer DenseReformer
			if kind != QuantizationFP16 {
				rotator, err := NewFHTRotatorFromSigns(4, []byte{0x13, 0x57, 0x9b, 0xdf})
				require.NoError(t, err)
				reformer, err = NewRotationReformer(rotator)
				require.NoError(t, err)
			}
			owned, err := NewScalarQuantizedFlatIndex(ctx, 4, MetricCosine, kind, reformer, candidates)
			require.NoError(t, err)
			borrowed, err := NewScalarQuantizedFlatIndexWithBorrowedVectors(ctx, 4, MetricCosine, kind, reformer, candidates)
			require.NoError(t, err)
			require.Nil(t, borrowed.vectors.originals)
			require.Same(t, &candidates[0].Vector[0], &borrowed.vectors.originalRows[0][0])
			key, original := candidates[0].Key, slices.Clone(candidates[0].Vector)
			candidates[0] = Candidate{Key: 9999}
			gotVector, found := borrowed.Vector(key)
			require.True(t, found)
			require.Equal(t, original, gotVector)
			gotVector[0] += 100
			again, _ := borrowed.Vector(key)
			require.Equal(t, original, again)
			for _, filter := range []func(uint64) bool{nil, func(key uint64) bool { return key%3 != 0 }} {
				options := SearchOptions{TopK: 10, Filter: filter}
				want, err := owned.SearchWithOptions(ctx, original, options)
				require.NoError(t, err)
				got, err := borrowed.SearchWithOptions(ctx, original, options)
				require.NoError(t, err)
				require.Equal(t, want, got)
				ownedRefiner, err := NewOriginalVectorRefiner(owned, MetricCosine)
				require.NoError(t, err)
				borrowedRefiner, err := NewOriginalVectorRefiner(borrowed, MetricCosine)
				require.NoError(t, err)
				wantRefined, err := RefinedSearch(ctx, owned, ownedRefiner, original, options, 100)
				require.NoError(t, err)
				gotRefined, err := RefinedSearch(ctx, borrowed, borrowedRefiner, original, options, 100)
				require.NoError(t, err)
				require.Equal(t, wantRefined, gotRefined)
			}
		})
	}
}

func TestQuantizedFlatBorrowedVectorValidation(t *testing.T) {
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	for _, tc := range []struct {
		name       string
		ctx        context.Context
		dimension  int
		candidates []Candidate
	}{
		{"nil context", nil, 4, nil},
		{"canceled", canceled, 4, quantizedIndexCandidates(2)},
		{"dimension", context.Background(), 0, nil},
		{"short row", context.Background(), 4, []Candidate{{Key: 1, Vector: []float32{1}}}},
		{"duplicate", context.Background(), 4, []Candidate{{Key: 1, Vector: []float32{1, 2, 3, 4}}, {Key: 1, Vector: []float32{4, 3, 2, 1}}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewScalarQuantizedFlatIndexWithBorrowedVectors(tc.ctx, tc.dimension, MetricL2, QuantizationInt4, nil, tc.candidates)
			require.Error(t, err)
		})
	}
}
