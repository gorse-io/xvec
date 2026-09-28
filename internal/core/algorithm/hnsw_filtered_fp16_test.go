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
	"math"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestScalarQuantizedHNSWFilteredFP16Prefetch(t *testing.T) {
	candidates := quantizedIndexCandidates(DefaultHNSWBruteForceThreshold + 200)
	base := buildDenseHNSWFromCandidates(t, candidates)
	index, err := NewScalarQuantizedHNSWIndex(context.Background(), base, QuantizationFP16, nil)
	require.NoError(t, err)
	for _, radius := range []float32{0, 20} {
		options := HNSWSearchOptions{SearchOptions: SearchOptions{TopK: 20, Radius: radius, Filter: func(key uint64) bool { return key%2 == 0 }}, EF: 120}
		for _, q := range []int{19, 70, 123} {
			want, err := index.SearchHNSW(context.Background(), candidates[q].Vector, options)
			require.NoError(t, err)
			for _, hint := range [][2]uint32{{8, 0}, {8, 1}, {math.MaxUint32, math.MaxUint32}} {
				prefetched := options
				prefetched.PrefetchOffset, prefetched.PrefetchLines = hint[0], hint[1]
				got, err := index.SearchHNSW(context.Background(), candidates[q].Vector, prefetched)
				require.NoError(t, err)
				require.Equal(t, want, got)
			}
		}
	}
}
