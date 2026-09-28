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
	"github.com/stretchr/testify/require"
	"testing"
)

func TestQuantizedFlatKeySearchMatchesFilteredScan(t *testing.T) {
	ctx := context.Background()
	candidates := quantizedIndexCandidates(80)
	for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine} {
		for _, kind := range []Quantization{QuantizationInt4, QuantizationInt8, QuantizationFP16} {
			t.Run(fmt.Sprintf("%v/%v", metric, kind), func(t *testing.T) {
				index, err := NewScalarQuantizedFlatIndex(ctx, 4, metric, kind, nil, candidates)
				require.NoError(t, err)
				keys := []uint64{candidates[5].Key, candidates[17].Key, candidates[29].Key, candidates[5].Key, ^uint64(0)}
				selected := map[uint64]bool{}
				for _, key := range keys {
					selected[key] = true
				}
				for _, radius := range []float32{0, 1, 20} {
					options := SearchOptions{TopK: 5, Radius: radius, Filter: func(key uint64) bool { return key != candidates[17].Key }}
					full := options
					full.Filter = func(key uint64) bool { return selected[key] && options.Filter(key) }
					expected, err := index.SearchWithOptions(ctx, candidates[5].Vector, full)
					require.NoError(t, err)
					got, err := index.SearchKeysWithOptions(ctx, candidates[5].Vector, keys, options)
					require.NoError(t, err)
					require.Equal(t, expected, got)
				}
				empty, err := index.SearchKeysWithOptions(ctx, candidates[0].Vector, nil, SearchOptions{TopK: 5})
				require.NoError(t, err)
				require.Empty(t, empty)
				_, err = index.SearchKeysWithOptions(nil, candidates[0].Vector, keys, SearchOptions{TopK: 5})
				require.Error(t, err)
				_, err = index.SearchKeysWithOptions(ctx, []float32{1}, keys, SearchOptions{TopK: 5})
				require.Error(t, err)
				_, err = index.SearchKeysWithOptions(ctx, candidates[0].Vector, keys, SearchOptions{})
				require.ErrorIs(t, err, ErrInvalidTopK)
				canceled, cancel := context.WithCancel(ctx)
				cancel()
				_, err = index.SearchKeysWithOptions(canceled, candidates[0].Vector, keys, SearchOptions{TopK: 5})
				require.ErrorIs(t, err, context.Canceled)
			})
		}
	}
}
