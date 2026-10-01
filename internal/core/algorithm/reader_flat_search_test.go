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
	"github.com/stretchr/testify/require"
	"testing"
)

type flatSearchTestReader [][]float32

func (r flatSearchTestReader) ReadVector(position int, destination []float32) error {
	copy(destination, r[position])
	return nil
}

func TestDenseReaderSearchMatchesFlat(t *testing.T) {
	ctx := context.Background()
	keys := []uint64{11, 2, 7, 9}
	reader := flatSearchTestReader{{1, 2}, {2, 1}, {1, 2}, {-1, 0}}
	query := []float32{1, 1}
	for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine} {
		t.Run(fmt.Sprint(metric), func(t *testing.T) {
			flat, err := NewDenseFlatIndex(2, metric)
			require.NoError(t, err)
			for position, key := range keys {
				require.NoError(t, flat.Add(ctx, key, reader[position]))
			}
			for _, radius := range []float32{0, 2} {
				for _, filter := range []CandidateFilter{nil, func(key uint64) bool { return key%2 == 1 }} {
					options := SearchOptions{TopK: 2, Filter: filter, Radius: radius}
					want, err := flat.SearchWithOptions(ctx, query, options)
					require.NoError(t, err)
					got, err := SearchDenseReader(ctx, metric, query, keys, reader, options)
					require.NoError(t, err)
					require.Equal(t, want, got)
					groups := GroupByOptions{GroupCount: 2, TopKPerGroup: 1, Filter: filter, Radius: radius, Resolve: func(key uint64) (string, bool) { return fmt.Sprint(key % 3), key != 9 }}
					wantGroups, err := flat.SearchGroups(ctx, query, groups)
					require.NoError(t, err)
					gotGroups, err := SearchDenseReaderGroups(ctx, metric, query, keys, reader, groups)
					require.NoError(t, err)
					require.Equal(t, wantGroups, gotGroups)
				}
			}
		})
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	_, err := SearchDenseReader(canceled, MetricL2, query, keys, reader, SearchOptions{TopK: 2})
	require.ErrorIs(t, err, context.Canceled)
	failure := errors.New("read failed")
	_, err = SearchDenseReader(ctx, MetricL2, query, keys, failingFlatReader{failure}, SearchOptions{TopK: 2})
	require.ErrorIs(t, err, failure)
}

type failingFlatReader struct{ err error }

func (r failingFlatReader) ReadVector(int, []float32) error { return r.err }
