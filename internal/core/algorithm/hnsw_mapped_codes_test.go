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
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestHNSWMappedCodesCloseAndSharedFlatView(t *testing.T) {
	ctx := context.Background()
	candidates := quantizedIndexCandidates(80)
	base := buildDenseHNSWFromCandidates(t, candidates)
	path := filepath.Join(t.TempDir(), "index.hnsw")
	require.NoError(t, base.Save(ctx, path))
	for _, view := range []string{"hnsw", "flat"} {
		t.Run(view, func(t *testing.T) {
			index, err := OpenScalarQuantizedHNSWIndexWithBorrowedVectors(ctx, path, QuantizationFP16, nil, candidates, true)
			require.NoError(t, err)
			t.Cleanup(func() { require.NoError(t, index.Close()) })
			require.Len(t, index.vectors.mappedCodes, len(candidates)*4*2)
			flat := index.FlatIndex()
			entered, release := make(chan struct{}), make(chan struct{})
			var once sync.Once
			filter := func(uint64) bool {
				once.Do(func() { close(entered); <-release })
				return true
			}
			query := candidates[3].Vector
			searchDone := make(chan error, 1)
			go func() {
				var err error
				options := SearchOptions{TopK: 4, Filter: filter}
				if view == "flat" {
					_, err = flat.SearchKeysWithOptions(ctx, query, []uint64{candidates[0].Key}, options)
				} else {
					_, err = index.SearchWithOptions(ctx, query, options)
				}
				searchDone <- err
			}()
			select {
			case <-entered:
			case <-time.After(5 * time.Second):
				t.Fatal("search did not reach filter")
			}
			closeDone := make(chan error, 1)
			go func() { closeDone <- index.Close() }()
			select {
			case err := <-closeDone:
				t.Fatalf("Close released an arena still used by a search: %v", err)
			case <-time.After(20 * time.Millisecond):
			}
			close(release)
			require.NoError(t, <-searchDone)
			require.NoError(t, <-closeDone)
			require.NoError(t, index.Close())
			require.Nil(t, index.vectors.mappedCodes)
			require.Nil(t, index.vectors.codes)
			_, err = index.Search(ctx, query, 4)
			require.ErrorIs(t, err, ErrScalarQuantizedIndexClosed)
			_, err = index.SearchHNSWGroups(ctx, query, HNSWGroupSearchOptions{})
			require.ErrorIs(t, err, ErrScalarQuantizedIndexClosed)
			_, err = flat.Search(ctx, query, 0)
			require.ErrorIs(t, err, ErrScalarQuantizedIndexClosed)
			_, err = flat.SearchWithOptions(ctx, query, SearchOptions{TopK: 4})
			require.ErrorIs(t, err, ErrScalarQuantizedIndexClosed)
			_, err = flat.SearchGroups(ctx, query, GroupByOptions{})
			require.ErrorIs(t, err, ErrScalarQuantizedIndexClosed)
		})
	}
}
