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
	"encoding/binary"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestFP16CodeCacheBudgetEvictionAndOwnership(t *testing.T) {
	for _, budget := range []int{8, 128, 160, 256} {
		cache := newFP16CodeCache(4, 1024, budget)
		var held [][]byte
		for pass := range 4 {
			for position := range 1024 {
				scratch := make([]byte, 8)
				binary.LittleEndian.PutUint64(scratch, uint64(position+pass*1024))
				code := cache.put(position, scratch)
				held = append(held, code)
				// Cached values must own their memory, even after scratch is reused.
				if cache.get(position) != nil {
					scratch[0] ^= 255
				}
			}
		}
		bytes := 0
		for n := range cache.shards {
			shard := &cache.shards[n]
			for _, code := range shard.codes {
				bytes += len(code)
			}
			require.Equal(t, len(shard.slots), len(shard.codes))
		}
		require.LessOrEqual(t, bytes, budget)
		for position, code := range held {
			require.Equal(t, uint64(position), binary.LittleEndian.Uint64(code))
		}
	}
}

func TestFP16CodeCacheConcurrentEviction(t *testing.T) {
	cache := newFP16CodeCache(4, 1024, 128)
	var wg sync.WaitGroup
	for worker := range 8 {
		wg.Go(func() {
			scratch := make([]byte, 8)
			for pass := range 1024 {
				position := (pass*17 + worker) % 1024
				binary.LittleEndian.PutUint64(scratch, uint64(position))
				code := cache.put(position, scratch)
				if binary.LittleEndian.Uint64(code) != uint64(position) {
					t.Error("code changed after publication")
				}
				if code := cache.get(position); code != nil && binary.LittleEndian.Uint64(code) != uint64(position) {
					t.Error("cache returned another position")
				}
			}
		})
	}
	wg.Wait()
}
