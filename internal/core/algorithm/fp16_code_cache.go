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
	"slices"
	"sync"
)

const fp16CacheShards = 16

// fp16CodeCache bounds retained code bytes per field. Sharded FIFO replacement
// avoids a collection-sized arena and global locking on sparse scan hits.
// Entries are immutable; eviction never overwrites storage held by a reader.
type fp16CodeCache struct {
	shards   [fp16CacheShards]fp16CodeCacheShard
	capacity int
}

type fp16CodeCacheShard struct {
	mu    sync.RWMutex
	codes map[int][]byte
	slots []int
	next  int
}

func newFP16CodeCache(dimension, count, budget int) *fp16CodeCache {
	capacity := max(1, min(count, budget/(dimension*2)))
	return &fp16CodeCache{capacity: capacity}
}

func (c *fp16CodeCache) get(position int) []byte {
	shard := &c.shards[position%fp16CacheShards]
	shard.mu.RLock()
	code := shard.codes[position]
	shard.mu.RUnlock()
	return code
}

func (c *fp16CodeCache) put(position int, code []byte) []byte {
	shard := &c.shards[position%fp16CacheShards]
	shard.mu.Lock()
	defer shard.mu.Unlock()
	if existing := shard.codes[position]; existing != nil {
		return existing
	}
	limit := c.capacity / fp16CacheShards
	if position%fp16CacheShards < c.capacity%fp16CacheShards {
		limit++
	}
	if limit == 0 {
		return code
	}
	if shard.codes == nil {
		shard.codes = make(map[int][]byte)
		shard.slots = make([]int, 0, limit)
	}
	owned := slices.Clone(code)
	if len(shard.slots) < limit {
		shard.slots = append(shard.slots, position)
	} else {
		delete(shard.codes, shard.slots[shard.next])
		shard.slots[shard.next] = position
		shard.next = (shard.next + 1) % limit
	}
	shard.codes[position] = owned
	return owned
}
