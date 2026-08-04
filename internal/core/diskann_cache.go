// Copyright 2026-present the zvec-go project
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
	"container/list"
	"errors"
	"sync"
)

// DiskANNCacheStats is a point-in-time snapshot of cache activity.
type DiskANNCacheStats struct {
	Hits      uint64
	Misses    uint64
	Evictions uint64
}

type diskANNCacheEntry struct {
	id   uint32
	node DiskANNNode
}

// DiskANNNodeCache is a bounded concurrency-safe LRU of immutable node copies.
type DiskANNNodeCache struct {
	mu       sync.Mutex
	capacity int
	items    map[uint32]*list.Element
	order    *list.List
	stats    DiskANNCacheStats
}

func NewDiskANNNodeCache(capacity int) (*DiskANNNodeCache, error) {
	if capacity < 0 {
		return nil, errors.New("core: negative DiskANN cache capacity")
	}
	return &DiskANNNodeCache{capacity: capacity, items: make(map[uint32]*list.Element), order: list.New()}, nil
}

func (c *DiskANNNodeCache) Get(id uint32) (DiskANNNode, bool) {
	if c == nil {
		return DiskANNNode{}, false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	element, found := c.items[id]
	if !found {
		c.stats.Misses++
		return DiskANNNode{}, false
	}
	c.order.MoveToFront(element)
	c.stats.Hits++
	return cloneDiskANNNode(element.Value.(diskANNCacheEntry).node), true
}

func (c *DiskANNNodeCache) Put(node DiskANNNode) {
	if c == nil || c.capacity == 0 {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if element, found := c.items[node.ID]; found {
		element.Value = diskANNCacheEntry{id: node.ID, node: cloneDiskANNNode(node)}
		c.order.MoveToFront(element)
		return
	}
	element := c.order.PushFront(diskANNCacheEntry{id: node.ID, node: cloneDiskANNNode(node)})
	c.items[node.ID] = element
	if c.order.Len() > c.capacity {
		oldest := c.order.Back()
		entry := oldest.Value.(diskANNCacheEntry)
		delete(c.items, entry.id)
		c.order.Remove(oldest)
		c.stats.Evictions++
	}
}

func (c *DiskANNNodeCache) Len() int {
	if c == nil {
		return 0
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.order.Len()
}

func (c *DiskANNNodeCache) Capacity() int {
	if c == nil {
		return 0
	}
	return c.capacity
}

func (c *DiskANNNodeCache) Stats() DiskANNCacheStats {
	if c == nil {
		return DiskANNCacheStats{}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.stats
}

func (c *DiskANNNodeCache) Clear() {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.items = make(map[uint32]*list.Element)
	c.order.Init()
}
