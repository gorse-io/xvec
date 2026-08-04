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
	"context"
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/gorse-io/zvec/internal/ailego"
)

// AddSparse incrementally inserts one unique key and canonical sparse vector.
// The insertion is planned on a private graph generation and becomes visible
// in one commit, so cancellation never exposes partial CSR or topology state.
func (i *SparseHNSWIndex) AddSparse(ctx context.Context, key uint64, vector SparseVector) error {
	if i == nil {
		return errors.New("core: nil sparse HNSW index")
	}
	if ctx == nil {
		return errors.New("core: nil sparse HNSW add context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if _, err := ailego.SparseInnerProduct(vector.Indices, vector.Values, nil, nil); err != nil {
		return fmt.Errorf("core: validate incremental sparse HNSW vector: %w", err)
	}

	i.streamMu.Lock()
	defer i.streamMu.Unlock()
	if err := ctx.Err(); err != nil {
		return err
	}

	i.mu.RLock()
	if _, exists := i.positions[key]; exists {
		i.mu.RUnlock()
		return fmt.Errorf("%w: %d", ErrDuplicateKey, key)
	}
	if len(i.keys) == maxPlatformInt() || uint64(len(i.keys)) >= math.MaxUint32 ||
		len(vector.Indices) > maxPlatformInt()-len(i.indices) {
		i.mu.RUnlock()
		return ErrSparseHNSWCapacity
	}
	working, err := cloneSparseHNSWIndex(ctx, i)
	i.mu.RUnlock()
	if err != nil {
		return err
	}

	random := splitMix64{state: working.levelRNGState}
	level := sampleHNSWLevel(&random, working.options.M)
	position := len(working.keys)
	working.keys = append(working.keys, key)
	working.indices = append(working.indices, vector.Indices...)
	working.values = append(working.values, vector.Values...)
	working.offsets = append(working.offsets, len(working.indices))
	working.positions[key] = position
	working.levels = append(working.levels, level)
	working.neighbors = append(working.neighbors, make([][]int, level+1))
	working.levelRNGState = random.state
	if err := working.insertBuiltNode(ctx, position); err != nil {
		return fmt.Errorf("core: insert incremental sparse HNSW node %d: %w", position, err)
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	i.mu.Lock()
	if err := ctx.Err(); err != nil {
		i.mu.Unlock()
		return err
	}
	i.keys = working.keys
	i.offsets = working.offsets
	i.indices = working.indices
	i.values = working.values
	i.positions = working.positions
	i.levels = working.levels
	i.neighbors = working.neighbors
	i.entryPoint = working.entryPoint
	i.maxLevel = working.maxLevel
	i.levelRNGState = working.levelRNGState
	i.mu.Unlock()
	return nil
}

// cloneSparseHNSWIndex copies one complete generation. Callers that clone a
// live streamable index must hold its read or write lock for the duration.
func cloneSparseHNSWIndex(ctx context.Context, source *SparseHNSWIndex) (*SparseHNSWIndex, error) {
	if ctx == nil {
		return nil, errors.New("core: nil sparse HNSW clone context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if source == nil {
		return nil, fmt.Errorf("%w: nil index", ErrInvalidSparseHNSWFile)
	}
	clone := &SparseHNSWIndex{
		options:       source.options,
		keys:          slices.Clone(source.keys),
		offsets:       slices.Clone(source.offsets),
		indices:       slices.Clone(source.indices),
		values:        slices.Clone(source.values),
		positions:     make(map[uint64]int, len(source.positions)),
		levels:        slices.Clone(source.levels),
		neighbors:     make([][][]int, len(source.neighbors)),
		entryPoint:    source.entryPoint,
		maxLevel:      source.maxLevel,
		levelRNGState: source.levelRNGState,
	}
	for key, position := range source.positions {
		clone.positions[key] = position
	}
	for position, levels := range source.neighbors {
		if position&127 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		clone.neighbors[position] = make([][]int, len(levels))
		for level, neighbors := range levels {
			clone.neighbors[position][level] = slices.Clone(neighbors)
		}
	}
	return clone, nil
}

var (
	_ SparseStreamer = (*SparseHNSWIndex)(nil)
	_ SparseIndex    = (*SparseHNSWIndex)(nil)
)
