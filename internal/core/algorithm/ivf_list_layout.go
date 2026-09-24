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

import "context"

// packCosineLists places each list's originals next to one another so exact
// scans can stream through memory. Routing, list iteration order, and vector
// bits are unchanged. Call before caching magnitudes and publishing the index.
// Incremental additions still append normally; reopening packs them again.
func (i *IVFIndex) packCosineLists(ctx context.Context) error {
	if !i.cosineDotRouting || i.fp16 {
		return nil
	}
	packed := true
	next := 0
	for _, list := range i.lists {
		for _, position := range list.positions {
			if next&1023 == 0 {
				if err := ctx.Err(); err != nil {
					return err
				}
			}
			packed = packed && position == next
			next++
		}
	}
	if packed {
		return nil
	}

	keys := make([]uint64, len(i.keys))
	vectors := make([]float32, len(i.vectors))
	positions := make(map[uint64]int, len(i.keys))
	listForPosition := make([]int, len(i.keys))
	listPositions := make([]int, len(i.keys))
	lists := make([]ivfList, len(i.lists))
	next = 0
	for cluster, list := range i.lists {
		start := next
		for _, position := range list.positions {
			if next&1023 == 0 {
				if err := ctx.Err(); err != nil {
					return err
				}
			}
			key := i.keys[position]
			keys[next] = key
			positions[key] = next
			copy(vectors[next*i.dimension:(next+1)*i.dimension],
				i.vectors[position*i.dimension:(position+1)*i.dimension])
			listForPosition[next] = cluster
			listPositions[next] = next
			next++
		}
		// Limit capacity so Add cannot overwrite the next list's positions.
		lists[cluster].positions = listPositions[start:next:next]
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	// Commit together: a canceled build must leave builder-owned storage intact.
	i.keys, i.vectors, i.positions = keys, vectors, positions
	i.listForPosition, i.lists = listForPosition, lists
	return nil
}
