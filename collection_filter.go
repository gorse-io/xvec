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

package xvec

import (
	"context"
	"fmt"
	"slices"

	"github.com/gorse-io/xvec/internal/db/sqlengine"
)

// evaluateSnapshotFilters intersects inverted/forward candidates with the
// snapshot's existing visibility masks. Work scales with matches, not with a
// new traversal of every live document on every query.
func evaluateSnapshotFilters(ctx context.Context, plan *sqlengine.Plan, snapshot *collectionQuerySnapshot, ratio float32) (evaluatedSegmentFilters, error) {
	if err := ctx.Err(); err != nil {
		return evaluatedSegmentFilters{}, err
	}
	if plan == nil {
		return snapshot.liveFilter, nil
	}
	runtimes := make(map[uint64]*collectionSegmentRuntime, len(snapshot.runtimes))
	for _, runtime := range snapshot.runtimes {
		runtimes[runtime.segmentID] = runtime
	}
	result := evaluatedSegmentFilters{local: make(map[uint64]evaluatedFilter, len(snapshot.segments))}
	globalMatches := make(map[uint64]struct{})
	var globalOrdinals []uint32
	for _, segment := range snapshot.segments {
		if err := ctx.Err(); err != nil {
			return evaluatedSegmentFilters{}, err
		}
		if len(segment.documents) == 0 {
			continue
		}
		live, found := snapshot.liveFilter.local[segment.metadata.ID]
		if !found {
			return evaluatedSegmentFilters{}, fmt.Errorf("live filter for segment %d is missing", segment.metadata.ID)
		}
		var cached sqlengine.IndexSet
		var documentOrdinals map[uint64]int
		if runtime := runtimes[segment.metadata.ID]; runtime != nil {
			var err error
			documentOrdinals = runtime.documentOrdinals
			cached, err = runtime.indexes.scalarIndexesForFields(ctx, plan.Fields())
			if err != nil {
				return evaluatedSegmentFilters{}, err
			}
		}
		local, err := evaluateFilterDocumentsByOrdinal(ctx, plan, segment.documents, ratio, documentOrdinals, cached)
		if err != nil {
			return evaluatedSegmentFilters{}, err
		}
		// A nil visibility predicate means every segment document is live.
		if live.predicate != nil {
			matches := make(map[uint64]struct{}, len(local.ordinals))
			accepted := local.ordinals[:0]
			for _, ordinal := range local.ordinals {
				if err := ctx.Err(); err != nil {
					return evaluatedSegmentFilters{}, err
				}
				key := segment.documents[ordinal].DocID
				if live.predicate(key) {
					matches[key] = struct{}{}
					accepted = append(accepted, ordinal)
				}
			}
			local.ordinals = accepted
			local.matched = uint64(len(accepted))
			local.predicate = func(key uint64) bool { _, ok := matches[key]; return ok }
		}
		local.total = live.matched
		result.local[segment.metadata.ID] = local
		// Empty writable segments do not contribute documents or live masks.
		// With one nonempty, wholly live segment, its local ordinals and
		// predicate are already global, even when empty segments also exist.
		if len(snapshot.liveFilter.local) == 1 && live.predicate == nil {
			result.global = local
			return result, nil
		}
		result.global.usedIndex = result.global.usedIndex || local.usedIndex
		for _, ordinal := range local.ordinals {
			if err := ctx.Err(); err != nil {
				return evaluatedSegmentFilters{}, err
			}
			key := segment.documents[ordinal].DocID
			globalOrdinal, ok := snapshot.documentOrdinals[key]
			if !ok {
				return evaluatedSegmentFilters{}, fmt.Errorf("live document %d is missing", key)
			}
			globalMatches[key] = struct{}{}
			globalOrdinals = append(globalOrdinals, uint32(globalOrdinal))
		}
	}
	slices.Sort(globalOrdinals)
	result.global.ordinals = globalOrdinals
	result.global.matched = uint64(len(globalMatches))
	result.global.total = uint64(len(snapshot.documents))
	result.global.present = true
	result.global.predicate = func(key uint64) bool { _, ok := globalMatches[key]; return ok }
	return result, nil
}
