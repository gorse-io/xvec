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

package sqlengine

import "github.com/gorse-io/xvec/internal/ailego/container"

// Each posting appears in at most one aggregate. Unlike cumulative prefix
// bitmaps, disjoint blocks bound extra storage to one more copy of the postings.
const invertedRangeBlockSize = 256

// buildRangeBlocks follows zvec's pre-aggregated range-posting approach.
// Aggregates are derived from validated postings on Seal and Open, so existing
// persisted indexes benefit without a format change. Call before publication.
func (i *InvertedIndex) buildRangeBlocks() {
	if !i.field.RangeOptimized || i.field.Array || len(i.ordered) < invertedRangeBlockSize {
		return
	}
	switch i.field.Kind {
	case ValueInt32, ValueInt64, ValueUint32, ValueUint64:
	default:
		return
	}
	i.rangeBlocks = make([]*container.Bitmap, len(i.ordered)/invertedRangeBlockSize)
	for block := range i.rangeBlocks {
		bitmap := container.NewBitmap(0)
		start := block * invertedRangeBlockSize
		for _, key := range i.ordered[start : start+invertedRangeBlockSize] {
			bitmap.Or(i.postings[key])
		}
		i.rangeBlocks[block] = bitmap
	}
}

// unionOrderedRange merges full blocks and only scans individual postings at
// the boundaries. Cached bitmaps are never returned or modified by a query.
// Small ranges still use the original postings, with no aggregate-copy cost.
func (i *InvertedIndex) unionOrderedRange(bitmap *container.Bitmap, start, end int) {
	for start < end {
		if len(i.rangeBlocks) != 0 && start%invertedRangeBlockSize == 0 &&
			end-start >= invertedRangeBlockSize {
			bitmap.Or(i.rangeBlocks[start/invertedRangeBlockSize])
			start += invertedRangeBlockSize
		} else {
			bitmap.Or(i.postings[i.ordered[start]])
			start++
		}
	}
}
