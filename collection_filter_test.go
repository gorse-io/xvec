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
	"testing"

	"github.com/gorse-io/xvec/internal/db/index/common"
	"github.com/stretchr/testify/require"
)

func TestSnapshotFilterCandidatesMatchForwardEvaluation(t *testing.T) {
	ctx := context.Background()
	schema := NewCollectionSchema("filter_candidates",
		FieldSchema{Name: "label", DataType: DataTypeString, Nullable: true, Index: NewInvertIndexParams()},
		FieldSchema{Name: "number", DataType: DataTypeInt64, Index: NewInvertIndexParams()},
		FieldSchema{Name: "plain", DataType: DataTypeString},
		FieldSchema{Name: "tags", DataType: DataTypeArrayString, Nullable: true, Index: NewInvertIndexParams()},
	)
	docs := make([]Document, 24)
	for j := range docs {
		var label any = "keep"
		if j%3 == 0 {
			label = "drop"
		}
		if j%7 == 0 {
			label = nil
		}
		var tags any = StringArray{"a", "b"}
		if j%3 == 0 {
			tags = StringArray{}
		}
		if j%5 == 0 {
			tags = nil
		}
		docs[j] = Document{DocID: uint64(101 + j), PrimaryKey: fmt.Sprint(j), Fields: map[string]any{"label": label, "number": int64(j), "plain": fmt.Sprintf("text-%d", j%4), "tags": tags}}
	}
	segments := []collectionSegmentDocuments{
		{metadata: common.SegmentMetadata{ID: 1}, documents: docs[:12]},
		{metadata: common.SegmentMetadata{ID: 2}, documents: docs[12:]},
	}
	expressions := []string{"label = 'keep'", "label = 'absent'", "label != 'keep'", "label IS NULL", "label IS NOT NULL", "number >= 5 AND number < 19", "label = 'keep' AND plain = 'text-1'", "label = 'drop' OR plain = 'text-2'", "label = 'keep' OR number = 0", "tags CONTAIN_ANY ('b')", "array_length(tags) = 0", "plain LIKE '%xt-1'", "number < 0", "label NOT IN ('keep')"}
	for _, visibility := range []string{"all", "single", "single-empty", "some", "none"} {
		t.Run(visibility, func(t *testing.T) {
			localSegments := segments
			if visibility == "single" || visibility == "single-empty" {
				localSegments = []collectionSegmentDocuments{{metadata: common.SegmentMetadata{ID: 1}, documents: docs}}
			}
			if visibility == "single-empty" {
				localSegments = append([]collectionSegmentDocuments{{metadata: common.SegmentMetadata{ID: 2}}}, localSegments...)
				localSegments = append(localSegments, collectionSegmentDocuments{metadata: common.SegmentMetadata{ID: 3}})
			}
			var live []Document
			for j, doc := range docs {
				if visibility == "all" || visibility == "single" || visibility == "single-empty" || visibility == "some" && j%4 != 0 {
					live = append(live, doc)
				}
			}
			mask, err := evaluateSegmentFilters(ctx, nil, live, localSegments, nil, .9)
			require.NoError(t, err)
			snapshot := &collectionQuerySnapshot{schema: schema, documents: live, documentOrdinals: indexDocumentOrdinals(live), segments: localSegments, liveFilter: mask}
			for _, expr := range expressions {
				for _, ratio := range []float32{.1, .9} {
					t.Run(fmt.Sprintf("%s/%.1f", expr, ratio), func(t *testing.T) {
						plan, err := buildFilterPlan(expr, schema)
						require.NoError(t, err)
						// Zero threshold forces row-by-row SQL evaluation, independently of the
						// exact bitmap path and snapshot visibility intersection being tested.
						expected, err := evaluateFilterDocuments(ctx, plan, live, 0)
						require.NoError(t, err)
						bitmapFilter, err := evaluateFilterDocumentsByOrdinal(ctx, plan, live, ratio, indexDocumentOrdinals(live))
						require.NoError(t, err)
						require.True(t, slices.Equal(expected.ordinals, bitmapFilter.ordinals))
						for _, doc := range docs {
							require.Equal(t, expected.predicate(doc.DocID), bitmapFilter.predicate(doc.DocID))
						}
						got, err := evaluateSnapshotFilters(ctx, plan, snapshot, ratio)
						require.NoError(t, err)
						require.Equal(t, expected.matched, got.global.matched)
						require.Equal(t, expected.total, got.global.total)
						require.True(t, slices.Equal(expected.ordinals, got.global.ordinals))
						for _, doc := range docs {
							require.Equal(t, expected.predicate(doc.DocID), got.global.predicate(doc.DocID))
						}
						for _, segment := range localSegments {
							if len(segment.documents) == 0 {
								_, found := got.local[segment.metadata.ID]
								require.False(t, found)
								continue
							}
							local := got.local[segment.metadata.ID]
							var want []uint32
							for ordinal, doc := range segment.documents {
								if expected.predicate(doc.DocID) {
									want = append(want, uint32(ordinal))
								}
							}
							require.True(t, slices.Equal(want, local.ordinals))
							require.Equal(t, uint64(len(want)), local.matched)
							for _, doc := range segment.documents {
								require.Equal(t, expected.predicate(doc.DocID), local.predicate(doc.DocID))
							}
						}
					})
				}
			}
			canceled, cancel := context.WithCancel(ctx)
			cancel()
			_, err = evaluateSnapshotFilters(canceled, nil, snapshot, .9)
			require.ErrorIs(t, err, context.Canceled)
		})
	}
}
