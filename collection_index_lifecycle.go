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
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"

	core "github.com/gorse-io/xvec/internal/core/algorithm"
	"github.com/gorse-io/xvec/internal/db/index/common"
)

// prepareSegmentRuntimesLocked is called at open/write/maintenance boundaries.
// Native indexes are only opened here; missing ANN artifacts use writer Flat.
func (c *Collection) prepareSegmentRuntimesLocked(ctx context.Context) error {
	if appended, err := c.appendOwnedMutableRuntimeLocked(ctx); appended || err != nil {
		return err
	}
	segments, err := c.segmentDocumentsLocked(ctx)
	if err != nil {
		return err
	}
	_, err = c.segmentRuntimeIndexesLocked(ctx, segments)
	return err
}

func (c *Collection) cachedSegmentRuntimeIndexesLocked(ctx context.Context, segments []collectionSegmentDocuments) ([]*collectionSegmentRuntime, error) {
	c.indexMu.RLock()
	defer c.indexMu.RUnlock()
	runtimes := make([]*collectionSegmentRuntime, 0, len(segments))
	for _, segment := range segments {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if len(segment.documents) == 0 {
			continue
		}
		key, err := collectionRuntimeKeyFor(c.schema, segment.documents)
		if err != nil {
			return nil, err
		}
		runtime := c.segmentIndexes[segment.metadata.ID]
		if runtime == nil || runtime.key != key {
			return nil, fmt.Errorf("segment %d has no prepared query runtime", segment.metadata.ID)
		}
		runtimes = append(runtimes, runtime)
	}
	return runtimes, nil
}

// Append-only Flat storage can be shared with in-flight snapshots: searches
// bound keys to their snapshot's physical range before applying visibility.
func boundCollectionFilter(documents []Document, filter core.CandidateFilter) core.CandidateFilter {
	if len(documents) == 0 {
		return func(uint64) bool { return false }
	}
	first, last := documents[0].DocID, documents[len(documents)-1].DocID
	return func(key uint64) bool { return key >= first && key <= last && (filter == nil || filter(key)) }
}

func (i *collectionRuntimeIndexes) searchVectorSpec(field string, configured collectionVectorIndex) collectionVectorIndex {
	if i.writerFlat[field] {
		configured.indexType = IndexTypeFlat
		configured.quantize = QuantizeTypeUndefined
		configured.rotate = false
	}
	return configured
}

func (i *collectionRuntimeIndexes) appendWriterFlat(ctx context.Context, field FieldSchema, spec collectionVectorIndex, documents []Document, previous *collectionRuntimeIndexes) error {
	if spec.indexType != IndexTypeFlat {
		i.writerFlat[field.Name] = true
	}
	start := 0
	if field.DataType.IsDenseVector() {
		var index *core.DenseFlatIndex
		if previous != nil {
			index, _ = previous.denseExact[field.Name].(*core.DenseFlatIndex)
			if index != nil {
				start = previous.key.count
			}
		}
		if index == nil {
			var err error
			if field.DataType == DataTypeVectorFP16 {
				index, err = core.NewDenseFlatIndexFP16FromBorrowedRows(ctx, int(field.Dimension), spec.metric, nil, nil)
			} else {
				index, err = core.NewDenseFlatIndex(int(field.Dimension), spec.metric)
			}
			if err != nil {
				return err
			}
		}
		for _, document := range documents[start:] {
			raw, found := document.Fields[field.Name]
			if !found || raw == nil {
				continue
			}
			var addErr error
			if vector, ok := raw.(VectorFP16); ok {
				addErr = index.AddBorrowedFP16(ctx, document.DocID, nativeFP16Bits(vector))
			} else {
				vector, err := denseValueToFloat32Borrowed(raw)
				if err != nil {
					return err
				}
				addErr = index.Add(ctx, document.DocID, vector)
			}
			if addErr != nil && !errors.Is(addErr, core.ErrDuplicateKey) {
				return addErr
			}
		}
		i.denseExact[field.Name], i.denseFlat[field.Name], i.denseNative[field.Name] = index, index, index
	} else {
		var index *core.SparseFlatIndex
		if previous != nil {
			index = previous.sparseExact[field.Name]
			if index != nil {
				start = previous.key.count
			}
		}
		if index == nil {
			var err error
			index, err = core.NewSparseFlatIndex(core.MetricIP)
			if err != nil {
				return err
			}
		}
		for _, document := range documents[start:] {
			raw, found := document.Fields[field.Name]
			if !found || raw == nil {
				continue
			}
			vector, err := sparseValueToCore(raw)
			if err != nil {
				return err
			}
			if err := index.AddSparse(ctx, document.DocID, vector); err != nil && !errors.Is(err, core.ErrDuplicateKey) {
				return err
			}
		}
		i.sparseExact[field.Name], i.sparseFlat[field.Name], i.sparseNative[field.Name] = index, index, index
	}
	return nil
}

func (c *Collection) segmentNativeArtifactsComplete(snapshot common.SegmentIndexSnapshotMetadata) bool {
	kinds := make(map[string]bool, len(snapshot.Artifacts))
	for _, artifact := range snapshot.Artifacts {
		kinds[collectionIndexArtifactKey(artifact.Field, artifact.Kind)] = true
	}
	for _, field := range c.schema.Fields {
		if (field.IndexType() == IndexTypeFTS && !kinds[collectionIndexArtifactKey(field.Name, collectionFTSArtifactKind)]) ||
			(field.IndexType() == IndexTypeInvert && !kinds[collectionIndexArtifactKey(field.Name, collectionInvertArtifactKind)]) {
			return false
		}
		if field.DataType.IsVector() && field.IndexType() != IndexTypeFlat &&
			!kinds[collectionIndexArtifactKey(field.Name, collectionVectorArtifactKind(field.IndexType()))] {
			return false
		}
	}
	return true
}

type collectionCandidateReader []core.Candidate

func (r collectionCandidateReader) ReadVector(position int, destination []float32) error {
	copy(destination, r[position].Vector)
	return nil
}
func (i *collectionOriginalDenseIndex) vectorReader() core.DenseVectorReader {
	if i.reader != nil {
		return i.reader
	}
	return collectionCandidateReader(i.candidates)
}

// Index artifacts depend on their field definition. A schema-only change must
// not discard another field's already-built index.
func (c *Collection) rebindUnchangedIndexArtifacts(nextSchema CollectionSchema, encodedSchema []byte) ([]common.SegmentIndexSnapshotMetadata, error) {
	oldKey, err := collectionRuntimeKeyFor(c.schema, nil)
	if err != nil {
		return nil, err
	}
	oldHash := hex.EncodeToString(oldKey.schemaHash[:])
	nextHash := sha256.Sum256(encodedSchema)
	var retained []common.SegmentIndexSnapshotMetadata
	for _, snapshot := range c.store.Manifest().SegmentIndexSnapshots {
		if snapshot.SchemaSHA256 != oldHash || c.schema.Name != nextSchema.Name {
			continue
		}
		artifacts := make([]common.IndexArtifactMetadata, 0, len(snapshot.Artifacts))
		for _, artifact := range snapshot.Artifacts {
			before, oldFound := c.schema.Field(artifact.Field)
			after, newFound := nextSchema.Field(artifact.Field)
			if oldFound && newFound && equalFieldSchema(before, after) {
				artifacts = append(artifacts, artifact)
			}
		}
		if len(artifacts) == 0 {
			continue
		}
		snapshot.Artifacts = artifacts
		snapshot.SchemaSHA256 = hex.EncodeToString(nextHash[:])
		retained = append(retained, snapshot)
	}
	return retained, nil
}
