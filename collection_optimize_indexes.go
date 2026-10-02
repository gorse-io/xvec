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
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/gorse-io/xvec/internal/db"
	"github.com/gorse-io/xvec/internal/db/index/common"
)

type collectionArtifactBuilder func(context.Context, CollectionSchema, []Document, int, uint32) (*collectionRuntimeIndexes, error)

// The caller holds maintenanceMu, so schema changes, physical rewrites and
// Close cannot invalidate this source. Only snapshot/publication hold mu;
// writes, deletes and queries can continue throughout ANN construction.
func (c *Collection) buildAndPublishNativeIndexes(ctx context.Context, workers int, build collectionArtifactBuilder, columns ...string) error {
	c.mu.RLock()
	segments, err := c.segmentDocumentsLocked(ctx)
	schema := c.schema.Clone()
	manifest := c.store.Manifest()
	buildSchema := schema.Clone()
	if len(columns) != 0 {
		buildSchema.Fields = nil
		for _, field := range schema.Fields {
			for _, column := range columns {
				if field.Name == column {
					buildSchema.Fields = append(buildSchema.Fields, field)
				}
			}
		}
	}
	owner := &Collection{path: c.path, schema: buildSchema, options: c.options}
	sources := make([]*collectionSegmentRuntime, 0, len(c.segmentIndexes))
	if err == nil {
		for _, source := range c.segmentIndexes {
			source.retain()
			sources = append(sources, source)
		}
	}
	c.mu.RUnlock()
	if err != nil {
		return err
	}
	defer func() {
		for _, source := range sources {
			_ = c.releaseSegmentRuntime(source)
		}
	}()

	existing := make(map[uint64]common.SegmentIndexSnapshotMetadata, len(manifest.SegmentIndexSnapshots))
	for _, snapshot := range manifest.SegmentIndexSnapshots {
		existing[snapshot.SegmentID] = snapshot
	}
	prepared := make(map[uint64]*collectionSegmentRuntime)
	snapshots := make(map[uint64]common.SegmentIndexSnapshotMetadata)
	created := make([]string, 0)
	published := false
	defer func() {
		if !published {
			for _, runtime := range prepared {
				_ = c.releaseSegmentRuntime(runtime)
			}
			for _, path := range created {
				_ = os.RemoveAll(path)
			}
		}
	}()

	for _, segment := range segments {
		if err := ctx.Err(); err != nil {
			return err
		}
		if segment.mutable || len(segment.documents) == 0 {
			continue
		}
		key, err := collectionRuntimeKeyFor(schema, segment.documents)
		if err != nil {
			return err
		}
		if snapshot, found := existing[segment.metadata.ID]; found &&
			owner.segmentIndexSnapshotFilesExist(segment.metadata, key, snapshot) && owner.segmentNativeArtifactsComplete(snapshot) {
			continue
		}
		indexes, err := build(ctx, buildSchema, segment.documents, workers, owner.options.MaxBufferSize)
		if err != nil {
			return fmt.Errorf("build indexes for segment %d: %w", segment.metadata.ID, err)
		}
		artifacts, paths, writeErr := owner.writeSegmentRuntimeArtifacts(ctx, segment.metadata.ID, indexes)
		closeErr := indexes.Close()
		created = append(created, paths...)
		if err := errors.Join(writeErr, closeErr); err != nil {
			return err
		}
		if len(columns) != 0 {
			if previous, found := existing[segment.metadata.ID]; found && owner.segmentIndexSnapshotFilesExist(segment.metadata, key, previous) {
				for _, artifact := range previous.Artifacts {
					replaced := false
					for _, column := range columns {
						if artifact.Field == column {
							replaced = true
						}
					}
					if !replaced {
						artifacts = append(artifacts, artifact)
					}
				}
			}
		}
		if len(artifacts) == 0 {
			continue
		}
		pathsByField := make(map[string]string, len(artifacts))
		for _, artifact := range artifacts {
			pathsByField[collectionIndexArtifactKey(artifact.Field, artifact.Kind)] = filepath.Join(owner.path, filepath.FromSlash(artifact.File))
		}
		// Open the replacement before acquiring the publication lock.
		opened, err := buildCollectionRuntimeIndexes(ctx, schema, segment.documents, workers, owner.options.MaxBufferSize, owner.options.EnableMmap, pathsByField)
		if err != nil {
			return err
		}
		opened.key = key
		runtime := &collectionSegmentRuntime{segmentID: segment.metadata.ID, key: key, indexes: opened, documents: segment.documents, documentOrdinals: indexDocumentOrdinals(segment.documents)}
		runtime.refs.Store(1)
		prepared[segment.metadata.ID] = runtime
		snapshots[segment.metadata.ID] = common.SegmentIndexSnapshotMetadata{
			SegmentID: segment.metadata.ID, SchemaSHA256: hex.EncodeToString(key.schemaHash[:]),
			DocumentCount: uint64(key.count), MinDocumentID: segment.metadata.MinDocID,
			MaxDocumentID: segment.metadata.MaxDocID, Artifacts: artifacts,
		}
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return err
	}
	// Preserve artifacts of segments flushed/rotated while these indexes built.
	current := c.store.Manifest()
	next := make([]common.SegmentIndexSnapshotMetadata, 0, len(current.PersistedSegments))
	for _, snapshot := range current.SegmentIndexSnapshots {
		if _, replaced := snapshots[snapshot.SegmentID]; !replaced {
			next = append(next, snapshot)
		}
	}
	for _, segment := range current.PersistedSegments {
		if snapshot, found := snapshots[segment.ID]; found {
			next = append(next, snapshot)
		}
	}
	if len(snapshots) == 0 {
		return c.store.PruneObsoleteArtifacts(ctx)
	}
	committed, publishErr := c.store.PublishSegmentIndexSnapshots(ctx, next)
	if !committed {
		return publishErr
	}
	published = true
	c.invalidateQuerySnapshotLocked()
	c.indexMu.Lock()
	for id, runtime := range prepared {
		previous := c.segmentIndexes[id]
		c.segmentIndexes[id] = runtime
		c.indexBuildCount++
		_ = c.releaseSegmentRuntime(previous)
	}
	c.indexMu.Unlock()
	if publishErr != nil {
		return publishErr
	}
	return c.store.PruneObsoleteArtifacts(ctx)
}

// The caller holds maintenanceMu. Prepare every replacement runtime and its
// artifacts before taking mu, including Flat/scalar/FTS state and ANN opening.
func (c *Collection) buildAndPublishCompaction(ctx context.Context, workers int, compaction *db.Compaction) error {
	owner := &Collection{path: c.path, schema: c.schema.Clone(), options: c.options}
	prepared := make(map[uint64]*collectionSegmentRuntime)
	snapshots := make([]common.SegmentIndexSnapshotMetadata, 0)
	created := make([]string, 0)
	committed := false
	defer func() {
		if !committed {
			for _, runtime := range prepared {
				_ = c.releaseSegmentRuntime(runtime)
			}
			for _, path := range created {
				_ = os.RemoveAll(path)
			}
		}
	}()
	err := compaction.VisitOutputs(func(snapshot db.SegmentSnapshot) error {
		documents := make([]Document, len(snapshot.Documents))
		for index, record := range snapshot.Documents {
			if err := ctx.Err(); err != nil {
				return err
			}
			document, err := decodeStoredDocument(record)
			if err != nil {
				return err
			}
			if err := document.Validate(owner.schema); err != nil {
				return err
			}
			documents[index] = document
		}
		key, err := collectionRuntimeKeyFor(owner.schema, documents)
		if err != nil {
			return err
		}
		indexes, err := buildCollectionArtifactIndexes(ctx, owner.schema, documents, workers, owner.options.MaxBufferSize)
		if err != nil {
			return err
		}
		artifacts, paths, writeErr := owner.writeSegmentRuntimeArtifacts(ctx, snapshot.Metadata.ID, indexes)
		created = append(created, paths...)
		if err := errors.Join(writeErr, indexes.Close()); err != nil {
			return err
		}
		pathsByField := make(map[string]string, len(artifacts))
		for _, artifact := range artifacts {
			pathsByField[collectionIndexArtifactKey(artifact.Field, artifact.Kind)] = filepath.Join(owner.path, filepath.FromSlash(artifact.File))
		}
		opened, err := buildCollectionRuntimeIndexes(ctx, owner.schema, documents, workers, owner.options.MaxBufferSize, owner.options.EnableMmap, pathsByField)
		if err != nil {
			return err
		}
		opened.key = key
		runtime := &collectionSegmentRuntime{segmentID: snapshot.Metadata.ID, key: key, indexes: opened,
			documents: documents, documentOrdinals: indexDocumentOrdinals(documents)}
		runtime.refs.Store(1)
		prepared[runtime.segmentID] = runtime
		if len(artifacts) != 0 {
			snapshots = append(snapshots, common.SegmentIndexSnapshotMetadata{
				SegmentID: runtime.segmentID, SchemaSHA256: hex.EncodeToString(key.schemaHash[:]),
				DocumentCount: uint64(key.count), MinDocumentID: snapshot.Metadata.MinDocID,
				MaxDocumentID: snapshot.Metadata.MaxDocID, Artifacts: artifacts,
			})
		}
		return nil
	})
	if err != nil {
		return err
	}

	// Phase 3: merge against the current storage state and publish prepared
	// runtimes. Existing query leases retain their previous immutable indexes.
	c.mu.Lock()
	committed, err = compaction.Commit(ctx, snapshots)
	if !committed {
		c.mu.Unlock()
		return err
	}
	c.invalidateQuerySnapshotLocked()
	manifest := c.store.Manifest()
	retained := make(map[uint64]bool, len(manifest.PersistedSegments)+1)
	for _, segment := range manifest.PersistedSegments {
		retained[segment.ID] = true
	}
	if manifest.WritingSegment != nil {
		retained[manifest.WritingSegment.ID] = true
	}
	c.indexMu.Lock()
	for id, runtime := range c.segmentIndexes {
		if !retained[id] {
			delete(c.segmentIndexes, id)
			_ = c.releaseSegmentRuntime(runtime)
		}
	}
	for id, runtime := range prepared {
		c.segmentIndexes[id] = runtime
		c.indexBuildCount++
	}
	c.indexMu.Unlock()
	c.mu.Unlock()
	if err != nil {
		return err
	}
	return c.store.PruneObsoleteArtifacts(ctx)
}
