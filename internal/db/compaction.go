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

package db

import (
	"bytes"
	"context"
	"errors"
	"math"

	"github.com/gorse-io/xvec/internal/db/index/common"
	segmentstore "github.com/gorse-io/xvec/internal/db/index/segment"
)

// Compaction prepares replacements for a sealed segment snapshot without
// holding the collection lock. The owner serializes maintenance (including
// artifact pruning) until Close, but may continue writes and Flush. Methods on
// one Compaction must be called sequentially. Close releases unpublished data.
type Compaction struct {
	store     *CollectionStore
	manifest  common.Manifest
	inputs    []*segmentstore.ImmutableSegment
	deletes   *common.DeleteStore
	outputs   []*segmentstore.ImmutableSegment
	created   []string
	purged    []uint64
	prepared  bool
	changed   bool
	committed bool
	closed    bool
}

// BeginCompaction captures only segment handles and logical deletions. Flush
// must first checkpoint all existing WAL records, including delete-only WALs,
// so recovery never replays operations referring to a purged document.
func (c *CollectionStore) BeginCompaction(ctx context.Context) (*Compaction, error) {
	if c == nil || ctx == nil {
		return nil, errors.New("db: nil compaction collection or context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.requireWritableLocked(); err != nil {
		return nil, err
	}
	if c.wal.HasRecords() || c.manager.Writing().Metadata().DocCount != 0 {
		return nil, errors.New("db: flush before beginning compaction")
	}
	if c.versions.Current().Generation == math.MaxUint64 {
		return nil, errors.New("db: manifest generation space is exhausted")
	}
	return &Compaction{store: c, manifest: c.versions.Current(),
		inputs: c.manager.ImmutableSegments(), deletes: c.manager.Deletes().Clone()}, nil
}

// Prepare copies live encoded records and writes/opens their replacement
// segments outside the collection lock. Newly deleted versions stay in the
// replacements and are hidden by the current delete set at publication.
func (p *Compaction) Prepare(ctx context.Context) error {
	if p == nil || p.closed || p.prepared || ctx == nil {
		return errors.New("db: invalid compaction preparation")
	}
	var documents []segmentstore.StoredDocument
	for _, input := range p.inputs {
		if err := input.VisitDocuments(func(records []segmentstore.StoredDocument) error {
			for _, record := range records {
				if err := ctx.Err(); err != nil {
					return err
				}
				if p.deletes.IsDeleted(record.DocID) {
					p.purged = append(p.purged, record.DocID)
				} else {
					documents = append(documents, record.Clone())
				}
			}
			return nil
		}); err != nil {
			return err
		}
	}
	runs := rewriteDocumentRuns(documents, p.manifest.SegmentMaxDocuments)
	p.changed = len(p.purged) != 0 || len(runs) != len(p.inputs)
	if !p.changed {
		for index, run := range runs {
			metadata := p.inputs[index].Metadata()
			if metadata.MinDocID != run[0].DocID || metadata.MaxDocID != run[len(run)-1].DocID || metadata.DocCount != uint64(len(run)) {
				p.changed = true
				break
			}
		}
	}
	if !p.changed {
		p.prepared = true
		return ctx.Err()
	}

	// Reserve IDs in memory, as zvec's allocator does. The next publication or
	// Flush persists the high-water mark; orphaned files from a crash are never
	// overwritten because availableArtifact skips existing generations.
	c := p.store
	c.mu.Lock()
	if err := c.requireWritableLocked(); err != nil {
		c.mu.Unlock()
		return err
	}
	current := c.versions.Current()
	firstID := max(current.NextSegmentID, c.reservedNextSegmentID)
	if uint64(len(runs)) > math.MaxUint64-firstID {
		c.mu.Unlock()
		return errors.New("db: segment ID space is exhausted")
	}
	c.reservedNextSegmentID = firstID + uint64(len(runs))
	c.mu.Unlock()

	for index, run := range runs {
		id := firstID + uint64(index)
		writing, err := segmentstore.NewWriteSegment(id, run[0].DocID, uint64(len(run)))
		if err != nil {
			return err
		}
		for _, record := range run {
			if err := writing.ApplyExpected(ctx, record.DocID, record.PrimaryKey, record.Payload); err != nil {
				return err
			}
		}
		relative, err := c.availableArtifact(func(generation uint64) string {
			return segmentFileName(id, generation)
		}, p.manifest.Generation+1)
		if err != nil {
			return err
		}
		p.created = append(p.created, collectionPath(c.dir, relative))
		output, err := writing.SnapshotWithMmap(ctx, c.dir, relative, p.manifest.EnableMmap)
		if err != nil {
			return err
		}
		p.outputs = append(p.outputs, output)
	}
	p.prepared = true
	return ctx.Err()
}

// Changed reports whether Prepare produced a different physical layout.
func (p *Compaction) Changed() bool { return p.changed }

// VisitOutputs exposes the prepared data for building indexes before commit.
// Borrowed records are valid only within the callback.
func (p *Compaction) VisitOutputs(visit func(SegmentSnapshot) error) error {
	if p == nil || !p.prepared || p.closed || visit == nil {
		return errors.New("db: invalid compaction output visitor")
	}
	for _, output := range p.outputs {
		if err := output.VisitDocuments(func(records []segmentstore.StoredDocument) error {
			return visit(SegmentSnapshot{Metadata: output.Metadata(), Documents: records})
		}); err != nil {
			return err
		}
	}
	return nil
}

// Commit replaces only the captured inputs, preserving the current writing
// segment, WAL, IDMap, later flushed segments, and concurrent deletions. Index
// snapshots must refer exclusively to the already prepared replacement data.
// committed is true after CURRENT changes, even on a post-commit sync error.
func (p *Compaction) Commit(ctx context.Context, indexes []common.SegmentIndexSnapshotMetadata) (committed bool, err error) {
	if p == nil || !p.prepared || !p.changed || p.closed || p.committed || ctx == nil {
		return false, errors.New("db: invalid compaction commit")
	}
	if err := ctx.Err(); err != nil {
		return false, err
	}
	c := p.store
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.requireWritableLocked(); err != nil {
		return false, err
	}
	current := c.versions.Current()
	if !bytes.Equal(current.Schema, p.manifest.Schema) {
		return false, errors.New("db: schema changed during compaction")
	}
	inputs := make(map[uint64]*segmentstore.ImmutableSegment, len(p.inputs))
	for _, input := range p.inputs {
		inputs[input.ID()] = input
	}
	remaining := make([]*segmentstore.ImmutableSegment, 0)
	for _, segment := range c.manager.ImmutableSegments() {
		if input, found := inputs[segment.ID()]; found {
			if input != segment {
				return false, errors.New("db: compaction input changed")
			}
			delete(inputs, segment.ID())
		} else {
			remaining = append(remaining, segment)
		}
	}
	if len(inputs) != 0 {
		return false, errors.New("db: compaction input was removed")
	}
	deletes := c.manager.Deletes().Clone()
	for _, id := range p.purged {
		if _, err := deletes.Restore(ctx, id); err != nil {
			return false, err
		}
	}
	nextManager := segmentstore.NewSegmentManager(c.manager.PrimaryKeys(), deletes)
	for _, segment := range append(remaining, p.outputs...) {
		if err := nextManager.AddImmutable(segment); err != nil {
			return false, err
		}
	}
	if err := nextManager.SetWriting(c.manager.Writing()); err != nil {
		return false, err
	}
	nextEngine, err := NewWriteEngine(nextManager, c.wal)
	if err != nil {
		return false, err
	}
	next := current.Clone()
	next.PersistedSegments = nextManager.ImmutableMetadata()
	next.NextSegmentID = max(next.NextSegmentID, c.reservedNextSegmentID)
	inputIDs := make(map[uint64]bool, len(p.inputs))
	for _, input := range p.inputs {
		inputIDs[input.ID()] = true
	}
	next.SegmentIndexSnapshots = nil
	for _, snapshot := range current.SegmentIndexSnapshots {
		if !inputIDs[snapshot.SegmentID] {
			next.SegmentIndexSnapshots = append(next.SegmentIndexSnapshots, snapshot)
		}
	}
	outputIDs := make(map[uint64]bool, len(p.outputs))
	for _, output := range p.outputs {
		outputIDs[output.ID()] = true
	}
	for _, snapshot := range indexes {
		if !outputIDs[snapshot.SegmentID] {
			return false, errors.New("db: compaction index references a non-output segment")
		}
	}
	next.SegmentIndexSnapshots = append(next.SegmentIndexSnapshots, common.CloneSegmentIndexSnapshots(indexes)...)
	if err := next.Validate(); err != nil {
		return false, err
	}
	if err := validateSegmentIndexFiles(c.dir, indexes); err != nil {
		return false, err
	}

	// Keep the checkpoint's delete set separate from in-memory/WAL changes:
	// replay must start from the exact current IDMap checkpoint. Only deletions
	// already checkpointed before BeginCompaction can refer to purged versions.
	deletePath := ""
	defer func() {
		if !committed && deletePath != "" {
			_ = removeCollectionArtifact(deletePath)
		}
	}()
	if len(p.purged) != 0 {
		checkpoint, err := common.LoadDeleteStore(ctx, collectionPath(c.dir, common.DeleteSnapshotName(current.DeleteSnapshotGeneration)))
		if err != nil {
			return false, err
		}
		for _, id := range p.purged {
			if _, err := checkpoint.Restore(ctx, id); err != nil {
				return false, err
			}
		}
		next.DeleteSnapshotGeneration, err = c.nextDeleteSnapshotGeneration(current.DeleteSnapshotGeneration)
		if err != nil {
			return false, err
		}
		deletePath = collectionPath(c.dir, common.DeleteSnapshotName(next.DeleteSnapshotGeneration))
		if err := checkpoint.WriteSnapshot(ctx, deletePath); err != nil {
			return false, err
		}
	}
	if err := c.wal.Sync(ctx); err != nil {
		return false, err
	}
	_, publishErr := c.versions.Publish(ctx, next)
	committed = c.versions.Current().Generation != current.Generation
	if !committed {
		return false, publishErr
	}
	c.manager, c.engine = nextManager, nextEngine
	p.committed = true
	var closeErrors []error
	for _, input := range p.inputs {
		closeErrors = append(closeErrors, input.Close())
	}
	return true, errors.Join(publishErr, errors.Join(closeErrors...))
}

// Close discards uncommitted replacements. It never deletes a committed file.
func (p *Compaction) Close() error {
	if p == nil || p.closed {
		return nil
	}
	p.closed = true
	if p.committed {
		return nil
	}
	var errs []error
	for _, output := range p.outputs {
		errs = append(errs, output.Close())
	}
	for _, path := range p.created {
		errs = append(errs, removeCollectionArtifact(path))
	}
	return errors.Join(errs...)
}
