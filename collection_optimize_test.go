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

package zvec

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/gorse-io/zvec/internal/ailego"
)

func TestOptimizeCompactsLiveDocumentsAndPrunesArtifacts(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "optimize")
	collection, err := CreateAndOpen(ctx, path, testPublicCollectionSchema(), NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}

	documents := []Document{
		testPublicDocument("a", "alpha", "low", 1, 1, []float32{1, 0}),
		testPublicDocument("b", "bravo", "high", 2, 2, []float32{2, 0}),
		testPublicDocument("c", "charlie", "low", 3, 3, []float32{3, 0}),
		testPublicDocument("d", "delta", "high", 4, 4, []float32{4, 0}),
		testPublicDocument("e", "echo", "low", 5, 5, []float32{5, 0}),
		testPublicDocument("f", "foxtrot", "high", 6, 6, []float32{6, 0}),
	}
	wantIDs := make(map[string]uint64, len(documents))
	for start := 0; start < len(documents); start += 2 {
		results, insertErr := collection.Insert(ctx, documents[start:start+2])
		if insertErr != nil {
			t.Fatal(insertErr)
		}
		for index := range results {
			wantIDs[results[index].PrimaryKey] = results[index].DocID
		}
		if err := collection.Flush(ctx); err != nil {
			t.Fatal(err)
		}
	}
	initial := collection.store.Manifest()
	if len(initial.PersistedSegments) != 3 {
		t.Fatalf("initial segments = %d", len(initial.PersistedSegments))
	}
	unknown := filepath.Join(path, "segments", "application", "note.txt")
	if err := os.MkdirAll(filepath.Dir(unknown), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(unknown, []byte("retain me"), 0o644); err != nil {
		t.Fatal(err)
	}
	outside := t.TempDir()
	escapeTarget := filepath.Join(outside, "data-external.seg")
	if err := os.WriteFile(escapeTarget, []byte("external"), 0o644); err != nil {
		t.Fatal(err)
	}
	escapeLink := filepath.Join(path, "segments", "escape")
	symlinkCreated := os.Symlink(outside, escapeLink) == nil

	before, err := collection.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 6,
		Projection: Projection{OutputFields: []string{"title"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := collection.Optimize(ctx, OptimizeOptions{Concurrency: 3}); err != nil {
		t.Fatal(err)
	}
	optimized := collection.store.Manifest()
	if optimized.Generation <= initial.Generation || len(optimized.PersistedSegments) != 1 || optimized.WritingSegmentStartDocID != 6 {
		t.Fatalf("optimized manifest = %#v", optimized)
	}
	assertOptimizeArtifacts(t, path, 1)
	if content, err := os.ReadFile(unknown); err != nil || string(content) != "retain me" {
		t.Fatalf("unknown artifact = %q, %v", content, err)
	}
	if symlinkCreated {
		if content, err := os.ReadFile(escapeTarget); err != nil || string(content) != "external" {
			t.Fatalf("symlink escape target = %q, %v", content, err)
		}
	}
	after, err := collection.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 6,
		Projection: Projection{OutputFields: []string{"title"}},
	})
	if err != nil || !reflect.DeepEqual(after, before) {
		t.Fatalf("query after optimize = %#v, %v; want %#v", after, err, before)
	}
	assertOptimizeDocumentIDs(t, ctx, collection, wantIDs)

	// A canonical collection is a manifest no-op, while prune remains safe to
	// retry for a process that stopped just after an earlier publication.
	generation := optimized.Generation
	if err := collection.Optimize(ctx, OptimizeOptions{Concurrency: 1}); err != nil {
		t.Fatal(err)
	}
	if got := collection.store.Manifest().Generation; got != generation {
		t.Fatalf("no-op optimize advanced generation to %d", got)
	}

	updated, err := collection.Update(ctx, []Document{{PrimaryKey: "a", Fields: map[string]any{"rating": int32(10)}}})
	if err != nil {
		t.Fatal(err)
	}
	wantIDs["a"] = updated[0].DocID
	if _, err := collection.Delete(ctx, []string{"e"}); err != nil {
		t.Fatal(err)
	}
	delete(wantIDs, "e")
	temporary := testPublicDocument("temporary", "temporary", "low", 7, 7, []float32{7, 0})
	inserted, err := collection.Insert(ctx, []Document{temporary})
	if err != nil || inserted[0].DocID != 7 {
		t.Fatalf("temporary insert = %#v, %v", inserted, err)
	}
	if _, err := collection.Delete(ctx, []string{"temporary"}); err != nil {
		t.Fatal(err)
	}
	if err := collection.Optimize(ctx, OptimizeOptions{Concurrency: 2}); err != nil {
		t.Fatal(err)
	}
	optimized = collection.store.Manifest()
	if len(optimized.PersistedSegments) != 2 || optimized.WritingSegmentStartDocID != 8 {
		t.Fatalf("optimized gapped manifest = %#v", optimized)
	}
	assertOptimizeArtifacts(t, path, 2)
	assertOptimizeDocumentIDs(t, ctx, collection, wantIDs)
	fetched, err := collection.Fetch(ctx, []string{"a", "e", "temporary"}, Projection{})
	if err != nil || fetched[0] == nil || fetched[0].Fields["rating"] != int32(10) || fetched[1] != nil || fetched[2] != nil {
		t.Fatalf("fetch after delete reclamation = %#v, %v", fetched, err)
	}

	next := testPublicDocument("next", "next", "low", 8, 8, []float32{8, 0})
	inserted, err = collection.Insert(ctx, []Document{next})
	if err != nil || inserted[0].DocID != 8 {
		t.Fatalf("next insert = %#v, %v", inserted, err)
	}
	wantIDs["next"] = 8
	if err := collection.Close(); err != nil {
		t.Fatal(err)
	}
	collection, err = Open(ctx, path, NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	defer collection.Close()
	assertOptimizeDocumentIDs(t, ctx, collection, wantIDs)
	if collection.Stats().DocumentCount != uint64(len(wantIDs)) {
		t.Fatalf("reopened document count = %d", collection.Stats().DocumentCount)
	}
}

func TestOptimizeFullyDeletedCollectionKeepsDocumentIDsMonotonic(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "optimize-empty")
	collection, err := CreateAndOpen(ctx, path, testPublicCollectionSchema(), NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	documents := []Document{
		testPublicDocument("a", "a", "low", 1, 1, []float32{1, 0}),
		testPublicDocument("b", "b", "high", 2, 2, []float32{2, 0}),
	}
	if _, err := collection.Insert(ctx, documents); err != nil {
		t.Fatal(err)
	}
	if err := collection.Flush(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := collection.Delete(ctx, []string{"a", "b"}); err != nil {
		t.Fatal(err)
	}
	if err := collection.Optimize(ctx, OptimizeOptions{}); err != nil {
		t.Fatal(err)
	}
	manifest := collection.store.Manifest()
	if len(manifest.PersistedSegments) != 0 || manifest.WritingSegmentStartDocID != 2 || collection.Stats().DocumentCount != 0 {
		t.Fatalf("fully deleted manifest = %#v, stats = %#v", manifest, collection.Stats())
	}
	assertOptimizeArtifacts(t, path, 0)
	if err := collection.Close(); err != nil {
		t.Fatal(err)
	}
	collection, err = Open(ctx, path, NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	defer collection.Close()
	inserted, err := collection.Insert(ctx, []Document{testPublicDocument("c", "c", "low", 3, 3, []float32{3, 0})})
	if err != nil || inserted[0].DocID != 2 {
		t.Fatalf("insert after full reclamation = %#v, %v", inserted, err)
	}
}

func TestOptimizeValidationUnsupportedIndexesAndRollback(t *testing.T) {
	ctx := context.Background()
	var nilCollection *Collection
	if err := nilCollection.Optimize(ctx, OptimizeOptions{}); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("nil collection Optimize = %v", err)
	}

	path := filepath.Join(t.TempDir(), "optimize-errors")
	collection, err := CreateAndOpen(ctx, path, testPublicCollectionSchema(), NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	if err := collection.Optimize(nil, OptimizeOptions{}); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("nil context Optimize = %v", err)
	}
	if err := collection.Optimize(ctx, OptimizeOptions{Concurrency: -1}); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("negative concurrency Optimize = %v", err)
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	if err := collection.Optimize(canceled, OptimizeOptions{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled Optimize = %v", err)
	}
	initialGeneration := collection.store.Manifest().Generation
	if err := collection.Optimize(ctx, OptimizeOptions{}); err != nil {
		t.Fatal(err)
	}
	if got := collection.store.Manifest().Generation; got != initialGeneration {
		t.Fatalf("empty Optimize advanced generation to %d", got)
	}
	if _, err := collection.Insert(ctx, []Document{testPublicDocument("a", "a", "low", 1, 1, []float32{1, 0})}); err != nil {
		t.Fatal(err)
	}
	initialGeneration = collection.store.Manifest().Generation
	versionLock, err := ailego.AcquireFileLock(ctx, filepath.Join(path, ".version.lock"), ailego.LockExclusive)
	if err != nil {
		t.Fatal(err)
	}
	deadline, cancel := context.WithTimeout(ctx, 75*time.Millisecond)
	err = collection.Optimize(deadline, OptimizeOptions{Concurrency: 2})
	cancel()
	if !errors.Is(err, context.DeadlineExceeded) {
		_ = versionLock.Close()
		t.Fatalf("blocked Optimize = %v", err)
	}
	if err := versionLock.Close(); err != nil {
		t.Fatal(err)
	}
	if got := collection.store.Manifest().Generation; got != initialGeneration {
		t.Fatalf("failed Optimize published generation %d", got)
	}
	fetched, err := collection.Fetch(ctx, []string{"a"}, Projection{})
	if err != nil || fetched[0] == nil || fetched[0].DocID != 0 {
		t.Fatalf("document after Optimize rollback = %#v, %v", fetched, err)
	}
	if err := collection.Close(); err != nil {
		t.Fatal(err)
	}
	if err := collection.Optimize(ctx, OptimizeOptions{}); !errors.Is(err, ErrFailedPrecondition) {
		t.Fatalf("closed Optimize = %v", err)
	}
	readOnlyOptions := NewCollectionOptions()
	readOnlyOptions.ReadOnly = true
	collection, err = Open(ctx, path, readOnlyOptions)
	if err != nil {
		t.Fatal(err)
	}
	if err := collection.Optimize(ctx, OptimizeOptions{}); !errors.Is(err, ErrPermissionDenied) {
		t.Fatalf("read-only Optimize = %v", err)
	}
	if err := collection.Close(); err != nil {
		t.Fatal(err)
	}

	unsupported := []struct {
		name  string
		field FieldSchema
		value any
	}{
		{name: "HNSW", field: FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewHNSWIndexParams(MetricTypeIP)}, value: VectorFP32{1, 0}},
		{name: "quantized Flat", field: quantizedOptimizeField(), value: VectorFP32{1, 0}},
		{name: "FTS", field: FieldSchema{Name: "text", DataType: DataTypeString, Index: NewFTSIndexParams()}, value: "alpha"},
		{name: "binary INVERT", field: FieldSchema{Name: "data", DataType: DataTypeBinary, Index: NewInvertIndexParams()}, value: Binary{1, 2}},
	}
	for _, testCase := range unsupported {
		t.Run(testCase.name, func(t *testing.T) {
			schema := NewCollectionSchema("unsupported_optimize", testCase.field)
			schema.MaxDocsPerSegment = MinMaxDocsPerSegment
			collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "collection"), schema, NewCollectionOptions())
			if err != nil {
				t.Fatal(err)
			}
			defer collection.Close()
			if _, err := collection.Insert(ctx, []Document{{PrimaryKey: "a", Fields: map[string]any{testCase.field.Name: testCase.value}}}); err != nil {
				t.Fatal(err)
			}
			generation := collection.store.Manifest().Generation
			if err := collection.Optimize(ctx, OptimizeOptions{}); !errors.Is(err, ErrNotSupported) {
				t.Fatalf("Optimize = %v", err)
			}
			if collection.store.Manifest().Generation != generation || collection.Stats().DocumentCount != 1 {
				t.Fatalf("unsupported Optimize changed state: %#v", collection.store.Manifest())
			}
		})
	}
}

func quantizedOptimizeField() FieldSchema {
	params := NewFlatIndexParams(MetricTypeIP)
	params.Quantize = QuantizeTypeFP16
	return FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: params}
}

func assertOptimizeDocumentIDs(t *testing.T, ctx context.Context, collection *Collection, want map[string]uint64) {
	t.Helper()
	keys := make([]string, 0, len(want))
	for key := range want {
		keys = append(keys, key)
	}
	fetched, err := collection.Fetch(ctx, keys, Projection{})
	if err != nil {
		t.Fatal(err)
	}
	for index, key := range keys {
		if fetched[index] == nil || fetched[index].DocID != want[key] {
			t.Fatalf("document %q = %#v, want ID %d", key, fetched[index], want[key])
		}
	}
}

func assertOptimizeArtifacts(t *testing.T, path string, segments int) {
	t.Helper()
	patterns := map[string]int{
		filepath.Join(path, "segments", "*", "data-*.seg"): segments,
		filepath.Join(path, "wal", "*.wal"):                1,
		filepath.Join(path, "wal", "*.wal.lock"):           1,
		filepath.Join(path, "snapshots", "primary-*.snap"): 1,
		filepath.Join(path, "snapshots", "delete-*.snap"):  1,
	}
	for pattern, want := range patterns {
		matches, err := filepath.Glob(pattern)
		if err != nil {
			t.Fatal(err)
		}
		filtered := matches[:0]
		for _, name := range matches {
			parent, statErr := os.Lstat(filepath.Dir(name))
			if statErr != nil {
				t.Fatal(statErr)
			}
			if parent.Mode()&os.ModeSymlink == 0 {
				filtered = append(filtered, name)
			}
		}
		matches = filtered
		if len(matches) != want {
			t.Fatalf("artifacts %q = %v, want %d", pattern, matches, want)
		}
	}
}
