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
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/gorse-io/zvec/internal/ailego"
)

const (
	atomicRecoveryPathEnv     = "ZVEC_ATOMIC_RECOVERY_PATH"
	atomicRecoveryMutationEnv = "ZVEC_ATOMIC_RECOVERY_MUTATION"
	atomicRecoveryPhaseEnv    = "ZVEC_ATOMIC_RECOVERY_PHASE"
)

type atomicRecoveryCase struct {
	name        string
	dataRewrite bool
}

func TestAtomicDDLAndOptimizeCrashRecovery(t *testing.T) {
	if path := os.Getenv(atomicRecoveryPathEnv); path != "" {
		runAtomicRecoveryChild(path, os.Getenv(atomicRecoveryMutationEnv), os.Getenv(atomicRecoveryPhaseEnv))
		return
	}

	tests := []atomicRecoveryCase{
		{name: "create_index"},
		{name: "drop_index"},
		{name: "add_column", dataRewrite: true},
		{name: "alter_column", dataRewrite: true},
		{name: "drop_column", dataRewrite: true},
		{name: "optimize", dataRewrite: true},
	}
	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			t.Run("before_current", func(t *testing.T) {
				path, generation, current, initialSegments := createAtomicRecoveryFixture(t)
				versionLock, err := ailego.AcquireFileLock(context.Background(), filepath.Join(path, ".version.lock"), ailego.LockExclusive)
				if err != nil {
					t.Fatal(err)
				}
				lockClosed := false
				defer func() {
					if !lockClosed {
						_ = versionLock.Close()
					}
				}()

				command := startAtomicRecoveryChild(t, path, testCase.name, "before_current")
				waitForAtomicMarker(t, command, filepath.Join(path, ".atomic-blocked"))
				if testCase.dataRewrite {
					waitForAdditionalSegment(t, command, path, initialSegments)
				}
				after, err := os.ReadFile(filepath.Join(path, "CURRENT"))
				if err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(after, current) {
					t.Fatal("CURRENT changed before the held publication boundary")
				}
				killAtomicRecoveryChild(t, command)
				if err := versionLock.Close(); err != nil {
					t.Fatal(err)
				}
				lockClosed = true

				collection, err := Open(context.Background(), path, NewCollectionOptions())
				if err != nil {
					t.Fatal(err)
				}
				defer collection.Close()
				if got := collection.store.Manifest().Generation; got != generation {
					t.Fatalf("pre-commit recovery generation = %d, want %d", got, generation)
				}
				assertAtomicInitialState(t, collection)
				assertAtomicContinuedWrite(t, collection, testCase.name, false)
			})

			t.Run("after_current", func(t *testing.T) {
				path, generation, _, _ := createAtomicRecoveryFixture(t)
				blocker := ""
				if testCase.name == "optimize" {
					// Force the post-publication prune to stop. The child still
					// observes a newer CURRENT and waits to be killed with an open
					// collection handle, leaving cleanup for recovery to retry.
					blocker = filepath.Join(path, "wal", "99999999999999999999-99999999999999999999.wal")
					if err := os.Mkdir(blocker, 0o755); err != nil {
						t.Fatal(err)
					}
				}
				command := startAtomicRecoveryChild(t, path, testCase.name, "after_current")
				waitForAtomicMarker(t, command, filepath.Join(path, ".atomic-committed"))
				killAtomicRecoveryChild(t, command)

				collection, err := Open(context.Background(), path, NewCollectionOptions())
				if err != nil {
					t.Fatal(err)
				}
				defer collection.Close()
				if got := collection.store.Manifest().Generation; got <= generation {
					t.Fatalf("post-commit recovery generation = %d, initial %d", got, generation)
				}
				assertAtomicCommittedState(t, collection, testCase.name)
				if blocker != "" {
					if err := os.Remove(blocker); err != nil {
						t.Fatal(err)
					}
					committedGeneration := collection.store.Manifest().Generation
					if err := collection.Optimize(context.Background(), OptimizeOptions{}); err != nil {
						t.Fatal(err)
					}
					if got := collection.store.Manifest().Generation; got != committedGeneration {
						t.Fatalf("cleanup retry advanced generation to %d", got)
					}
					assertOptimizeArtifacts(t, path, 1)
				}
				assertAtomicContinuedWrite(t, collection, testCase.name, true)
			})
		})
	}
}

func runAtomicRecoveryChild(path, mutation, phase string) {
	collection, err := Open(context.Background(), path, NewCollectionOptions())
	if err != nil {
		os.Exit(91)
	}
	initialGeneration := collection.store.Manifest().Generation
	if err := os.WriteFile(filepath.Join(path, ".atomic-started"), []byte(mutation), 0o600); err != nil {
		os.Exit(92)
	}
	if phase == "before_current" {
		result := make(chan error, 1)
		go func() {
			result <- runAtomicRecoveryMutation(collection, mutation)
		}()
		select {
		case <-result:
			os.Exit(93)
		case <-time.After(150 * time.Millisecond):
			if err := os.WriteFile(filepath.Join(path, ".atomic-blocked"), []byte(mutation), 0o600); err != nil {
				os.Exit(94)
			}
			for {
				time.Sleep(time.Second)
			}
		}
	}

	mutationErr := runAtomicRecoveryMutation(collection, mutation)
	if collection.store.Manifest().Generation <= initialGeneration {
		os.Exit(95)
	}
	// Optimize may report the deliberately injected post-commit prune error.
	// Every other successful publication must return nil.
	if mutationErr != nil && mutation != "optimize" {
		os.Exit(96)
	}
	if err := os.WriteFile(filepath.Join(path, ".atomic-committed"), []byte(mutation), 0o600); err != nil {
		os.Exit(97)
	}
	for {
		time.Sleep(time.Second)
	}
}

func runAtomicRecoveryMutation(collection *Collection, mutation string) error {
	ctx := context.Background()
	switch mutation {
	case "create_index":
		return collection.CreateIndex(ctx, "rating", NewInvertIndexParams(), CreateIndexOptions{Concurrency: 2})
	case "drop_index":
		return collection.DropIndex(ctx, "indexed")
	case "add_column":
		return collection.AddColumn(ctx, FieldSchema{Name: "added", DataType: DataTypeInt64}, "rating + 1", AddColumnOptions{Concurrency: 2})
	case "alter_column":
		replacement := FieldSchema{Name: "renamed", DataType: DataTypeInt64}
		return collection.AlterColumn(ctx, "alter_me", "", &replacement, AlterColumnOptions{Concurrency: 2})
	case "drop_column":
		return collection.DropColumn(ctx, "drop_me")
	case "optimize":
		return collection.Optimize(ctx, OptimizeOptions{Concurrency: 2})
	default:
		return fmt.Errorf("unknown atomic recovery mutation %q", mutation)
	}
}

func createAtomicRecoveryFixture(t *testing.T) (path string, generation uint64, current []byte, segments int) {
	t.Helper()
	ctx := context.Background()
	path = filepath.Join(t.TempDir(), "atomic-recovery")
	options := NewCollectionOptions()
	options.WALSyncEvery = 1
	collection, err := CreateAndOpen(ctx, path, atomicRecoverySchema(), options)
	if err != nil {
		t.Fatal(err)
	}
	documents := []Document{
		atomicRecoveryDocument("a", "alpha", 1, 1, 11, 21, 4),
		atomicRecoveryDocument("b", "bravo", 2, 2, 12, 22, 2),
		atomicRecoveryDocument("c", "charlie", 3, 3, 13, 23, 3),
	}
	if _, err := collection.Insert(ctx, documents[:1]); err != nil {
		t.Fatal(err)
	}
	if err := collection.Flush(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := collection.Insert(ctx, documents[1:2]); err != nil {
		t.Fatal(err)
	}
	if err := collection.Flush(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := collection.Insert(ctx, documents[2:]); err != nil {
		t.Fatal(err)
	}
	updated, err := collection.Update(ctx, []Document{{PrimaryKey: "a", Fields: map[string]any{"rating": int32(10)}}})
	if err != nil || updated[0].DocID != 3 {
		t.Fatalf("fixture update = %#v, %v", updated, err)
	}
	if _, err := collection.Delete(ctx, []string{"b"}); err != nil {
		t.Fatal(err)
	}
	generation = collection.store.Manifest().Generation
	if err := collection.Close(); err != nil {
		t.Fatal(err)
	}
	current, err = os.ReadFile(filepath.Join(path, "CURRENT"))
	if err != nil {
		t.Fatal(err)
	}
	segments = countAtomicSegments(t, path)
	if segments != 2 {
		t.Fatalf("fixture segments = %d", segments)
	}
	return path, generation, current, segments
}

func atomicRecoverySchema() CollectionSchema {
	schema := NewCollectionSchema("atomic_recovery",
		FieldSchema{Name: "title", DataType: DataTypeString},
		FieldSchema{Name: "rating", DataType: DataTypeInt32},
		FieldSchema{Name: "indexed", DataType: DataTypeInt32, Index: NewInvertIndexParams()},
		FieldSchema{Name: "alter_me", DataType: DataTypeInt32},
		FieldSchema{Name: "drop_me", DataType: DataTypeInt32},
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewFlatIndexParams(MetricTypeIP)},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	return schema
}

func atomicRecoveryDocument(primaryKey, title string, rating, indexed, alter, drop int32, score float32) Document {
	return Document{PrimaryKey: primaryKey, Fields: map[string]any{
		"title": title, "rating": rating, "indexed": indexed,
		"alter_me": alter, "drop_me": drop,
		"embedding": VectorFP32{score, 0},
	}}
}

func assertAtomicInitialState(t *testing.T, collection *Collection) {
	t.Helper()
	if !reflect.DeepEqual(collection.Schema(), atomicRecoverySchema()) {
		t.Fatalf("initial schema = %#v", collection.Schema())
	}
	assertAtomicLiveDocuments(t, collection)
}

func assertAtomicCommittedState(t *testing.T, collection *Collection, mutation string) {
	t.Helper()
	schema := collection.Schema()
	switch mutation {
	case "create_index":
		field, _ := schema.Field("rating")
		if field.IndexType() != IndexTypeInvert {
			t.Fatalf("committed rating index = %#v", field.Index)
		}
	case "drop_index":
		field, _ := schema.Field("indexed")
		if field.IndexType() != IndexTypeUndefined {
			t.Fatalf("committed indexed index = %#v", field.Index)
		}
	case "add_column":
		field, found := schema.Field("added")
		if !found || field.DataType != DataTypeInt64 {
			t.Fatalf("committed added field = %#v, %v", field, found)
		}
	case "alter_column":
		_, oldFound := schema.Field("alter_me")
		field, newFound := schema.Field("renamed")
		if oldFound || !newFound || field.DataType != DataTypeInt64 {
			t.Fatalf("committed alter schema = %#v", schema)
		}
	case "drop_column":
		if _, found := schema.Field("drop_me"); found {
			t.Fatalf("committed drop schema = %#v", schema)
		}
	case "optimize":
		manifest := collection.store.Manifest()
		if len(manifest.PersistedSegments) != 1 || manifest.PersistedSegments[0].MinDocID != 2 ||
			manifest.PersistedSegments[0].MaxDocID != 3 || manifest.PersistedSegments[0].DocCount != 2 ||
			manifest.WritingSegmentStartDocID != 4 {
			t.Fatalf("committed optimize manifest = %#v", manifest)
		}
	}
	assertAtomicLiveDocuments(t, collection)
	fetched, err := collection.Fetch(context.Background(), []string{"a", "c"}, Projection{})
	if err != nil {
		t.Fatal(err)
	}
	switch mutation {
	case "add_column":
		if fetched[0].Fields["added"] != int64(11) || fetched[1].Fields["added"] != int64(4) {
			t.Fatalf("committed add payloads = %#v", fetched)
		}
	case "alter_column":
		if fetched[0].Fields["renamed"] != int64(11) || fetched[1].Fields["renamed"] != int64(13) {
			t.Fatalf("committed alter payloads = %#v", fetched)
		}
		if _, found := fetched[0].Fields["alter_me"]; found {
			t.Fatalf("committed alter retained old payload = %#v", fetched[0])
		}
	case "drop_column":
		if _, found := fetched[0].Fields["drop_me"]; found {
			t.Fatalf("committed drop retained payload = %#v", fetched[0])
		}
	}
}

func assertAtomicLiveDocuments(t *testing.T, collection *Collection) {
	t.Helper()
	fetched, err := collection.Fetch(context.Background(), []string{"a", "b", "c"}, Projection{IncludeVectors: true})
	if err != nil || fetched[0] == nil || fetched[0].DocID != 3 || fetched[1] != nil || fetched[2] == nil || fetched[2].DocID != 2 {
		t.Fatalf("recovered live documents = %#v, %v", fetched, err)
	}
	if fetched[0].Fields["rating"] != int32(10) || fetched[2].Fields["rating"] != int32(3) {
		t.Fatalf("recovered ratings = %#v", fetched)
	}
	results, err := collection.Query(context.Background(), VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 10, Filter: "indexed >= 1",
	})
	if err != nil || !reflect.DeepEqual(documentKeys(results), []string{"a", "c"}) {
		t.Fatalf("recovered query = %v, %v", documentKeys(results), err)
	}
}

func assertAtomicContinuedWrite(t *testing.T, collection *Collection, mutation string, committed bool) {
	t.Helper()
	document := atomicRecoveryDocument("d", "durable", 4, 4, 14, 24, 1)
	if committed {
		switch mutation {
		case "add_column":
			document.Fields["added"] = int64(5)
		case "alter_column":
			delete(document.Fields, "alter_me")
			document.Fields["renamed"] = int64(14)
		case "drop_column":
			delete(document.Fields, "drop_me")
		}
	}
	inserted, err := collection.Insert(context.Background(), []Document{document})
	if err != nil || inserted[0].DocID != 4 {
		t.Fatalf("continued write = %#v, %v", inserted, err)
	}
}

func startAtomicRecoveryChild(t *testing.T, path, mutation, phase string) *exec.Cmd {
	t.Helper()
	command := exec.Command(os.Args[0], "-test.run=^TestAtomicDDLAndOptimizeCrashRecovery$")
	command.Env = append(os.Environ(),
		atomicRecoveryPathEnv+"="+path,
		atomicRecoveryMutationEnv+"="+mutation,
		atomicRecoveryPhaseEnv+"="+phase,
	)
	if err := command.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if command.ProcessState == nil {
			_ = command.Process.Kill()
			_ = command.Wait()
		}
	})
	return command
}

func waitForAtomicMarker(t *testing.T, command *exec.Cmd, marker string) {
	t.Helper()
	deadline := time.NewTimer(10 * time.Second)
	defer deadline.Stop()
	ticker := time.NewTicker(10 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			if _, err := os.Stat(marker); err == nil {
				return
			} else if !errors.Is(err, os.ErrNotExist) {
				t.Fatal(err)
			}
		case <-deadline.C:
			t.Fatalf("child %d did not create marker %q", command.Process.Pid, marker)
		}
	}
}

func waitForAdditionalSegment(t *testing.T, command *exec.Cmd, path string, initial int) {
	t.Helper()
	deadline := time.NewTimer(10 * time.Second)
	defer deadline.Stop()
	ticker := time.NewTicker(10 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			if countAtomicSegments(t, path) > initial {
				return
			}
		case <-deadline.C:
			t.Fatalf("child %d did not create pre-commit segment artifacts", command.Process.Pid)
		}
	}
}

func countAtomicSegments(t *testing.T, path string) int {
	t.Helper()
	files, err := filepath.Glob(filepath.Join(path, "segments", "*", "data-*.seg"))
	if err != nil {
		t.Fatal(err)
	}
	return len(files)
}

func killAtomicRecoveryChild(t *testing.T, command *exec.Cmd) {
	t.Helper()
	if err := command.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	if err := command.Wait(); err == nil {
		t.Fatal("killed atomic recovery child exited successfully")
	}
}
