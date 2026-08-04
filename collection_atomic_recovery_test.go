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
	"testing"
	"time"

	"github.com/gorse-io/zvec/internal/ailego"
	"github.com/stretchr/testify/require"
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
				require.NoError(t, err)

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
				require.NoError(t, err)
				require.True(t, bytes.Equal(after, current),
					"CURRENT changed before the held publication boundary")

				killAtomicRecoveryChild(t, command)
				{
					err := versionLock.Close()
					require.NoError(t, err)
				}

				lockClosed = true

				collection, err := Open(context.Background(), path, NewCollectionOptions())
				require.NoError(t, err)

				defer collection.Close()
				{
					got := collection.store.Manifest().Generation
					require.Equal(t, generation, got)
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
					{
						err := os.Mkdir(blocker, 0o755)
						require.NoError(t, err)
					}
				}
				command := startAtomicRecoveryChild(t, path, testCase.name, "after_current")
				waitForAtomicMarker(t, command, filepath.Join(path, ".atomic-committed"))
				killAtomicRecoveryChild(t, command)

				collection, err := Open(context.Background(), path, NewCollectionOptions())
				require.NoError(t, err)

				defer collection.Close()
				{
					got := collection.store.Manifest().Generation
					require.True(t, got > generation)
				}

				assertAtomicCommittedState(t, collection, testCase.name)
				if blocker != "" {
					{
						err := os.Remove(blocker)
						require.NoError(t, err)
					}

					committedGeneration := collection.store.Manifest().Generation
					{
						err := collection.Optimize(context.Background(), OptimizeOptions{})
						require.NoError(t, err)
					}
					{
						got := collection.store.Manifest().Generation
						require.Equal(t, committedGeneration, got)
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
	require.NoError(t, err)

	documents := []Document{
		atomicRecoveryDocument("a", "alpha", 1, 1, 11, 21, 4),
		atomicRecoveryDocument("b", "bravo", 2, 2, 12, 22, 2),
		atomicRecoveryDocument("c", "charlie", 3, 3, 13, 23, 3),
	}
	{
		_, err := collection.Insert(ctx, documents[:1])
		require.NoError(t, err)
	}
	{
		err := collection.Flush(ctx)
		require.NoError(t, err)
	}
	{
		_, err := collection.Insert(ctx, documents[1:2])
		require.NoError(t, err)
	}
	{
		err := collection.Flush(ctx)
		require.NoError(t, err)
	}
	{
		_, err := collection.Insert(ctx, documents[2:])
		require.NoError(t, err)
	}

	updated, err := collection.Update(ctx, []Document{{PrimaryKey: "a", Fields: map[string]any{"rating": int32(10)}}})
	require.NoError(t, err)
	require.True(t, updated[0].DocID == 3)
	{
		_, err := collection.Delete(ctx, []string{"b"})
		require.NoError(t, err)
	}

	generation = collection.store.Manifest().Generation
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	current, err = os.ReadFile(filepath.Join(path, "CURRENT"))
	require.NoError(t, err)

	segments = countAtomicSegments(t, path)
	require.True(t, segments == 2)

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
	require.Equal(t, atomicRecoverySchema(), collection.Schema())

	assertAtomicLiveDocuments(t, collection)
}

func assertAtomicCommittedState(t *testing.T, collection *Collection, mutation string) {
	t.Helper()
	schema := collection.Schema()
	switch mutation {
	case "create_index":
		field, _ := schema.Field("rating")
		require.Equal(t, IndexTypeInvert, field.IndexType())

	case "drop_index":
		field, _ := schema.Field("indexed")
		require.Equal(t, IndexTypeUndefined, field.IndexType())

	case "add_column":
		field, found := schema.Field("added")
		require.True(t, found)
		require.Equal(t, DataTypeInt64, field.DataType)

	case "alter_column":
		_, oldFound := schema.Field("alter_me")
		field, newFound := schema.Field("renamed")
		require.False(t, oldFound)
		require.True(t, newFound)
		require.Equal(t, DataTypeInt64, field.DataType)

	case "drop_column":
		{
			_, found := schema.Field("drop_me")
			require.False(t, found)
		}

	case "optimize":
		manifest := collection.store.Manifest()
		require.Len(t, manifest.PersistedSegments, 1)
		require.True(t, manifest.PersistedSegments[0].MinDocID == 2)
		require.True(t, manifest.PersistedSegments[0].MaxDocID == 3)
		require.True(t, manifest.PersistedSegments[0].DocCount == 2)
		require.True(t, manifest.WritingSegmentStartDocID == 4)
	}
	assertAtomicLiveDocuments(t, collection)
	fetched, err := collection.Fetch(context.Background(), []string{"a", "c"}, Projection{})
	require.NoError(t, err)

	switch mutation {
	case "add_column":
		require.Equal(t, int64(11), fetched[0].Fields["added"])
		require.Equal(t, int64(4), fetched[1].Fields["added"])

	case "alter_column":
		require.Equal(t, int64(11), fetched[0].Fields["renamed"])
		require.Equal(t, int64(13), fetched[1].Fields["renamed"])
		{
			_, found := fetched[0].Fields["alter_me"]
			require.False(t, found)
		}

	case "drop_column":
		{
			_, found := fetched[0].Fields["drop_me"]
			require.False(t, found)
		}
	}
}

func assertAtomicLiveDocuments(t *testing.T, collection *Collection) {
	t.Helper()
	fetched, err := collection.Fetch(context.Background(), []string{"a", "b", "c"}, Projection{IncludeVectors: true})
	require.NoError(t, err)
	require.NotNil(t, fetched[0])
	require.True(t, fetched[0].DocID == 3)
	require.Nil(t, fetched[1])
	require.NotNil(t, fetched[2])
	require.True(t, fetched[2].DocID == 2)
	require.Equal(t, int32(10), fetched[0].Fields["rating"])
	require.Equal(t, int32(3), fetched[2].Fields["rating"])

	results, err := collection.Query(context.Background(), VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 10, Filter: "indexed >= 1",
	})
	require.NoError(t, err)
	require.Equal(t, []string{"a", "c"}, documentKeys(results))
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
	require.NoError(t, err)
	require.True(t, inserted[0].DocID == 4)
}

func startAtomicRecoveryChild(t *testing.T, path, mutation, phase string) *exec.Cmd {
	t.Helper()
	command := exec.Command(os.Args[0], "-test.run=^TestAtomicDDLAndOptimizeCrashRecovery$")
	command.Env = append(os.Environ(),
		atomicRecoveryPathEnv+"="+path,
		atomicRecoveryMutationEnv+"="+mutation,
		atomicRecoveryPhaseEnv+"="+phase,
	)
	{
		err := command.Start()
		require.NoError(t, err)
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
				require.NoError(t, err)
			}
		case <-deadline.C:
			require.FailNowf(t, "atomic marker timeout", "child %d did not create marker %q", command.Process.Pid, marker)
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
			require.FailNowf(t, "segment artifact timeout", "child %d did not create pre-commit segment artifacts", command.Process.Pid)
		}
	}
}

func countAtomicSegments(t *testing.T, path string) int {
	t.Helper()
	files, err := filepath.Glob(filepath.Join(path, "segments", "*", "data-*.seg"))
	require.NoError(t, err)

	return len(files)
}

func killAtomicRecoveryChild(t *testing.T, command *exec.Cmd) {
	t.Helper()
	{
		err := command.Process.Kill()
		require.NoError(t, err)
	}
	{
		err := command.Wait()
		require.Error(t, err,
			"killed atomic recovery child exited successfully")
	}
}
