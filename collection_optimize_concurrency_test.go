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
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/gorse-io/xvec/internal/db"
	"github.com/stretchr/testify/require"
)

func TestOptimizeCompactionAllowsQueriesAndMutations(t *testing.T) {
	for _, stage := range []string{"before_prepare", "after_prepare"} {
		t.Run(stage, func(t *testing.T) {
			ctx := context.Background()
			path := filepath.Join(t.TempDir(), "concurrent-optimize")
			schema := NewCollectionSchema("concurrent_optimize",
				FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewHNSWIndexParams(MetricTypeIP)},
				FieldSchema{Name: "rating", DataType: DataTypeInt32, Index: NewInvertIndexParams()},
				FieldSchema{Name: "text", DataType: DataTypeString, Index: FTSIndexParams{Tokenizer: "whitespace"}},
			)
			c, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
			require.NoError(t, err)
			defer func() { require.NoError(t, c.Close()) }()
			makeDoc := func(key string, rating int32) Document {
				return Document{PrimaryKey: key, Fields: map[string]any{
					"embedding": VectorFP32{float32(rating), 0}, "rating": rating, "text": "shared " + key,
				}}
			}
			_, err = c.Insert(ctx, []Document{makeDoc("a", 1), makeDoc("b", 2)})
			require.NoError(t, err)
			require.NoError(t, c.Flush(ctx))
			_, err = c.Insert(ctx, []Document{makeDoc("c", 3), makeDoc("d", 4)})
			require.NoError(t, err)
			require.NoError(t, c.Flush(ctx))
			_, err = c.Delete(ctx, []string{"b"})
			require.NoError(t, err)

			entered, resume := make(chan struct{}), make(chan struct{})
			var once sync.Once
			release := func() { once.Do(func() { close(resume) }) }
			done := make(chan error, 1)
			go func() {
				done <- c.optimize(ctx, OptimizeOptions{Concurrency: 2}, func(p *db.Compaction, ctx context.Context) error {
					if stage == "after_prepare" {
						if err := p.Prepare(ctx); err != nil {
							return err
						}
					}
					close(entered)
					<-resume
					if stage == "before_prepare" {
						return p.Prepare(ctx)
					}
					return nil
				})
			}()
			defer release()
			select {
			case <-entered:
			case err := <-done:
				t.Fatalf("optimize stopped before preparation: %v", err)
			case <-time.After(5 * time.Second):
				t.Fatal("optimize did not reach compaction")
			}
			operations := make(chan error, 1)
			go func() {
				query := VectorQuery{Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 10}
				if _, err := c.Query(ctx, query); err != nil {
					operations <- err
					return
				}
				if _, err := c.Insert(ctx, []Document{makeDoc("e", 5)}); err != nil {
					operations <- err
					return
				}
				if _, err := c.Update(ctx, []Document{makeDoc("a", 10)}); err != nil {
					operations <- err
					return
				}
				if _, err := c.Delete(ctx, []string{"c"}); err != nil {
					operations <- err
					return
				}
				if _, err := c.Upsert(ctx, []Document{makeDoc("b", 20)}); err != nil {
					operations <- err
					return
				}
				_, err := c.Query(ctx, query)
				operations <- err
			}()
			select {
			case err := <-operations:
				require.NoError(t, err)
			case <-time.After(5 * time.Second):
				t.Fatal("query or mutation waited for compaction")
			}
			_, err = c.CreateIterator(ctx, NewIteratorOptions())
			require.ErrorIs(t, err, ErrFailedPrecondition)
			release()
			require.NoError(t, <-done)

			assertLive := func() {
				t.Helper()
				query := VectorQuery{Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 10, Filter: "rating >= 4"}
				results, err := c.Query(ctx, query)
				require.NoError(t, err)
				require.Equal(t, []string{"b", "a", "e", "d"}, documentKeys(results))
				results, err = c.Query(ctx, VectorQuery{Field: "text", FTS: &FTSClause{Query: "shared"}, TopK: 10})
				require.NoError(t, err)
				require.ElementsMatch(t, []string{"a", "b", "d", "e"}, documentKeys(results))
				fetched, err := c.Fetch(ctx, []string{"a", "b", "c", "d", "e"}, Projection{})
				require.NoError(t, err)
				require.Equal(t, int32(10), fetched[0].Fields["rating"])
				require.Equal(t, int32(20), fetched[1].Fields["rating"])
				require.Nil(t, fetched[2])
				require.Equal(t, uint64(3), fetched[3].DocID)
				require.Equal(t, uint64(4), fetched[4].DocID)
			}
			assertLive()
			require.InDelta(t, .25, c.Stats().IndexCompleteness["embedding"], .0001)
			require.NoError(t, c.Close())
			c, err = Open(ctx, path, NewCollectionOptions())
			require.NoError(t, err)
			assertLive()
			require.NoError(t, c.Optimize(ctx, OptimizeOptions{}))
			assertLive()
			require.Equal(t, float32(1), c.Stats().IndexCompleteness["embedding"])
		})
	}
}

func TestOptimizePublishesWhileOldQueryRetainsSnapshot(t *testing.T) {
	ctx := context.Background()
	c, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "old-query"), testMultiQuerySchema(), NewCollectionOptions())
	require.NoError(t, err)
	defer func() { require.NoError(t, c.Close()) }()
	_, err = c.Insert(ctx, testMultiQueryDocuments())
	require.NoError(t, err)
	require.NoError(t, c.Flush(ctx))
	query := VectorQuery{Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 10}
	before, err := c.Query(ctx, query)
	require.NoError(t, err)
	snapshot := c.querySnapshot.Load()
	runtime := snapshot.runtimes[0]
	blocking := &blockingCollectionDenseIndex{
		collectionDenseIndex: runtime.indexes.denseNative["embedding"],
		started:              make(chan struct{}), release: make(chan struct{}), closed: make(chan struct{}),
	}
	runtime.indexes.denseNative["embedding"] = blocking
	runtime.indexes.denseFlat["embedding"] = blocking
	var once sync.Once
	release := func() { once.Do(func() { close(blocking.release) }) }
	defer release()
	queried := make(chan []Document, 1)
	queryErrors := make(chan error, 1)
	go func() {
		documents, err := c.Query(ctx, query)
		queried <- documents
		queryErrors <- err
	}()
	select {
	case <-blocking.started:
	case <-time.After(5 * time.Second):
		t.Fatal("query did not enter the index")
	}
	_, err = c.Update(ctx, []Document{{PrimaryKey: "a", Fields: map[string]any{"rating": int32(42)}}})
	require.NoError(t, err)
	_, err = c.Delete(ctx, []string{"b"})
	require.NoError(t, err)
	optimized := make(chan error, 1)
	go func() { optimized <- c.Optimize(ctx, OptimizeOptions{}) }()
	select {
	case err := <-optimized:
		require.NoError(t, err)
	case <-time.After(5 * time.Second):
		t.Fatal("optimize publication waited for the old query")
	}
	select {
	case <-blocking.closed:
		t.Fatal("old query's index closed before its snapshot was released")
	default:
	}
	release()
	require.NoError(t, <-queryErrors)
	require.Equal(t, before, <-queried)
	select {
	case <-blocking.closed:
	case <-time.After(5 * time.Second):
		t.Fatal("retired index was not released after the old query completed")
	}
	after, err := c.Fetch(ctx, []string{"a", "b"}, Projection{})
	require.NoError(t, err)
	require.Equal(t, int32(42), after[0].Fields["rating"])
	require.Nil(t, after[1])
}

func TestOptimizeCanceledPreparationRetainsConcurrentWrites(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "cancel-prepare")
	c, err := CreateAndOpen(ctx, path, testMultiQuerySchema(), NewCollectionOptions())
	require.NoError(t, err)
	defer func() { require.NoError(t, c.Close()) }()
	_, err = c.Insert(ctx, testMultiQueryDocuments())
	require.NoError(t, err)
	_, err = c.Delete(ctx, []string{"b"})
	require.NoError(t, err)
	canceled, cancel := context.WithCancel(ctx)
	defer cancel()
	err = c.optimize(canceled, OptimizeOptions{}, func(p *db.Compaction, ctx context.Context) error {
		if err := p.Prepare(ctx); err != nil {
			return err
		}
		if _, err := c.Update(ctx, []Document{{PrimaryKey: "a", Fields: map[string]any{"rating": int32(42)}}}); err != nil {
			return err
		}
		cancel()
		return ctx.Err()
	})
	require.ErrorIs(t, err, context.Canceled)
	require.NoError(t, c.Close())
	c, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)
	fetched, err := c.Fetch(ctx, []string{"a", "b"}, Projection{})
	require.NoError(t, err)
	require.Equal(t, int32(42), fetched[0].Fields["rating"])
	require.Nil(t, fetched[1])
	require.NoError(t, c.Optimize(ctx, OptimizeOptions{}))
}

func TestOptimizeConcurrentCrashRecovery(t *testing.T) {
	const pathEnv = "XVEC_OPTIMIZE_CRASH_PATH"
	const phaseEnv = "XVEC_OPTIMIZE_CRASH_PHASE"
	ctx := context.Background()
	if path := os.Getenv(pathEnv); path != "" {
		options := NewCollectionOptions()
		options.WALSyncEvery = 1
		c, err := Open(ctx, path, options)
		require.NoError(t, err)
		pause := func() {
			require.NoError(t, os.WriteFile(filepath.Join(path, ".concurrent-crash-ready"), nil, 0o600))
			for {
				time.Sleep(time.Second)
			}
		}
		err = c.optimize(ctx, OptimizeOptions{}, func(p *db.Compaction, ctx context.Context) error {
			if err := p.Prepare(ctx); err != nil {
				return err
			}
			if _, err := c.Update(ctx, []Document{{PrimaryKey: "a", Fields: map[string]any{"rating": int32(42)}}}); err != nil {
				return err
			}
			if _, err := c.Delete(ctx, []string{"c"}); err != nil {
				return err
			}
			if _, err := c.Insert(ctx, []Document{atomicRecoveryDocument("e", "new", 5, 5, 5, 5, 5)}); err != nil {
				return err
			}
			if os.Getenv(phaseEnv) == "before_commit" {
				pause()
			}
			return nil
		})
		require.NoError(t, err)
		pause()
		return
	}
	for _, phase := range []string{"before_commit", "after_commit"} {
		t.Run(phase, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "crash")
			options := NewCollectionOptions()
			options.WALSyncEvery = 1
			c, err := CreateAndOpen(ctx, path, atomicRecoverySchema(), options)
			require.NoError(t, err)
			_, err = c.Insert(ctx, []Document{
				atomicRecoveryDocument("a", "a", 1, 1, 1, 1, 1),
				atomicRecoveryDocument("b", "b", 2, 2, 2, 2, 2),
				atomicRecoveryDocument("c", "c", 3, 3, 3, 3, 3),
			})
			require.NoError(t, err)
			_, err = c.Delete(ctx, []string{"b"})
			require.NoError(t, err)
			require.NoError(t, c.Flush(ctx))
			generation := c.store.Manifest().Generation
			require.NoError(t, c.Close())
			command := exec.Command(os.Args[0], "-test.run=^TestOptimizeConcurrentCrashRecovery$")
			command.Env = append(os.Environ(), pathEnv+"="+path, phaseEnv+"="+phase)
			require.NoError(t, command.Start())
			t.Cleanup(func() {
				if command.ProcessState == nil {
					_ = command.Process.Kill()
					_ = command.Wait()
				}
			})
			waitForAtomicMarker(t, command, filepath.Join(path, ".concurrent-crash-ready"))
			killAtomicRecoveryChild(t, command)
			c, err = Open(ctx, path, NewCollectionOptions())
			require.NoError(t, err)
			defer func() { require.NoError(t, c.Close()) }()
			if phase == "before_commit" {
				require.Equal(t, generation, c.store.Manifest().Generation)
			} else {
				require.Greater(t, c.store.Manifest().Generation, generation)
			}
			fetched, err := c.Fetch(ctx, []string{"a", "b", "c", "e"}, Projection{})
			require.NoError(t, err)
			require.Equal(t, int32(42), fetched[0].Fields["rating"])
			require.Equal(t, uint64(3), fetched[0].DocID)
			require.Nil(t, fetched[1])
			require.Nil(t, fetched[2])
			require.Equal(t, uint64(4), fetched[3].DocID)
			require.NoError(t, c.Optimize(ctx, OptimizeOptions{}))
			assertOptimizeArtifacts(t, path, 1)
		})
	}
}
