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

package core

import (
	"context"
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"reflect"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gorse-io/zvec/internal/ailego"
)

func TestHNSWIncrementalMatchesOneShotBuild(t *testing.T) {
	t.Parallel()
	inputs := hnswBuildInputs(180)
	options := DefaultHNSWBuildOptions(MetricL2)
	options.M = 8
	options.EFConstruction = 40
	options.Seed = 0x3141592653589793

	streamed := buildHNSWPrefix(t, options, inputs[:120])
	for _, input := range inputs[120:] {
		if err := streamed.Add(context.Background(), input.Key, input.Vector); err != nil {
			t.Fatal(err)
		}
	}
	oneShot := buildHNSWPrefix(t, options, inputs)
	assertSameHNSWIndex(t, streamed, oneShot)

	path := filepath.Join(t.TempDir(), "streamed.hnsw")
	if err := streamed.Save(context.Background(), path); err != nil {
		t.Fatal(err)
	}
	reopened, err := OpenHNSWIndex(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	assertSameHNSWIndex(t, reopened, oneShot)

	next := Candidate{Key: 999999, Vector: []float32{3.25, 7.5, 1.125}}
	if err := reopened.Add(context.Background(), next.Key, next.Vector); err != nil {
		t.Fatal(err)
	}
	all := append(slices.Clone(inputs), next)
	want := buildHNSWPrefix(t, options, all)
	assertSameHNSWIndex(t, reopened, want)
}

func TestHNSWIncrementalEmptyAndOwnership(t *testing.T) {
	t.Parallel()
	options := DefaultHNSWBuildOptions(MetricCosine)
	options.M = 4
	options.EFConstruction = 16
	options.Seed = 7
	builder, err := NewHNSWBuilder(2, options)
	if err != nil {
		t.Fatal(err)
	}
	index, err := builder.Build(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	vector := []float32{1, 2}
	if err := index.Add(context.Background(), 17, vector); err != nil {
		t.Fatal(err)
	}
	vector[0] = -100
	stored, found := index.Vector(17)
	if !found || !slices.Equal(stored, []float32{1, 2}) {
		t.Fatalf("stored incremental vector = %v, %v", stored, found)
	}
	entry, found := index.EntryPoint()
	if !found || entry != 17 || index.Len() != 1 {
		t.Fatalf("single streamed graph = entry %d/%v, len %d", entry, found, index.Len())
	}
	results, err := index.Search(context.Background(), []float32{1, 2}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(results, []Result{{Key: 17, Score: 0}}) {
		t.Fatalf("single streamed search = %#v", results)
	}
}

func TestHNSWIncrementalFailuresAreAtomic(t *testing.T) {
	t.Parallel()
	index := persistedHNSWIndex(t, MetricL2, 300)
	before, err := encodeHNSWIndex(context.Background(), index)
	if err != nil {
		t.Fatal(err)
	}

	if err := index.Add(nil, 100000, []float32{1, 2, 3}); err == nil {
		t.Fatal("nil context succeeded")
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if err := index.Add(canceled, 100000, []float32{1, 2, 3}); !errors.Is(err, context.Canceled) {
		t.Fatalf("pre-canceled error = %v", err)
	}
	if err := index.Add(context.Background(), 100000, []float32{1, 2}); !errors.Is(err, ailego.ErrDimensionMismatch) {
		t.Fatalf("dimension error = %v", err)
	}
	if err := index.Add(context.Background(), 100000, []float32{1, float32(math.NaN()), 3}); !errors.Is(err, ailego.ErrNonFiniteVector) {
		t.Fatalf("finite error = %v", err)
	}
	if err := index.Add(context.Background(), index.keys[0], []float32{1, 2, 3}); !errors.Is(err, ErrDuplicateKey) {
		t.Fatalf("duplicate error = %v", err)
	}

	// This context cancels after cloning has begun, exercising rollback after
	// private topology state has already been allocated and copied.
	midClone := newCancelAfterChecks(5)
	if err := index.Add(midClone, 100000, []float32{1, 2, 3}); !errors.Is(err, context.Canceled) {
		t.Fatalf("mid-clone cancellation error = %v", err)
	}
	midTraversal := newCancelAfterChecks(7)
	if err := index.Add(midTraversal, 100000, []float32{1, 2, 3}); !errors.Is(err, context.Canceled) {
		t.Fatalf("mid-traversal cancellation error = %v", err)
	}
	after, err := encodeHNSWIndex(context.Background(), index)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(after, before) {
		t.Fatal("failed incremental add changed graph generation")
	}
	var nilIndex *HNSWIndex
	if err := nilIndex.Add(context.Background(), 1, []float32{1}); err == nil {
		t.Fatal("nil index add succeeded")
	}

	// A failed add must not consume a sampled level. The next successful add
	// therefore still matches a one-shot build of the same sequence.
	next := Candidate{Key: 100000, Vector: []float32{1, 2, 3}}
	if err := index.Add(context.Background(), next.Key, next.Vector); err != nil {
		t.Fatal(err)
	}
	inputs := append(hnswBuildInputs(300), next)
	want := buildHNSWPrefix(t, index.options, inputs)
	assertSameHNSWIndex(t, index, want)
}

func TestHNSWConcurrentStreamingSearchAndSave(t *testing.T) {
	options := DefaultHNSWBuildOptions(MetricL2)
	options.M = 6
	options.EFConstruction = 30
	options.Seed = 123
	builder, err := NewHNSWBuilder(3, options)
	if err != nil {
		t.Fatal(err)
	}
	index, err := builder.Build(context.Background())
	if err != nil {
		t.Fatal(err)
	}

	dir := t.TempDir()
	errCh := make(chan error, 32)
	var writers sync.WaitGroup
	for worker := 0; worker < 4; worker++ {
		writers.Add(1)
		go func(worker int) {
			defer writers.Done()
			for value := 0; value < 50; value++ {
				key := uint64(worker*100 + value + 1)
				vector := []float32{float32(worker + 1), float32(value + 1), float32(worker + value + 1)}
				if err := index.Add(context.Background(), key, vector); err != nil {
					errCh <- err
					return
				}
				if _, found := index.Vector(key); !found {
					errCh <- fmt.Errorf("key %d missing immediately after add", key)
					return
				}
			}
		}(worker)
	}

	stopReaders := make(chan struct{})
	var readers sync.WaitGroup
	for worker := 0; worker < 4; worker++ {
		readers.Add(1)
		go func() {
			defer readers.Done()
			for {
				select {
				case <-stopReaders:
					return
				default:
				}
				if _, err := index.Search(context.Background(), []float32{1, 2, 3}, 10); err != nil {
					errCh <- err
					return
				}
				if entry, found := index.EntryPoint(); found {
					if _, found := index.Level(entry); !found {
						errCh <- fmt.Errorf("entry key %d has no level", entry)
						return
					}
				}
			}
		}()
	}
	readers.Add(1)
	go func() {
		defer readers.Done()
		for generation := 0; generation < 10; generation++ {
			path := filepath.Join(dir, fmt.Sprintf("snapshot-%02d.hnsw", generation))
			if err := index.Save(context.Background(), path); err != nil {
				errCh <- err
				return
			}
			if _, err := OpenHNSWIndex(context.Background(), path); err != nil {
				errCh <- err
				return
			}
		}
	}()

	writers.Wait()
	close(stopReaders)
	readers.Wait()
	close(errCh)
	for err := range errCh {
		t.Fatal(err)
	}
	if index.Len() != 200 {
		t.Fatalf("final length = %d, want 200", index.Len())
	}
	path := filepath.Join(dir, "final.hnsw")
	if err := index.Save(context.Background(), path); err != nil {
		t.Fatal(err)
	}
	reopened, err := OpenHNSWIndex(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	assertSameHNSWIndex(t, reopened, index)
}

func buildHNSWPrefix(t testing.TB, options HNSWBuildOptions, inputs []Candidate) *HNSWIndex {
	t.Helper()
	dimension := 3
	if len(inputs) != 0 {
		dimension = len(inputs[0].Vector)
	}
	builder, err := NewHNSWBuilder(dimension, options)
	if err != nil {
		t.Fatal(err)
	}
	for _, input := range inputs {
		if err := builder.Add(context.Background(), input.Key, input.Vector); err != nil {
			t.Fatal(err)
		}
	}
	index, err := builder.Build(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	return index
}

type cancelAfterChecks struct {
	limit int32
	calls atomic.Int32
	done  chan struct{}
	once  sync.Once
}

func newCancelAfterChecks(limit int32) *cancelAfterChecks {
	return &cancelAfterChecks{limit: limit, done: make(chan struct{})}
}

func (c *cancelAfterChecks) Deadline() (time.Time, bool) { return time.Time{}, false }
func (c *cancelAfterChecks) Done() <-chan struct{}       { return c.done }
func (c *cancelAfterChecks) Value(any) any               { return nil }
func (c *cancelAfterChecks) Err() error {
	if c.calls.Add(1) < c.limit {
		return nil
	}
	c.once.Do(func() { close(c.done) })
	return context.Canceled
}
