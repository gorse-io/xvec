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
	"sync"
	"testing"

	"github.com/gorse-io/zvec/internal/ailego"
)

func TestIVFIncrementalBootstrapAndAssignment(t *testing.T) {
	t.Parallel()
	options := DefaultIVFBuildOptions(MetricIP)
	options.NList = 3
	builder, err := NewIVFBuilder(2, options)
	if err != nil {
		t.Fatal(err)
	}
	index, err := builder.Build(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	inputs := []Candidate{
		{Key: 1, Vector: []float32{1, 0}},
		{Key: 2, Vector: []float32{0, 2}},
		{Key: 3, Vector: []float32{-1, 0}},
		{Key: 4, Vector: []float32{0, 1.5}},
	}
	for _, input := range inputs {
		if err := index.Add(context.Background(), input.Key, input.Vector); err != nil {
			t.Fatal(err)
		}
	}
	if index.Len() != 4 || index.NList() != 3 {
		t.Fatalf("size = %d vectors/%d lists", index.Len(), index.NList())
	}
	for key, want := range map[uint64]int{1: 0, 2: 1, 3: 2, 4: 1} {
		if got, found := index.ListForKey(key); !found || got != want {
			t.Fatalf("key %d list = %d, %v; want %d", key, got, found, want)
		}
	}
	if got := index.model.counts; !reflect.DeepEqual(got, []int{1, 2, 1}) {
		t.Fatalf("centroid counts = %v", got)
	}
	if got := index.TrainingCost(); got != -9 {
		t.Fatalf("incremental objective = %v, want -9", got)
	}
	if index.TrainingIterations() != 0 || index.TrainingConverged() {
		t.Fatal("bootstrap unexpectedly reports a completed training round")
	}

	query := []float32{0, 1}
	got, err := index.SearchIVF(context.Background(), query, IVFSearchOptions{
		SearchOptions: SearchOptions{TopK: len(inputs)},
		NProbe:        index.NList(),
	})
	if err != nil {
		t.Fatal(err)
	}
	want, err := TopK(context.Background(), MetricIP, query, inputs, len(inputs))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("incremental full-probe search = %#v, want %#v", got, want)
	}

	path := filepath.Join(t.TempDir(), "streamed.ivf")
	if err := index.Save(context.Background(), path); err != nil {
		t.Fatal(err)
	}
	reopened, err := OpenIVFIndex(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	assertSameIVFIndex(t, reopened, index)
}

func TestIVFIncrementalFixedCentroidsAndOwnership(t *testing.T) {
	t.Parallel()
	index := persistedIVFIndex(t, MetricL2, 2)
	centroids := index.Centroids()
	vector := []float32{10, 11, 12}
	if err := index.Add(context.Background(), 1000, vector); err != nil {
		t.Fatal(err)
	}
	vector[0] = -100
	stored, found := index.Vector(1000)
	if !found || stored[0] != 10 {
		t.Fatalf("stored incremental vector = %v, %v", stored, found)
	}
	if !reflect.DeepEqual(index.Centroids(), centroids) {
		t.Fatal("incremental add changed a full trained centroid set")
	}
	list, found := index.ListForKey(1000)
	if !found {
		t.Fatal("incremental key has no list")
	}
	candidates, err := index.List(list)
	if err != nil {
		t.Fatal(err)
	}
	if candidates[len(candidates)-1].Key != 1000 {
		t.Fatalf("incremental list order = %#v", candidates)
	}
}

func TestIVFIncrementalValidationIsAtomic(t *testing.T) {
	t.Parallel()
	index := persistedIVFIndex(t, MetricL2, 2)
	originalLen := index.Len()
	if err := index.Add(nil, 1000, []float32{1, 2, 3}); err == nil {
		t.Fatal("nil context succeeded")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := index.Add(ctx, 1000, []float32{1, 2, 3}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled error = %v", err)
	}
	if err := index.Add(context.Background(), 1000, []float32{1, 2}); !errors.Is(err, ailego.ErrDimensionMismatch) {
		t.Fatalf("dimension error = %v", err)
	}
	if err := index.Add(context.Background(), 1000, []float32{1, float32(math.NaN()), 3}); !errors.Is(err, ailego.ErrNonFiniteVector) {
		t.Fatalf("finite error = %v", err)
	}
	if err := index.Add(context.Background(), index.keys[0], []float32{1, 2, 3}); !errors.Is(err, ErrDuplicateKey) {
		t.Fatalf("duplicate error = %v", err)
	}
	if index.Len() != originalLen {
		t.Fatalf("failed adds changed length to %d", index.Len())
	}
	var nilIndex *IVFIndex
	if err := nilIndex.Add(context.Background(), 1, []float32{1}); err == nil {
		t.Fatal("nil index add succeeded")
	}
}

func TestIVFConcurrentStreamingSearchAndSave(t *testing.T) {
	options := DefaultIVFBuildOptions(MetricL2)
	options.NList = 8
	options.NIterations = 4
	builder, err := NewIVFBuilder(3, options)
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
	for worker := 0; worker < 8; worker++ {
		writers.Add(1)
		go func(worker int) {
			defer writers.Done()
			for value := 0; value < 100; value++ {
				key := uint64(worker*100 + value + 1)
				vector := []float32{float32(worker), float32(value), float32(worker + value)}
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
				nprobe := max(1, index.NList())
				_, err := index.SearchIVF(context.Background(), []float32{1, 2, 3}, IVFSearchOptions{
					SearchOptions: SearchOptions{TopK: 10}, NProbe: nprobe,
				})
				if err != nil {
					errCh <- err
					return
				}
			}
		}()
	}
	readers.Add(1)
	go func() {
		defer readers.Done()
		for generation := 0; generation < 20; generation++ {
			path := filepath.Join(dir, fmt.Sprintf("snapshot-%02d.ivf", generation))
			if err := index.Save(context.Background(), path); err != nil {
				errCh <- err
				return
			}
			if _, err := OpenIVFIndex(context.Background(), path); err != nil {
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
	if index.Len() != 800 {
		t.Fatalf("final length = %d, want 800", index.Len())
	}
	path := filepath.Join(dir, "final.ivf")
	if err := index.Save(context.Background(), path); err != nil {
		t.Fatal(err)
	}
	reopened, err := OpenIVFIndex(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	if reopened.Len() != index.Len() || reopened.NList() != index.NList() {
		t.Fatalf("reopened size = %d/%d, want %d/%d", reopened.Len(), reopened.NList(), index.Len(), index.NList())
	}
}
