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
	"math"
	"reflect"
	"slices"
	"testing"

	"github.com/gorse-io/zvec/internal/ailego"
)

func TestIVFBuildPartitionsVectors(t *testing.T) {
	t.Parallel()
	options := DefaultIVFBuildOptions(MetricL2)
	options.NList = 2
	options.NIterations = 20
	options.Seed = 7
	builder, err := NewIVFBuilder(2, options)
	if err != nil {
		t.Fatal(err)
	}
	input := []Candidate{
		{Key: 10, Vector: []float32{0, 0}},
		{Key: 11, Vector: []float32{0, 1}},
		{Key: 20, Vector: []float32{10, 10}},
		{Key: 21, Vector: []float32{10, 11}},
	}
	for _, candidate := range input {
		if err := builder.Add(context.Background(), candidate.Key, candidate.Vector); err != nil {
			t.Fatal(err)
		}
	}
	index, err := builder.Build(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if index.Dimension() != 2 || index.Metric() != MetricL2 || index.Len() != 4 || index.NList() != 2 {
		t.Fatalf("metadata = (%d, %d, %d, %d)", index.Dimension(), index.Metric(), index.Len(), index.NList())
	}
	centroids := index.Centroids()
	if len(centroids) != 2 {
		t.Fatalf("centroids = %v", centroids)
	}
	firstList, ok := index.ListForKey(10)
	if !ok {
		t.Fatal("key 10 is not assigned")
	}
	if list, _ := index.ListForKey(11); list != firstList {
		t.Fatalf("near vectors split across lists: %d and %d", firstList, list)
	}
	secondList, ok := index.ListForKey(20)
	if !ok || secondList == firstList {
		t.Fatalf("far cluster list = %d, first = %d", secondList, firstList)
	}
	if list, _ := index.ListForKey(21); list != secondList {
		t.Fatalf("near vectors split across lists: %d and %d", secondList, list)
	}
	for _, list := range []int{firstList, secondList} {
		candidates, err := index.List(list)
		if err != nil {
			t.Fatal(err)
		}
		if len(candidates) != 2 || candidates[0].Key > candidates[1].Key {
			t.Fatalf("list %d candidates = %#v", list, candidates)
		}
	}
	if index.TrainingIterations() == 0 || math.IsNaN(index.TrainingCost()) || math.IsInf(index.TrainingCost(), 0) {
		t.Fatalf("training stats = (%d, %g)", index.TrainingIterations(), index.TrainingCost())
	}
}

func TestIVFBuildOwnsOriginalVectors(t *testing.T) {
	t.Parallel()
	options := DefaultIVFBuildOptions(MetricL2)
	options.NList = 1
	builder, _ := NewIVFBuilder(2, options)
	input := []float32{1, 2}
	if err := builder.Add(context.Background(), 7, input); err != nil {
		t.Fatal(err)
	}
	input[0] = 99
	index, err := builder.Build(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	vector, found := index.Vector(7)
	if !found || !slices.Equal(vector, []float32{1, 2}) {
		t.Fatalf("vector = %v, %v", vector, found)
	}
	vector[0] = 88
	candidates, _ := index.List(0)
	candidates[0].Vector[0] = 77
	centroids := index.Centroids()
	centroids[0][0] = 66
	vector, _ = index.Vector(7)
	if vector[0] != 1 || index.Centroids()[0][0] != 1 {
		t.Fatal("IVF accessors expose mutable index state")
	}
}

func TestIVFBuildDeterministicAcrossWorkers(t *testing.T) {
	t.Parallel()
	build := func(workers int) *IVFIndex {
		options := DefaultIVFBuildOptions(MetricL2)
		options.NList = 7
		options.NIterations = 8
		options.Seed = 123
		options.Workers = workers
		builder, err := NewIVFBuilder(3, options)
		if err != nil {
			t.Fatal(err)
		}
		for index := 0; index < 100; index++ {
			vector := []float32{float32(index % 11), float32(index%7) / 2, float32(index%5) - 2}
			if err := builder.Add(context.Background(), uint64(index+1), vector); err != nil {
				t.Fatal(err)
			}
		}
		built, err := builder.Build(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		return built
	}
	one, many := build(1), build(8)
	if !reflect.DeepEqual(one.Centroids(), many.Centroids()) || one.TrainingCost() != many.TrainingCost() {
		t.Fatal("IVF training differs across worker counts")
	}
	for key := uint64(1); key <= 100; key++ {
		left, _ := one.ListForKey(key)
		right, _ := many.ListForKey(key)
		if left != right {
			t.Fatalf("key %d list = %d and %d", key, left, right)
		}
	}
}

func TestIVFBuildEmptyAndClusterCap(t *testing.T) {
	t.Parallel()
	options := DefaultIVFBuildOptions(MetricL2)
	builder, _ := NewIVFBuilder(3, options)
	empty, err := builder.Build(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if empty.Len() != 0 || empty.NList() != 0 || len(empty.Centroids()) != 0 {
		t.Fatalf("empty index = (%d, %d, %v)", empty.Len(), empty.NList(), empty.Centroids())
	}
	if _, err := empty.List(0); !errors.Is(err, ErrInvalidIVFList) {
		t.Fatalf("empty list error = %v", err)
	}

	options.NList = 100
	builder, _ = NewIVFBuilder(1, options)
	for key := uint64(1); key <= 3; key++ {
		if err := builder.Add(context.Background(), key, []float32{float32(key)}); err != nil {
			t.Fatal(err)
		}
	}
	index, err := builder.Build(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if index.NList() != 3 {
		t.Fatalf("effective NList = %d, want 3", index.NList())
	}
}

func TestIVFBuilderLifecycleAndValidation(t *testing.T) {
	t.Parallel()
	valid := DefaultIVFBuildOptions(MetricL2)
	for _, options := range []IVFBuildOptions{
		{},
		func() IVFBuildOptions { value := valid; value.Metric = 0; return value }(),
		func() IVFBuildOptions { value := valid; value.NList = 0; return value }(),
		func() IVFBuildOptions { value := valid; value.NIterations = 0; return value }(),
		func() IVFBuildOptions { value := valid; value.Tolerance = -1; return value }(),
		func() IVFBuildOptions { value := valid; value.Tolerance = math.NaN(); return value }(),
	} {
		if _, err := NewIVFBuilder(2, options); !errors.Is(err, ErrInvalidIVFOptions) {
			t.Errorf("options %#v error = %v", options, err)
		}
	}
	if _, err := NewIVFBuilder(0, valid); !errors.Is(err, ErrInvalidDimension) {
		t.Fatalf("dimension error = %v", err)
	}
	builder, _ := NewIVFBuilder(2, valid)
	if err := builder.Add(nil, 1, []float32{1, 2}); err == nil {
		t.Fatal("nil add context accepted")
	}
	if err := builder.Add(context.Background(), 1, []float32{1}); !errors.Is(err, ailego.ErrDimensionMismatch) {
		t.Fatalf("vector dimension error = %v", err)
	}
	if err := builder.Add(context.Background(), 1, []float32{1, float32(math.Inf(1))}); !errors.Is(err, ailego.ErrNonFiniteVector) {
		t.Fatalf("non-finite error = %v", err)
	}
	if err := builder.Add(context.Background(), 1, []float32{1, 2}); err != nil {
		t.Fatal(err)
	}
	if err := builder.Add(context.Background(), 1, []float32{3, 4}); !errors.Is(err, ErrDuplicateKey) {
		t.Fatalf("duplicate error = %v", err)
	}
	index, err := builder.Build(context.Background())
	if err != nil || index.Len() != 1 {
		t.Fatalf("build = %v, %v", index, err)
	}
	if _, err := builder.Build(context.Background()); !errors.Is(err, ErrBuilderClosed) {
		t.Fatalf("second build error = %v", err)
	}
	if err := builder.Add(context.Background(), 2, []float32{3, 4}); !errors.Is(err, ErrBuilderClosed) {
		t.Fatalf("post-build add error = %v", err)
	}
	if _, found := index.Vector(99); found {
		t.Fatal("missing vector found")
	}
	if _, found := index.ListForKey(99); found {
		t.Fatal("missing list assignment found")
	}
	if _, err := index.List(-1); !errors.Is(err, ErrInvalidIVFList) {
		t.Fatalf("negative list error = %v", err)
	}
}

func TestIVFBuilderCancellationCanRetry(t *testing.T) {
	t.Parallel()
	options := DefaultIVFBuildOptions(MetricL2)
	options.NList = 2
	builder, _ := NewIVFBuilder(1, options)
	for key := uint64(1); key <= 4; key++ {
		if err := builder.Add(context.Background(), key, []float32{float32(key)}); err != nil {
			t.Fatal(err)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := builder.Build(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled build error = %v", err)
	}
	index, err := builder.Build(context.Background())
	if err != nil || index.Len() != 4 {
		t.Fatalf("retry = %v, %v", index, err)
	}
}

func TestIVFBuildOptionsAreValueSemantic(t *testing.T) {
	t.Parallel()
	options := DefaultIVFBuildOptions(MetricIP)
	options.NList = 2
	builder, _ := NewIVFBuilder(1, options)
	for key, value := range map[uint64]float32{1: -2, 2: -1, 3: 1, 4: 2} {
		if err := builder.Add(context.Background(), key, []float32{value}); err != nil {
			t.Fatal(err)
		}
	}
	options.NList = 99
	index, err := builder.Build(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if index.BuildOptions().NList != 2 || index.Metric() != MetricIP {
		t.Fatalf("options = %#v", index.BuildOptions())
	}
}
