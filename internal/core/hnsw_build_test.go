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

func TestHNSWBuildOptionsAndValidation(t *testing.T) {
	t.Parallel()
	defaults := DefaultHNSWBuildOptions(MetricCosine)
	if defaults.M != 50 || defaults.EFConstruction != 500 || defaults.Metric != MetricCosine {
		t.Fatalf("defaults = %#v", defaults)
	}
	valid := DefaultHNSWBuildOptions(MetricL2)
	for _, options := range []HNSWBuildOptions{
		{},
		func() HNSWBuildOptions { value := valid; value.Metric = 0; return value }(),
		func() HNSWBuildOptions { value := valid; value.M = 0; return value }(),
		func() HNSWBuildOptions { value := valid; value.M = MaxHNSWM + 1; return value }(),
		func() HNSWBuildOptions { value := valid; value.EFConstruction = value.M - 1; return value }(),
	} {
		if _, err := NewHNSWBuilder(3, options); !errors.Is(err, ErrInvalidHNSWOptions) {
			t.Fatalf("options %#v error = %v", options, err)
		}
	}
	if _, err := NewHNSWBuilder(0, valid); !errors.Is(err, ErrInvalidDimension) {
		t.Fatalf("zero dimension error = %v", err)
	}
	if _, err := NewHNSWBuilder(MaxRotationDimension+1, valid); !errors.Is(err, ErrInvalidDimension) {
		t.Fatalf("large dimension error = %v", err)
	}
}

func TestHNSWBuildGraphInvariants(t *testing.T) {
	t.Parallel()
	for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine, MetricMIPSL2} {
		options := DefaultHNSWBuildOptions(metric)
		options.M = 4
		options.EFConstruction = 16
		options.Seed = 0x5eed
		builder, err := NewHNSWBuilder(3, options)
		if err != nil {
			t.Fatal(err)
		}
		inputs := hnswBuildInputs(80)
		for _, input := range inputs {
			if err := builder.Add(context.Background(), input.Key, input.Vector); err != nil {
				t.Fatal(err)
			}
		}
		index, err := builder.Build(context.Background())
		if err != nil {
			t.Fatalf("metric %d: %v", metric, err)
		}
		if index.Dimension() != 3 || index.Metric() != metric || index.Len() != len(inputs) || index.BuildOptions() != options {
			t.Fatalf("metric %d metadata differs", metric)
		}
		entryKey, found := index.EntryPoint()
		if !found {
			t.Fatalf("metric %d has no entry point", metric)
		}
		entryLevel, _ := index.Level(entryKey)
		if entryLevel != index.MaxLevel() {
			t.Fatalf("metric %d entry level = %d, max = %d", metric, entryLevel, index.MaxLevel())
		}
		assertHNSWGraphInvariants(t, index)
	}
}

func TestHNSWBuildDeterministicAndOwned(t *testing.T) {
	t.Parallel()
	inputs := hnswBuildInputs(120)
	build := func() *HNSWIndex {
		options := DefaultHNSWBuildOptions(MetricL2)
		options.M = 3
		options.EFConstruction = 12
		options.Seed = 42
		builder, err := NewHNSWBuilder(3, options)
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
	first, second := build(), build()
	if first.entryPoint != second.entryPoint || first.maxLevel != second.maxLevel || first.levelRNGState != second.levelRNGState ||
		!reflect.DeepEqual(first.levels, second.levels) || !reflect.DeepEqual(first.neighbors, second.neighbors) {
		t.Fatal("fixed seed and insertion order produced different HNSW graphs")
	}

	original, found := first.Vector(inputs[0].Key)
	if !found {
		t.Fatal("first input missing")
	}
	inputs[0].Vector[0] = -999
	if got, _ := first.Vector(inputs[0].Key); got[0] != original[0] {
		t.Fatal("builder did not own input vector")
	}
	original[0] = -888
	if got, _ := first.Vector(inputs[0].Key); got[0] == -888 {
		t.Fatal("Vector exposed mutable storage")
	}
	neighbors, err := first.Neighbors(first.keys[1], 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(neighbors) != 0 {
		neighbors[0] = math.MaxUint64
		again, err := first.Neighbors(first.keys[1], 0)
		if err != nil {
			t.Fatal(err)
		}
		if slices.Equal(neighbors, again) {
			t.Fatal("Neighbors exposed mutable storage")
		}
	}
}

func TestHNSWBuildEmptySingleAndLevels(t *testing.T) {
	t.Parallel()
	options := DefaultHNSWBuildOptions(MetricL2)
	options.M = 2
	options.EFConstruction = 4
	options.Seed = 9
	builder, _ := NewHNSWBuilder(2, options)
	empty, err := builder.Build(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if empty.Len() != 0 || empty.MaxLevel() != -1 {
		t.Fatalf("empty metadata = len %d, level %d", empty.Len(), empty.MaxLevel())
	}
	if _, found := empty.EntryPoint(); found {
		t.Fatal("empty graph has entry point")
	}
	if _, err := empty.Neighbors(1, 0); !errors.Is(err, ErrHNSWKeyNotFound) {
		t.Fatalf("unknown key error = %v", err)
	}

	builder, _ = NewHNSWBuilder(2, options)
	if err := builder.Add(context.Background(), 17, []float32{1, 2}); err != nil {
		t.Fatal(err)
	}
	single, err := builder.Build(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	entry, found := single.EntryPoint()
	if !found || entry != 17 {
		t.Fatalf("single entry = %d, %v", entry, found)
	}
	level, _ := single.Level(17)
	for current := 0; current <= level; current++ {
		neighbors, err := single.Neighbors(17, current)
		if err != nil || len(neighbors) != 0 {
			t.Fatalf("single level %d neighbors = %v, %v", current, neighbors, err)
		}
	}
	if _, err := single.Neighbors(17, level+1); !errors.Is(err, ErrInvalidHNSWLevel) {
		t.Fatalf("invalid level error = %v", err)
	}
}

func TestHNSWBuilderLifecycleAndErrors(t *testing.T) {
	t.Parallel()
	options := DefaultHNSWBuildOptions(MetricL2)
	options.M = 2
	options.EFConstruction = 8
	builder, _ := NewHNSWBuilder(2, options)
	if err := builder.Add(nil, 1, []float32{1, 2}); err == nil {
		t.Fatal("nil add context succeeded")
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if err := builder.Add(canceled, 1, []float32{1, 2}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled add error = %v", err)
	}
	if err := builder.Add(context.Background(), 1, []float32{1}); !errors.Is(err, ailego.ErrDimensionMismatch) {
		t.Fatalf("dimension error = %v", err)
	}
	if err := builder.Add(context.Background(), 1, []float32{1, float32(math.NaN())}); !errors.Is(err, ailego.ErrNonFiniteVector) {
		t.Fatalf("finite error = %v", err)
	}
	if err := builder.Add(context.Background(), 1, []float32{1, 2}); err != nil {
		t.Fatal(err)
	}
	if err := builder.Add(context.Background(), 1, []float32{2, 3}); !errors.Is(err, ErrDuplicateKey) {
		t.Fatalf("duplicate error = %v", err)
	}
	if _, err := builder.Build(canceled); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled build error = %v", err)
	}
	index, err := builder.Build(context.Background())
	if err != nil || index.Len() != 1 {
		t.Fatalf("retry build = %#v, %v", index, err)
	}
	if err := builder.Add(context.Background(), 2, []float32{2, 3}); !errors.Is(err, ErrBuilderClosed) {
		t.Fatalf("closed add error = %v", err)
	}
	if _, err := builder.Build(context.Background()); !errors.Is(err, ErrBuilderClosed) {
		t.Fatalf("closed build error = %v", err)
	}
	var nilBuilder *HNSWBuilder
	if err := nilBuilder.Add(context.Background(), 1, []float32{1}); err == nil {
		t.Fatal("nil builder add succeeded")
	}
	if _, err := nilBuilder.Build(context.Background()); err == nil {
		t.Fatal("nil builder build succeeded")
	}
}

func TestHNSWLevelSamplingDeterministicAndBounded(t *testing.T) {
	t.Parallel()
	first := splitMix64{state: 123}
	second := splitMix64{state: 123}
	levels := make([]int, 10000)
	upper := 0
	for index := range levels {
		levels[index] = sampleHNSWLevel(&first, 4)
		if got := sampleHNSWLevel(&second, 4); got != levels[index] {
			t.Fatalf("sample %d = %d, want %d", index, got, levels[index])
		}
		if levels[index] < 0 || levels[index] > MaxHNSWLevel {
			t.Fatalf("sample %d out of range: %d", index, levels[index])
		}
		if levels[index] > 0 {
			upper++
		}
	}
	if upper < 2000 || upper > 3000 {
		t.Fatalf("upper-level sample count = %d, want approximately 2500", upper)
	}
}

func BenchmarkHNSWBuild(b *testing.B) {
	inputs := hnswBuildInputs(1000)
	options := DefaultHNSWBuildOptions(MetricL2)
	options.M = 16
	options.EFConstruction = 100
	for b.Loop() {
		builder, err := NewHNSWBuilder(3, options)
		if err != nil {
			b.Fatal(err)
		}
		for _, input := range inputs {
			if err := builder.Add(context.Background(), input.Key, input.Vector); err != nil {
				b.Fatal(err)
			}
		}
		if _, err := builder.Build(context.Background()); err != nil {
			b.Fatal(err)
		}
	}
}

func hnswBuildInputs(count int) []Candidate {
	inputs := make([]Candidate, count)
	for index := range inputs {
		value := float32(index + 1)
		inputs[index] = Candidate{
			Key: uint64(index*17 + 3),
			Vector: []float32{
				float32((index*7)%31) + 0.25,
				float32((index*13)%37) + 0.5,
				value/float32(count+1) + 0.75,
			},
		}
	}
	return inputs
}

func assertHNSWGraphInvariants(t testing.TB, index *HNSWIndex) {
	t.Helper()
	if len(index.keys) != len(index.levels) || len(index.keys) != len(index.neighbors) || len(index.positions) != len(index.keys) {
		t.Fatal("inconsistent HNSW top-level storage")
	}
	maxLevel := -1
	for position, key := range index.keys {
		if mapped := index.positions[key]; mapped != position {
			t.Fatalf("key %d maps to %d, want %d", key, mapped, position)
		}
		level := index.levels[position]
		maxLevel = max(maxLevel, level)
		if level < 0 || level > MaxHNSWLevel || len(index.neighbors[position]) != level+1 {
			t.Fatalf("node %d level storage is invalid", position)
		}
		for currentLevel, neighbors := range index.neighbors[position] {
			limit := index.options.M
			if currentLevel == 0 {
				limit *= 2
			}
			if len(neighbors) > limit {
				t.Fatalf("node %d level %d degree = %d, limit %d", position, currentLevel, len(neighbors), limit)
			}
			seen := make(map[int]struct{}, len(neighbors))
			for _, neighbor := range neighbors {
				if neighbor < 0 || neighbor >= len(index.keys) || neighbor == position || index.levels[neighbor] < currentLevel {
					t.Fatalf("node %d level %d has invalid neighbor %d", position, currentLevel, neighbor)
				}
				if _, duplicate := seen[neighbor]; duplicate {
					t.Fatalf("node %d level %d repeats neighbor %d", position, currentLevel, neighbor)
				}
				seen[neighbor] = struct{}{}
			}
		}
	}
	if maxLevel != index.maxLevel || (len(index.keys) != 0 && index.levels[index.entryPoint] != maxLevel) {
		t.Fatalf("entry/max level mismatch: entry %d max %d, derived %d", index.entryPoint, index.maxLevel, maxLevel)
	}
}
