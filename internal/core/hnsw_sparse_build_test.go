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

func TestSparseHNSWBuildOptionsAndValidation(t *testing.T) {
	t.Parallel()
	defaults := DefaultSparseHNSWBuildOptions()
	if defaults.Metric != MetricIP || defaults.M != DefaultHNSWM || defaults.EFConstruction != DefaultHNSWEFConstruction {
		t.Fatalf("defaults = %#v", defaults)
	}
	for _, options := range []HNSWBuildOptions{
		{},
		DefaultHNSWBuildOptions(MetricL2),
		func() HNSWBuildOptions { value := defaults; value.M = 0; return value }(),
		func() HNSWBuildOptions { value := defaults; value.EFConstruction = value.M - 1; return value }(),
	} {
		if _, err := NewSparseHNSWBuilder(options); !errors.Is(err, ErrInvalidHNSWOptions) {
			t.Fatalf("options %#v error = %v", options, err)
		}
	}
}

func TestSparseHNSWBuildGraphInvariants(t *testing.T) {
	t.Parallel()
	options := DefaultSparseHNSWBuildOptions()
	options.M = 5
	options.EFConstruction = 24
	options.Seed = 0x5eed
	inputs := sparseHNSWBuildInputs(120)
	index := buildSparseHNSW(t, options, inputs)
	if index.Metric() != MetricIP || index.Len() != len(inputs) || index.BuildOptions() != options {
		t.Fatalf("metadata = metric %d, len %d, options %#v", index.Metric(), index.Len(), index.BuildOptions())
	}
	entryKey, found := index.EntryPoint()
	if !found {
		t.Fatal("graph has no entry point")
	}
	entryLevel, _ := index.Level(entryKey)
	if entryLevel != index.MaxLevel() {
		t.Fatalf("entry level = %d, max = %d", entryLevel, index.MaxLevel())
	}
	assertSparseHNSWGraphInvariants(t, index)
}

func TestSparseHNSWBuildDeterministicAndOwned(t *testing.T) {
	t.Parallel()
	inputs := sparseHNSWBuildInputs(140)
	options := DefaultSparseHNSWBuildOptions()
	options.M = 4
	options.EFConstruction = 20
	options.Seed = 42
	first := buildSparseHNSW(t, options, inputs)
	second := buildSparseHNSW(t, options, inputs)
	if first.entryPoint != second.entryPoint || first.maxLevel != second.maxLevel || first.levelRNGState != second.levelRNGState ||
		!slices.Equal(first.levels, second.levels) || !reflect.DeepEqual(first.neighbors, second.neighbors) {
		t.Fatal("fixed seed and insertion order produced different sparse HNSW graphs")
	}

	original, found := first.SparseVector(inputs[0].key)
	if !found {
		t.Fatal("first input missing")
	}
	inputs[0].vector.Indices[0] = math.MaxUint32
	inputs[0].vector.Values[0] = -999
	if got, _ := first.SparseVector(inputs[0].key); !reflect.DeepEqual(got, original) {
		t.Fatal("builder did not own sparse input")
	}
	original.Indices[0] = math.MaxUint32
	original.Values[0] = -888
	if got, _ := first.SparseVector(inputs[0].key); got.Indices[0] == math.MaxUint32 || got.Values[0] == -888 {
		t.Fatal("SparseVector exposed mutable storage")
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

func TestSparseHNSWBuildEmptySingleAndLevels(t *testing.T) {
	t.Parallel()
	options := DefaultSparseHNSWBuildOptions()
	options.M = 2
	options.EFConstruction = 8
	options.Seed = 9
	builder, _ := NewSparseHNSWBuilder(options)
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

	builder, _ = NewSparseHNSWBuilder(options)
	if err := builder.AddSparse(context.Background(), 17, SparseVector{}); err != nil {
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
	vector, found := single.SparseVector(17)
	if !found || len(vector.Indices) != 0 || len(vector.Values) != 0 {
		t.Fatalf("empty sparse vector = %#v, %v", vector, found)
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

func TestSparseHNSWBuilderLifecycleAndErrors(t *testing.T) {
	t.Parallel()
	options := DefaultSparseHNSWBuildOptions()
	options.M = 3
	options.EFConstruction = 12
	builder, _ := NewSparseHNSWBuilder(options)
	valid := SparseVector{Indices: []uint32{1, 9}, Values: []float32{2, 3}}
	if err := builder.AddSparse(nil, 1, valid); err == nil {
		t.Fatal("nil add context succeeded")
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if err := builder.AddSparse(canceled, 1, valid); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled add error = %v", err)
	}
	invalid := []struct {
		vector SparseVector
		want   error
	}{
		{SparseVector{Indices: []uint32{1}, Values: nil}, ailego.ErrDimensionMismatch},
		{SparseVector{Indices: []uint32{2, 1}, Values: []float32{1, 2}}, ailego.ErrInvalidSparseOrder},
		{SparseVector{Indices: []uint32{1, 1}, Values: []float32{1, 2}}, ailego.ErrInvalidSparseOrder},
		{SparseVector{Indices: []uint32{1}, Values: []float32{float32(math.NaN())}}, ailego.ErrNonFiniteVector},
	}
	for _, test := range invalid {
		if err := builder.AddSparse(context.Background(), 1, test.vector); !errors.Is(err, test.want) {
			t.Fatalf("vector %#v error = %v", test.vector, err)
		}
	}
	if err := builder.AddSparse(context.Background(), 1, valid); err != nil {
		t.Fatal(err)
	}
	if err := builder.AddSparse(context.Background(), 1, valid); !errors.Is(err, ErrDuplicateKey) {
		t.Fatalf("duplicate error = %v", err)
	}
	if _, err := builder.Build(canceled); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled build error = %v", err)
	}
	index, err := builder.Build(context.Background())
	if err != nil || index.Len() != 1 {
		t.Fatalf("retry build = %#v, %v", index, err)
	}
	if err := builder.AddSparse(context.Background(), 2, SparseVector{}); !errors.Is(err, ErrBuilderClosed) {
		t.Fatalf("closed add error = %v", err)
	}
	if _, err := builder.Build(context.Background()); !errors.Is(err, ErrBuilderClosed) {
		t.Fatalf("closed build error = %v", err)
	}
	var nilBuilder *SparseHNSWBuilder
	if err := nilBuilder.AddSparse(context.Background(), 1, SparseVector{}); err == nil {
		t.Fatal("nil builder add succeeded")
	}
	if _, err := nilBuilder.Build(context.Background()); err == nil {
		t.Fatal("nil builder build succeeded")
	}
}

func BenchmarkSparseHNSWBuild(b *testing.B) {
	inputs := sparseHNSWBuildInputs(1000)
	options := DefaultSparseHNSWBuildOptions()
	options.M = 16
	options.EFConstruction = 100
	for b.Loop() {
		builder, err := NewSparseHNSWBuilder(options)
		if err != nil {
			b.Fatal(err)
		}
		for _, input := range inputs {
			if err := builder.AddSparse(context.Background(), input.key, input.vector); err != nil {
				b.Fatal(err)
			}
		}
		if _, err := builder.Build(context.Background()); err != nil {
			b.Fatal(err)
		}
	}
}

type sparseHNSWInput struct {
	key    uint64
	vector SparseVector
}

func sparseHNSWBuildInputs(count int) []sparseHNSWInput {
	inputs := make([]sparseHNSWInput, count)
	for position := range inputs {
		inputs[position] = sparseHNSWInput{
			key: uint64(position*19 + 5),
			vector: SparseVector{
				Indices: []uint32{uint32(position % 31), uint32(100 + position%37), uint32(200 + position%43)},
				Values: []float32{
					float32(position%7) + 0.25,
					float32(position%11) + 0.5,
					float32(position%13) + 0.75,
				},
			},
		}
	}
	return inputs
}

func buildSparseHNSW(t testing.TB, options HNSWBuildOptions, inputs []sparseHNSWInput) *SparseHNSWIndex {
	t.Helper()
	builder, err := NewSparseHNSWBuilder(options)
	if err != nil {
		t.Fatal(err)
	}
	for _, input := range inputs {
		if err := builder.AddSparse(context.Background(), input.key, input.vector); err != nil {
			t.Fatal(err)
		}
	}
	index, err := builder.Build(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	return index
}

func assertSparseHNSWGraphInvariants(t testing.TB, index *SparseHNSWIndex) {
	t.Helper()
	count := len(index.keys)
	if len(index.offsets) != count+1 || len(index.indices) != len(index.values) ||
		len(index.positions) != count || len(index.levels) != count || len(index.neighbors) != count {
		t.Fatal("inconsistent sparse HNSW top-level storage")
	}
	if index.offsets[0] != 0 || index.offsets[count] != len(index.indices) {
		t.Fatal("inconsistent sparse HNSW offsets")
	}
	derivedMax := -1
	for position, key := range index.keys {
		if mapped := index.positions[key]; mapped != position {
			t.Fatalf("key %d maps to %d, want %d", key, mapped, position)
		}
		if index.offsets[position] > index.offsets[position+1] {
			t.Fatalf("node %d has descending CSR offsets", position)
		}
		vector := index.sparseVectorAt(position)
		if _, err := ailego.SparseInnerProduct(vector.Indices, vector.Values, nil, nil); err != nil {
			t.Fatalf("node %d vector is invalid: %v", position, err)
		}
		level := index.levels[position]
		derivedMax = max(derivedMax, level)
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
				if neighbor < 0 || neighbor >= count || neighbor == position || index.levels[neighbor] < currentLevel {
					t.Fatalf("node %d level %d has invalid neighbor %d", position, currentLevel, neighbor)
				}
				if _, duplicate := seen[neighbor]; duplicate {
					t.Fatalf("node %d level %d repeats neighbor %d", position, currentLevel, neighbor)
				}
				seen[neighbor] = struct{}{}
			}
		}
	}
	if derivedMax != index.maxLevel || (count != 0 && index.levels[index.entryPoint] != derivedMax) {
		t.Fatalf("entry/max level mismatch: entry %d max %d, derived %d", index.entryPoint, index.maxLevel, derivedMax)
	}
}
