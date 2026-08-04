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
	"reflect"
	"slices"
	"testing"
)

func TestScalarQuantizedDiskANNMatchesScalarTruthAndRefinesOriginals(t *testing.T) {
	const dimension = 8
	candidates := diskANNIndexCandidates(36, dimension)
	query := slices.Clone(candidates[17].Vector)
	query[1] += .137

	for _, kind := range []Quantization{QuantizationFP16, QuantizationInt8, QuantizationInt4} {
		for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine, MetricMIPSL2} {
			t.Run(quantizationName(kind)+"/"+diskANNMetricName(metric), func(t *testing.T) {
				var reformer DenseReformer
				if kind != QuantizationFP16 {
					signs := make([]byte, 4*((dimension+7)/8))
					for position := range signs {
						signs[position] = byte(position*73 + int(kind)*19)
					}
					rotator, err := NewFHTRotatorFromSigns(dimension, signs)
					if err != nil {
						t.Fatal(err)
					}
					reformer, err = NewRotationReformer(rotator)
					if err != nil {
						t.Fatal(err)
					}
				}

				options := DefaultDiskANNBuildOptions(metric)
				options.MaxDegree, options.ListSize, options.PQChunks = 8, len(candidates), 4
				options.Workers, options.CacheCapacity = 2, len(candidates)
				index, err := NewScalarQuantizedDiskANNIndex(
					context.Background(), dimension, options, kind, reformer, candidates,
				)
				if err != nil {
					t.Fatal(err)
				}
				defer index.Close()
				truth, err := NewScalarQuantizedFlatIndex(
					context.Background(), dimension, metric, kind, reformer, candidates,
				)
				if err != nil {
					t.Fatal(err)
				}
				if index.Dimension() != dimension || index.Metric() != metric || index.Len() != len(candidates) || index.PQChunks() != 4 {
					t.Fatalf("metadata = dimension %d metric %d len %d chunks %d", index.Dimension(), index.Metric(), index.Len(), index.PQChunks())
				}

				search := SearchOptions{TopK: 12, Filter: func(key uint64) bool { return key%3 != 0 }}
				want, err := truth.SearchWithOptions(context.Background(), query, search)
				if err != nil {
					t.Fatal(err)
				}
				linear, err := index.SearchDiskANN(context.Background(), query, DiskANNSearchOptions{
					SearchOptions: search, ListSize: len(candidates), Linear: true,
				})
				if err != nil {
					t.Fatal(err)
				}
				graph, err := index.SearchDiskANN(context.Background(), query, DiskANNSearchOptions{
					SearchOptions: search, ListSize: len(candidates),
				})
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(linear, want) || !reflect.DeepEqual(graph, want) {
					t.Fatalf("DiskANN = linear %#v graph %#v, scalar truth %#v", linear, graph, want)
				}

				boundedOptions := search
				boundedOptions.Radius = want[6].Score
				if metric == MetricIP && boundedOptions.Radius <= 0 {
					boundedOptions.Radius = .0001
				}
				bounded, err := index.SearchDiskANN(context.Background(), query, DiskANNSearchOptions{
					SearchOptions: boundedOptions, ListSize: len(candidates),
				})
				if err != nil {
					t.Fatal(err)
				}
				for _, result := range bounded {
					if !scoreWithinRadius(metric, result.Score, boundedOptions.Radius) {
						t.Fatalf("radius admitted %#v", result)
					}
				}

				all, err := index.SearchDiskANN(context.Background(), query, DiskANNSearchOptions{
					SearchOptions: SearchOptions{TopK: len(candidates)}, ListSize: len(candidates), Linear: true,
				})
				if err != nil {
					t.Fatal(err)
				}
				refiner, err := NewOriginalVectorRefiner(index, metric)
				if err != nil {
					t.Fatal(err)
				}
				refined, err := refiner.Refine(context.Background(), query, all, SearchOptions{TopK: 5})
				if err != nil {
					t.Fatal(err)
				}
				originalTruth, err := TopK(context.Background(), metric, query, candidates, 5)
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(refined, originalTruth) {
					t.Fatalf("refined = %#v, original truth %#v", refined, originalTruth)
				}
				original, found := index.Vector(candidates[9].Key)
				if !found || !slices.Equal(original, candidates[9].Vector) {
					t.Fatalf("original vector = %v, %v", original, found)
				}
				original[0]++
				again, _ := index.Vector(candidates[9].Key)
				if slices.Equal(original, again) {
					t.Fatal("Vector returned an alias")
				}
			})
		}
	}
}

func TestScalarQuantizedDiskANNValidationCancellationAndClose(t *testing.T) {
	options := DefaultDiskANNBuildOptions(MetricL2)
	options.MaxDegree, options.ListSize, options.PQChunks = 4, 8, 2
	candidates := []Candidate{{Key: 1, Vector: []float32{1, 2, 3, 4}}}
	if _, err := NewScalarQuantizedDiskANNIndex(nil, 4, options, QuantizationFP16, nil, candidates); err == nil {
		t.Fatal("nil context succeeded")
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := NewScalarQuantizedDiskANNIndex(canceled, 4, options, QuantizationFP16, nil, candidates); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled build = %v", err)
	}
	if _, err := NewScalarQuantizedDiskANNIndex(context.Background(), 4, options, QuantizationFP16, nil,
		[]Candidate{{Key: 1, Vector: []float32{70000, 0, 0, 0}}}); !errors.Is(err, ErrQuantizationOverflow) {
		t.Fatalf("FP16 overflow = %v", err)
	}
	if _, err := NewScalarQuantizedDiskANNIndex(context.Background(), 3, options, QuantizationInt4, nil,
		[]Candidate{{Key: 1, Vector: []float32{1, 2, 3}}}); !errors.Is(err, ErrOddInt4Dimension) {
		t.Fatalf("odd INT4 dimension = %v", err)
	}
	index, err := NewScalarQuantizedDiskANNIndex(context.Background(), 4, options, QuantizationFP16, nil, candidates)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := index.Search(context.Background(), []float32{1, 2, 3, 4}, 0); err != nil {
		t.Fatalf("zero top-k = %v", err)
	}
	if _, err := index.SearchWithOptions(context.Background(), []float32{1, 2, 3, 4}, SearchOptions{}); !errors.Is(err, ErrInvalidTopK) {
		t.Fatalf("invalid top-k = %v", err)
	}
	if err := index.Close(); err != nil {
		t.Fatal(err)
	}
	if err := index.Close(); err != nil {
		t.Fatalf("second Close = %v", err)
	}
	if _, err := index.Search(context.Background(), []float32{1, 2, 3, 4}, 1); !errors.Is(err, ErrDiskANNClosed) {
		t.Fatalf("search after Close = %v", err)
	}
}

func quantizationName(kind Quantization) string {
	switch kind {
	case QuantizationFP16:
		return "FP16"
	case QuantizationInt8:
		return "INT8"
	case QuantizationInt4:
		return "INT4"
	default:
		return "unknown"
	}
}
