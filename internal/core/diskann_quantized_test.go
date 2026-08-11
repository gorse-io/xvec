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

package core

import (
	"context"
	"slices"
	"testing"

	"github.com/stretchr/testify/require"
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
					require.NoError(t, err)

					reformer, err = NewRotationReformer(rotator)
					require.NoError(t, err)
				}

				options := DefaultDiskANNBuildOptions(metric)
				options.MaxDegree, options.ListSize, options.PQChunks = 8, len(candidates), 4
				options.Workers, options.CacheCapacity = 2, len(candidates)
				index, err := NewScalarQuantizedDiskANNIndex(
					context.Background(), dimension, options, kind, reformer, candidates,
				)
				require.NoError(t, err)

				defer index.Close()
				truth, err := NewScalarQuantizedFlatIndex(
					context.Background(), dimension, metric, kind, reformer, candidates,
				)
				require.NoError(t, err)
				require.Equal(t, dimension, index.Dimension())
				require.Equal(t, metric, index.Metric())
				require.Len(t, candidates, index.Len())
				require.True(t, index.PQChunks() == 4)

				search := SearchOptions{TopK: 12, Filter: func(key uint64) bool { return key%3 != 0 }}
				want, err := truth.SearchWithOptions(context.Background(), query, search)
				require.NoError(t, err)

				linear, err := index.SearchDiskANN(context.Background(), query, DiskANNSearchOptions{
					SearchOptions: search, ListSize: len(candidates), Linear: true,
				})
				require.NoError(t, err)

				graph, err := index.SearchDiskANN(context.Background(), query, DiskANNSearchOptions{
					SearchOptions: search, ListSize: len(candidates),
				})
				require.NoError(t, err)
				require.Equal(t, want, linear)
				require.Equal(t, want, graph)

				boundedOptions := search
				boundedOptions.Radius = want[6].Score
				if metric == MetricIP && boundedOptions.Radius <= 0 {
					boundedOptions.Radius = .0001
				}
				bounded, err := index.SearchDiskANN(context.Background(), query, DiskANNSearchOptions{
					SearchOptions: boundedOptions, ListSize: len(candidates),
				})
				require.NoError(t, err)

				for _, result := range bounded {
					require.True(t, scoreWithinRadius(metric, result.Score, boundedOptions.Radius))
				}

				all, err := index.SearchDiskANN(context.Background(), query, DiskANNSearchOptions{
					SearchOptions: SearchOptions{TopK: len(candidates)}, ListSize: len(candidates), Linear: true,
				})
				require.NoError(t, err)

				refiner, err := NewOriginalVectorRefiner(index, metric)
				require.NoError(t, err)

				refined, err := refiner.Refine(context.Background(), query, all, SearchOptions{TopK: 5})
				require.NoError(t, err)

				originalTruth, err := TopK(context.Background(), metric, query, candidates, 5)
				require.NoError(t, err)
				require.Equal(t, originalTruth, refined)

				original, found := index.Vector(candidates[9].Key)
				require.True(t, found)
				require.True(t, slices.Equal(original, candidates[9].Vector))

				original[0]++
				again, _ := index.Vector(candidates[9].Key)
				require.False(t, slices.Equal(original, again),
					"Vector returned an alias")
			})
		}
	}
}

func TestScalarQuantizedDiskANNValidationCancellationAndClose(t *testing.T) {
	options := DefaultDiskANNBuildOptions(MetricL2)
	options.MaxDegree, options.ListSize, options.PQChunks = 4, 8, 2
	candidates := []Candidate{{Key: 1, Vector: []float32{1, 2, 3, 4}}}
	{
		_, err := NewScalarQuantizedDiskANNIndex(nil, 4, options, QuantizationFP16, nil, candidates)
		require.Error(t, err,
			"nil context succeeded")
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	{
		_, err := NewScalarQuantizedDiskANNIndex(canceled, 4, options, QuantizationFP16, nil, candidates)
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		_, err := NewScalarQuantizedDiskANNIndex(context.Background(), 4, options, QuantizationFP16, nil,
			[]Candidate{{Key: 1, Vector: []float32{70000, 0, 0, 0}}})
		require.ErrorIs(t, err, ErrQuantizationOverflow)
	}
	{
		_, err := NewScalarQuantizedDiskANNIndex(context.Background(), 3, options, QuantizationInt4, nil,
			[]Candidate{{Key: 1, Vector: []float32{1, 2, 3}}})
		require.ErrorIs(t, err, ErrOddInt4Dimension)
	}

	index, err := NewScalarQuantizedDiskANNIndex(context.Background(), 4, options, QuantizationFP16, nil, candidates)
	require.NoError(t, err)
	{
		_, err := index.Search(context.Background(), []float32{1, 2, 3, 4}, 0)
		require.NoError(t, err)
	}
	{
		_, err := index.SearchWithOptions(context.Background(), []float32{1, 2, 3, 4}, SearchOptions{})
		require.ErrorIs(t, err, ErrInvalidTopK)
	}
	{
		err := index.Close()
		require.NoError(t, err)
	}
	{
		err := index.Close()
		require.NoError(t, err)
	}
	{
		_, err := index.Search(context.Background(), []float32{1, 2, 3, 4}, 1)
		require.ErrorIs(t, err, ErrDiskANNClosed)
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
