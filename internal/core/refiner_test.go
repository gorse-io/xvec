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
	"math"
	"slices"
	"testing"

	"github.com/gorse-io/xvec/internal/ailego"
	"github.com/stretchr/testify/require"
)

func TestOriginalVectorRefiner(t *testing.T) {
	t.Parallel()
	provider, err := NewDenseFlatIndex(2, MetricL2)
	require.NoError(t, err)

	for _, candidate := range []Candidate{
		{Key: 10, Vector: []float32{3, 0}},
		{Key: 20, Vector: []float32{1, 0}},
		{Key: 30, Vector: []float32{2, 0}},
	} {
		{
			err := provider.Add(context.Background(), candidate.Key, candidate.Vector)
			require.NoError(t, err)
		}
	}
	refiner, err := NewOriginalVectorRefiner(provider, MetricL2)
	require.NoError(t, err)

	approximate := []Result{{Key: 10, Score: 0}, {Key: 30, Score: 1}, {Key: 20, Score: 99}, {Key: 20, Score: -1}}
	results, err := refiner.Refine(context.Background(), []float32{0, 0}, approximate, SearchOptions{TopK: 2})
	require.NoError(t, err)

	want := []Result{{Key: 20, Score: 1}, {Key: 30, Score: 4}}
	require.True(t, slices.Equal(results, want))
}

func TestOriginalVectorRefinerFilterAndRadius(t *testing.T) {
	t.Parallel()
	provider, _ := NewDenseFlatIndex(1, MetricIP)
	for key, value := range map[uint64]float32{1: 1, 2: 2, 3: 3} {
		{
			err := provider.Add(context.Background(), key, []float32{value})
			require.NoError(t, err)
		}
	}
	refiner, _ := NewOriginalVectorRefiner(provider, MetricIP)
	results, err := refiner.Refine(context.Background(), []float32{1}, []Result{{Key: 1}, {Key: 2}, {Key: 3}}, SearchOptions{
		TopK: 3, Radius: 2, Filter: func(key uint64) bool { return key != 3 },
	})
	require.NoError(t, err)
	require.True(t, slices.Equal(results, []Result{{Key: 2, Score: 2}}))
}

func TestOriginalVectorRefinerValidation(t *testing.T) {
	t.Parallel()
	{
		_, err := NewOriginalVectorRefiner(nil, MetricL2)
		require.Error(t, err,
			"nil provider accepted")
	}

	provider, _ := NewDenseFlatIndex(2, MetricL2)
	refiner, _ := NewOriginalVectorRefiner(provider, MetricL2)
	{
		_, err := refiner.Refine(nil, []float32{0, 0}, nil, SearchOptions{TopK: 1})
		require.Error(t, err,
			"nil context accepted")
	}
	{
		_, err := refiner.Refine(context.Background(), []float32{0}, nil, SearchOptions{TopK: 1})
		require.ErrorIs(t, err, ErrInvalidDimension)
	}
	{
		_, err := refiner.Refine(context.Background(), []float32{0, float32(math.NaN())}, nil, SearchOptions{TopK: 1})
		require.ErrorIs(t, err, ailego.ErrNonFiniteVector)
	}
	{
		_, err := refiner.Refine(context.Background(), []float32{0, 0}, []Result{{Key: 99}}, SearchOptions{TopK: 1})
		require.ErrorIs(t, err, ErrMissingRefineVector)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	{
		_, err := refiner.Refine(ctx, []float32{0, 0}, nil, SearchOptions{TopK: 1})
		require.ErrorIs(t, err, context.Canceled)
	}
}

func TestRefinementCandidateCount(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		topK  int
		scale float32
		want  int
	}{
		{10, 10, 100},
		{3, 1.9, 5},
		{1, .1, 1},
	} {
		got, err := RefinementCandidateCount(test.topK, test.scale)
		require.NoError(t, err)
		require.Equal(t, test.want, got)
	}
	{
		_, err := RefinementCandidateCount(0, 1)
		require.ErrorIs(t, err, ErrInvalidTopK)
	}

	for _, scale := range []float32{0, -1, float32(math.NaN()), float32(math.Inf(1)), math.MaxFloat32} {
		{
			_, err := RefinementCandidateCount(2, scale)
			require.ErrorIs(t, err, ErrInvalidRefinerScale)
		}
	}
}

func TestRefinedSearch(t *testing.T) {
	t.Parallel()
	provider, _ := NewDenseFlatIndex(1, MetricL2)
	for key, value := range map[uint64]float32{1: 1, 2: 2, 3: 3, 4: 4} {
		{
			err := provider.Add(context.Background(), key, []float32{value})
			require.NoError(t, err)
		}
	}
	base := &recordingDenseSearcher{
		metric:  MetricL2,
		results: []Result{{Key: 4, Score: 0}, {Key: 2, Score: 1}, {Key: 1, Score: 2}, {Key: 3, Score: 3}},
	}
	refiner, _ := NewOriginalVectorRefiner(provider, MetricL2)
	results, err := RefinedSearch(context.Background(), base, refiner, []float32{0}, SearchOptions{TopK: 2, Radius: 5}, 2)
	require.NoError(t, err)
	require.True(t, slices.Equal(results, []Result{{Key: 1, Score: 1}, {Key: 2, Score: 4}}))
	require.True(t, base.options.TopK == 4)
	require.True(t, base.options.Radius == 0)
}

func TestRefinedSearchValidation(t *testing.T) {
	t.Parallel()
	provider, _ := NewDenseFlatIndex(1, MetricL2)
	refiner, _ := NewOriginalVectorRefiner(provider, MetricL2)
	base := &recordingDenseSearcher{metric: MetricIP}
	{
		_, err := RefinedSearch(context.Background(), base, refiner, []float32{0}, SearchOptions{TopK: 1}, 2)
		require.Error(t, err,
			"metric mismatch accepted")
	}
	{
		_, err := RefinedSearch(context.Background(), nil, refiner, []float32{0}, SearchOptions{TopK: 1}, 2)
		require.Error(t, err,
			"nil base accepted")
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	{
		_, err := RefinedSearch(ctx, base, refiner, []float32{0}, SearchOptions{TopK: 1}, 2)
		require.ErrorIs(t, err, context.Canceled)
	}
}

type recordingDenseSearcher struct {
	metric  Metric
	results []Result
	options SearchOptions
}

func (s *recordingDenseSearcher) Metric() Metric { return s.metric }
func (s *recordingDenseSearcher) SearchWithOptions(_ context.Context, _ []float32, options SearchOptions) ([]Result, error) {
	s.options = options
	count := min(options.TopK, len(s.results))
	return slices.Clone(s.results[:count]), nil
}
