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

	"github.com/gorse-io/xvec/internal/ailego/math"
	"github.com/stretchr/testify/require"
)

func TestPQChunkOffsetsAndOptions(t *testing.T) {
	{
		got := pqChunkOffsets(10, 3)
		require.True(t, slices.Equal(got, []int{0, 4, 7, 10}))
	}

	defaults := DefaultPQOptions(MetricL2)
	require.Equal(t, MetricL2, defaults.Metric)
	require.True(t, defaults.Chunks == 0)
	require.True(t, defaults.MaxTrainSamples == 200_000)
	require.True(t, defaults.MaxIterations == 12)

	invalid := []PQOptions{
		DefaultPQOptions(MetricCosine),
		DefaultPQOptions(MetricMIPSL2),
		{Metric: MetricL2, Chunks: -1, MaxTrainSamples: 1, MaxIterations: 1},
		{Metric: MetricL2, MaxTrainSamples: 0, MaxIterations: 1},
		{Metric: MetricL2, MaxTrainSamples: 1, MaxIterations: 0},
		{Metric: MetricL2, MaxTrainSamples: 1, MaxIterations: 1, Workers: -1},
	}
	for _, options := range invalid {
		{
			err := options.Validate()
			require.ErrorIs(t, err, ErrInvalidPQOptions)
		}
	}
	{
		err := invalid[0].Validate()
		require.ErrorIs(t, err, ErrPQUnsupportedMetric)
	}
}

func TestPQKMC2SeedFixture(t *testing.T) {
	vectors := [][]float32{{0}, {1}, {2}, {3}, {4}, {5}}
	centroids, err := initializePQKMC2(context.Background(), vectors, 3, MetricL2, 0)
	require.NoError(t, err)

	want := [][]float32{{1}, {5}, {3}}
	require.Equal(t, want, centroids)
}

func TestPQTrainDeterminismPrefixSamplingAndOwnership(t *testing.T) {
	vectors := pqTrainingVectors(96, 8)
	options := DefaultPQOptions(MetricL2)
	options.MaxTrainSamples = 64
	options.MaxIterations = 4
	options.Seed = 0x123456789abcdef0
	options.Workers = 1
	one, err := TrainPQ(context.Background(), vectors, options)
	require.NoError(t, err)
	require.True(t, one.Dimension() == 8)
	require.True(t, one.Chunks() == 4)
	require.Equal(t, MetricL2, one.Metric())
	require.True(t, slices.Equal(one.ChunkOffsets(), []int{0, 2, 4, 6, 8}))
	require.Len(t, one.Pivots(), PQCentroidCount*8)

	options.Workers = 8
	many, err := TrainPQ(context.Background(), vectors, options)
	require.NoError(t, err)
	require.Equal(t, many.State(), one.State(),
		"PQ training differs across worker counts")

	changedTail := cloneVectors(vectors)
	for index := options.MaxTrainSamples; index < len(changedTail); index++ {
		for component := range changedTail[index] {
			changedTail[index][component] += 1000
		}
	}
	prefix, err := TrainPQ(context.Background(), changedTail, options)
	require.NoError(t, err)
	require.Equal(t, prefix.State(), one.State(),
		"vectors beyond MaxTrainSamples changed PQ model")

	code, err := one.Encode(vectors[0])
	require.NoError(t, err)

	decoded, err := one.Decode(code)
	require.NoError(t, err)

	assertFloatSlicesClose(t, decoded, vectors[0], 1e-6)
	vectors[0][0] = 9999
	state := one.State()
	state.ChunkOffsets[1] = 1
	state.Pivots[0] = 9999
	offsets := one.ChunkOffsets()
	offsets[1] = 1
	pivots := one.Pivots()
	pivots[0] = 9999
	require.True(t, one.ChunkOffsets()[1] == 2,
		"PQ model aliases input, state, or accessor slices")
	require.False(t, one.Pivots()[0] == 9999,
		"PQ model aliases input, state, or accessor slices")
}

func TestPQTrainingBuildsLossyCodebook(t *testing.T) {
	vectors := make([][]float32, 320)
	for index := range vectors {
		vectors[index] = []float32{
			float32(index) / 17,
			float32((index*index+11*index)%997) / 31,
			float32(index) / 23,
			float32((37*index+index*index)%991) / 29,
		}
	}
	options := DefaultPQOptions(MetricL2)
	options.Chunks = 2
	options.MaxTrainSamples = len(vectors)
	options.MaxIterations = 4
	options.Workers = 4
	model, err := TrainPQ(context.Background(), vectors, options)
	require.NoError(t, err)

	codes, err := model.EncodeBatch(context.Background(), vectors, 4)
	require.NoError(t, err)

	distinct := make(map[[2]byte]struct{})
	var distortion float64
	for index, code := range codes {
		encoded := code.Bytes()
		distinct[[2]byte{encoded[0], encoded[1]}] = struct{}{}
		decoded, err := model.Decode(code)
		require.NoError(t, err)

		score, err := MetricL2.Compute(vectors[index], decoded)
		require.NoError(t, err)

		distortion += float64(score)
	}
	require.True(t, len(distinct) >= 200)
	{
		average := distortion / float64(len(vectors))
		require.True(t, average > 0)
		require.True(t, average <= 10)
	}
}

func TestPQL2EncodingDecodeAndDistanceTableFixture(t *testing.T) {
	model := pqFixtureModel(t, MetricL2)
	vector := []float32{2.2, 0, 0, 3.6}
	code, err := model.Encode(vector)
	require.NoError(t, err)
	require.True(t, slices.Equal(code.Bytes(), []byte{2, 4}))
	require.True(t, code.Chunks() == 2)

	bytes := code.Bytes()
	bytes[0] = 99
	require.True(t, code.Bytes()[0] == 2,
		"PQ code accessor aliases storage")

	restoredCode, err := model.Code([]byte{2, 4})
	require.NoError(t, err)

	decoded, err := model.Decode(restoredCode)
	require.NoError(t, err)
	require.True(t, slices.Equal(decoded, []float32{2, 0, 0, 4}))

	query := []float32{1, 0, 0, 5}
	table, err := model.DistanceTable(query)
	require.NoError(t, err)
	require.Equal(t, MetricL2, table.Metric())
	require.True(t, table.Chunks() == 2)
	require.True(t, table.Centroids() == 256)
	require.Len(t, table.Values(), 512)

	values := table.Values()
	require.True(t, values[2] == 1)
	require.True(t, values[PQCentroidCount+4] == 1)

	values[2] = 999
	require.True(t, table.Values()[2] == 1,
		"distance table accessor aliases storage")

	score, err := table.Lookup(code)
	require.NoError(t, err)
	require.True(t, score == 2)

	direct, err := model.Distance(query, code)
	require.NoError(t, err)
	require.Equal(t, score, direct)

	want, err := MetricL2.Compute(query, decoded)
	require.NoError(t, err)
	require.Equal(t, score, want)
}

func TestPQInnerProductTableAndBatch(t *testing.T) {
	model := pqFixtureModel(t, MetricIP)
	vectors := [][]float32{{0, 2, 3, 0}, {2, 0, 0, 4}}
	codes, err := model.EncodeBatch(context.Background(), vectors, 4)
	require.NoError(t, err)
	require.True(t, slices.Equal(codes[0].Bytes(), []byte{1, 0}))
	require.True(t, slices.Equal(codes[1].Bytes(), []byte{0, 1}))

	query := []float32{0, 4, 2, 0}
	table, err := model.DistanceTable(query)
	require.NoError(t, err)

	scores, err := table.LookupBatch(context.Background(), codes, 3)
	require.NoError(t, err)
	require.True(t, slices.Equal(scores, []float32{6, 0}))

	for index, code := range codes {
		decoded, err := model.Decode(code)
		require.NoError(t, err)

		want, err := MetricIP.Compute(query, decoded)
		require.NoError(t, err)
		require.Equal(t, want, scores[index])
	}
}

func TestPQInnerProductTraining(t *testing.T) {
	vectors := pqTrainingVectors(80, 4)
	options := DefaultPQOptions(MetricIP)
	options.Chunks = 2
	options.MaxTrainSamples = 64
	options.MaxIterations = 4
	options.Workers = 4
	model, err := TrainPQ(context.Background(), vectors, options)
	require.NoError(t, err)

	code, err := model.Encode(vectors[71])
	require.NoError(t, err)

	decoded, err := model.Decode(code)
	require.NoError(t, err)

	table, err := model.DistanceTable(vectors[17])
	require.NoError(t, err)

	got, err := table.Lookup(code)
	require.NoError(t, err)

	want, err := MetricIP.Compute(vectors[17], decoded)
	require.NoError(t, err)
	require.InDelta(t, want, got, 1e-5)
}

func TestPQRestoreMismatchAndValidation(t *testing.T) {
	model := pqFixtureModel(t, MetricL2)
	state := model.State()
	restored, err := RestorePQModel(state)
	require.NoError(t, err)

	code, err := model.Encode([]float32{2, 0, 0, 4})
	require.NoError(t, err)
	{
		_, err := restored.Decode(code)
		require.NoError(t, err)
	}

	state.Pivots[0]++
	different, err := RestorePQModel(state)
	require.NoError(t, err)
	{
		_, err := different.Decode(code)
		require.ErrorIs(t, err, ErrPQModelMismatch)
	}

	table, err := different.DistanceTable([]float32{0, 0, 0, 0})
	require.NoError(t, err)
	{
		_, err := table.Lookup(code)
		require.ErrorIs(t, err, ErrPQModelMismatch)
	}
	{
		_, err := model.Code([]byte{1})
		require.ErrorIs(t, err, ErrInvalidPQCode)
	}
	{
		_, err := model.Decode(PQCode{})
		require.ErrorIs(t, err, ErrInvalidPQCode)
	}

	badStates := []PQModelState{
		{},
		{Dimension: 4, Metric: MetricCosine, ChunkOffsets: []int{0, 4}, Pivots: make([]float32, 1024)},
		{Dimension: 4, Metric: MetricL2, ChunkOffsets: []int{1, 4}, Pivots: make([]float32, 1024)},
		{Dimension: 4, Metric: MetricL2, ChunkOffsets: []int{0, 2, 2, 4}, Pivots: make([]float32, 1024)},
		{Dimension: 4, Metric: MetricL2, ChunkOffsets: []int{0, 4}, Pivots: make([]float32, 1023)},
	}
	nonFinite := model.State()
	nonFinite.Pivots[0] = float32(math.NaN())
	badStates = append(badStates, nonFinite)
	for _, bad := range badStates {
		{
			_, err := RestorePQModel(bad)
			require.ErrorIs(t, err, ErrInvalidPQModel)
		}
	}
	wideDimension := 4097
	wide, err := RestorePQModel(PQModelState{
		Dimension: wideDimension, Metric: MetricL2,
		ChunkOffsets: pqChunkOffsets(wideDimension, wideDimension),
		Pivots:       make([]float32, PQCentroidCount*wideDimension),
	})
	require.NoError(t, err)
	require.Equal(t, wideDimension, wide.Chunks())
}

func TestPQContextDimensionAndOverflowErrors(t *testing.T) {
	options := DefaultPQOptions(MetricL2)
	{
		_, err := TrainPQ(nil, [][]float32{{1, 2}}, options)
		require.Error(t, err,
			"nil training context succeeded")
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	{
		_, err := TrainPQ(canceled, [][]float32{{1, 2}}, options)
		require.ErrorIs(t, err, context.Canceled)
	}

	midTraining := newCancelAfterChecks(4)
	{
		_, err := TrainPQ(midTraining, pqTrainingVectors(64, 4), options)
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		_, err := TrainPQ(context.Background(), nil, options)
		require.ErrorIs(t, err, ErrEmptyTrainingSet)
	}
	{
		_, err := TrainPQ(context.Background(), [][]float32{{1}}, options)
		require.ErrorIs(t, err, ErrInvalidPQOptions)
	}

	options.Chunks = 3
	{
		_, err := TrainPQ(context.Background(), [][]float32{{1, 2}}, options)
		require.ErrorIs(t, err, ErrInvalidPQOptions)
	}

	model := pqFixtureModel(t, MetricL2)
	var nilModel *PQModel
	{
		_, err := nilModel.Encode([]float32{1})
		require.ErrorIs(t, err, ErrInvalidPQModel)
	}
	{
		_, err := nilModel.DistanceTable([]float32{1})
		require.ErrorIs(t, err, ErrInvalidPQModel)
	}
	{
		_, err := model.Encode([]float32{1, 2})
		require.ErrorIs(t, err, mathutil.ErrDimensionMismatch)
	}
	{
		_, err := model.DistanceTable([]float32{1, 2})
		require.ErrorIs(t, err, mathutil.ErrDimensionMismatch)
	}
	{
		_, err := model.EncodeBatch(nil, nil, 0)
		require.Error(t, err,
			"nil batch encode context succeeded")
	}
	{
		_, err := model.EncodeBatch(canceled, nil, 0)
		require.ErrorIs(t, err, context.Canceled)
	}

	table, err := model.DistanceTable([]float32{0, 0, 0, 0})
	require.NoError(t, err)
	{
		_, err := table.LookupBatch(nil, nil, 0)
		require.Error(t, err,
			"nil batch lookup context succeeded")
	}
	{
		_, err := table.LookupBatch(canceled, nil, 0)
		require.ErrorIs(t, err, context.Canceled)
	}

	var nilTable *PQDistanceTable
	{
		_, err := nilTable.Lookup(PQCode{})
		require.ErrorIs(t, err, ErrInvalidPQModel)
	}

	overflow := &PQDistanceTable{
		modelFingerprint: 1, metric: MetricL2, chunks: 3,
		values: make([]float32, 3*PQCentroidCount),
	}
	for chunk := 0; chunk < 3; chunk++ {
		overflow.values[chunk*PQCentroidCount] = math.MaxFloat32 / 2
	}
	{
		_, err := overflow.Lookup(PQCode{modelFingerprint: 1, codes: []byte{0, 0, 0}})
		require.ErrorIs(t, err, ErrPQScoreOverflow)
	}
}

func FuzzRestorePQModel(f *testing.F) {
	f.Add([]byte{4, 2, byte(MetricL2), 0, 0, 0, 0})
	f.Add([]byte{8, 3, byte(MetricIP), 0, 0, 128, 63})
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) < 7 {
			return
		}
		dimension := int(data[0]%8) + 1
		chunks := int(data[1]%byte(dimension)) + 1
		offsets := pqChunkOffsets(dimension, chunks)
		if data[3]&1 != 0 && len(offsets) > 2 {
			offsets[1] = int(data[4]) % (dimension + 1)
		}
		value := math.Float32frombits(uint32(data[3]) | uint32(data[4])<<8 | uint32(data[5])<<16 | uint32(data[6])<<24)
		pivots := make([]float32, PQCentroidCount*dimension)
		for index := range pivots {
			pivots[index] = value
		}
		model, err := RestorePQModel(PQModelState{
			Dimension: dimension, Metric: Metric(data[2]), ChunkOffsets: offsets, Pivots: pivots,
		})
		if err != nil {
			return
		}
		code, err := model.Encode(make([]float32, dimension))
		if err != nil {
			return
		}
		table, err := model.DistanceTable(make([]float32, dimension))
		if err != nil {
			return
		}
		_, _ = table.Lookup(code)
	})
}

func BenchmarkPQDistanceLookup(b *testing.B) {
	model := pqFixtureModel(b, MetricL2)
	code, err := model.Code([]byte{2, 4})
	if err != nil {
		require.NoError(b, err)
	}

	table, err := model.DistanceTable([]float32{1, 0, 0, 5})
	if err != nil {
		require.NoError(b, err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		{
			_, err := table.Lookup(code)
			if err != nil {
				require.NoError(b, err)
			}
		}
	}
}

func BenchmarkPQTrain(b *testing.B) {
	vectors := pqTrainingVectors(2048, 64)
	options := DefaultPQOptions(MetricL2)
	options.Chunks = 32
	options.MaxTrainSamples = len(vectors)
	options.MaxIterations = 4
	options.Workers = 4
	b.ReportAllocs()
	for b.Loop() {
		if _, err := TrainPQ(context.Background(), vectors, options); err != nil {
			require.NoError(b, err)
		}
	}
}

func pqFixtureModel(t testing.TB, metric Metric) *PQModel {
	t.Helper()
	pivots := make([]float32, PQCentroidCount*4)
	for centroid := 0; centroid < PQCentroidCount; centroid++ {
		if metric == MetricL2 {
			pivots[centroid*4] = float32(centroid)
			pivots[centroid*4+3] = float32(centroid)
		} else {
			pivots[centroid*4] = 1
			pivots[centroid*4+2] = 1
		}
	}
	if metric == MetricIP {
		pivots[4], pivots[5], pivots[6], pivots[7] = 0, 1, 0, 1
	}
	model, err := RestorePQModel(PQModelState{
		Dimension: 4, Metric: metric, ChunkOffsets: []int{0, 2, 4}, Pivots: pivots,
	})
	require.NoError(t, err)

	return model
}

func pqTrainingVectors(count, dimension int) [][]float32 {
	vectors := make([][]float32, count)
	for index := range vectors {
		vectors[index] = make([]float32, dimension)
		for component := range vectors[index] {
			vectors[index][component] = float32((index*17+component*13)%101)/7 - 5
		}
	}
	return vectors
}
