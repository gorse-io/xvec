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

func TestPQChunkOffsetsAndOptions(t *testing.T) {
	if got := pqChunkOffsets(10, 3); !slices.Equal(got, []int{0, 4, 7, 10}) {
		t.Fatalf("chunk offsets = %v", got)
	}
	defaults := DefaultPQOptions(MetricL2)
	if defaults.Metric != MetricL2 || defaults.Chunks != 0 || defaults.MaxTrainSamples != 200_000 || defaults.MaxIterations != 12 {
		t.Fatalf("defaults = %#v", defaults)
	}
	invalid := []PQOptions{
		DefaultPQOptions(MetricCosine),
		DefaultPQOptions(MetricMIPSL2),
		{Metric: MetricL2, Chunks: -1, MaxTrainSamples: 1, MaxIterations: 1},
		{Metric: MetricL2, MaxTrainSamples: 0, MaxIterations: 1},
		{Metric: MetricL2, MaxTrainSamples: 1, MaxIterations: 0},
		{Metric: MetricL2, MaxTrainSamples: 1, MaxIterations: 1, Workers: -1},
	}
	for _, options := range invalid {
		if err := options.Validate(); !errors.Is(err, ErrInvalidPQOptions) {
			t.Fatalf("options %#v error = %v", options, err)
		}
	}
	if err := invalid[0].Validate(); !errors.Is(err, ErrPQUnsupportedMetric) {
		t.Fatalf("unsupported metric error = %v", err)
	}
}

func TestPQKMC2SeedFixture(t *testing.T) {
	vectors := [][]float32{{0}, {1}, {2}, {3}, {4}, {5}}
	centroids, err := initializePQKMC2(context.Background(), vectors, 3, MetricL2, 0)
	if err != nil {
		t.Fatal(err)
	}
	want := [][]float32{{1}, {5}, {3}}
	if !reflect.DeepEqual(centroids, want) {
		t.Fatalf("KMC2 centroids = %v", centroids)
	}
}

func TestPQTrainDeterminismPrefixSamplingAndOwnership(t *testing.T) {
	vectors := pqTrainingVectors(96, 8)
	options := DefaultPQOptions(MetricL2)
	options.MaxTrainSamples = 64
	options.MaxIterations = 4
	options.Seed = 0x123456789abcdef0
	options.Workers = 1
	one, err := TrainPQ(context.Background(), vectors, options)
	if err != nil {
		t.Fatal(err)
	}
	if one.Dimension() != 8 || one.Chunks() != 4 || one.Metric() != MetricL2 ||
		!slices.Equal(one.ChunkOffsets(), []int{0, 2, 4, 6, 8}) || len(one.Pivots()) != PQCentroidCount*8 {
		t.Fatalf("model metadata = dim %d chunks %d metric %d offsets %v pivots %d", one.Dimension(), one.Chunks(), one.Metric(), one.ChunkOffsets(), len(one.Pivots()))
	}
	options.Workers = 8
	many, err := TrainPQ(context.Background(), vectors, options)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(one.State(), many.State()) {
		t.Fatal("PQ training differs across worker counts")
	}

	changedTail := cloneVectors(vectors)
	for index := options.MaxTrainSamples; index < len(changedTail); index++ {
		for component := range changedTail[index] {
			changedTail[index][component] += 1000
		}
	}
	prefix, err := TrainPQ(context.Background(), changedTail, options)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(one.State(), prefix.State()) {
		t.Fatal("vectors beyond MaxTrainSamples changed PQ model")
	}

	code, err := one.Encode(vectors[0])
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := one.Decode(code)
	if err != nil {
		t.Fatal(err)
	}
	assertFloatSlicesClose(t, decoded, vectors[0], 1e-6)
	vectors[0][0] = 9999
	state := one.State()
	state.ChunkOffsets[1] = 1
	state.Pivots[0] = 9999
	offsets := one.ChunkOffsets()
	offsets[1] = 1
	pivots := one.Pivots()
	pivots[0] = 9999
	if one.ChunkOffsets()[1] != 2 || one.Pivots()[0] == 9999 {
		t.Fatal("PQ model aliases input, state, or accessor slices")
	}
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
	if err != nil {
		t.Fatal(err)
	}
	codes, err := model.EncodeBatch(context.Background(), vectors, 4)
	if err != nil {
		t.Fatal(err)
	}
	distinct := make(map[[2]byte]struct{})
	var distortion float64
	for index, code := range codes {
		encoded := code.Bytes()
		distinct[[2]byte{encoded[0], encoded[1]}] = struct{}{}
		decoded, err := model.Decode(code)
		if err != nil {
			t.Fatal(err)
		}
		score, err := MetricL2.Compute(vectors[index], decoded)
		if err != nil {
			t.Fatal(err)
		}
		distortion += float64(score)
	}
	if len(distinct) < 200 {
		t.Fatalf("only %d distinct product codes", len(distinct))
	}
	if average := distortion / float64(len(vectors)); average <= 0 || average > 10 {
		t.Fatalf("average reconstruction distortion = %g", average)
	}
}

func TestPQL2EncodingDecodeAndDistanceTableFixture(t *testing.T) {
	model := pqFixtureModel(t, MetricL2)
	vector := []float32{2.2, 0, 0, 3.6}
	code, err := model.Encode(vector)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(code.Bytes(), []byte{2, 4}) || code.Chunks() != 2 {
		t.Fatalf("code = %v", code.Bytes())
	}
	bytes := code.Bytes()
	bytes[0] = 99
	if code.Bytes()[0] != 2 {
		t.Fatal("PQ code accessor aliases storage")
	}
	restoredCode, err := model.Code([]byte{2, 4})
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := model.Decode(restoredCode)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(decoded, []float32{2, 0, 0, 4}) {
		t.Fatalf("decoded = %v", decoded)
	}
	query := []float32{1, 0, 0, 5}
	table, err := model.DistanceTable(query)
	if err != nil {
		t.Fatal(err)
	}
	if table.Metric() != MetricL2 || table.Chunks() != 2 || table.Centroids() != 256 || len(table.Values()) != 512 {
		t.Fatalf("table metadata = metric %d chunks %d centroids %d values %d", table.Metric(), table.Chunks(), table.Centroids(), len(table.Values()))
	}
	values := table.Values()
	if values[2] != 1 || values[PQCentroidCount+4] != 1 {
		t.Fatalf("selected table entries = %g, %g", values[2], values[PQCentroidCount+4])
	}
	values[2] = 999
	if table.Values()[2] != 1 {
		t.Fatal("distance table accessor aliases storage")
	}
	score, err := table.Lookup(code)
	if err != nil || score != 2 {
		t.Fatalf("lookup = %g, %v", score, err)
	}
	direct, err := model.Distance(query, code)
	if err != nil || direct != score {
		t.Fatalf("direct distance = %g, %v", direct, err)
	}
	want, err := MetricL2.Compute(query, decoded)
	if err != nil || want != score {
		t.Fatalf("decoded distance = %g, %v", want, err)
	}
}

func TestPQInnerProductTableAndBatch(t *testing.T) {
	model := pqFixtureModel(t, MetricIP)
	vectors := [][]float32{{0, 2, 3, 0}, {2, 0, 0, 4}}
	codes, err := model.EncodeBatch(context.Background(), vectors, 4)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(codes[0].Bytes(), []byte{1, 0}) || !slices.Equal(codes[1].Bytes(), []byte{0, 1}) {
		t.Fatalf("IP codes = %v, %v", codes[0].Bytes(), codes[1].Bytes())
	}
	query := []float32{0, 4, 2, 0}
	table, err := model.DistanceTable(query)
	if err != nil {
		t.Fatal(err)
	}
	scores, err := table.LookupBatch(context.Background(), codes, 3)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(scores, []float32{6, 0}) {
		t.Fatalf("IP scores = %v", scores)
	}
	for index, code := range codes {
		decoded, err := model.Decode(code)
		if err != nil {
			t.Fatal(err)
		}
		want, err := MetricIP.Compute(query, decoded)
		if err != nil || scores[index] != want {
			t.Fatalf("score %d = %g, want %g, err %v", index, scores[index], want, err)
		}
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
	if err != nil {
		t.Fatal(err)
	}
	code, err := model.Encode(vectors[71])
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := model.Decode(code)
	if err != nil {
		t.Fatal(err)
	}
	table, err := model.DistanceTable(vectors[17])
	if err != nil {
		t.Fatal(err)
	}
	got, err := table.Lookup(code)
	if err != nil {
		t.Fatal(err)
	}
	want, err := MetricIP.Compute(vectors[17], decoded)
	if err != nil || math.Abs(float64(got-want)) > 1e-5 {
		t.Fatalf("trained IP score = %g, want %g, err %v", got, want, err)
	}
}

func TestPQRestoreMismatchAndValidation(t *testing.T) {
	model := pqFixtureModel(t, MetricL2)
	state := model.State()
	restored, err := RestorePQModel(state)
	if err != nil {
		t.Fatal(err)
	}
	code, err := model.Encode([]float32{2, 0, 0, 4})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := restored.Decode(code); err != nil {
		t.Fatalf("restored model rejected matching code: %v", err)
	}
	state.Pivots[0]++
	different, err := RestorePQModel(state)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := different.Decode(code); !errors.Is(err, ErrPQModelMismatch) {
		t.Fatalf("model mismatch error = %v", err)
	}
	table, err := different.DistanceTable([]float32{0, 0, 0, 0})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := table.Lookup(code); !errors.Is(err, ErrPQModelMismatch) {
		t.Fatalf("table mismatch error = %v", err)
	}
	if _, err := model.Code([]byte{1}); !errors.Is(err, ErrInvalidPQCode) {
		t.Fatalf("short code error = %v", err)
	}
	if _, err := model.Decode(PQCode{}); !errors.Is(err, ErrInvalidPQCode) {
		t.Fatalf("zero code error = %v", err)
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
		if _, err := RestorePQModel(bad); !errors.Is(err, ErrInvalidPQModel) {
			t.Fatalf("bad state error = %v", err)
		}
	}
	wideDimension := 4097
	wide, err := RestorePQModel(PQModelState{
		Dimension: wideDimension, Metric: MetricL2,
		ChunkOffsets: pqChunkOffsets(wideDimension, wideDimension),
		Pivots:       make([]float32, PQCentroidCount*wideDimension),
	})
	if err != nil || wide.Chunks() != wideDimension {
		t.Fatalf("wide model restore = %#v, %v", wide, err)
	}
}

func TestPQContextDimensionAndOverflowErrors(t *testing.T) {
	options := DefaultPQOptions(MetricL2)
	if _, err := TrainPQ(nil, [][]float32{{1, 2}}, options); err == nil {
		t.Fatal("nil training context succeeded")
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := TrainPQ(canceled, [][]float32{{1, 2}}, options); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled training error = %v", err)
	}
	midTraining := newCancelAfterChecks(4)
	if _, err := TrainPQ(midTraining, pqTrainingVectors(64, 4), options); !errors.Is(err, context.Canceled) {
		t.Fatalf("mid-training cancellation error = %v", err)
	}
	if _, err := TrainPQ(context.Background(), nil, options); !errors.Is(err, ErrEmptyTrainingSet) {
		t.Fatalf("empty training error = %v", err)
	}
	if _, err := TrainPQ(context.Background(), [][]float32{{1}}, options); !errors.Is(err, ErrInvalidPQOptions) {
		t.Fatalf("one-dimensional auto chunks error = %v", err)
	}
	options.Chunks = 3
	if _, err := TrainPQ(context.Background(), [][]float32{{1, 2}}, options); !errors.Is(err, ErrInvalidPQOptions) {
		t.Fatalf("too many chunks error = %v", err)
	}
	model := pqFixtureModel(t, MetricL2)
	var nilModel *PQModel
	if _, err := nilModel.Encode([]float32{1}); !errors.Is(err, ErrInvalidPQModel) {
		t.Fatalf("nil model encode error = %v", err)
	}
	if _, err := nilModel.DistanceTable([]float32{1}); !errors.Is(err, ErrInvalidPQModel) {
		t.Fatalf("nil model table error = %v", err)
	}
	if _, err := model.Encode([]float32{1, 2}); !errors.Is(err, ailego.ErrDimensionMismatch) {
		t.Fatalf("encode dimension error = %v", err)
	}
	if _, err := model.DistanceTable([]float32{1, 2}); !errors.Is(err, ailego.ErrDimensionMismatch) {
		t.Fatalf("table dimension error = %v", err)
	}
	if _, err := model.EncodeBatch(nil, nil, 0); err == nil {
		t.Fatal("nil batch encode context succeeded")
	}
	if _, err := model.EncodeBatch(canceled, nil, 0); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled batch encode error = %v", err)
	}
	table, err := model.DistanceTable([]float32{0, 0, 0, 0})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := table.LookupBatch(nil, nil, 0); err == nil {
		t.Fatal("nil batch lookup context succeeded")
	}
	if _, err := table.LookupBatch(canceled, nil, 0); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled batch lookup error = %v", err)
	}
	var nilTable *PQDistanceTable
	if _, err := nilTable.Lookup(PQCode{}); !errors.Is(err, ErrInvalidPQModel) {
		t.Fatalf("nil table lookup error = %v", err)
	}
	overflow := &PQDistanceTable{
		modelFingerprint: 1, metric: MetricL2, chunks: 3,
		values: make([]float32, 3*PQCentroidCount),
	}
	for chunk := 0; chunk < 3; chunk++ {
		overflow.values[chunk*PQCentroidCount] = math.MaxFloat32 / 2
	}
	if _, err := overflow.Lookup(PQCode{modelFingerprint: 1, codes: []byte{0, 0, 0}}); !errors.Is(err, ErrPQScoreOverflow) {
		t.Fatalf("overflow lookup error = %v", err)
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
		b.Fatal(err)
	}
	table, err := model.DistanceTable([]float32{1, 0, 0, 5})
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, err := table.Lookup(code); err != nil {
			b.Fatal(err)
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
	if err != nil {
		t.Fatal(err)
	}
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
