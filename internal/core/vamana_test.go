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
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"

	"github.com/gorse-io/zvec/internal/ailego"
	"github.com/stretchr/testify/require"
)

func TestVamanaBuildOptionsGraphDeterminismAndOwnership(t *testing.T) {
	defaults := DefaultVamanaBuildOptions(MetricCosine)
	require.True(t, defaults.MaxDegree == 64)
	require.True(t, defaults.SearchListSize == 100)
	require.True(t, defaults.Alpha == 1.2)
	require.True(t, defaults.MaxOcclusionSize == 750)
	require.Equal(t, MetricCosine, defaults.Metric)

	valid := DefaultVamanaBuildOptions(MetricL2)
	invalid := []VamanaBuildOptions{
		{},
		func() VamanaBuildOptions { value := valid; value.MaxDegree = 0; return value }(),
		func() VamanaBuildOptions { value := valid; value.MaxDegree = MaxVamanaDegree + 1; return value }(),
		func() VamanaBuildOptions { value := valid; value.SearchListSize = value.MaxDegree - 1; return value }(),
		func() VamanaBuildOptions { value := valid; value.Alpha = float32(math.NaN()); return value }(),
		func() VamanaBuildOptions { value := valid; value.MaxOcclusionSize = 0; return value }(),
	}
	for _, options := range invalid {
		{
			_, err := NewVamanaBuilder(3, options)
			require.ErrorIs(t, err, ErrInvalidVamanaOptions)
		}
	}
	{
		_, err := NewVamanaBuilder(0, valid)
		require.ErrorIs(t, err, ErrInvalidDimension)
	}

	inputs := hnswBuildInputs(160)
	build := func(saturate bool) *VamanaIndex {
		options := DefaultVamanaBuildOptions(MetricL2)
		options.MaxDegree = 8
		options.SearchListSize = 40
		options.MaxOcclusionSize = 60
		options.SaturateGraph = saturate
		return buildVamana(t, inputs, options)
	}
	first, second := build(false), build(false)
	require.Equal(t, second.neighbors, first.neighbors,

		"Vamana build is not deterministic")
	require.Equal(t, second.entryPoint, first.entryPoint,

		"Vamana build is not deterministic")
	require.Equal(t, second.neighborDistances, first.neighborDistances,
		"Vamana build is not deterministic")
	require.True(t, first.Dimension() == 3,
		"Vamana metadata differs")
	require.Equal(t, MetricL2, first.Metric(),
		"Vamana metadata differs")
	require.Len(t, inputs, first.Len(),
		"Vamana metadata differs")
	require.True(t, first.BuildOptions().MaxDegree == 8,
		"Vamana metadata differs")

	entry, found := first.EntryPoint()
	require.True(t, found,
		"Vamana entry point missing")
	require.Equal(t, first.keys[first.entryPoint], entry,
		"Vamana entry point missing")

	assertVamanaGraphInvariants(t, first)
	for _, adjacent := range build(true).neighbors {
		require.False(t, len(adjacent) == 0,
			"saturated graph retained an empty neighbor list")
	}
	original, _ := first.Vector(inputs[0].Key)
	inputs[0].Vector[0]++
	again, _ := first.Vector(inputs[0].Key)
	require.Equal(t, again[0], original[0],
		"builder did not own input vector")

	original[0]++
	again, _ = first.Vector(inputs[0].Key)
	require.NotEqual(t, again[0], original[0],
		"Vector exposed mutable storage")
}

func TestVamanaSearchMetricsFilterRadiusAndRecall(t *testing.T) {
	for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine, MetricMIPSL2} {
		inputs := hnswBuildInputs(180)
		options := DefaultVamanaBuildOptions(metric)
		options.MaxDegree = 8
		options.SearchListSize = 40
		options.MaxOcclusionSize = 60
		index := buildVamana(t, inputs, options)
		query := slices.Clone(inputs[77].Vector)
		search := VamanaSearchOptions{
			SearchOptions: SearchOptions{TopK: 12, Filter: func(key uint64) bool { return key%3 != 0 }},
			EFSearch:      80,
		}
		got, err := index.SearchVamana(context.Background(), query, search)
		require.NoError(t, err)

		want, err := topKCandidatesWithOptions(context.Background(), metric, query, search.SearchOptions, len(inputs), func(position int) Candidate {
			return inputs[position]
		}, true)
		require.NoError(t, err)
		require.Equal(t, want, got)
	}
	inputs := hnswBuildInputs(80)
	options := DefaultVamanaBuildOptions(MetricL2)
	options.MaxDegree, options.SearchListSize = 8, 32
	index := buildVamana(t, inputs, options)
	target := inputs[17]
	bounded, err := index.SearchVamana(context.Background(), target.Vector, VamanaSearchOptions{
		SearchOptions: SearchOptions{TopK: 5, Radius: .01, Filter: func(key uint64) bool { return key == target.Key }},
		EFSearch:      40,
	})
	require.NoError(t, err)
	require.Equal(t, []Result{{Key: target.Key, Score: 0}}, bounded)

	inputs = hnswRaBitQCandidates(DefaultVamanaBruteForceThreshold+120, 64)
	for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine, MetricMIPSL2} {
		options := DefaultVamanaBuildOptions(metric)
		options.MaxDegree = 16
		options.SearchListSize = 80
		options.MaxOcclusionSize = 160
		index := buildVamana(t, inputs, options)
		var matches, total int
		for queryIndex := 0; queryIndex < 10; queryIndex++ {
			query := inputs[(queryIndex*79+17)%len(inputs)].Vector
			truth, err := TopK(context.Background(), metric, query, inputs, 10)
			require.NoError(t, err)

			got, err := index.SearchVamana(context.Background(), query, VamanaSearchOptions{SearchOptions: SearchOptions{TopK: 10}, EFSearch: 100})
			require.NoError(t, err)

			if metric == MetricL2 && queryIndex == 0 {
				prefetched, err := index.SearchVamana(context.Background(), query, VamanaSearchOptions{
					SearchOptions: SearchOptions{TopK: 10}, EFSearch: 100,
					PrefetchOffset: math.MaxUint32, PrefetchLines: math.MaxUint32,
				})
				require.NoError(t, err)
				require.Equal(t, got, prefetched)
			}
			matches += resultOverlap(got, truth)
			total += len(truth)
		}
		{
			recall := float64(matches) / float64(total)
			require.True(t, recall >= .80)
		}
	}
}

func TestVamanaRobustPrunePinnedGeometry(t *testing.T) {
	options := DefaultVamanaBuildOptions(MetricL2)
	options.MaxDegree, options.SearchListSize, options.MaxOcclusionSize = 3, 4, 4
	index := &VamanaIndex{
		dimension: 2, options: options,
		keys: []uint64{1, 2, 3, 4}, vectors: []float32{0, 0, 1, 0, 1.1, 0, 0, 2},
	}
	candidates := []vamanaDistanceNode{{position: 1, distance: 1}, {position: 2, distance: 1.21}, {position: 3, distance: 4}}
	selected, err := index.robustPrune(context.Background(), 0, candidates)
	require.NoError(t, err)
	require.Equal(t, []vamanaDistanceNode{{position: 1, distance: 1}, {position: 3, distance: 4}}, selected)

	index.options.SaturateGraph = true
	selected, err = index.robustPrune(context.Background(), 0, candidates)
	require.NoError(t, err)
	require.Equal(t, []vamanaDistanceNode{{position: 1, distance: 1}, {position: 3, distance: 4}, {position: 2, distance: 1.21}}, selected)
}

func TestVamanaEmptyIncrementalAndValidation(t *testing.T) {
	options := DefaultVamanaBuildOptions(MetricL2)
	options.MaxDegree = 4
	options.SearchListSize = 12
	builder, err := NewVamanaBuilder(3, options)
	require.NoError(t, err)

	index, err := builder.Build(context.Background())
	require.NoError(t, err)
	require.True(t, index.Len() == 0,
		"empty Vamana index is not empty")
	{
		_, found := index.EntryPoint()
		require.False(t, found,
			"empty Vamana index has entry point")
	}

	validSearch := VamanaSearchOptions{SearchOptions: SearchOptions{TopK: 1}, EFSearch: 10}
	{
		got, err := index.SearchVamana(context.Background(), []float32{0, 0, 0}, validSearch)
		require.NoError(t, err)
		require.Len(t, got, 0)
	}

	vector := []float32{1, 2, 3}
	{
		err := index.Add(context.Background(), 7, vector)
		require.NoError(t, err)
	}
	require.True(t, index.Len() == 1)
	{
		err := index.Add(context.Background(), 7, vector)
		require.ErrorIs(t, err, ErrDuplicateKey)
	}
	{
		err := index.Add(context.Background(), 8, vector[:2])
		require.ErrorIs(t, err, ailego.ErrDimensionMismatch)
	}
	{
		_, err := index.SearchVamana(nil, vector, validSearch)
		require.Error(t, err,
			"nil search context succeeded")
	}

	validSearch.EFSearch = 0
	{
		_, err := index.SearchVamana(context.Background(), vector, validSearch)
		require.ErrorIs(t, err, ErrInvalidVamanaEF)
	}
}

func buildVamana(t testing.TB, inputs []Candidate, options VamanaBuildOptions) *VamanaIndex {
	t.Helper()
	dimension := 3
	if len(inputs) != 0 {
		dimension = len(inputs[0].Vector)
	}
	builder, err := NewVamanaBuilder(dimension, options)
	require.NoError(t, err)

	for _, input := range inputs {
		{
			err := builder.Add(context.Background(), input.Key, input.Vector)
			require.NoError(t, err)
		}
	}
	index, err := builder.Build(context.Background())
	require.NoError(t, err)

	return index
}

func assertVamanaGraphInvariants(t testing.TB, index *VamanaIndex) {
	t.Helper()
	{
		err := validateVamanaIndex(context.Background(), index)
		require.NoError(t, err)
	}

	for position, adjacent := range index.neighbors {
		seen := make(map[int]struct{}, len(adjacent))
		for _, neighbor := range adjacent {
			require.NotEqual(t, position, neighbor)
			{
				_, found := seen[neighbor]
				require.False(t, found)
			}

			seen[neighbor] = struct{}{}
		}
	}
}

func TestVamanaIncrementalFailuresAreAtomic(t *testing.T) {
	options := DefaultVamanaBuildOptions(MetricL2)
	options.MaxDegree, options.SearchListSize = 8, 32
	index := buildVamana(t, hnswBuildInputs(100), options)
	before, err := encodeVamanaIndex(context.Background(), index)
	require.NoError(t, err)

	vector := []float32{1, 2, 3}
	{
		err := index.Add(nil, 999999, vector)
		require.Error(t, err,
			"nil Add context succeeded")
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	{
		err := index.Add(canceled, 999999, vector)
		require.Equal(t, context.Canceled, err)
	}

	midGeneration := newCancelAfterChecks(4)
	{
		err := index.Add(midGeneration, 999999, vector)
		require.Equal(t, context.Canceled, err)
	}

	after, err := encodeVamanaIndex(context.Background(), index)
	require.NoError(t, err)
	require.True(t, slices.Equal(after, before),
		"failed Add changed Vamana generation")
	{
		err := index.Add(context.Background(), 999999, vector)
		require.NoError(t, err)
	}
	require.True(t, index.Len() == 101)
}

func TestVamanaConcurrentAddSearchSaveAndOpen(t *testing.T) {
	options := DefaultVamanaBuildOptions(MetricCosine)
	options.MaxDegree, options.SearchListSize = 8, 32
	index := buildVamana(t, nil, options)
	dir := t.TempDir()
	errCh := make(chan error, 32)
	var writers sync.WaitGroup
	for worker := 0; worker < 3; worker++ {
		writers.Add(1)
		go func(worker int) {
			defer writers.Done()
			for value := 0; value < 12; value++ {
				vector := []float32{float32(worker + 1), float32(value + 1), float32(worker + value + 1)}
				key := uint64(worker*100 + value + 1)
				if err := index.Add(context.Background(), key, vector); err != nil {
					errCh <- err
					return
				}
			}
		}(worker)
	}
	stop := make(chan struct{})
	var readers sync.WaitGroup
	for worker := 0; worker < 3; worker++ {
		readers.Add(1)
		go func() {
			defer readers.Done()
			for {
				select {
				case <-stop:
					return
				default:
				}
				if _, err := index.Search(context.Background(), []float32{1, 2, 3}, 5); err != nil {
					errCh <- err
					return
				}
			}
		}()
	}
	readers.Add(1)
	go func() {
		defer readers.Done()
		for generation := 0; generation < 8; generation++ {
			path := filepath.Join(dir, fmt.Sprintf("snapshot-%02d.vamana", generation))
			if err := index.Save(context.Background(), path); err != nil {
				errCh <- err
				return
			}
			if _, err := OpenVamanaIndex(context.Background(), path); err != nil {
				errCh <- err
				return
			}
		}
	}()
	writers.Wait()
	close(stop)
	readers.Wait()
	close(errCh)
	for err := range errCh {
		require.NoError(t, err)
	}
	require.True(t, index.Len() == 36)
}

func TestScalarQuantizedVamanaSearch(t *testing.T) {
	inputs := hnswBuildInputs(180)
	options := DefaultVamanaBuildOptions(MetricL2)
	options.MaxDegree, options.SearchListSize = 8, 32
	base := buildVamana(t, inputs, options)
	index, err := NewScalarQuantizedVamanaIndex(context.Background(), base, QuantizationInt8, nil)
	require.NoError(t, err)

	flat, err := NewScalarQuantizedFlatIndex(context.Background(), 3, MetricL2, QuantizationInt8, nil, inputs)
	require.NoError(t, err)

	query := []float32{7.25, 11.5, 1.1}
	search := VamanaSearchOptions{SearchOptions: SearchOptions{TopK: 15, Filter: func(key uint64) bool { return key%2 == 1 }}, EFSearch: 80}
	got, err := index.SearchVamana(context.Background(), query, search)
	require.NoError(t, err)

	want, err := flat.SearchWithOptions(context.Background(), query, search.SearchOptions)
	require.NoError(t, err)
	require.Equal(t, want, got)
	{
		vector, found := index.Vector(inputs[0].Key)
		require.True(t, found,
			"quantized Vamana lost original vector")
		require.Equal(t, inputs[0].Vector, vector,
			"quantized Vamana lost original vector")
	}
	{
		_, err := NewScalarQuantizedVamanaIndex(nil, base, QuantizationInt8, nil)
		require.Error(t, err,
			"nil quantization context succeeded")
	}
	{
		_, err := NewScalarQuantizedVamanaIndex(context.Background(), nil, QuantizationInt8, nil)
		require.Error(t, err,
			"nil source index succeeded")
	}
}

func BenchmarkVamanaSearch(b *testing.B) {
	inputs := hnswRaBitQCandidates(2_000, 64)
	options := DefaultVamanaBuildOptions(MetricL2)
	options.MaxDegree, options.SearchListSize, options.MaxOcclusionSize = 16, 80, 160
	index := buildVamana(b, inputs, options)
	query := inputs[713].Vector
	search := VamanaSearchOptions{SearchOptions: SearchOptions{TopK: 10}, EFSearch: 100}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		{
			_, err := index.SearchVamana(context.Background(), query, search)
			if err != nil {
				require.NoError(b, err)
			}
		}
	}
}

func TestVamanaPersistenceRoundTripReplaceAndIncrement(t *testing.T) {
	options := DefaultVamanaBuildOptions(MetricCosine)
	options.MaxDegree, options.SearchListSize, options.MaxOcclusionSize = 8, 40, 80
	index := buildVamana(t, hnswRaBitQCandidates(180, 70), options)
	path := filepath.Join(t.TempDir(), "vectors.vamana")
	{
		err := index.Save(context.Background(), path)
		require.NoError(t, err)
	}

	info, err := os.Stat(path)
	require.NoError(t, err)
	assertPrivateFileMode(t, info.Mode())
	opened, err := OpenVamanaIndex(context.Background(), path)
	require.NoError(t, err)

	assertSameVamanaIndex(t, opened, index)
	query := hnswRaBitQCandidates(1, 70)[0].Vector
	search := VamanaSearchOptions{SearchOptions: SearchOptions{TopK: 20}, EFSearch: 80}
	want, err := index.SearchVamana(context.Background(), query, search)
	require.NoError(t, err)

	got, err := opened.SearchVamana(context.Background(), query, search)
	require.NoError(t, err)
	require.Equal(t, want, got)

	next := hnswRaBitQCandidates(1, 70)[0]
	next.Key = 999999
	{
		err := opened.Add(context.Background(), next.Key, next.Vector)
		require.NoError(t, err)
	}
	{
		err := opened.Save(context.Background(), path)
		require.NoError(t, err)
	}

	reopened, err := OpenVamanaIndex(context.Background(), path)
	require.NoError(t, err)

	assertSameVamanaIndex(t, reopened, opened)

	replacementOptions := DefaultVamanaBuildOptions(MetricIP)
	replacementOptions.MaxDegree, replacementOptions.SearchListSize = 4, 16
	replacement := buildVamana(t, hnswBuildInputs(40), replacementOptions)
	{
		err := replacement.Save(context.Background(), path)
		require.NoError(t, err)
	}

	reopened, err = OpenVamanaIndex(context.Background(), path)
	require.NoError(t, err)

	assertSameVamanaIndex(t, reopened, replacement)
}

func TestVamanaPersistenceEmptyCancellationAndErrors(t *testing.T) {
	index := buildVamana(t, nil, DefaultVamanaBuildOptions(MetricL2))
	largeOptions := DefaultVamanaBuildOptions(MetricL2)
	largeOptions.MaxDegree, largeOptions.SearchListSize = 8, 40
	largeIndex := buildVamana(t, hnswRaBitQCandidates(300, 70), largeOptions)
	dir := t.TempDir()
	path := filepath.Join(dir, "empty.vamana")
	{
		err := index.Save(context.Background(), path)
		require.NoError(t, err)
	}

	original, err := os.ReadFile(path)
	require.NoError(t, err)

	opened, err := OpenVamanaIndex(context.Background(), path)
	require.NoError(t, err)

	assertSameVamanaIndex(t, opened, index)
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	{
		err := index.Save(canceled, path)
		require.ErrorIs(t, err, context.Canceled)
	}

	after, err := os.ReadFile(path)
	require.NoError(t, err)
	require.True(t, slices.Equal(after, original),
		"canceled Save changed artifact")
	{
		_, err := OpenVamanaIndex(canceled, path)
		require.ErrorIs(t, err, context.Canceled)
	}

	midSave := newCancelAfterChecks(7)
	{
		err := largeIndex.Save(midSave, path)
		require.ErrorIs(t, err, context.Canceled)
	}

	after, err = os.ReadFile(path)
	require.NoError(t, err)
	require.True(t, slices.Equal(after, original),
		"mid-save cancellation changed artifact")

	midEncode := newCancelAfterChecks(5)
	{
		_, err := encodeVamanaIndex(midEncode, largeIndex)
		require.ErrorIs(t, err, context.Canceled)
	}

	largeEncoded, err := encodeVamanaIndex(context.Background(), largeIndex)
	require.NoError(t, err)

	midDecode := newCancelAfterChecks(3)
	{
		_, err := decodeVamanaIndex(midDecode, largeEncoded)
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		err := index.Save(nil, path)
		require.Error(t, err,
			"nil Save context succeeded")
	}
	{
		_, err := OpenVamanaIndex(nil, path)
		require.Error(t, err,
			"nil Open context succeeded")
	}
	{
		_, err := encodeVamanaIndex(nil, index)
		require.Error(t, err,
			"nil encode context succeeded")
	}
	{
		_, err := decodeVamanaIndex(nil, original)
		require.Error(t, err,
			"nil decode context succeeded")
	}
	{
		err := index.Save(context.Background(), "")
		require.ErrorIs(t, err, ErrInvalidVamanaFile)
	}
	{
		_, err := OpenVamanaIndex(context.Background(), "")
		require.ErrorIs(t, err, ErrInvalidVamanaFile)
	}
	{
		_, err := OpenVamanaIndex(context.Background(), filepath.Join(dir, "missing"))
		require.ErrorIs(t, err, os.ErrNotExist)
	}

	var nilIndex *VamanaIndex
	{
		err := nilIndex.Save(context.Background(), filepath.Join(dir, "nil"))
		require.ErrorIs(t, err, ErrInvalidVamanaFile)
	}
}

func TestVamanaPersistenceDetectsCorruption(t *testing.T) {
	options := DefaultVamanaBuildOptions(MetricL2)
	options.MaxDegree, options.SearchListSize = 4, 16
	valid, err := encodeVamanaIndex(context.Background(), buildVamana(t, hnswBuildInputs(32), options))
	require.NoError(t, err)

	for _, cut := range []int{0, 1, vamanaHeaderSize - 1, vamanaHeaderSize, len(valid) - 1} {
		{
			_, err := decodeVamanaIndex(context.Background(), valid[:cut])
			require.ErrorIs(t, err, ErrInvalidVamanaFile)
		}
	}
	trailing := append(slices.Clone(valid), 0)
	{
		_, err := decodeVamanaIndex(context.Background(), trailing)
		require.ErrorIs(t, err, ErrInvalidVamanaFile)
	}

	badMagic := slices.Clone(valid)
	badMagic[0] ^= 1
	{
		_, err := decodeVamanaIndex(context.Background(), badMagic)
		require.ErrorIs(t, err, ErrInvalidVamanaFile)
	}

	badVersion := slices.Clone(valid)
	binary.LittleEndian.PutUint16(badVersion[8:10], vamanaFileVersion+1)
	{
		_, err := decodeVamanaIndex(context.Background(), badVersion)
		require.ErrorIs(t, err, ErrUnsupportedVamanaVersion)
	}

	badHeader := slices.Clone(valid)
	badHeader[56] ^= 1
	{
		_, err := decodeVamanaIndex(context.Background(), badHeader)
		require.ErrorIs(t, err, ErrVamanaChecksumMismatch)
	}

	badPayload := slices.Clone(valid)
	badPayload[len(badPayload)-1] ^= 1
	{
		_, err := decodeVamanaIndex(context.Background(), badPayload)
		require.ErrorIs(t, err, ErrVamanaChecksumMismatch)
	}

	badSelfLoop := slices.Clone(valid)
	count := int(binary.LittleEndian.Uint64(badSelfLoop[32:40]))
	dimension := int(binary.LittleEndian.Uint32(badSelfLoop[48:52]))
	adjacencyOffset := vamanaHeaderSize + count*8 + count*dimension*4
	degree := int(binary.LittleEndian.Uint32(badSelfLoop[adjacencyOffset : adjacencyOffset+4]))
	require.False(t, degree == 0,
		"fixture entry has no neighbors")

	binary.LittleEndian.PutUint32(badSelfLoop[adjacencyOffset+4:adjacencyOffset+8], 0)
	refreshVamanaChecksums(badSelfLoop)
	{
		_, err := decodeVamanaIndex(context.Background(), badSelfLoop)
		require.ErrorIs(t, err, ErrInvalidVamanaFile)
	}

	badDegree := slices.Clone(valid)
	binary.LittleEndian.PutUint32(badDegree[adjacencyOffset:adjacencyOffset+4], uint32(options.MaxDegree+1))
	refreshVamanaChecksums(badDegree)
	{
		_, err := decodeVamanaIndex(context.Background(), badDegree)
		require.ErrorIs(t, err, ErrInvalidVamanaFile)
	}

	badDuplicateKey := slices.Clone(valid)
	copy(badDuplicateKey[vamanaHeaderSize+8:vamanaHeaderSize+16], badDuplicateKey[vamanaHeaderSize:vamanaHeaderSize+8])
	refreshVamanaChecksums(badDuplicateKey)
	{
		_, err := decodeVamanaIndex(context.Background(), badDuplicateKey)
		require.ErrorIs(t, err, ErrInvalidVamanaFile)
	}

	badVector := slices.Clone(valid)
	vectorOffset := vamanaHeaderSize + count*8
	binary.LittleEndian.PutUint32(badVector[vectorOffset:vectorOffset+4], math.Float32bits(float32(math.NaN())))
	refreshVamanaChecksums(badVector)
	{
		_, err := decodeVamanaIndex(context.Background(), badVector)
		require.ErrorIs(t, err, ErrInvalidVamanaFile)
	}

	badEntry := slices.Clone(valid)
	binary.LittleEndian.PutUint64(badEntry[72:80], uint64(count))
	refreshVamanaChecksums(badEntry)
	{
		_, err := decodeVamanaIndex(context.Background(), badEntry)
		require.ErrorIs(t, err, ErrInvalidVamanaFile)
	}

	badNeighbor := slices.Clone(valid)
	binary.LittleEndian.PutUint32(badNeighbor[adjacencyOffset+4:adjacencyOffset+8], uint32(count))
	refreshVamanaChecksums(badNeighbor)
	{
		_, err := decodeVamanaIndex(context.Background(), badNeighbor)
		require.ErrorIs(t, err, ErrInvalidVamanaFile)
	}

	badDuplicateNeighbor := slices.Clone(valid)
	duplicateOffset, found := vamanaAdjacencyWithDegree(badDuplicateNeighbor, 2)
	require.True(t, found,
		"fixture has no node with two neighbors")

	copy(badDuplicateNeighbor[duplicateOffset+8:duplicateOffset+12], badDuplicateNeighbor[duplicateOffset+4:duplicateOffset+8])
	refreshVamanaChecksums(badDuplicateNeighbor)
	{
		_, err := decodeVamanaIndex(context.Background(), badDuplicateNeighbor)
		require.ErrorIs(t, err, ErrInvalidVamanaFile)
	}

	badCount := slices.Clone(valid)
	binary.LittleEndian.PutUint64(badCount[32:40], uint64(math.MaxUint32)+1)
	refreshVamanaChecksums(badCount)
	{
		_, err := decodeVamanaIndex(context.Background(), badCount)
		require.ErrorIs(t, err, ErrInvalidVamanaFile)
	}
}

func FuzzVamanaDecode(f *testing.F) {
	options := DefaultVamanaBuildOptions(MetricL2)
	options.MaxDegree, options.SearchListSize = 4, 12
	valid, err := encodeVamanaIndex(context.Background(), buildVamana(f, hnswBuildInputs(8), options))
	require.NoError(f, err)

	f.Add(valid)
	f.Add([]byte("not-vamana"))
	f.Fuzz(func(t *testing.T, encoded []byte) {
		index, err := decodeVamanaIndex(context.Background(), encoded)
		if err != nil {
			return
		}
		{
			err := validateVamanaIndex(context.Background(), index)
			require.NoError(t, err)
		}
	})
}

func refreshVamanaChecksums(encoded []byte) {
	header := encoded[:vamanaHeaderSize]
	payload := encoded[vamanaHeaderSize:]
	binary.LittleEndian.PutUint32(header[80:84], ailego.CRC32C(payload))
	binary.LittleEndian.PutUint32(header[124:128], ailego.CRC32C(header[:124]))
}

func vamanaAdjacencyWithDegree(encoded []byte, minimum int) (int, bool) {
	count := int(binary.LittleEndian.Uint64(encoded[32:40]))
	dimension := int(binary.LittleEndian.Uint32(encoded[48:52]))
	offset := vamanaHeaderSize + count*8 + count*dimension*4
	for range count {
		degree := int(binary.LittleEndian.Uint32(encoded[offset : offset+4]))
		if degree >= minimum {
			return offset, true
		}
		offset += 4 + degree*4
	}
	return 0, false
}

func assertSameVamanaIndex(t testing.TB, got, want *VamanaIndex) {
	t.Helper()
	require.Equal(t, want.dimension, got.dimension,

		"Vamana indexes differ")
	require.Equal(t, want.options, got.options,

		"Vamana indexes differ")
	require.Equal(t, want.entryPoint, got.entryPoint,

		"Vamana indexes differ")
	require.True(t, slices.Equal(got.keys, want.keys),

		"Vamana indexes differ")
	require.True(t, slices.Equal(got.vectors, want.vectors),

		"Vamana indexes differ")
	require.Equal(t, want.positions, got.positions,

		"Vamana indexes differ")
	require.Equal(t, want.neighbors, got.neighbors,

		"Vamana indexes differ")
	require.Equal(t, want.neighborDistances, got.neighborDistances,
		"Vamana indexes differ")
}
