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
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/gorse-io/xvec/internal/ailego"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDiskANNBuildSearchMetricsFilterRadiusAndRefiner(t *testing.T) {
	for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine, MetricMIPSL2} {
		t.Run(diskANNMetricName(metric), func(t *testing.T) {
			candidates := diskANNIndexCandidates(72, 8)
			options := DefaultDiskANNBuildOptions(metric)
			options.MaxDegree, options.ListSize, options.PQChunks = 8, 24, 4
			options.CacheCapacity = len(candidates)
			index := buildDiskANNIndex(t, candidates, options)
			require.True(t, index.Dimension() == 8)
			require.Equal(t, metric, index.Metric())
			require.Len(t, candidates, index.Len())
			require.True(t, index.PQChunks() == 4)
			{
				got, ok := index.EntryPoint()
				require.True(t, ok)
				require.True(t, slices.ContainsFunc(candidates, func(candidate Candidate) bool { return candidate.Key == got }))
			}

			query := slices.Clone(candidates[17].Vector)
			linearOptions := DiskANNSearchOptions{SearchOptions: SearchOptions{TopK: 12}, ListSize: len(candidates), Linear: true}
			want, err := index.SearchDiskANN(context.Background(), query, linearOptions)
			require.NoError(t, err)

			graphOptions := linearOptions
			graphOptions.Linear = false
			got, err := index.SearchDiskANN(context.Background(), query, graphOptions)
			require.NoError(t, err)
			require.Equal(t, want, got)
			require.Equal(t, candidates[17].Key, got[0].Key)

			filter := func(key uint64) bool { return key%2 == 0 }
			filteredOptions := graphOptions
			filteredOptions.Filter = filter
			filtered, err := index.SearchDiskANN(context.Background(), query, filteredOptions)
			require.NoError(t, err)

			for _, result := range filtered {
				require.True(t, filter(result.Key))
			}
			require.False(t, len(filtered) == 0,
				"filter removed every result")

			radiusOptions := graphOptions
			radiusOptions.Radius = want[min(5, len(want)-1)].Score
			if metric == MetricIP && radiusOptions.Radius <= 0 {
				radiusOptions.Radius = 0.0001
			}
			radius, err := index.SearchDiskANN(context.Background(), query, radiusOptions)
			require.NoError(t, err)

			for _, result := range radius {
				require.True(t, scoreWithinRadius(metric, result.Score, radiusOptions.Radius))
			}

			refiner, err := NewOriginalVectorRefiner(index, metric)
			require.NoError(t, err)

			refined, err := refiner.Refine(context.Background(), query, got, SearchOptions{TopK: 5})
			require.NoError(t, err)
			require.Equal(t, want[:5], refined)

			vector, found := index.Vector(candidates[9].Key)
			require.True(t, found,
				"original vector provider differs")
			require.True(t, slices.Equal(vector, candidates[9].Vector),
				"original vector provider differs")

			vector[0]++
			again, _ := index.Vector(candidates[9].Key)
			require.False(t, slices.Equal(vector, again),
				"vector provider returned an alias")
		})
	}
}

func TestDiskANNBuilderEmptyValidationCancellationAndClose(t *testing.T) {
	defaults := DefaultDiskANNBuildOptions(MetricL2)
	require.True(t, defaults.MaxDegree == 100)
	require.True(t, defaults.ListSize == 50)
	require.True(t, defaults.PQChunks == 0)
	require.True(t, defaults.CacheCapacity == 1024)

	invalid := []DiskANNBuildOptions{
		{},
		func() DiskANNBuildOptions { value := defaults; value.MaxDegree = 0; return value }(),
		func() DiskANNBuildOptions { value := defaults; value.ListSize = 0; return value }(),
		func() DiskANNBuildOptions { value := defaults; value.PQChunks = -1; return value }(),
		func() DiskANNBuildOptions { value := defaults; value.Workers = -1; return value }(),
		func() DiskANNBuildOptions { value := defaults; value.CacheCapacity = -1; return value }(),
	}
	for _, options := range invalid {
		{
			_, err := NewDiskANNBuilder(8, options)
			require.ErrorIs(t, err, ErrInvalidDiskANNOptions)
		}
	}
	{
		_, err := NewDiskANNBuilder(0, defaults)
		require.ErrorIs(t, err, ErrInvalidDimension)
	}

	tooManyChunks := defaults
	tooManyChunks.PQChunks = 9
	{
		_, err := NewDiskANNBuilder(8, tooManyChunks)
		require.ErrorIs(t, err, ErrInvalidDiskANNOptions)
	}

	builder, err := NewDiskANNBuilder(4, defaults)
	require.NoError(t, err)
	{
		err := builder.Add(context.Background(), 7, []float32{1, 2, 3, 4})
		require.NoError(t, err)
	}
	{
		err := builder.Add(context.Background(), 7, []float32{4, 3, 2, 1})
		require.ErrorIs(t, err, ErrDuplicateKey)
	}
	{
		err := builder.Add(context.Background(), 8, []float32{1})
		require.Error(t, err,
			"invalid vector succeeded")
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	{
		err := builder.Add(canceled, 8, []float32{1, 2, 3, 4})
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		_, err := builder.Build(canceled)
		require.ErrorIs(t, err, context.Canceled)
	}

	index, err := builder.Build(context.Background())
	require.NoError(t, err)
	{
		_, err := builder.Build(context.Background())
		require.ErrorIs(t, err, ErrBuilderClosed)
	}
	{
		err := builder.Add(context.Background(), 8, []float32{1, 2, 3, 4})
		require.ErrorIs(t, err, ErrBuilderClosed)
	}
	{
		_, err := index.SearchDiskANN(context.Background(), []float32{1, 2, 3, 4}, DiskANNSearchOptions{})
		require.ErrorIs(t, err, ErrInvalidDiskANNListSize)
	}
	{
		_, err := index.SearchWithOptions(context.Background(), []float32{1, 2, 3, 4}, SearchOptions{})
		require.ErrorIs(t, err, ErrInvalidTopK)
	}
	{
		_, err := index.SearchDiskANN(canceled, []float32{1, 2, 3, 4}, DiskANNSearchOptions{SearchOptions: SearchOptions{TopK: 1}, ListSize: 1})
		require.ErrorIs(t, err, context.Canceled)
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
		_, err := index.SearchDiskANN(context.Background(), []float32{1, 2, 3, 4}, DiskANNSearchOptions{SearchOptions: SearchOptions{TopK: 1}, ListSize: 1})
		require.ErrorIs(t, err, ErrDiskANNClosed)
	}
	{
		_, found := index.Vector(7)
		require.False(t, found,
			"closed provider returned a vector")
	}

	emptyBuilder, err := NewDiskANNBuilder(4, defaults)
	require.NoError(t, err)

	empty, err := emptyBuilder.Build(context.Background())
	require.NoError(t, err)

	results, err := empty.SearchDiskANN(context.Background(), []float32{0, 0, 0, 0}, DiskANNSearchOptions{SearchOptions: SearchOptions{TopK: 1}, ListSize: 1})
	require.NoError(t, err)
	require.Len(t, results, 0)
	require.True(t, empty.PQChunks() == 0)
}

func TestDiskANNWarmCacheBoundAndSearchOwnership(t *testing.T) {
	options := DefaultDiskANNBuildOptions(MetricL2)
	options.MaxDegree, options.ListSize, options.PQChunks, options.CacheCapacity = 6, 16, 3, 7
	index := buildDiskANNIndex(t, diskANNIndexCandidates(40, 6), options)
	warmed, err := index.WarmCache(context.Background(), 20)
	require.NoError(t, err)
	require.True(t, warmed == 7)
	require.True(t, index.nodes.cache.Len() == 7)
	{
		_, err := index.WarmCache(context.Background(), -1)
		require.ErrorIs(t, err, ErrInvalidDiskANNOptions)
	}

	before := index.CacheStats()
	query := diskANNIndexCandidates(1, 6)[0].Vector
	first, err := index.SearchDiskANN(context.Background(), query, DiskANNSearchOptions{SearchOptions: SearchOptions{TopK: 5}, ListSize: 30})
	require.NoError(t, err)

	second, err := index.SearchDiskANN(context.Background(), query, DiskANNSearchOptions{SearchOptions: SearchOptions{TopK: 5}, ListSize: 30})
	require.NoError(t, err)
	require.Equal(t, second, first)
	require.True(t, index.CacheStats().Hits > before.Hits)
}

func TestDiskANNApproximateRecall(t *testing.T) {
	const count, dimension, queryCount, topK = 320, 16, 8, 10
	options := DefaultDiskANNBuildOptions(MetricL2)
	options.MaxDegree, options.ListSize, options.PQChunks, options.CacheCapacity = 12, 48, 8, 64
	candidates := diskANNIndexCandidates(count, dimension)
	index := buildDiskANNIndex(t, candidates, options)
	hits := 0
	for queryIndex := 0; queryIndex < queryCount; queryIndex++ {
		query := candidates[(queryIndex*37+11)%count].Vector
		want, err := index.SearchDiskANN(context.Background(), query, DiskANNSearchOptions{
			SearchOptions: SearchOptions{TopK: topK}, ListSize: count, Linear: true,
		})
		require.NoError(t, err)

		got, err := index.SearchDiskANN(context.Background(), query, DiskANNSearchOptions{
			SearchOptions: SearchOptions{TopK: topK}, ListSize: 64,
		})
		require.NoError(t, err)

		truth := make(map[uint64]struct{}, len(want))
		for _, result := range want {
			truth[result.Key] = struct{}{}
		}
		for _, result := range got {
			if _, found := truth[result.Key]; found {
				hits++
			}
		}
	}
	recall := float64(hits) / (queryCount * topK)
	require.True(t, recall >= 0.80)
}

func TestDiskANNConcurrentSearchAndProviderReads(t *testing.T) {
	options := DefaultDiskANNBuildOptions(MetricCosine)
	options.MaxDegree, options.ListSize, options.PQChunks, options.CacheCapacity = 8, 24, 4, 32
	candidates := diskANNIndexCandidates(96, 8)
	index := buildDiskANNIndex(t, candidates, options)
	query := candidates[31].Vector
	search := DiskANNSearchOptions{SearchOptions: SearchOptions{TopK: 10}, ListSize: 64}
	want, err := index.SearchDiskANN(context.Background(), query, search)
	require.NoError(t, err)

	var wait sync.WaitGroup
	errorsCh := make(chan error, 8)
	for worker := 0; worker < 8; worker++ {
		wait.Add(1)
		go func(worker int) {
			defer wait.Done()
			for iteration := 0; iteration < 20; iteration++ {
				got, err := index.SearchDiskANN(context.Background(), query, search)
				if err != nil {
					errorsCh <- err
					return
				}
				if !assert.Equal(t, want, got) {
					errorsCh <- errors.New("concurrent DiskANN result differs")
					return
				}
				candidate := candidates[(worker+iteration)%len(candidates)]
				if vector, found := index.Vector(candidate.Key); !found || !slices.Equal(vector, candidate.Vector) {
					errorsCh <- errors.New("concurrent DiskANN provider read differs")
					return
				}
			}
		}(worker)
	}
	wait.Wait()
	close(errorsCh)
	for err := range errorsCh {
		require.NoError(t, err)
	}
}

func buildDiskANNIndex(t testing.TB, candidates []Candidate, options DiskANNBuildOptions) *DiskANNIndex {
	t.Helper()
	dimension := 8
	if len(candidates) != 0 {
		dimension = len(candidates[0].Vector)
	}
	builder, err := NewDiskANNBuilder(dimension, options)
	require.NoError(t, err)

	for _, candidate := range candidates {
		{
			err := builder.Add(context.Background(), candidate.Key, candidate.Vector)
			require.NoError(t, err)
		}
	}
	index, err := builder.Build(context.Background())
	require.NoError(t, err)

	return index
}

func diskANNIndexCandidates(count, dimension int) []Candidate {
	result := make([]Candidate, count)
	for position := range result {
		vector := make([]float32, dimension)
		for component := range vector {
			angle := float64((position+3)*(component+5)) * 0.071
			vector[component] = float32(math.Sin(angle) + 0.45*math.Cos(angle*0.37) + float64(position%9)*0.025)
		}
		result[position] = Candidate{Key: uint64(1000 + position*7), Vector: vector}
	}
	return result
}

func diskANNMetricName(metric Metric) string {
	switch metric {
	case MetricL2:
		return "L2"
	case MetricIP:
		return "IP"
	case MetricCosine:
		return "Cosine"
	case MetricMIPSL2:
		return "MIPSL2"
	default:
		return "invalid"
	}
}

func TestDiskANNPersistenceRoundTripSearchCacheAndReplace(t *testing.T) {
	options := DefaultDiskANNBuildOptions(MetricCosine)
	options.MaxDegree, options.ListSize, options.PQChunks, options.CacheCapacity = 8, 24, 4, 20
	candidates := diskANNIndexCandidates(96, 8)
	index := buildDiskANNIndex(t, candidates, options)
	dir := t.TempDir()
	path := filepath.Join(dir, "vectors.diskann")
	{
		err := index.Save(context.Background(), path)
		require.NoError(t, err)
	}

	info, err := os.Stat(path)
	require.NoError(t, err)
	assertPrivateFileMode(t, info.Mode())
	opened, err := OpenDiskANNIndex(context.Background(), path, 24, 4)
	require.NoError(t, err)

	defer opened.Close()
	require.Equal(t, index.Dimension(), opened.Dimension())
	require.Equal(t, index.Metric(), opened.Metric())
	require.Equal(t, index.Len(), opened.Len())
	require.Equal(t, index.PQChunks(), opened.PQChunks())
	require.Equal(t, DiskANNBuildOptions{
		Metric: MetricCosine, MaxDegree: 8, ListSize: 24, PQChunks: 4, Workers: 4, CacheCapacity: 24,
	}, opened.BuildOptions())

	query := candidates[31].Vector
	search := DiskANNSearchOptions{SearchOptions: SearchOptions{TopK: 15}, ListSize: 72}
	want, err := index.SearchDiskANN(context.Background(), query, search)
	require.NoError(t, err)

	got, err := opened.SearchDiskANN(context.Background(), query, search)
	require.NoError(t, err)
	require.Equal(t, want, got)
	mapped, err := OpenDiskANNIndexWithMmap(context.Background(), path, 24, 4, true)
	require.NoError(t, err)
	mappedResults, err := mapped.SearchDiskANN(context.Background(), query, search)
	require.NoError(t, err)
	require.Equal(t, want, mappedResults)
	require.NoError(t, mapped.Close())
	{
		warmed, err := opened.WarmCache(context.Background(), 10)
		require.NoError(t, err)
		require.True(t, warmed == 10)
	}

	vector, found := opened.Vector(candidates[40].Key)
	require.True(t, found,
		"opened original vector differs")
	require.True(t, slices.Equal(vector, candidates[40].Vector),
		"opened original vector differs")

	copyPath := filepath.Join(dir, "copy.diskann")
	{
		err := opened.Save(context.Background(), copyPath)
		require.NoError(t, err)
	}

	copyIndex, err := OpenDiskANNIndex(context.Background(), copyPath, 0, 1)
	require.NoError(t, err)

	copyResults, err := copyIndex.SearchDiskANN(context.Background(), query, search)
	require.NoError(t, err)
	require.Equal(t, want, copyResults,
		"saved-opened index changed results")
	{
		err := copyIndex.Close()
		require.NoError(t, err)
	}
	{
		_, err := copyIndex.SearchDiskANN(context.Background(), query, search)
		require.ErrorIs(t, err, ErrDiskANNClosed)
	}

	replacementOptions := DefaultDiskANNBuildOptions(MetricIP)
	replacementOptions.MaxDegree, replacementOptions.ListSize, replacementOptions.PQChunks = 5, 12, 2
	replacement := buildDiskANNIndex(t, diskANNIndexCandidates(40, 6), replacementOptions)
	{
		err := replacement.Save(context.Background(), copyPath)
		require.NoError(t, err)
	}

	reopened, err := OpenDiskANNIndex(context.Background(), copyPath, 0, 2)
	require.NoError(t, err)

	defer reopened.Close()
	require.Equal(t, MetricIP, reopened.Metric())
	require.True(t, reopened.Dimension() == 6)
	require.True(t, reopened.Len() == 40)
}

func TestDiskANNPersistenceMIPSL2TraversalState(t *testing.T) {
	options := DefaultDiskANNBuildOptions(MetricMIPSL2)
	options.MaxDegree, options.ListSize, options.PQChunks = 6, 16, 3
	candidates := diskANNIndexCandidates(48, 6)
	index := buildDiskANNIndex(t, candidates, options)
	path := filepath.Join(t.TempDir(), "mips.diskann")
	{
		err := index.Save(context.Background(), path)
		require.NoError(t, err)
	}

	opened, err := OpenDiskANNIndex(context.Background(), path, 8, 2)
	require.NoError(t, err)

	defer opened.Close()
	require.Len(t, opened.codeNorms, len(candidates))
	require.Equal(t, MetricL2, opened.traversalMetric)

	search := DiskANNSearchOptions{SearchOptions: SearchOptions{TopK: 10}, ListSize: len(candidates)}
	want, err := index.SearchDiskANN(context.Background(), candidates[19].Vector, search)
	require.NoError(t, err)

	got, err := opened.SearchDiskANN(context.Background(), candidates[19].Vector, search)
	require.NoError(t, err)
	require.Equal(t, want, got)
}

func TestDiskANNPersistenceEmptyCancellationAndErrors(t *testing.T) {
	options := DefaultDiskANNBuildOptions(MetricL2)
	options.MaxDegree, options.ListSize = 4, 8
	empty := buildDiskANNIndex(t, nil, options)
	dir := t.TempDir()
	path := filepath.Join(dir, "empty.diskann")
	{
		err := empty.Save(context.Background(), path)
		require.NoError(t, err)
	}

	original, err := os.ReadFile(path)
	require.NoError(t, err)

	opened, err := OpenDiskANNIndex(context.Background(), path, 0, 0)
	require.NoError(t, err)
	require.True(t, opened.Len() == 0,
		"empty metadata differs")
	require.True(t, opened.PQChunks() == 0,
		"empty metadata differs")
	{
		err := opened.Close()
		require.NoError(t, err)
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	{
		err := empty.Save(canceled, path)
		require.ErrorIs(t, err, context.Canceled)
	}

	after, err := os.ReadFile(path)
	require.NoError(t, err)
	require.True(t, slices.Equal(after, original),
		"canceled Save changed artifact")

	nonEmptyOptions := DefaultDiskANNBuildOptions(MetricL2)
	nonEmptyOptions.MaxDegree, nonEmptyOptions.ListSize, nonEmptyOptions.PQChunks = 6, 16, 4
	nonEmpty := buildDiskANNIndex(t, diskANNIndexCandidates(96, 8), nonEmptyOptions)
	midSave := newCancelAfterChecks(5)
	{
		err := nonEmpty.Save(midSave, path)
		require.ErrorIs(t, err, context.Canceled)
	}

	after, err = os.ReadFile(path)
	require.NoError(t, err)
	require.True(t, slices.Equal(after, original),
		"mid-save cancellation changed artifact")

	midOpen := newCancelAfterChecks(5)
	{
		_, err := OpenDiskANNIndex(midOpen, path, 0, 1)
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		_, err := OpenDiskANNIndex(canceled, path, 0, 0)
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		err := empty.Save(nil, path)
		require.Error(t, err,
			"nil Save context succeeded")
	}
	{
		_, err := OpenDiskANNIndex(nil, path, 0, 0)
		require.Error(t, err,
			"nil Open context succeeded")
	}
	{
		err := empty.Save(context.Background(), "")
		require.ErrorIs(t, err, ErrInvalidDiskANNFile)
	}
	{
		_, err := OpenDiskANNIndex(context.Background(), "", 0, 0)
		require.ErrorIs(t, err, ErrInvalidDiskANNFile)
	}
	{
		_, err := OpenDiskANNIndex(context.Background(), filepath.Join(dir, "missing"), 0, 0)
		require.ErrorIs(t, err, os.ErrNotExist)
	}
	{
		_, err := OpenDiskANNIndex(context.Background(), path, -1, 0)
		require.ErrorIs(t, err, ErrInvalidDiskANNOptions)
	}

	var nilIndex *DiskANNIndex
	{
		err := nilIndex.Save(context.Background(), path)
		require.ErrorIs(t, err, ErrInvalidDiskANNFile)
	}
}

func TestDiskANNPersistenceDetectsHeaderSectionAndRecordCorruption(t *testing.T) {
	options := DefaultDiskANNBuildOptions(MetricL2)
	options.MaxDegree, options.ListSize, options.PQChunks = 5, 12, 3
	valid, err := encodeDiskANNIndex(context.Background(), buildDiskANNIndex(t, diskANNIndexCandidates(24, 6), options))
	require.NoError(t, err)

	for _, cut := range []int{0, 1, diskANNIndexHeaderSize - 1, diskANNIndexHeaderSize, len(valid) - 1} {
		{
			_, err := openDiskANNIndexReader(context.Background(), bytes.NewReader(valid[:cut]), int64(cut), 0, 1, nil)
			require.Error(t, err)
		}
	}
	trailing := append(slices.Clone(valid), 0)
	{
		_, err := openDiskANNIndexReader(context.Background(), bytes.NewReader(trailing), int64(len(trailing)), 0, 1, nil)
		require.ErrorIs(t, err, ErrInvalidDiskANNFile)
	}

	badMagic := slices.Clone(valid)
	badMagic[0] ^= 1
	{
		_, err := openDiskANNIndexReader(context.Background(), bytes.NewReader(badMagic), int64(len(badMagic)), 0, 1, nil)
		require.ErrorIs(t, err, ErrInvalidDiskANNFile)
	}

	badVersion := slices.Clone(valid)
	binary.LittleEndian.PutUint16(badVersion[8:10], diskANNIndexFileVersion+1)
	{
		_, err := openDiskANNIndexReader(context.Background(), bytes.NewReader(badVersion), int64(len(badVersion)), 0, 1, nil)
		require.ErrorIs(t, err, ErrUnsupportedDiskANNIndexVersion)
	}

	badHeader := slices.Clone(valid)
	badHeader[40] ^= 1
	{
		_, err := openDiskANNIndexReader(context.Background(), bytes.NewReader(badHeader), int64(len(badHeader)), 0, 1, nil)
		require.ErrorIs(t, err, ErrDiskANNIndexChecksumMismatch)
	}

	for _, offset := range []int{
		int(binary.LittleEndian.Uint64(valid[64:72])),
		int(binary.LittleEndian.Uint64(valid[80:88])),
		int(binary.LittleEndian.Uint64(valid[96:104])),
		int(binary.LittleEndian.Uint64(valid[112:120])),
		int(binary.LittleEndian.Uint64(valid[128:136])),
	} {
		corrupt := slices.Clone(valid)
		corrupt[offset] ^= 1
		{
			_, err := openDiskANNIndexReader(context.Background(), bytes.NewReader(corrupt), int64(len(corrupt)), 0, 1, nil)
			require.ErrorIs(t, err, ErrDiskANNIndexChecksumMismatch)
		}
	}

	// Refresh both whole-section checksums but leave the node record checksum
	// stale. Open accepts the complete sections; the random read detects the
	// damaged record at its own integrity boundary.
	recordCorrupt := slices.Clone(valid)
	nodesOffset := int(binary.LittleEndian.Uint64(recordCorrupt[128:136]))
	nodesLength := int(binary.LittleEndian.Uint64(recordCorrupt[136:144]))
	recordCorrupt[nodesOffset+diskANNNodeHeaderSize+4] ^= 1
	nodeHeader := recordCorrupt[nodesOffset : nodesOffset+diskANNNodeHeaderSize]
	nodeData := recordCorrupt[nodesOffset+diskANNNodeHeaderSize : nodesOffset+nodesLength]
	binary.LittleEndian.PutUint32(nodeHeader[72:76], ailego.CRC32C(nodeData))
	binary.LittleEndian.PutUint32(nodeHeader[diskANNNodeHeaderCRCPos:], ailego.CRC32C(nodeHeader[:diskANNNodeHeaderCRCPos]))
	binary.LittleEndian.PutUint32(recordCorrupt[160:164], ailego.CRC32C(recordCorrupt[nodesOffset:nodesOffset+nodesLength]))
	binary.LittleEndian.PutUint32(recordCorrupt[diskANNIndexHeaderCRCPos:], ailego.CRC32C(recordCorrupt[:diskANNIndexHeaderCRCPos]))
	opened, err := openDiskANNIndexReader(context.Background(), bytes.NewReader(recordCorrupt), int64(len(recordCorrupt)), 0, 1, nil)
	require.NoError(t, err)
	{
		_, err := opened.nodes.ReadNode(context.Background(), 0)
		require.ErrorIs(t, err, ErrDiskANNChecksumMismatch)
	}
}

func FuzzDiskANNIndexFile(f *testing.F) {
	options := DefaultDiskANNBuildOptions(MetricL2)
	options.MaxDegree, options.ListSize, options.PQChunks = 3, 6, 2
	valid, err := encodeDiskANNIndex(context.Background(), buildDiskANNIndex(f, diskANNIndexCandidates(8, 4), options))
	require.NoError(f, err)

	f.Add(valid)
	f.Add([]byte("not-a-diskann-index"))
	f.Fuzz(func(t *testing.T, encoded []byte) {
		if len(encoded) > 4<<20 {
			return
		}
		index, err := openDiskANNIndexReader(context.Background(), bytes.NewReader(encoded), int64(len(encoded)), 0, 1, nil)
		if err == nil {
			_ = index.Close()
		}
	})
}

func BenchmarkDiskANNSearchWarmCache(b *testing.B) {
	options := DefaultDiskANNBuildOptions(MetricL2)
	options.MaxDegree, options.ListSize, options.PQChunks, options.CacheCapacity = 8, 24, 4, 128
	candidates := diskANNIndexCandidates(128, 8)
	index := buildDiskANNIndex(b, candidates, options)
	_, _ = index.WarmCache(context.Background(), 128)
	query := candidates[27].Vector
	search := DiskANNSearchOptions{SearchOptions: SearchOptions{TopK: 10}, ListSize: 48}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		{
			_, err := index.SearchDiskANN(context.Background(), query, search)
			if err != nil {
				require.NoError(b, err)
			}
		}
	}
}

const (
	diskANNSaveCrashHelperEnv = "ZVEC_DISKANN_SAVE_CRASH_HELPER"
	diskANNSaveSourceEnv      = "ZVEC_DISKANN_SAVE_SOURCE"
	diskANNSaveTargetEnv      = "ZVEC_DISKANN_SAVE_TARGET"
)

// TestV04DiskANNAtomicSaveProcessKill kills a writer only after its temporary
// artifact exists. The published path must still contain one complete,
// checksummed generation—never a prefix of either generation.
func TestV04DiskANNAtomicSaveProcessKill(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping DiskANN subprocess fault injection in short mode")
	}
	dir := t.TempDir()
	target := filepath.Join(dir, "current.diskann")
	oldOptions := DefaultDiskANNBuildOptions(MetricL2)
	oldOptions.MaxDegree, oldOptions.ListSize, oldOptions.PQChunks = 4, 8, 2
	oldIndex := buildDiskANNIndex(t, diskANNIndexCandidates(8, 4), oldOptions)
	{
		err := oldIndex.Save(context.Background(), target)
		require.NoError(t, err)
	}

	// A large fixed-record layout keeps the atomic writer inside its chunked
	// write boundary long enough for the parent to observe and kill it without
	// requiring a production-only fault hook.
	source := filepath.Join(dir, "replacement.diskann")
	newOptions := DefaultDiskANNBuildOptions(MetricL2)
	newOptions.MaxDegree, newOptions.ListSize, newOptions.PQChunks = 32_767, 32_767, 4
	newIndex := buildDiskANNIndex(t, diskANNIndexCandidates(192, 8), newOptions)
	{
		err := newIndex.Save(context.Background(), source)
		require.NoError(t, err)
	}

	command := exec.Command(os.Args[0], "-test.run=^TestV04DiskANNSaveCrashHelper$")
	command.Env = append(os.Environ(),
		diskANNSaveCrashHelperEnv+"=1",
		diskANNSaveSourceEnv+"="+source,
		diskANNSaveTargetEnv+"="+target,
	)
	{
		err := command.Start()
		require.NoError(t, err)
	}

	done := make(chan error, 1)
	go func() { done <- command.Wait() }()

	deadline := time.Now().Add(30 * time.Second)
	killed := false
	for !killed {
		select {
		case err := <-done:
			require.FailNowf(t, "DiskANN crash helper exited before kill boundary", "%v", err)
		default:
		}
		temps, err := filepath.Glob(filepath.Join(dir, ".zvec-atomic-*.tmp"))
		require.NoError(t, err)

		if len(temps) != 0 {
			{
				err := command.Process.Kill()
				require.NoError(t, err)
			}

			<-done
			killed = true
			break
		}
		if time.Now().After(deadline) {
			_ = command.Process.Kill()
			<-done
			require.FailNow(t, "DiskANN helper did not reach the atomic write boundary")
		}
		time.Sleep(time.Millisecond)
	}

	opened, err := OpenDiskANNIndex(context.Background(), target, 0, 1)
	require.NoError(t, err)

	defer opened.Close()
	require.False(t, opened.Len() != 8 && opened.Len() != 192)
}

func TestV04DiskANNSaveCrashHelper(t *testing.T) {
	if os.Getenv(diskANNSaveCrashHelperEnv) != "1" {
		return
	}
	index, err := OpenDiskANNIndex(
		context.Background(), os.Getenv(diskANNSaveSourceEnv), 0, 1,
	)
	if err != nil {
		os.Exit(2)
	}
	if err := index.Save(context.Background(), os.Getenv(diskANNSaveTargetEnv)); err != nil {
		os.Exit(3)
	}
	os.Exit(0)
}
