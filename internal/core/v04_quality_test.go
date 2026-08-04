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
	"fmt"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// TestV04DiskANNRecallAndReopenMatrix locks approximate quality and identical
// graph behavior across reopen for every public dense metric.
func TestV04DiskANNRecallAndReopenMatrix(t *testing.T) {
	const count, dimension, queryCount, topK = 384, 24, 12, 10
	candidates := diskANNIndexCandidates(count, dimension)
	for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine, MetricMIPSL2} {
		t.Run(diskANNMetricName(metric), func(t *testing.T) {
			options := DefaultDiskANNBuildOptions(metric)
			options.MaxDegree, options.ListSize, options.PQChunks = 16, 64, 12
			options.CacheCapacity = 96
			index := buildDiskANNIndex(t, candidates, options)
			path := filepath.Join(t.TempDir(), "quality.diskann")
			{
				err := index.Save(context.Background(), path)
				require.NoError(t, err)
			}

			reopened, err := OpenDiskANNIndex(context.Background(), path, 96, 4)
			require.NoError(t, err)

			defer reopened.Close()

			matched := 0
			for queryIndex := range queryCount {
				query := candidates[(queryIndex*31+17)%count].Vector
				truth, err := index.SearchDiskANN(context.Background(), query, DiskANNSearchOptions{
					SearchOptions: SearchOptions{TopK: topK}, ListSize: count, Linear: true,
				})
				require.NoError(t, err)

				search := DiskANNSearchOptions{
					SearchOptions: SearchOptions{TopK: topK}, ListSize: 160,
				}
				built, err := index.SearchDiskANN(context.Background(), query, search)
				require.NoError(t, err)

				opened, err := reopened.SearchDiskANN(context.Background(), query, search)
				require.NoError(t, err)
				require.Equal(t, built, opened)

				matched += resultOverlap(built, truth)
			}
			recall := float64(matched) / float64(queryCount*topK)
			t.Logf("DiskANN recall@%d = %.3f", topK, recall)
			require.True(t, recall >= .80)
		})
	}
}

type diskANNReadBudget struct {
	data          []byte
	enabled       atomic.Bool
	inFlight      atomic.Int64
	maxInFlight   atomic.Int64
	concurrent    atomic.Int64
	maxConcurrent atomic.Int64
	maxRequest    atomic.Int64
}

func (r *diskANNReadBudget) ReadAt(buffer []byte, offset int64) (int, error) {
	if !r.enabled.Load() {
		return bytes.NewReader(r.data).ReadAt(buffer, offset)
	}
	request := int64(len(buffer))
	updateAtomicMaximum(&r.maxRequest, request)
	updateAtomicMaximum(&r.maxInFlight, r.inFlight.Add(request))
	updateAtomicMaximum(&r.maxConcurrent, r.concurrent.Add(1))
	defer r.inFlight.Add(-request)
	defer r.concurrent.Add(-1)
	time.Sleep(time.Millisecond)
	return bytes.NewReader(r.data).ReadAt(buffer, offset)
}

func updateAtomicMaximum(target *atomic.Int64, value int64) {
	for previous := target.Load(); value > previous; previous = target.Load() {
		if target.CompareAndSwap(previous, value) {
			return
		}
	}
}

// TestV04DiskANNReadResourceBounds exercises a multi-sector layout through an
// instrumented ReaderAt. Search must respect both the worker limit and the
// pinned 128-sector aggregate batch budget.
func TestV04DiskANNReadResourceBounds(t *testing.T) {
	options := DefaultDiskANNBuildOptions(MetricL2)
	options.MaxDegree, options.ListSize, options.PQChunks = 2048, 64, 8
	options.CacheCapacity = 0
	candidates := diskANNIndexCandidates(128, 16)
	index := buildDiskANNIndex(t, candidates, options)
	encoded, err := encodeDiskANNIndex(context.Background(), index)
	require.NoError(t, err)

	for _, workers := range []int{3, 256} {
		t.Run(fmt.Sprintf("workers=%d", workers), func(t *testing.T) {
			reader := &diskANNReadBudget{data: encoded}
			opened, err := openDiskANNIndexReader(context.Background(), reader, int64(len(encoded)), 0, workers, nil)
			require.NoError(t, err)

			defer opened.Close()
			reader.enabled.Store(true)
			results, err := opened.SearchDiskANN(context.Background(), candidates[57].Vector, DiskANNSearchOptions{
				SearchOptions: SearchOptions{TopK: 10}, ListSize: len(candidates),
			})
			require.NoError(t, err)
			require.Len(t, results, 10)

			sectorsPerNode := opened.nodes.layout.SectorsPerNode()
			batchNodes := opened.diskANNReadBatchSize()
			require.True(t, batchNodes*sectorsPerNode <= MaxDiskANNReadSectors)
			require.True(t, (batchNodes+1)*sectorsPerNode > MaxDiskANNReadSectors)
			{
				got := reader.maxConcurrent.Load()
				require.True(t, got <= int64(workers))
			}
			{
				got, limit := reader.maxInFlight.Load(), int64(MaxDiskANNReadSectors*DiskANNSectorSize)
				require.True(t, got <= limit)
			}
			{
				got, limit := reader.maxRequest.Load(), int64(sectorsPerNode*DiskANNSectorSize)
				require.True(t, got <= limit)
			}
		})
	}
}

// BenchmarkV04DenseSearchQuality compares the v0.4 indexes on one corpus and
// reports recall beside latency and allocation data.
func BenchmarkV04DenseSearchQuality(b *testing.B) {
	candidates := diskANNIndexCandidates(1500, 64)
	query := candidates[731].Vector
	truth, err := TopK(context.Background(), MetricL2, query, candidates, 10)
	if err != nil {
		require.NoError(b, err)
	}

	flat, err := NewDenseFlatIndex(64, MetricL2)
	if err != nil {
		require.NoError(b, err)
	}

	for _, candidate := range candidates {
		{
			err := flat.Add(context.Background(), candidate.Key, candidate.Vector)
			if err != nil {
				require.NoError(b, err)
			}
		}
	}

	rabitqOptions := DefaultHNSWRaBitQBuildOptions(MetricL2)
	rabitqOptions.M, rabitqOptions.EFConstruction = 12, 96
	rabitqOptions.Clusters, rabitqOptions.SampleCount = 8, 512
	rabitqBuilder, err := NewHNSWRaBitQBuilder(64, rabitqOptions)
	if err != nil {
		require.NoError(b, err)
	}

	for _, candidate := range candidates {
		{
			err := rabitqBuilder.Add(context.Background(), candidate.Key, candidate.Vector)
			if err != nil {
				require.NoError(b, err)
			}
		}
	}
	rabitq, err := rabitqBuilder.Build(context.Background())
	if err != nil {
		require.NoError(b, err)
	}

	vamanaOptions := DefaultVamanaBuildOptions(MetricL2)
	vamanaOptions.MaxDegree, vamanaOptions.SearchListSize = 12, 96
	vamanaBuilder, err := NewVamanaBuilder(64, vamanaOptions)
	if err != nil {
		require.NoError(b, err)
	}

	for _, candidate := range candidates {
		{
			err := vamanaBuilder.Add(context.Background(), candidate.Key, candidate.Vector)
			if err != nil {
				require.NoError(b, err)
			}
		}
	}
	vamana, err := vamanaBuilder.Build(context.Background())
	if err != nil {
		require.NoError(b, err)
	}

	diskOptions := DefaultDiskANNBuildOptions(MetricL2)
	diskOptions.MaxDegree, diskOptions.ListSize, diskOptions.PQChunks = 12, 96, 32
	diskOptions.CacheCapacity = 512
	disk := buildDiskANNIndex(b, candidates, diskOptions)
	defer disk.Close()
	path := filepath.Join(b.TempDir(), "quality.diskann")
	{
		err := disk.Save(context.Background(), path)
		if err != nil {
			require.NoError(b, err)
		}
	}

	info, err := os.Stat(path)
	if err != nil {
		require.NoError(b, err)
	}

	benchmarks := []struct {
		name          string
		artifactBytes int64
		search        func() ([]Result, error)
	}{
		{name: "Flat", search: func() ([]Result, error) { return flat.Search(context.Background(), query, 10) }},
		{name: "HNSW-RaBitQ/EF=160", search: func() ([]Result, error) {
			return rabitq.SearchHNSWRaBitQ(context.Background(), query, HNSWRaBitQSearchOptions{
				SearchOptions: SearchOptions{TopK: 10}, EF: 160, Refine: true,
			})
		}},
		{name: "Vamana/EF=160", search: func() ([]Result, error) {
			return vamana.SearchVamana(context.Background(), query, VamanaSearchOptions{
				SearchOptions: SearchOptions{TopK: 10}, EFSearch: 160,
			})
		}},
		{name: "DiskANN/ListSize=160", artifactBytes: info.Size(), search: func() ([]Result, error) {
			return disk.SearchDiskANN(context.Background(), query, DiskANNSearchOptions{
				SearchOptions: SearchOptions{TopK: 10}, ListSize: 160,
			})
		}},
	}
	for _, benchmark := range benchmarks {
		b.Run(benchmark.name, func(b *testing.B) {
			got, err := benchmark.search()
			if err != nil {
				require.NoError(b, err)
			}

			recall := float64(resultOverlap(got, truth)) / float64(len(truth))
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				{
					_, err := benchmark.search()
					if err != nil {
						require.NoError(b, err)
					}
				}
			}
			b.ReportMetric(recall, "recall@10")
			if benchmark.artifactBytes != 0 {
				b.ReportMetric(float64(benchmark.artifactBytes), "artifact_B")
			}
		})
	}
}

func BenchmarkV04DiskANNBuild(b *testing.B) {
	candidates := diskANNIndexCandidates(512, 32)
	options := DefaultDiskANNBuildOptions(MetricL2)
	options.MaxDegree, options.ListSize, options.PQChunks = 16, 64, 16
	b.ReportAllocs()
	for b.Loop() {
		builder, err := NewDiskANNBuilder(32, options)
		if err != nil {
			require.NoError(b, err)
		}

		for _, candidate := range candidates {
			{
				err := builder.Add(context.Background(), candidate.Key, candidate.Vector)
				if err != nil {
					require.NoError(b, err)
				}
			}
		}
		index, err := builder.Build(context.Background())
		if err != nil {
			require.NoError(b, err)
		}
		{
			err := index.Close()
			if err != nil {
				require.NoError(b, err)
			}
		}
	}
}
