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
	"fmt"
	"path/filepath"
	"slices"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
)

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
