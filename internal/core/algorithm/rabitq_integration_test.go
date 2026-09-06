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
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/gorse-io/xvec/pkg/rabitq"
	"github.com/stretchr/testify/require"
)

func TestHNSWRaBitQUsesSplitSingleLayout(t *testing.T) {
	index := buildHNSWRaBitQ(t, hnswRaBitQCandidates(12, 70), hnswRaBitQTestOptions(MetricL2))
	require.NotEmpty(t, index.codes)
	for _, code := range index.codes {
		require.Len(t, code.binData, rabitq.BinDataBytes(index.model.paddedDimension))
		require.Len(t, code.exData, rabitq.ExDataBytes(index.model.paddedDimension, index.model.extraBits))
	}
	prepared, cluster, err := index.model.prepareAndCluster(index.base.vectors[:index.base.dimension])
	require.NoError(t, err)
	rotated, err := rotateRaBitQVector(index.model.rotator, index.model.dimension, index.model.paddedDimension, prepared)
	require.NoError(t, err)
	binData := make([]byte, rabitq.BinDataBytes(index.model.paddedDimension))
	exData := make([]byte, rabitq.ExDataBytes(index.model.paddedDimension, index.model.extraBits))
	require.NoError(t, rabitq.QuantizeSplitSingle(
		rotated, index.model.rotatedCentroids[cluster], index.model.paddedDimension, index.model.extraBits,
		binData, exData, index.model.rabitqMetric(), rabitq.RaBitQConfig{TConst: index.model.extraScale},
	))
	require.Equal(t, binData, index.codes[0].binData)
	require.Equal(t, exData, index.codes[0].exData)

	encoded, err := encodeHNSWRaBitQIndex(context.Background(), index)
	require.NoError(t, err)
	require.Equal(t, uint16(2), binary.LittleEndian.Uint16(encoded[8:10]))
	require.Equal(t,
		4+rabitq.BinDataBytes(index.model.paddedDimension)+rabitq.ExDataBytes(index.model.paddedDimension, index.model.extraBits),
		hnswRaBitQCodeRecordSize(index.model.paddedDimension, index.model.totalBits),
	)
}

func TestIVFRaBitQFastScanTailMetricsAndReopen(t *testing.T) {
	for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine} {
		t.Run(fmt.Sprintf("metric-%d", metric), func(t *testing.T) {
			ctx := context.Background()
			options := DefaultIVFRaBitQBuildOptions(metric)
			options.NList = 1
			options.TotalBits = 7
			options.MaxIterations = 4
			builder, err := NewIVFRaBitQBuilder(64, options)
			require.NoError(t, err)
			vectors := raBitQTestVectors(rabitq.BatchSize+3, 64)
			for position, vector := range vectors {
				require.NoError(t, builder.Add(ctx, uint64(position+1), vector))
			}
			index, err := builder.Build(ctx)
			require.NoError(t, err)
			require.Len(t, index.listCodes, 1)
			require.Len(t, index.listCodes[0].batchData, 2*rabitq.BatchDataBytes(index.model.paddedDimension))
			require.Len(t, index.listCodes[0].exData, len(vectors)*rabitq.ExDataBytes(index.model.paddedDimension, index.model.extraBits))

			search := IVFRaBitQSearchOptions{
				SearchOptions: SearchOptions{TopK: len(vectors)},
				NProbe:        1,
				Linear:        true,
			}
			want, err := index.SearchIVFRaBitQ(ctx, vectors[len(vectors)-1], search)
			require.NoError(t, err)
			require.Len(t, want, len(vectors), "unused FastScan tail lanes must not become results")
			seen := make(map[uint64]struct{}, len(want))
			for position, result := range want {
				require.GreaterOrEqual(t, result.Key, uint64(1))
				require.LessOrEqual(t, result.Key, uint64(len(vectors)))
				require.NotContains(t, seen, result.Key)
				seen[result.Key] = struct{}{}
				if position > 0 {
					require.False(t, resultBetter(metric, result, want[position-1]))
				}
			}
			candidates := make([]Candidate, len(vectors))
			for position, vector := range vectors {
				candidates[position] = Candidate{Key: uint64(position + 1), Vector: vector}
			}
			truth, err := TopK(ctx, metric, vectors[len(vectors)-1], candidates, 5)
			require.NoError(t, err)
			require.GreaterOrEqual(t, resultOverlap(want[:5], truth), 3)

			path := filepath.Join(t.TempDir(), "ivf-rabitq.idx")
			require.NoError(t, index.Save(ctx, path))
			record, err := os.ReadFile(path)
			require.NoError(t, err)
			require.Equal(t, uint32(2), binary.LittleEndian.Uint32(record[8:12]))
			require.Contains(t, string(record[ivfRaBitQHeaderSize:]), `"list_codes":`)
			require.NotContains(t, string(record[ivfRaBitQHeaderSize:]), `"codes":`)
			reopened, err := OpenIVFRaBitQIndex(ctx, path)
			require.NoError(t, err)
			got, err := reopened.SearchIVFRaBitQ(ctx, vectors[len(vectors)-1], search)
			require.NoError(t, err)
			require.Equal(t, want, got)
			require.Equal(t, index.listCodes, reopened.listCodes)
		})
	}
}

func TestIVFRaBitQEmptyFastScanSaveReopen(t *testing.T) {
	ctx := context.Background()
	options := DefaultIVFRaBitQBuildOptions(MetricL2)
	options.NList = 1
	builder, err := NewIVFRaBitQBuilder(64, options)
	require.NoError(t, err)
	index, err := builder.Build(ctx)
	require.NoError(t, err)
	require.Len(t, index.listCodes, 1)
	require.Empty(t, index.listCodes[0].batchData)

	path := filepath.Join(t.TempDir(), "empty.idx")
	require.NoError(t, index.Save(ctx, path))
	reopened, err := OpenIVFRaBitQIndex(ctx, path)
	require.NoError(t, err)
	results, err := reopened.SearchIVFRaBitQ(ctx, make([]float32, 64), IVFRaBitQSearchOptions{
		SearchOptions: SearchOptions{TopK: 1}, NProbe: 1,
	})
	require.NoError(t, err)
	require.Empty(t, results)
}
