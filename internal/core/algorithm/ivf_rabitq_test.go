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
	"bytes"
	"context"
	"encoding/binary"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/gorse-io/xvec/internal/ailego/hash"
	"github.com/stretchr/testify/require"
)

type cancelAfterRead struct {
	cancel context.CancelFunc
	done   bool
}

func (r *cancelAfterRead) Read(p []byte) (int, error) {
	if r.done {
		return 0, io.EOF
	}
	r.done = true
	for index := range p {
		p[index] = 1
	}
	r.cancel()
	return len(p), nil
}

func TestIVFRaBitQBuildSearchAndReopen(t *testing.T) {
	ctx := context.Background()
	options := DefaultIVFRaBitQBuildOptions(MetricL2)
	options.NList = 2
	options.TotalBits = 7
	options.SampleCount = 4
	builder, err := NewIVFRaBitQBuilder(64, options)
	require.NoError(t, err)
	for key := uint64(0); key < 8; key++ {
		vector := make([]float32, 64)
		vector[0] = float32(key)
		vector[1] = float32(key % 2)
		require.NoError(t, builder.Add(ctx, key, vector))
	}
	index, err := builder.Build(ctx)
	require.NoError(t, err)
	require.Equal(t, 8, index.Len())
	require.Equal(t, 2, index.NList())
	for position, code := range index.codes {
		require.Contains(t, index.base.lists[code.Cluster()].positions, position,
			"RaBitQ centroid and IVF list assignment differ")
	}

	query := make([]float32, 64)
	query[0] = 3
	results, err := index.SearchIVFRaBitQ(ctx, query, IVFRaBitQSearchOptions{
		SearchOptions: SearchOptions{TopK: 3}, NProbe: 2,
	})
	require.NoError(t, err)
	require.Len(t, results, 3)
	require.Contains(t, []uint64{results[0].Key, results[1].Key, results[2].Key}, uint64(3))

	path := filepath.Join(t.TempDir(), "ivf-rabitq.idx")
	require.NoError(t, index.Save(ctx, path))
	reopened, err := OpenIVFRaBitQIndex(ctx, path)
	require.NoError(t, err)
	reopenedResults, err := reopened.SearchIVFRaBitQ(ctx, query, IVFRaBitQSearchOptions{
		SearchOptions: SearchOptions{TopK: 3}, NProbe: 2,
	})
	require.NoError(t, err)
	require.Equal(t, results, reopenedResults)
}

func TestIVFRaBitQSmallIndexUsesEncodedBruteForce(t *testing.T) {
	ctx := context.Background()
	options := DefaultIVFRaBitQBuildOptions(MetricL2)
	options.NList = 2
	builder, err := NewIVFRaBitQBuilder(64, options)
	require.NoError(t, err)
	for key, value := range []float32{0, 1, 100, 101} {
		vector := make([]float32, 64)
		vector[0] = value
		require.NoError(t, builder.Add(ctx, uint64(key), vector))
	}
	index, err := builder.Build(ctx)
	require.NoError(t, err)
	query := make([]float32, 64)
	query[0] = 100
	results, err := index.SearchIVFRaBitQ(ctx, query, IVFRaBitQSearchOptions{
		SearchOptions: SearchOptions{TopK: 4}, NProbe: 1,
	})
	require.NoError(t, err)
	require.Len(t, results, index.Len(), "small indexes must scan all encoded clusters")
	require.Equal(t, uint64(2), results[0].Key)
}

func TestIVFRaBitQValidationRejectsListCodeMismatch(t *testing.T) {
	ctx := context.Background()
	options := DefaultIVFRaBitQBuildOptions(MetricL2)
	options.NList = 2
	builder, err := NewIVFRaBitQBuilder(64, options)
	require.NoError(t, err)
	for key := uint64(0); key < 4; key++ {
		vector := make([]float32, 64)
		vector[0] = float32(key * 100)
		require.NoError(t, builder.Add(ctx, key, vector))
	}
	index, err := builder.Build(ctx)
	require.NoError(t, err)
	index.base.listForPosition[0] = (index.codes[0].Cluster() + 1) % index.NList()
	require.ErrorIs(t, validateIVFRaBitQIndex(ctx, index), ErrInvalidIVFRaBitQFile)
}

func TestIVFRaBitQValidationRejectsModelOptionMismatch(t *testing.T) {
	ctx := context.Background()
	options := DefaultIVFRaBitQBuildOptions(MetricL2)
	options.NList = 1
	builder, err := NewIVFRaBitQBuilder(64, options)
	require.NoError(t, err)
	require.NoError(t, builder.Add(ctx, 1, make([]float32, 64)))
	index, err := builder.Build(ctx)
	require.NoError(t, err)
	index.model.metric = MetricIP
	require.ErrorIs(t, validateIVFRaBitQIndex(ctx, index), ErrInvalidIVFRaBitQFile)
	index.model.metric = MetricL2
	index.model.totalBits++
	require.ErrorIs(t, validateIVFRaBitQIndex(ctx, index), ErrInvalidIVFRaBitQFile)
}

func TestIVFRaBitQGroupSearchHonorsCanceledContextOnEmptyIndex(t *testing.T) {
	ctx := context.Background()
	options := DefaultIVFRaBitQBuildOptions(MetricL2)
	options.NList = 1
	builder, err := NewIVFRaBitQBuilder(64, options)
	require.NoError(t, err)
	index, err := builder.Build(ctx)
	require.NoError(t, err)
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	_, err = index.SearchIVFRaBitQGroups(canceled, make([]float32, 64), IVFRaBitQSearchOptions{
		SearchOptions: SearchOptions{TopK: 1}, NProbe: 1,
	}, GroupByOptions{GroupCount: 1, TopKPerGroup: 1, Resolve: func(uint64) (string, bool) { return "group", true }})
	require.ErrorIs(t, err, context.Canceled)
}

func TestIVFRaBitQNilSaveReturnsError(t *testing.T) {
	var index *IVFRaBitQIndex
	require.Error(t, index.Save(context.Background(), filepath.Join(t.TempDir(), "nil.idx")))
}

func TestReadIVFRaBitQRecordHonorsCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	reader := &cancelAfterRead{cancel: cancel}
	_, err := readIVFRaBitQRecord(ctx, reader, 2*ivfRaBitQReadChunkSize)
	require.ErrorIs(t, err, context.Canceled)

	_, err = readIVFRaBitQRecord(context.Background(), bytes.NewReader([]byte{1}), 2)
	require.ErrorIs(t, err, io.ErrUnexpectedEOF)
}

func TestIVFRaBitQLargeIndexUsesOnlyProbedLists(t *testing.T) {
	ctx := context.Background()
	options := DefaultIVFRaBitQBuildOptions(MetricL2)
	options.NList = 2
	builder, err := NewIVFRaBitQBuilder(64, options)
	require.NoError(t, err)
	for key := uint64(0); key < uint64(ivfRaBitQBruteForceThreshold+2); key++ {
		vector := make([]float32, 64)
		if key%2 == 1 {
			vector[0] = 1000
		}
		require.NoError(t, builder.Add(ctx, key, vector))
	}
	index, err := builder.Build(ctx)
	require.NoError(t, err)
	query := make([]float32, 64)
	query[0] = 1000
	probed, err := index.base.ProbedLists(ctx, query, 1)
	require.NoError(t, err)
	results, err := index.SearchIVFRaBitQ(ctx, query, IVFRaBitQSearchOptions{
		SearchOptions: SearchOptions{TopK: index.Len()}, NProbe: 1,
	})
	require.NoError(t, err)
	require.NotEmpty(t, results)
	require.Less(t, len(results), index.Len())
	for _, result := range results {
		require.Equal(t, probed[0], index.base.listForPosition[int(result.Key)])
	}
}

func TestDecodeIVFRaBitQPayloadPreservesCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := decodeIVFRaBitQPayload(ctx, []byte("{}"))
	require.ErrorIs(t, err, context.Canceled)
}

func TestIVFRaBitQOpenRejectsTrailingJSONValue(t *testing.T) {
	ctx := context.Background()
	options := DefaultIVFRaBitQBuildOptions(MetricL2)
	options.NList = 1
	builder, err := NewIVFRaBitQBuilder(64, options)
	require.NoError(t, err)
	require.NoError(t, builder.Add(ctx, 1, make([]float32, 64)))
	index, err := builder.Build(ctx)
	require.NoError(t, err)
	path := filepath.Join(t.TempDir(), "trailing.idx")
	require.NoError(t, index.Save(ctx, path))
	record, err := os.ReadFile(path)
	require.NoError(t, err)
	payload := append(append([]byte(nil), record[ivfRaBitQHeaderSize:]...), []byte("{}")...)
	header := append([]byte(nil), record[:ivfRaBitQHeaderSize]...)
	binary.LittleEndian.PutUint64(header[12:20], uint64(len(payload)))
	binary.LittleEndian.PutUint32(header[20:24], hashutil.CRC32C(payload))
	binary.LittleEndian.PutUint32(header[28:32], hashutil.CRC32C(header[:28]))
	require.NoError(t, os.WriteFile(path, append(header, payload...), 0o600))
	_, err = OpenIVFRaBitQIndex(ctx, path)
	require.ErrorIs(t, err, ErrInvalidIVFRaBitQFile)
}
