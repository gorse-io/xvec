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
	"slices"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/gorse-io/xvec/internal/ailego/hash"
	"github.com/stretchr/testify/require"
)

func TestDiskANNPackedLayoutReadCacheAndOwnership(t *testing.T) {
	layout, nodes, encoded := diskANNFixture(t, 10, 4, 3)
	require.True(t, layout.RecordSize() == 36)
	require.Equal(t, DiskANNSectorSize/36, layout.NodesPerSector())
	require.True(t, layout.SectorsPerNode() == 1)
	require.True(t, layout.DataOffset() == 4096)
	require.True(t, layout.DataLength() == 4096)

	counter := &countingReaderAt{reader: bytes.NewReader(encoded)}
	reader, err := OpenDiskANNNodeReader(context.Background(), counter, int64(len(encoded)), 10, 4)
	require.NoError(t, err)
	{
		got := reader.Layout()
		require.True(t, got.Count() == 10)
		require.True(t, got.Dimension() == 4)
		require.True(t, got.MaxDegree() == 3)
		require.Equal(t, MetricL2, got.Metric())
		require.Equal(t, int64(len(encoded)), got.TotalLength())
	}

	counter.calls.Store(0)
	got, err := reader.ReadNodes(context.Background(), []uint32{5, 1, 5})
	require.NoError(t, err)
	require.True(t, counter.calls.Load() == 1)
	require.Equal(t, []DiskANNNode{nodes[5], nodes[1], nodes[5]}, got)

	got[0].Vector[0] = 999
	got[0].Neighbors[0] = 9
	again, err := reader.ReadNodes(context.Background(), []uint32{5, 1, 5})
	require.NoError(t, err)
	require.True(t, counter.calls.Load() == 1,
		"cache miss or aliased cached node")
	require.Equal(t, []DiskANNNode{nodes[5], nodes[1], nodes[5]}, again,
		"cache miss or aliased cached node")

	stats := reader.CacheStats()
	require.True(t, stats.Hits == 3)
	require.True(t, stats.Misses == 3)
	require.True(t, stats.Evictions == 0)
}

func TestDiskANNMultiSectorLayoutAndPartialReaderAt(t *testing.T) {
	layout, nodes, encoded := diskANNFixture(t, 3, 1024, 10)
	require.True(t, layout.NodesPerSector() == 0)
	require.True(t, layout.SectorsPerNode() == 2)
	require.True(t, layout.RecordSize() == 4144)
	require.Equal(t, int64(6*DiskANNSectorSize), layout.DataLength())

	partial := &partialReaderAt{data: encoded, maximum: 127}
	reader, err := OpenDiskANNNodeReader(context.Background(), partial, int64(len(encoded)), 0, 3)
	require.NoError(t, err)

	got, err := reader.ReadNodes(context.Background(), []uint32{2, 0, 1})
	require.NoError(t, err)
	require.Equal(t, []DiskANNNode{nodes[2], nodes[0], nodes[1]}, got,
		"multi-sector nodes differ")

	for id := range nodes {
		spec, err := layout.readSpec(uint32(id))
		require.NoError(t, err)
		require.Equal(t, 2*DiskANNSectorSize, spec.length)
		require.True(t, spec.recordOffset == 0)
		require.Equal(t, int64(diskANNNodeHeaderSize+id*2*DiskANNSectorSize), spec.offset)
	}
}

func TestDiskANNLayoutEmptyAndValidation(t *testing.T) {
	empty, err := NewDiskANNLayout(MetricIP, 0, 2, 1)
	require.NoError(t, err)

	encoded, err := encodeDiskANNNodeFile(context.Background(), empty, nil)
	require.NoError(t, err)

	reader, err := OpenDiskANNNodeReader(context.Background(), bytes.NewReader(encoded), int64(len(encoded)), 0, 0)
	require.NoError(t, err)
	require.True(t, reader.Layout().DataLength() == 0,
		"empty layout has data")
	{
		got, err := reader.ReadNodes(context.Background(), nil)
		require.NoError(t, err)
		require.Len(t, got, 0)
	}

	invalid := []struct {
		metric                 Metric
		count, dimension, maxD int
	}{
		{0, 1, 2, 1}, {MetricL2, -1, 2, 1}, {MetricL2, 1, 0, 1},
		{MetricL2, 1, MaxRotationDimension + 1, 1}, {MetricL2, 1, 2, 0},
	}
	for _, value := range invalid {
		{
			_, err := NewDiskANNLayout(value.metric, value.count, value.dimension, value.maxD)
			require.ErrorIs(t, err, ErrInvalidDiskANNLayout)
		}
	}
	{
		_, err := reader.ReadNode(context.Background(), 0)
		require.ErrorIs(t, err, ErrInvalidDiskANNNode)
	}
}

func TestDiskANNNodeEncodingValidationAndCorruption(t *testing.T) {
	layout, nodes, encoded := diskANNFixture(t, 6, 3, 3)
	badNodes := []DiskANNNode{
		{ID: 6, Vector: []float32{1, 2, 3}},
		{ID: 0, Vector: []float32{1, 2}},
		{ID: 0, Vector: []float32{1, 2, 3}, Neighbors: []uint32{0}},
		{ID: 0, Vector: []float32{1, 2, 3}, Neighbors: []uint32{1, 1}},
		{ID: 0, Vector: []float32{1, 2, 3}, Neighbors: []uint32{1, 2, 3, 4}},
	}
	for _, node := range badNodes {
		{
			_, err := layout.encodeNode(node)
			require.ErrorIs(t, err, ErrInvalidDiskANNNode)
		}
	}
	{
		_, err := encodeDiskANNNodeFile(context.Background(), layout, nodes[:5])
		require.ErrorIs(t, err, ErrInvalidDiskANNLayout)
	}

	outOfOrder := slices.Clone(nodes)
	outOfOrder[0], outOfOrder[1] = outOfOrder[1], outOfOrder[0]
	{
		_, err := encodeDiskANNNodeFile(context.Background(), layout, outOfOrder)
		require.ErrorIs(t, err, ErrInvalidDiskANNNode)
	}

	corruptPayload := slices.Clone(encoded)
	corruptPayload[diskANNNodeHeaderSize+4] ^= 1
	{
		_, err := OpenDiskANNNodeReader(context.Background(), bytes.NewReader(corruptPayload), int64(len(corruptPayload)), 0, 1)
		require.ErrorIs(t, err, ErrDiskANNChecksumMismatch)
	}

	corruptRecord := slices.Clone(corruptPayload)
	refreshDiskANNDataChecksum(corruptRecord)
	reader, err := OpenDiskANNNodeReader(context.Background(), bytes.NewReader(corruptRecord), int64(len(corruptRecord)), 0, 1)
	require.NoError(t, err)
	{
		_, err := reader.ReadNode(context.Background(), 0)
		require.ErrorIs(t, err, ErrDiskANNChecksumMismatch)
	}

	semantic := slices.Clone(encoded)
	spec, _ := layout.readSpec(0)
	recordStart := int(spec.offset) + spec.recordOffset
	degreeOffset := recordStart + layout.dimension*4
	binary.LittleEndian.PutUint32(semantic[degreeOffset:degreeOffset+4], uint32(layout.maxDegree+1))
	refreshDiskANNRecordChecksum(semantic[recordStart : recordStart+layout.recordSize])
	refreshDiskANNDataChecksum(semantic)
	reader, err = OpenDiskANNNodeReader(context.Background(), bytes.NewReader(semantic), int64(len(semantic)), 0, 1)
	require.NoError(t, err)
	{
		_, err := reader.ReadNode(context.Background(), 0)
		require.ErrorIs(t, err, ErrInvalidDiskANNNode)
	}
}

func TestDiskANNHeaderCorruptionTruncationAndCancellation(t *testing.T) {
	layout, nodes, valid := diskANNFixture(t, 6, 3, 3)
	for _, size := range []int{0, 1, diskANNNodeHeaderSize - 1, len(valid) - 1} {
		{
			_, err := OpenDiskANNNodeReader(context.Background(), bytes.NewReader(valid[:size]), int64(size), 0, 1)
			require.Error(t, err)
		}
	}
	trailing := append(slices.Clone(valid), 0)
	{
		_, err := OpenDiskANNNodeReader(context.Background(), bytes.NewReader(trailing), int64(len(trailing)), 0, 1)
		require.ErrorIs(t, err, ErrInvalidDiskANNLayout)
	}

	badMagic := slices.Clone(valid)
	badMagic[0] ^= 1
	{
		_, err := OpenDiskANNNodeReader(context.Background(), bytes.NewReader(badMagic), int64(len(badMagic)), 0, 1)
		require.ErrorIs(t, err, ErrInvalidDiskANNLayout)
	}

	badVersion := slices.Clone(valid)
	binary.LittleEndian.PutUint16(badVersion[8:10], diskANNNodeFileVersion+1)
	{
		_, err := OpenDiskANNNodeReader(context.Background(), bytes.NewReader(badVersion), int64(len(badVersion)), 0, 1)
		require.ErrorIs(t, err, ErrUnsupportedDiskANNVersion)
	}

	badHeader := slices.Clone(valid)
	badHeader[48] ^= 1
	{
		_, err := OpenDiskANNNodeReader(context.Background(), bytes.NewReader(badHeader), int64(len(badHeader)), 0, 1)
		require.ErrorIs(t, err, ErrDiskANNChecksumMismatch)
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	{
		_, err := encodeDiskANNNodeFile(canceled, layout, nodes)
		require.ErrorIs(t, err, context.Canceled)
	}

	largeLayout, largeNodes, _ := diskANNFixture(t, 300, 3, 3)
	midEncode := newCancelAfterChecks(3)
	{
		_, err := encodeDiskANNNodeFile(midEncode, largeLayout, largeNodes)
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		_, err := OpenDiskANNNodeReader(canceled, bytes.NewReader(valid), int64(len(valid)), 0, 1)
		require.ErrorIs(t, err, context.Canceled)
	}

	reader, err := OpenDiskANNNodeReader(context.Background(), bytes.NewReader(valid), int64(len(valid)), 0, 1)
	require.NoError(t, err)
	{
		_, err := reader.ReadNodes(canceled, []uint32{0})
		require.ErrorIs(t, err, context.Canceled)
	}
}

func TestParallelReadAtOrderingPartialShortAndValidation(t *testing.T) {
	data := []byte("abcdefghijklmnopqrstuvwxyz")
	requests := []DiskANNReadRequest{{Offset: 10, Length: 4}, {Offset: 0, Length: 3}, {Offset: 20, Length: 6}}
	got, err := ParallelReadAt(context.Background(), &partialReaderAt{data: data, maximum: 2}, requests, 3)
	require.NoError(t, err)
	require.Equal(t, [][]byte{[]byte("klmn"), []byte("abc"), []byte("uvwxyz")}, got)
	{
		_, err := ParallelReadAt(context.Background(), bytes.NewReader(data), []DiskANNReadRequest{{Offset: 24, Length: 4}}, 1)
		require.ErrorIs(t, err, ErrDiskANNShortRead)
	}
	{
		_, err := ParallelReadAt(nil, bytes.NewReader(data), requests, 1)
		require.Error(t, err,
			"nil context succeeded")
	}
	{
		_, err := ParallelReadAt(context.Background(), nil, requests, 1)
		require.Error(t, err,
			"nil reader succeeded")
	}
	{
		_, err := ParallelReadAt(context.Background(), bytes.NewReader(data), []DiskANNReadRequest{{Offset: -1, Length: 1}}, 1)
		require.Error(t, err,
			"negative offset succeeded")
	}
}

func TestDiskANNNodeCacheEvictionConcurrencyAndOwnership(t *testing.T) {
	cache, err := NewDiskANNNodeCache(2)
	require.NoError(t, err)

	nodes := diskANNNodes(4, 2)
	cache.Put(nodes[0])
	cache.Put(nodes[1])
	{
		_, found := cache.Get(0)
		require.True(t, found,
			"missing cached node")
	}

	cache.Put(nodes[2])
	{
		_, found := cache.Get(1)
		require.False(t, found,
			"least-recently-used node was not evicted")
	}

	got, found := cache.Get(0)
	require.True(t, found,
		"recent node evicted")

	got.Vector[0] = 999
	again, _ := cache.Get(0)
	require.False(t, again.Vector[0] == 999,
		"cache returned aliased node")

	var wait sync.WaitGroup
	for worker := 0; worker < 8; worker++ {
		wait.Add(1)
		go func(worker int) {
			defer wait.Done()
			for iteration := 0; iteration < 100; iteration++ {
				cache.Put(nodes[(worker+iteration)%len(nodes)])
				_, _ = cache.Get(uint32(iteration % len(nodes)))
			}
		}(worker)
	}
	wait.Wait()
	require.True(t, cache.Len() <= cache.Capacity())
	require.False(t, cache.Stats().Evictions == 0)

	cache.Clear()
	require.True(t, cache.Len() == 0,
		"cache clear failed")
	{
		_, err := NewDiskANNNodeCache(-1)
		require.Error(t, err,
			"negative cache capacity succeeded")
	}
}

func FuzzDiskANNNodeFile(f *testing.F) {
	_, _, valid := diskANNFixture(f, 4, 3, 2)
	f.Add(valid)
	f.Add([]byte("not-diskann"))
	f.Fuzz(func(t *testing.T, encoded []byte) {
		if len(encoded) < diskANNNodeHeaderSize {
			return
		}
		layout, err := decodeDiskANNLayout(encoded[:diskANNNodeHeaderSize], int64(len(encoded)))
		if err != nil {
			return
		}
		if layout.count == 0 {
			return
		}
		spec, err := layout.readSpec(0)
		if err != nil || spec.offset > int64(len(encoded)) || int64(spec.length) > int64(len(encoded))-spec.offset {
			return
		}
		_, _ = layout.decodeNode(0, encoded[spec.offset+int64(spec.recordOffset):])
	})
}

func BenchmarkDiskANNWarmNodeRead(b *testing.B) {
	_, _, encoded := diskANNFixture(b, 100, 16, 8)
	reader, err := OpenDiskANNNodeReader(context.Background(), bytes.NewReader(encoded), int64(len(encoded)), 100, 4)
	if err != nil {
		require.NoError(b, err)
	}
	{
		_, err := reader.ReadNode(context.Background(), 50)
		if err != nil {
			require.NoError(b, err)
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		{
			_, err := reader.ReadNode(context.Background(), 50)
			if err != nil {
				require.NoError(b, err)
			}
		}
	}
}

func diskANNFixture(t testing.TB, count, dimension, maxDegree int) (DiskANNLayout, []DiskANNNode, []byte) {
	t.Helper()
	layout, err := NewDiskANNLayout(MetricL2, count, dimension, maxDegree)
	require.NoError(t, err)

	nodes := diskANNNodes(count, dimension)
	for index := range nodes {
		if count > 1 {
			nodes[index].Neighbors = []uint32{uint32((index + 1) % count)}
		}
	}
	encoded, err := encodeDiskANNNodeFile(context.Background(), layout, nodes)
	require.NoError(t, err)

	return layout, nodes, encoded
}

func diskANNNodes(count, dimension int) []DiskANNNode {
	nodes := make([]DiskANNNode, count)
	for index := range nodes {
		nodes[index].ID = uint32(index)
		nodes[index].Vector = make([]float32, dimension)
		for component := range nodes[index].Vector {
			nodes[index].Vector[component] = float32(index*dimension+component) / 7
		}
	}
	return nodes
}

func refreshDiskANNRecordChecksum(record []byte) {
	binary.LittleEndian.PutUint32(record[len(record)-4:], hashutil.CRC32C(record[:len(record)-4]))
}

func refreshDiskANNDataChecksum(encoded []byte) {
	header := encoded[:diskANNNodeHeaderSize]
	data := encoded[diskANNNodeHeaderSize:]
	binary.LittleEndian.PutUint32(header[72:76], hashutil.CRC32C(data))
	binary.LittleEndian.PutUint32(header[diskANNNodeHeaderCRCPos:], hashutil.CRC32C(header[:diskANNNodeHeaderCRCPos]))
}

type countingReaderAt struct {
	reader io.ReaderAt
	calls  atomic.Int64
}

func (r *countingReaderAt) ReadAt(buffer []byte, offset int64) (int, error) {
	r.calls.Add(1)
	return r.reader.ReadAt(buffer, offset)
}

type partialReaderAt struct {
	data    []byte
	maximum int
}

func (r *partialReaderAt) ReadAt(buffer []byte, offset int64) (int, error) {
	if offset < 0 || offset >= int64(len(r.data)) {
		return 0, io.EOF
	}
	length := min(len(buffer), r.maximum, len(r.data)-int(offset))
	copy(buffer, r.data[offset:int(offset)+length])
	if length == len(buffer) {
		return length, nil
	}
	return length, nil
}
