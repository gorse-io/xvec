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
	"io"
	"reflect"
	"slices"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/gorse-io/zvec/internal/ailego"
)

func TestDiskANNPackedLayoutReadCacheAndOwnership(t *testing.T) {
	layout, nodes, encoded := diskANNFixture(t, 10, 4, 3)
	if layout.RecordSize() != 36 || layout.NodesPerSector() != DiskANNSectorSize/36 ||
		layout.SectorsPerNode() != 1 || layout.DataOffset() != 4096 || layout.DataLength() != 4096 {
		t.Fatalf("packed layout = %#v", layout)
	}
	counter := &countingReaderAt{reader: bytes.NewReader(encoded)}
	reader, err := OpenDiskANNNodeReader(context.Background(), counter, int64(len(encoded)), 10, 4)
	if err != nil {
		t.Fatal(err)
	}
	if got := reader.Layout(); got.Count() != 10 || got.Dimension() != 4 || got.MaxDegree() != 3 ||
		got.Metric() != MetricL2 || got.TotalLength() != int64(len(encoded)) {
		t.Fatalf("opened layout = %#v", got)
	}
	counter.calls.Store(0)
	got, err := reader.ReadNodes(context.Background(), []uint32{5, 1, 5})
	if err != nil {
		t.Fatal(err)
	}
	if counter.calls.Load() != 1 {
		t.Fatalf("packed batch issued %d reads, want 1", counter.calls.Load())
	}
	if !reflect.DeepEqual(got, []DiskANNNode{nodes[5], nodes[1], nodes[5]}) {
		t.Fatalf("nodes = %#v", got)
	}
	got[0].Vector[0] = 999
	got[0].Neighbors[0] = 9
	again, err := reader.ReadNodes(context.Background(), []uint32{5, 1, 5})
	if err != nil {
		t.Fatal(err)
	}
	if counter.calls.Load() != 1 || !reflect.DeepEqual(again, []DiskANNNode{nodes[5], nodes[1], nodes[5]}) {
		t.Fatal("cache miss or aliased cached node")
	}
	stats := reader.CacheStats()
	if stats.Hits != 3 || stats.Misses != 3 || stats.Evictions != 0 {
		t.Fatalf("cache stats = %#v", stats)
	}
}

func TestDiskANNMultiSectorLayoutAndPartialReaderAt(t *testing.T) {
	layout, nodes, encoded := diskANNFixture(t, 3, 1024, 10)
	if layout.NodesPerSector() != 0 || layout.SectorsPerNode() != 2 ||
		layout.RecordSize() != 4144 || layout.DataLength() != 6*DiskANNSectorSize {
		t.Fatalf("multi-sector layout = %#v", layout)
	}
	partial := &partialReaderAt{data: encoded, maximum: 127}
	reader, err := OpenDiskANNNodeReader(context.Background(), partial, int64(len(encoded)), 0, 3)
	if err != nil {
		t.Fatal(err)
	}
	got, err := reader.ReadNodes(context.Background(), []uint32{2, 0, 1})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, []DiskANNNode{nodes[2], nodes[0], nodes[1]}) {
		t.Fatal("multi-sector nodes differ")
	}
	for id := range nodes {
		spec, err := layout.readSpec(uint32(id))
		if err != nil {
			t.Fatal(err)
		}
		if spec.length != 2*DiskANNSectorSize || spec.recordOffset != 0 ||
			spec.offset != int64(diskANNNodeHeaderSize+id*2*DiskANNSectorSize) {
			t.Fatalf("node %d read spec = %#v", id, spec)
		}
	}
}

func TestDiskANNLayoutEmptyAndValidation(t *testing.T) {
	empty, err := NewDiskANNLayout(MetricIP, 0, 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := encodeDiskANNNodeFile(context.Background(), empty, nil)
	if err != nil {
		t.Fatal(err)
	}
	reader, err := OpenDiskANNNodeReader(context.Background(), bytes.NewReader(encoded), int64(len(encoded)), 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if reader.Layout().DataLength() != 0 {
		t.Fatal("empty layout has data")
	}
	if got, err := reader.ReadNodes(context.Background(), nil); err != nil || len(got) != 0 {
		t.Fatalf("empty batch = %#v, %v", got, err)
	}
	invalid := []struct {
		metric                 Metric
		count, dimension, maxD int
	}{
		{0, 1, 2, 1}, {MetricL2, -1, 2, 1}, {MetricL2, 1, 0, 1},
		{MetricL2, 1, MaxRotationDimension + 1, 1}, {MetricL2, 1, 2, 0},
	}
	for _, value := range invalid {
		if _, err := NewDiskANNLayout(value.metric, value.count, value.dimension, value.maxD); !errors.Is(err, ErrInvalidDiskANNLayout) {
			t.Fatalf("layout %#v error = %v", value, err)
		}
	}
	if _, err := reader.ReadNode(context.Background(), 0); !errors.Is(err, ErrInvalidDiskANNNode) {
		t.Fatalf("empty node error = %v", err)
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
		if _, err := layout.encodeNode(node); !errors.Is(err, ErrInvalidDiskANNNode) {
			t.Fatalf("node %#v error = %v", node, err)
		}
	}
	if _, err := encodeDiskANNNodeFile(context.Background(), layout, nodes[:5]); !errors.Is(err, ErrInvalidDiskANNLayout) {
		t.Fatalf("short node set error = %v", err)
	}
	outOfOrder := slices.Clone(nodes)
	outOfOrder[0], outOfOrder[1] = outOfOrder[1], outOfOrder[0]
	if _, err := encodeDiskANNNodeFile(context.Background(), layout, outOfOrder); !errors.Is(err, ErrInvalidDiskANNNode) {
		t.Fatalf("out-of-order error = %v", err)
	}

	corruptPayload := slices.Clone(encoded)
	corruptPayload[diskANNNodeHeaderSize+4] ^= 1
	if _, err := OpenDiskANNNodeReader(context.Background(), bytes.NewReader(corruptPayload), int64(len(corruptPayload)), 0, 1); !errors.Is(err, ErrDiskANNChecksumMismatch) {
		t.Fatalf("payload checksum error = %v", err)
	}
	corruptRecord := slices.Clone(corruptPayload)
	refreshDiskANNDataChecksum(corruptRecord)
	reader, err := OpenDiskANNNodeReader(context.Background(), bytes.NewReader(corruptRecord), int64(len(corruptRecord)), 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := reader.ReadNode(context.Background(), 0); !errors.Is(err, ErrDiskANNChecksumMismatch) {
		t.Fatalf("record checksum error = %v", err)
	}

	semantic := slices.Clone(encoded)
	spec, _ := layout.readSpec(0)
	recordStart := int(spec.offset) + spec.recordOffset
	degreeOffset := recordStart + layout.dimension*4
	binary.LittleEndian.PutUint32(semantic[degreeOffset:degreeOffset+4], uint32(layout.maxDegree+1))
	refreshDiskANNRecordChecksum(semantic[recordStart : recordStart+layout.recordSize])
	refreshDiskANNDataChecksum(semantic)
	reader, err = OpenDiskANNNodeReader(context.Background(), bytes.NewReader(semantic), int64(len(semantic)), 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := reader.ReadNode(context.Background(), 0); !errors.Is(err, ErrInvalidDiskANNNode) {
		t.Fatalf("semantic corruption error = %v", err)
	}
}

func TestDiskANNHeaderCorruptionTruncationAndCancellation(t *testing.T) {
	layout, nodes, valid := diskANNFixture(t, 6, 3, 3)
	for _, size := range []int{0, 1, diskANNNodeHeaderSize - 1, len(valid) - 1} {
		if _, err := OpenDiskANNNodeReader(context.Background(), bytes.NewReader(valid[:size]), int64(size), 0, 1); err == nil {
			t.Fatalf("truncation %d succeeded", size)
		}
	}
	trailing := append(slices.Clone(valid), 0)
	if _, err := OpenDiskANNNodeReader(context.Background(), bytes.NewReader(trailing), int64(len(trailing)), 0, 1); !errors.Is(err, ErrInvalidDiskANNLayout) {
		t.Fatalf("trailing error = %v", err)
	}
	badMagic := slices.Clone(valid)
	badMagic[0] ^= 1
	if _, err := OpenDiskANNNodeReader(context.Background(), bytes.NewReader(badMagic), int64(len(badMagic)), 0, 1); !errors.Is(err, ErrInvalidDiskANNLayout) {
		t.Fatalf("magic error = %v", err)
	}
	badVersion := slices.Clone(valid)
	binary.LittleEndian.PutUint16(badVersion[8:10], diskANNNodeFileVersion+1)
	if _, err := OpenDiskANNNodeReader(context.Background(), bytes.NewReader(badVersion), int64(len(badVersion)), 0, 1); !errors.Is(err, ErrUnsupportedDiskANNVersion) {
		t.Fatalf("version error = %v", err)
	}
	badHeader := slices.Clone(valid)
	badHeader[48] ^= 1
	if _, err := OpenDiskANNNodeReader(context.Background(), bytes.NewReader(badHeader), int64(len(badHeader)), 0, 1); !errors.Is(err, ErrDiskANNChecksumMismatch) {
		t.Fatalf("header checksum error = %v", err)
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := encodeDiskANNNodeFile(canceled, layout, nodes); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled encode error = %v", err)
	}
	largeLayout, largeNodes, _ := diskANNFixture(t, 300, 3, 3)
	midEncode := newCancelAfterChecks(3)
	if _, err := encodeDiskANNNodeFile(midEncode, largeLayout, largeNodes); !errors.Is(err, context.Canceled) {
		t.Fatalf("mid-encode cancellation error = %v", err)
	}
	if _, err := OpenDiskANNNodeReader(canceled, bytes.NewReader(valid), int64(len(valid)), 0, 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled open error = %v", err)
	}
	reader, err := OpenDiskANNNodeReader(context.Background(), bytes.NewReader(valid), int64(len(valid)), 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := reader.ReadNodes(canceled, []uint32{0}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled read error = %v", err)
	}
}

func TestParallelReadAtOrderingPartialShortAndValidation(t *testing.T) {
	data := []byte("abcdefghijklmnopqrstuvwxyz")
	requests := []DiskANNReadRequest{{Offset: 10, Length: 4}, {Offset: 0, Length: 3}, {Offset: 20, Length: 6}}
	got, err := ParallelReadAt(context.Background(), &partialReaderAt{data: data, maximum: 2}, requests, 3)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, [][]byte{[]byte("klmn"), []byte("abc"), []byte("uvwxyz")}) {
		t.Fatalf("parallel reads = %q", got)
	}
	if _, err := ParallelReadAt(context.Background(), bytes.NewReader(data), []DiskANNReadRequest{{Offset: 24, Length: 4}}, 1); !errors.Is(err, ErrDiskANNShortRead) {
		t.Fatalf("short read error = %v", err)
	}
	if _, err := ParallelReadAt(nil, bytes.NewReader(data), requests, 1); err == nil {
		t.Fatal("nil context succeeded")
	}
	if _, err := ParallelReadAt(context.Background(), nil, requests, 1); err == nil {
		t.Fatal("nil reader succeeded")
	}
	if _, err := ParallelReadAt(context.Background(), bytes.NewReader(data), []DiskANNReadRequest{{Offset: -1, Length: 1}}, 1); err == nil {
		t.Fatal("negative offset succeeded")
	}
}

func TestDiskANNNodeCacheEvictionConcurrencyAndOwnership(t *testing.T) {
	cache, err := NewDiskANNNodeCache(2)
	if err != nil {
		t.Fatal(err)
	}
	nodes := diskANNNodes(4, 2)
	cache.Put(nodes[0])
	cache.Put(nodes[1])
	if _, found := cache.Get(0); !found {
		t.Fatal("missing cached node")
	}
	cache.Put(nodes[2])
	if _, found := cache.Get(1); found {
		t.Fatal("least-recently-used node was not evicted")
	}
	got, found := cache.Get(0)
	if !found {
		t.Fatal("recent node evicted")
	}
	got.Vector[0] = 999
	again, _ := cache.Get(0)
	if again.Vector[0] == 999 {
		t.Fatal("cache returned aliased node")
	}
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
	if cache.Len() > cache.Capacity() || cache.Stats().Evictions == 0 {
		t.Fatalf("cache state = len %d capacity %d stats %#v", cache.Len(), cache.Capacity(), cache.Stats())
	}
	cache.Clear()
	if cache.Len() != 0 {
		t.Fatal("cache clear failed")
	}
	if _, err := NewDiskANNNodeCache(-1); err == nil {
		t.Fatal("negative cache capacity succeeded")
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
		b.Fatal(err)
	}
	if _, err := reader.ReadNode(context.Background(), 50); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, err := reader.ReadNode(context.Background(), 50); err != nil {
			b.Fatal(err)
		}
	}
}

func diskANNFixture(t testing.TB, count, dimension, maxDegree int) (DiskANNLayout, []DiskANNNode, []byte) {
	t.Helper()
	layout, err := NewDiskANNLayout(MetricL2, count, dimension, maxDegree)
	if err != nil {
		t.Fatal(err)
	}
	nodes := diskANNNodes(count, dimension)
	for index := range nodes {
		if count > 1 {
			nodes[index].Neighbors = []uint32{uint32((index + 1) % count)}
		}
	}
	encoded, err := encodeDiskANNNodeFile(context.Background(), layout, nodes)
	if err != nil {
		t.Fatal(err)
	}
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
	binary.LittleEndian.PutUint32(record[len(record)-4:], ailego.CRC32C(record[:len(record)-4]))
}

func refreshDiskANNDataChecksum(encoded []byte) {
	header := encoded[:diskANNNodeHeaderSize]
	data := encoded[diskANNNodeHeaderSize:]
	binary.LittleEndian.PutUint32(header[72:76], ailego.CRC32C(data))
	binary.LittleEndian.PutUint32(header[diskANNNodeHeaderCRCPos:], ailego.CRC32C(header[:diskANNNodeHeaderCRCPos]))
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
