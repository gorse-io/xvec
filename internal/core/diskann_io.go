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
	"errors"
	"fmt"
	"io"

	"github.com/gorse-io/xvec/internal/ailego/hash"
	"github.com/gorse-io/xvec/internal/ailego/parallel"
)

var ErrDiskANNShortRead = errors.New("core: short DiskANN ReaderAt read")

// DiskANNReadRequest describes one exact random read.
type DiskANNReadRequest struct {
	Offset int64
	Length int
}

// ParallelReadAt executes exact ReaderAt requests concurrently and preserves
// request order. It is portable across regular files and custom ReaderAt
// implementations on Linux, macOS, and Windows.
func ParallelReadAt(ctx context.Context, reader io.ReaderAt, requests []DiskANNReadRequest, workers int) ([][]byte, error) {
	if ctx == nil {
		return nil, errors.New("core: nil parallel ReaderAt context")
	}
	if reader == nil {
		return nil, errors.New("core: nil ReaderAt")
	}
	if workers < 0 {
		return nil, errors.New("core: negative ReaderAt worker count")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	result := make([][]byte, len(requests))
	err := parallel.ParallelFor(ctx, len(requests), workers, func(ctx context.Context, index int) error {
		request := requests[index]
		if request.Offset < 0 || request.Length <= 0 {
			return fmt.Errorf("core: invalid ReaderAt request %d", index)
		}
		buffer := make([]byte, request.Length)
		if err := readFullAt(ctx, reader, buffer, request.Offset); err != nil {
			return fmt.Errorf("core: ReaderAt request %d: %w", index, err)
		}
		result[index] = buffer
		return nil
	})
	if err != nil {
		return nil, err
	}
	return result, nil
}

func readFullAt(ctx context.Context, reader io.ReaderAt, buffer []byte, offset int64) error {
	read := 0
	for read < len(buffer) {
		if err := ctx.Err(); err != nil {
			return err
		}
		n, err := reader.ReadAt(buffer[read:], offset+int64(read))
		if n < 0 || n > len(buffer)-read {
			return ErrDiskANNShortRead
		}
		read += n
		if read == len(buffer) {
			return nil
		}
		if err != nil {
			return fmt.Errorf("%w: %v", ErrDiskANNShortRead, err)
		}
		if n == 0 {
			return ErrDiskANNShortRead
		}
	}
	return nil
}

// DiskANNNodeReader validates one complete node artifact, then serves
// cache-aware batched random reads.
type DiskANNNodeReader struct {
	reader  io.ReaderAt
	layout  DiskANNLayout
	cache   *DiskANNNodeCache
	workers int
}

func OpenDiskANNNodeReader(ctx context.Context, reader io.ReaderAt, fileSize int64, cacheCapacity, workers int) (*DiskANNNodeReader, error) {
	if ctx == nil {
		return nil, errors.New("core: nil DiskANN open context")
	}
	if reader == nil || fileSize < diskANNNodeHeaderSize || cacheCapacity < 0 || workers < 0 {
		return nil, ErrInvalidDiskANNLayout
	}
	header := make([]byte, diskANNNodeHeaderSize)
	if err := readFullAt(ctx, reader, header, 0); err != nil {
		return nil, err
	}
	layout, err := decodeDiskANNLayout(header, fileSize)
	if err != nil {
		return nil, err
	}
	if err := verifyDiskANNData(ctx, reader, layout); err != nil {
		return nil, err
	}
	cache, err := NewDiskANNNodeCache(cacheCapacity)
	if err != nil {
		return nil, err
	}
	return &DiskANNNodeReader{reader: reader, layout: layout, cache: cache, workers: workers}, nil
}

func (r *DiskANNNodeReader) Layout() DiskANNLayout {
	if r == nil {
		return DiskANNLayout{}
	}
	return r.layout
}

func (r *DiskANNNodeReader) CacheStats() DiskANNCacheStats {
	if r == nil {
		return DiskANNCacheStats{}
	}
	return r.cache.Stats()
}

func (r *DiskANNNodeReader) ReadNode(ctx context.Context, nodeID uint32) (DiskANNNode, error) {
	nodes, err := r.ReadNodes(ctx, []uint32{nodeID})
	if err != nil {
		return DiskANNNode{}, err
	}
	return nodes[0], nil
}

func (r *DiskANNNodeReader) ReadNodes(ctx context.Context, nodeIDs []uint32) ([]DiskANNNode, error) {
	if r == nil || r.reader == nil {
		return nil, ErrInvalidDiskANNLayout
	}
	if ctx == nil {
		return nil, errors.New("core: nil DiskANN read context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	result := make([]DiskANNNode, len(nodeIDs))
	type group struct {
		spec    diskANNReadSpec
		indices []int
	}
	groups := make([]group, 0)
	groupForOffset := make(map[[2]int64]int)
	for index, nodeID := range nodeIDs {
		if node, found := r.cache.Get(nodeID); found {
			result[index] = node
			continue
		}
		spec, err := r.layout.readSpec(nodeID)
		if err != nil {
			return nil, err
		}
		key := [2]int64{spec.offset, int64(spec.length)}
		groupIndex, found := groupForOffset[key]
		if !found {
			groupIndex = len(groups)
			groupForOffset[key] = groupIndex
			groups = append(groups, group{spec: spec})
		}
		groups[groupIndex].indices = append(groups[groupIndex].indices, index)
	}
	requests := make([]DiskANNReadRequest, len(groups))
	for index, group := range groups {
		requests[index] = DiskANNReadRequest{Offset: group.spec.offset, Length: group.spec.length}
	}
	blocks, err := ParallelReadAt(ctx, r.reader, requests, r.workers)
	if err != nil {
		return nil, err
	}
	for groupIndex, group := range groups {
		for _, resultIndex := range group.indices {
			nodeID := nodeIDs[resultIndex]
			spec, _ := r.layout.readSpec(nodeID)
			node, err := r.layout.decodeNode(nodeID, blocks[groupIndex][spec.recordOffset:])
			if err != nil {
				return nil, err
			}
			result[resultIndex] = node
			r.cache.Put(node)
		}
	}
	return result, nil
}

func verifyDiskANNData(ctx context.Context, reader io.ReaderAt, layout DiskANNLayout) error {
	const chunkSize = 1 << 20
	bufferSize := chunkSize
	if layout.dataLength < chunkSize {
		bufferSize = int(layout.dataLength)
	}
	buffer := make([]byte, bufferSize)
	var crc uint32
	for offset := int64(0); offset < layout.dataLength; {
		if err := ctx.Err(); err != nil {
			return err
		}
		length := min(len(buffer), int(layout.dataLength-offset))
		if err := readFullAt(ctx, reader, buffer[:length], layout.dataOffset+offset); err != nil {
			return err
		}
		crc = hashutil.UpdateCRC32C(crc, buffer[:length])
		offset += int64(length)
	}
	if crc != layout.dataCRC {
		return fmt.Errorf("%w: data got %08x, want %08x", ErrDiskANNChecksumMismatch, crc, layout.dataCRC)
	}
	return nil
}
