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
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/gorse-io/xvec/internal/ailego/hash"
)

const (
	DiskANNSectorSize       = 4096
	MaxDiskANNReadSectors   = 128
	diskANNNodeFileVersion  = 1
	diskANNNodeHeaderSize   = DiskANNSectorSize
	diskANNNodeHeaderCRCPos = diskANNNodeHeaderSize - 4
)

var (
	diskANNNodeMagic = [8]byte{'Z', 'V', 'E', 'C', 'D', 'A', 'N', 'N'}

	ErrInvalidDiskANNLayout      = errors.New("core: invalid DiskANN layout")
	ErrInvalidDiskANNNode        = errors.New("core: invalid DiskANN node")
	ErrDiskANNChecksumMismatch   = errors.New("core: DiskANN checksum mismatch")
	ErrUnsupportedDiskANNVersion = errors.New("core: unsupported DiskANN format version")
)

// DiskANNLayout describes the sector-aligned random-access node section.
type DiskANNLayout struct {
	metric         Metric
	count          int
	dimension      int
	maxDegree      int
	recordSize     int
	nodesPerSector int
	sectorsPerNode int
	dataOffset     int64
	dataLength     int64
	dataCRC        uint32
}

// NewDiskANNLayout calculates the pinned packed-or-multi-sector node layout.
func NewDiskANNLayout(metric Metric, count, dimension, maxDegree int) (DiskANNLayout, error) {
	if !metric.valid() {
		return DiskANNLayout{}, fmt.Errorf("%w: invalid metric", ErrInvalidDiskANNLayout)
	}
	if count < 0 || uint64(count) > math.MaxUint32 {
		return DiskANNLayout{}, fmt.Errorf("%w: count must fit uint32 node IDs", ErrInvalidDiskANNLayout)
	}
	if dimension <= 0 || dimension > MaxRotationDimension {
		return DiskANNLayout{}, fmt.Errorf("%w: dimension must be in [1,%d]", ErrInvalidDiskANNLayout, MaxRotationDimension)
	}
	if maxDegree <= 0 || uint64(maxDegree) > math.MaxUint32 {
		return DiskANNLayout{}, fmt.Errorf("%w: MaxDegree must fit uint32", ErrInvalidDiskANNLayout)
	}
	record64 := uint64(dimension)*4 + 4 + uint64(maxDegree)*4 + 4
	if record64 > uint64(maxPlatformInt()) || record64 > math.MaxUint32 {
		return DiskANNLayout{}, fmt.Errorf("%w: record exceeds format capacity", ErrInvalidDiskANNLayout)
	}
	recordSize := int(record64)
	nodesPerSector, sectorsPerNode := DiskANNSectorSize/recordSize, 1
	var sectors uint64
	if nodesPerSector > 0 {
		sectors = (uint64(count) + uint64(nodesPerSector) - 1) / uint64(nodesPerSector)
	} else {
		sectorsPerNode = (recordSize + DiskANNSectorSize - 1) / DiskANNSectorSize
		if sectorsPerNode > maxPlatformInt()/DiskANNSectorSize {
			return DiskANNLayout{}, fmt.Errorf("%w: padded record exceeds platform capacity", ErrInvalidDiskANNLayout)
		}
		if uint64(count) > math.MaxUint64/uint64(sectorsPerNode) {
			return DiskANNLayout{}, fmt.Errorf("%w: data sector count overflow", ErrInvalidDiskANNLayout)
		}
		sectors = uint64(count) * uint64(sectorsPerNode)
	}
	if sectors > math.MaxInt64/DiskANNSectorSize {
		return DiskANNLayout{}, fmt.Errorf("%w: data length overflow", ErrInvalidDiskANNLayout)
	}
	return DiskANNLayout{
		metric: metric, count: count, dimension: dimension, maxDegree: maxDegree,
		recordSize: recordSize, nodesPerSector: nodesPerSector, sectorsPerNode: sectorsPerNode,
		dataOffset: diskANNNodeHeaderSize, dataLength: int64(sectors * DiskANNSectorSize),
	}, nil
}

func (l DiskANNLayout) Metric() Metric       { return l.metric }
func (l DiskANNLayout) Count() int           { return l.count }
func (l DiskANNLayout) Dimension() int       { return l.dimension }
func (l DiskANNLayout) MaxDegree() int       { return l.maxDegree }
func (l DiskANNLayout) RecordSize() int      { return l.recordSize }
func (l DiskANNLayout) NodesPerSector() int  { return l.nodesPerSector }
func (l DiskANNLayout) SectorsPerNode() int  { return l.sectorsPerNode }
func (l DiskANNLayout) DataOffset() int64    { return l.dataOffset }
func (l DiskANNLayout) DataLength() int64    { return l.dataLength }
func (l DiskANNLayout) TotalLength() int64   { return l.dataOffset + l.dataLength }
func (l DiskANNLayout) DataChecksum() uint32 { return l.dataCRC }

type diskANNReadSpec struct {
	offset       int64
	length       int
	recordOffset int
}

func (l DiskANNLayout) readSpec(nodeID uint32) (diskANNReadSpec, error) {
	if uint64(nodeID) >= uint64(l.count) {
		return diskANNReadSpec{}, fmt.Errorf("%w: node %d out of range", ErrInvalidDiskANNNode, nodeID)
	}
	if l.nodesPerSector > 0 {
		sector := uint64(nodeID) / uint64(l.nodesPerSector)
		return diskANNReadSpec{
			offset: l.dataOffset + int64(sector*DiskANNSectorSize), length: DiskANNSectorSize,
			recordOffset: int(uint64(nodeID)%uint64(l.nodesPerSector)) * l.recordSize,
		}, nil
	}
	sector := uint64(nodeID) * uint64(l.sectorsPerNode)
	return diskANNReadSpec{
		offset: l.dataOffset + int64(sector*DiskANNSectorSize),
		length: l.sectorsPerNode * DiskANNSectorSize,
	}, nil
}

// DiskANNNode is one original FP32 vector and its bounded outbound graph IDs.
type DiskANNNode struct {
	ID        uint32
	Vector    []float32
	Neighbors []uint32
}

func cloneDiskANNNode(node DiskANNNode) DiskANNNode {
	return DiskANNNode{ID: node.ID, Vector: slices.Clone(node.Vector), Neighbors: slices.Clone(node.Neighbors)}
}

func (l DiskANNLayout) encodeNode(node DiskANNNode) ([]byte, error) {
	if uint64(node.ID) >= uint64(l.count) || len(node.Vector) != l.dimension || len(node.Neighbors) > l.maxDegree {
		return nil, ErrInvalidDiskANNNode
	}
	record := make([]byte, l.recordSize)
	offset := 0
	for _, value := range node.Vector {
		if !finiteFloat32(value) {
			return nil, fmt.Errorf("%w: non-finite vector", ErrInvalidDiskANNNode)
		}
		binary.LittleEndian.PutUint32(record[offset:offset+4], math.Float32bits(value))
		offset += 4
	}
	binary.LittleEndian.PutUint32(record[offset:offset+4], uint32(len(node.Neighbors)))
	offset += 4
	seen := make(map[uint32]struct{}, len(node.Neighbors))
	for _, neighbor := range node.Neighbors {
		if uint64(neighbor) >= uint64(l.count) || neighbor == node.ID {
			return nil, fmt.Errorf("%w: neighbor out of range or self-loop", ErrInvalidDiskANNNode)
		}
		if _, found := seen[neighbor]; found {
			return nil, fmt.Errorf("%w: duplicate neighbor", ErrInvalidDiskANNNode)
		}
		seen[neighbor] = struct{}{}
		binary.LittleEndian.PutUint32(record[offset:offset+4], neighbor)
		offset += 4
	}
	binary.LittleEndian.PutUint32(record[len(record)-4:], hashutil.CRC32C(record[:len(record)-4]))
	return record, nil
}

func (l DiskANNLayout) decodeNode(nodeID uint32, record []byte) (DiskANNNode, error) {
	if uint64(nodeID) >= uint64(l.count) || len(record) < l.recordSize {
		return DiskANNNode{}, ErrInvalidDiskANNNode
	}
	record = record[:l.recordSize]
	if got, want := hashutil.CRC32C(record[:len(record)-4]), binary.LittleEndian.Uint32(record[len(record)-4:]); got != want {
		return DiskANNNode{}, fmt.Errorf("%w: node %d got %08x, want %08x", ErrDiskANNChecksumMismatch, nodeID, got, want)
	}
	node := DiskANNNode{ID: nodeID, Vector: make([]float32, l.dimension)}
	offset := 0
	for index := range node.Vector {
		value := math.Float32frombits(binary.LittleEndian.Uint32(record[offset : offset+4]))
		if !finiteFloat32(value) {
			return DiskANNNode{}, fmt.Errorf("%w: non-finite vector", ErrInvalidDiskANNNode)
		}
		node.Vector[index] = value
		offset += 4
	}
	degree := uint64(binary.LittleEndian.Uint32(record[offset : offset+4]))
	offset += 4
	if degree > uint64(l.maxDegree) {
		return DiskANNNode{}, fmt.Errorf("%w: degree out of range", ErrInvalidDiskANNNode)
	}
	node.Neighbors = make([]uint32, int(degree))
	seen := make(map[uint32]struct{}, len(node.Neighbors))
	for index := range node.Neighbors {
		neighbor := binary.LittleEndian.Uint32(record[offset : offset+4])
		offset += 4
		if uint64(neighbor) >= uint64(l.count) || neighbor == nodeID {
			return DiskANNNode{}, fmt.Errorf("%w: neighbor out of range or self-loop", ErrInvalidDiskANNNode)
		}
		if _, found := seen[neighbor]; found {
			return DiskANNNode{}, fmt.Errorf("%w: duplicate neighbor", ErrInvalidDiskANNNode)
		}
		seen[neighbor] = struct{}{}
		node.Neighbors[index] = neighbor
	}
	for ; offset < len(record)-4; offset++ {
		if record[offset] != 0 {
			return DiskANNNode{}, fmt.Errorf("%w: non-zero node padding", ErrInvalidDiskANNNode)
		}
	}
	return node, nil
}

func encodeDiskANNNodeFile(ctx context.Context, layout DiskANNLayout, nodes []DiskANNNode) ([]byte, error) {
	if ctx == nil {
		return nil, errors.New("core: nil DiskANN encode context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(nodes) != layout.count || layout.TotalLength() > int64(maxPlatformInt()) {
		return nil, ErrInvalidDiskANNLayout
	}
	data := make([]byte, int(layout.dataLength))
	for position, node := range nodes {
		if position&255 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		if node.ID != uint32(position) {
			return nil, fmt.Errorf("%w: nodes must be in ID order", ErrInvalidDiskANNNode)
		}
		record, err := layout.encodeNode(node)
		if err != nil {
			return nil, err
		}
		spec, _ := layout.readSpec(node.ID)
		start := int(spec.offset-layout.dataOffset) + spec.recordOffset
		copy(data[start:start+layout.recordSize], record)
	}
	layout.dataCRC = hashutil.CRC32C(data)
	header := layout.encodeHeader()
	return append(header, data...), nil
}

func (l DiskANNLayout) encodeHeader() []byte {
	header := make([]byte, diskANNNodeHeaderSize)
	copy(header[:8], diskANNNodeMagic[:])
	binary.LittleEndian.PutUint16(header[8:10], diskANNNodeFileVersion)
	binary.LittleEndian.PutUint16(header[10:12], diskANNNodeHeaderSize)
	binary.LittleEndian.PutUint64(header[16:24], uint64(l.TotalLength()))
	binary.LittleEndian.PutUint64(header[24:32], uint64(l.dataOffset))
	binary.LittleEndian.PutUint64(header[32:40], uint64(l.dataLength))
	binary.LittleEndian.PutUint64(header[40:48], uint64(l.count))
	binary.LittleEndian.PutUint32(header[48:52], uint32(l.dimension))
	binary.LittleEndian.PutUint32(header[52:56], uint32(l.maxDegree))
	binary.LittleEndian.PutUint32(header[56:60], uint32(l.recordSize))
	binary.LittleEndian.PutUint32(header[60:64], uint32(l.nodesPerSector))
	binary.LittleEndian.PutUint32(header[64:68], uint32(l.sectorsPerNode))
	header[68] = byte(l.metric)
	binary.LittleEndian.PutUint32(header[72:76], l.dataCRC)
	binary.LittleEndian.PutUint32(header[diskANNNodeHeaderCRCPos:], hashutil.CRC32C(header[:diskANNNodeHeaderCRCPos]))
	return header
}

func decodeDiskANNLayout(header []byte, fileSize int64) (DiskANNLayout, error) {
	if len(header) != diskANNNodeHeaderSize || !bytes.Equal(header[:8], diskANNNodeMagic[:]) {
		return DiskANNLayout{}, fmt.Errorf("%w: bad header", ErrInvalidDiskANNLayout)
	}
	if version := binary.LittleEndian.Uint16(header[8:10]); version != diskANNNodeFileVersion {
		return DiskANNLayout{}, fmt.Errorf("%w: %d", ErrUnsupportedDiskANNVersion, version)
	}
	if binary.LittleEndian.Uint16(header[10:12]) != diskANNNodeHeaderSize || binary.LittleEndian.Uint32(header[12:16]) != 0 ||
		!allZeroBytes(header[69:72]) || !allZeroBytes(header[76:diskANNNodeHeaderCRCPos]) {
		return DiskANNLayout{}, fmt.Errorf("%w: invalid reserved header fields", ErrInvalidDiskANNLayout)
	}
	if got, want := hashutil.CRC32C(header[:diskANNNodeHeaderCRCPos]), binary.LittleEndian.Uint32(header[diskANNNodeHeaderCRCPos:]); got != want {
		return DiskANNLayout{}, fmt.Errorf("%w: header got %08x, want %08x", ErrDiskANNChecksumMismatch, got, want)
	}
	total := binary.LittleEndian.Uint64(header[16:24])
	dataOffset := binary.LittleEndian.Uint64(header[24:32])
	dataLength := binary.LittleEndian.Uint64(header[32:40])
	count := binary.LittleEndian.Uint64(header[40:48])
	if total > math.MaxInt64 || dataOffset != diskANNNodeHeaderSize || dataLength > math.MaxInt64 ||
		total != dataOffset+dataLength || int64(total) != fileSize || count > math.MaxUint32 {
		return DiskANNLayout{}, fmt.Errorf("%w: invalid file lengths or count", ErrInvalidDiskANNLayout)
	}
	layout, err := NewDiskANNLayout(
		Metric(header[68]), int(count), int(binary.LittleEndian.Uint32(header[48:52])),
		int(binary.LittleEndian.Uint32(header[52:56])),
	)
	if err != nil {
		return DiskANNLayout{}, err
	}
	if uint32(layout.recordSize) != binary.LittleEndian.Uint32(header[56:60]) ||
		uint32(layout.nodesPerSector) != binary.LittleEndian.Uint32(header[60:64]) ||
		uint32(layout.sectorsPerNode) != binary.LittleEndian.Uint32(header[64:68]) ||
		uint64(layout.dataLength) != dataLength {
		return DiskANNLayout{}, fmt.Errorf("%w: inconsistent derived layout", ErrInvalidDiskANNLayout)
	}
	layout.dataCRC = binary.LittleEndian.Uint32(header[72:76])
	return layout, nil
}

func allZeroBytes(data []byte) bool {
	for _, value := range data {
		if value != 0 {
			return false
		}
	}
	return true
}
