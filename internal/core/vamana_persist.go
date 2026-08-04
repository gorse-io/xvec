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
	"fmt"
	"math"

	"github.com/gorse-io/zvec/internal/ailego"
)

const (
	vamanaFileVersion  = 1
	vamanaHeaderSize   = 128
	vamanaFlagSaturate = uint32(1)
)

var (
	vamanaFileMagic = [8]byte{'Z', 'V', 'E', 'C', 'V', 'M', 'N', 'A'}

	ErrInvalidVamanaFile        = errors.New("core: invalid Vamana file")
	ErrVamanaChecksumMismatch   = errors.New("core: Vamana checksum mismatch")
	ErrUnsupportedVamanaVersion = errors.New("core: unsupported Vamana file version")
)

// Save atomically publishes one complete native graph generation.
func (i *VamanaIndex) Save(ctx context.Context, path string) error {
	if ctx == nil {
		return errors.New("core: nil Vamana save context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if path == "" {
		return fmt.Errorf("%w: empty path", ErrInvalidVamanaFile)
	}
	if i == nil {
		return fmt.Errorf("%w: nil index", ErrInvalidVamanaFile)
	}
	i.mu.RLock()
	snapshot, err := cloneVamanaIndex(ctx, i)
	i.mu.RUnlock()
	if err != nil {
		return err
	}
	encoded, err := encodeVamanaIndex(ctx, snapshot)
	if err != nil {
		return err
	}
	if err := ailego.WriteFileAtomic(ctx, path, encoded, 0o600); err != nil {
		return fmt.Errorf("core: save Vamana file: %w", err)
	}
	return nil
}

// OpenVamanaIndex reads and verifies a native Go Vamana artifact.
func OpenVamanaIndex(ctx context.Context, path string) (*VamanaIndex, error) {
	if ctx == nil {
		return nil, errors.New("core: nil Vamana open context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if path == "" {
		return nil, fmt.Errorf("%w: empty path", ErrInvalidVamanaFile)
	}
	encoded, err := readHNSWFile(ctx, path)
	if err != nil {
		return nil, fmt.Errorf("core: read Vamana file: %w", err)
	}
	index, err := decodeVamanaIndex(ctx, encoded)
	if err != nil {
		return nil, fmt.Errorf("core: open Vamana file: %w", err)
	}
	return index, nil
}

func encodeVamanaIndex(ctx context.Context, index *VamanaIndex) ([]byte, error) {
	if ctx == nil {
		return nil, errors.New("core: nil Vamana encode context")
	}
	if err := validateVamanaIndex(ctx, index); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidVamanaFile, err)
	}
	if err := validateVamanaFormatOptions(index.options); err != nil {
		return nil, err
	}
	if uint64(len(index.keys)) > math.MaxUint32 {
		return nil, fmt.Errorf("%w: node count exceeds format capacity", ErrInvalidVamanaFile)
	}
	edgeCount := 0
	for _, adjacent := range index.neighbors {
		if edgeCount > maxPlatformInt()-len(adjacent) {
			return nil, fmt.Errorf("%w: edge count exceeds platform capacity", ErrInvalidVamanaFile)
		}
		edgeCount += len(adjacent)
	}
	payloadSize, err := checkedVamanaPayloadSize(len(index.keys), index.dimension, edgeCount)
	if err != nil {
		return nil, err
	}
	payload := make([]byte, 0, payloadSize)
	for position, key := range index.keys {
		if position&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		payload = binary.LittleEndian.AppendUint64(payload, key)
	}
	for position, value := range index.vectors {
		if position&16383 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		payload = binary.LittleEndian.AppendUint32(payload, math.Float32bits(value))
	}
	for position, adjacent := range index.neighbors {
		if position&255 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		payload = binary.LittleEndian.AppendUint32(payload, uint32(len(adjacent)))
		for _, neighbor := range adjacent {
			payload = binary.LittleEndian.AppendUint32(payload, uint32(neighbor))
		}
	}
	if len(payload) != payloadSize {
		return nil, fmt.Errorf("%w: internal payload length", ErrInvalidVamanaFile)
	}
	header := make([]byte, vamanaHeaderSize)
	copy(header[:8], vamanaFileMagic[:])
	binary.LittleEndian.PutUint16(header[8:10], vamanaFileVersion)
	binary.LittleEndian.PutUint16(header[10:12], vamanaHeaderSize)
	if index.options.SaturateGraph {
		binary.LittleEndian.PutUint32(header[12:16], vamanaFlagSaturate)
	}
	binary.LittleEndian.PutUint64(header[16:24], uint64(vamanaHeaderSize+payloadSize))
	binary.LittleEndian.PutUint64(header[24:32], uint64(payloadSize))
	binary.LittleEndian.PutUint64(header[32:40], uint64(len(index.keys)))
	binary.LittleEndian.PutUint64(header[40:48], uint64(edgeCount))
	binary.LittleEndian.PutUint32(header[48:52], uint32(index.dimension))
	header[52] = byte(index.options.Metric)
	binary.LittleEndian.PutUint32(header[56:60], uint32(index.options.MaxDegree))
	binary.LittleEndian.PutUint32(header[60:64], uint32(index.options.SearchListSize))
	binary.LittleEndian.PutUint32(header[64:68], uint32(index.options.MaxOcclusionSize))
	binary.LittleEndian.PutUint32(header[68:72], math.Float32bits(index.options.Alpha))
	entry := uint64(math.MaxUint64)
	if index.entryPoint >= 0 {
		entry = uint64(index.entryPoint)
	}
	binary.LittleEndian.PutUint64(header[72:80], entry)
	binary.LittleEndian.PutUint32(header[80:84], ailego.CRC32C(payload))
	binary.LittleEndian.PutUint32(header[124:128], ailego.CRC32C(header[:124]))
	return append(header, payload...), nil
}

func decodeVamanaIndex(ctx context.Context, encoded []byte) (*VamanaIndex, error) {
	if ctx == nil {
		return nil, errors.New("core: nil Vamana decode context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(encoded) < vamanaHeaderSize {
		return nil, fmt.Errorf("%w: truncated header", ErrInvalidVamanaFile)
	}
	header := encoded[:vamanaHeaderSize]
	if !bytes.Equal(header[:8], vamanaFileMagic[:]) {
		return nil, fmt.Errorf("%w: bad magic", ErrInvalidVamanaFile)
	}
	version := binary.LittleEndian.Uint16(header[8:10])
	if version != vamanaFileVersion {
		return nil, fmt.Errorf("%w: %d", ErrUnsupportedVamanaVersion, version)
	}
	flags := binary.LittleEndian.Uint32(header[12:16])
	if binary.LittleEndian.Uint16(header[10:12]) != vamanaHeaderSize || flags&^vamanaFlagSaturate != 0 ||
		!hnswAllZero(header[53:56]) || !hnswAllZero(header[84:124]) {
		return nil, fmt.Errorf("%w: invalid header fields", ErrInvalidVamanaFile)
	}
	if got, want := ailego.CRC32C(header[:124]), binary.LittleEndian.Uint32(header[124:128]); got != want {
		return nil, fmt.Errorf("%w: header got %08x, want %08x", ErrVamanaChecksumMismatch, got, want)
	}
	if binary.LittleEndian.Uint64(header[16:24]) != uint64(len(encoded)) ||
		binary.LittleEndian.Uint64(header[24:32]) != uint64(len(encoded)-vamanaHeaderSize) {
		return nil, fmt.Errorf("%w: inconsistent file length", ErrInvalidVamanaFile)
	}
	payload := encoded[vamanaHeaderSize:]
	if got, want := ailego.CRC32C(payload), binary.LittleEndian.Uint32(header[80:84]); got != want {
		return nil, fmt.Errorf("%w: payload got %08x, want %08x", ErrVamanaChecksumMismatch, got, want)
	}
	count64 := binary.LittleEndian.Uint64(header[32:40])
	edgeCount64 := binary.LittleEndian.Uint64(header[40:48])
	dimension64 := uint64(binary.LittleEndian.Uint32(header[48:52]))
	if count64 > math.MaxUint32 {
		return nil, fmt.Errorf("%w: node count exceeds format capacity", ErrInvalidVamanaFile)
	}
	for _, value := range []uint64{count64, edgeCount64, dimension64} {
		if value > uint64(maxPlatformInt()) {
			return nil, fmt.Errorf("%w: field exceeds platform capacity", ErrInvalidVamanaFile)
		}
	}
	count, edgeCount, dimension := int(count64), int(edgeCount64), int(dimension64)
	if dimension <= 0 || dimension > MaxRotationDimension {
		return nil, fmt.Errorf("%w: invalid dimension", ErrInvalidVamanaFile)
	}
	maxDegree64 := uint64(binary.LittleEndian.Uint32(header[56:60]))
	searchListSize64 := uint64(binary.LittleEndian.Uint32(header[60:64]))
	maxOcclusionSize64 := uint64(binary.LittleEndian.Uint32(header[64:68]))
	for _, value := range []uint64{maxDegree64, searchListSize64, maxOcclusionSize64} {
		if value > uint64(maxPlatformInt()) {
			return nil, fmt.Errorf("%w: option exceeds platform capacity", ErrInvalidVamanaFile)
		}
	}
	options := VamanaBuildOptions{
		Metric: Metric(header[52]), MaxDegree: int(maxDegree64),
		SearchListSize:   int(searchListSize64),
		MaxOcclusionSize: int(maxOcclusionSize64),
		Alpha:            math.Float32frombits(binary.LittleEndian.Uint32(header[68:72])), SaturateGraph: flags&vamanaFlagSaturate != 0,
	}
	if err := options.Validate(); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidVamanaFile, err)
	}
	expectedSize, err := checkedVamanaPayloadSize(count, dimension, edgeCount)
	if err != nil || expectedSize != len(payload) {
		return nil, fmt.Errorf("%w: invalid payload size", ErrInvalidVamanaFile)
	}
	entry64 := binary.LittleEndian.Uint64(header[72:80])
	entry := -1
	if count == 0 {
		if entry64 != math.MaxUint64 {
			return nil, fmt.Errorf("%w: empty graph entry point", ErrInvalidVamanaFile)
		}
	} else {
		if entry64 >= uint64(count) {
			return nil, fmt.Errorf("%w: entry point out of range", ErrInvalidVamanaFile)
		}
		entry = int(entry64)
	}
	index := &VamanaIndex{
		dimension: dimension, options: options, keys: make([]uint64, count),
		vectors: make([]float32, count*dimension), positions: make(map[uint64]int, count),
		neighbors: make([][]int, count), neighborDistances: make([][]float32, count), entryPoint: entry,
	}
	offset := 0
	for position := range index.keys {
		if position&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		key := binary.LittleEndian.Uint64(payload[offset : offset+8])
		offset += 8
		if _, found := index.positions[key]; found {
			return nil, fmt.Errorf("%w: duplicate key", ErrInvalidVamanaFile)
		}
		index.keys[position], index.positions[key] = key, position
	}
	for position := range index.vectors {
		if position&16383 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		index.vectors[position] = math.Float32frombits(binary.LittleEndian.Uint32(payload[offset : offset+4]))
		offset += 4
	}
	decodedEdges := 0
	for position := range index.neighbors {
		if position&255 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		degree := uint64(binary.LittleEndian.Uint32(payload[offset : offset+4]))
		offset += 4
		if degree > uint64(options.MaxDegree) || degree > uint64(edgeCount-decodedEdges) || degree > uint64((len(payload)-offset)/4) {
			return nil, fmt.Errorf("%w: invalid degree", ErrInvalidVamanaFile)
		}
		adjacent := make([]int, int(degree))
		for neighborIndex := range adjacent {
			if (decodedEdges+neighborIndex)&16383 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			neighbor := uint64(binary.LittleEndian.Uint32(payload[offset : offset+4]))
			offset += 4
			if neighbor >= uint64(count) {
				return nil, fmt.Errorf("%w: neighbor out of range", ErrInvalidVamanaFile)
			}
			adjacent[neighborIndex] = int(neighbor)
		}
		index.neighbors[position] = adjacent
		decodedEdges += len(adjacent)
	}
	if offset != len(payload) || decodedEdges != edgeCount {
		return nil, fmt.Errorf("%w: inconsistent edge payload", ErrInvalidVamanaFile)
	}
	for position, adjacent := range index.neighbors {
		if position&255 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		distances := make([]float32, len(adjacent))
		for neighborIndex, neighbor := range adjacent {
			distance, err := index.graphDistance(index.vectorAt(position), index.vectorAt(neighbor))
			if err != nil {
				return nil, fmt.Errorf("%w: neighbor distance: %v", ErrInvalidVamanaFile, err)
			}
			distances[neighborIndex] = distance
		}
		index.neighborDistances[position] = distances
	}
	if err := validateVamanaIndex(ctx, index); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidVamanaFile, err)
	}
	return index, nil
}

func validateVamanaFormatOptions(options VamanaBuildOptions) error {
	for _, value := range []int{options.MaxDegree, options.SearchListSize, options.MaxOcclusionSize} {
		if value < 0 || uint64(value) > math.MaxUint32 {
			return fmt.Errorf("%w: options exceed format capacity", ErrInvalidVamanaFile)
		}
	}
	return nil
}

func checkedVamanaPayloadSize(count, dimension, edgeCount int) (int, error) {
	if count < 0 || dimension <= 0 || edgeCount < 0 {
		return 0, fmt.Errorf("%w: invalid payload inputs", ErrInvalidVamanaFile)
	}
	perNode := uint64(12) + uint64(dimension)*4
	if uint64(edgeCount) > math.MaxUint64/4 || (perNode != 0 && uint64(count) > (math.MaxUint64-uint64(edgeCount)*4)/perNode) {
		return 0, fmt.Errorf("%w: payload size overflow", ErrInvalidVamanaFile)
	}
	total := uint64(count)*perNode + uint64(edgeCount)*4
	if total > uint64(maxPlatformInt()-vamanaHeaderSize) {
		return 0, fmt.Errorf("%w: payload exceeds platform capacity", ErrInvalidVamanaFile)
	}
	return int(total), nil
}
