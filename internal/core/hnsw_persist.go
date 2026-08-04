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
	"io"
	"math"
	"os"

	"github.com/gorse-io/zvec/internal/ailego"
)

const (
	hnswFileVersion = 1
	hnswHeaderSize  = 112
	hnswReadChunk   = 1 << 20

	hnswRecordFixedBytes = 12 // key uint64 plus maximum level uint32
	hnswLevelFixedBytes  = 4  // neighbor count uint32
)

var (
	hnswFileMagic = [8]byte{'Z', 'V', 'E', 'C', 'H', 'N', 'S', 'W'}

	// ErrInvalidHNSWFile reports a structurally or semantically invalid native
	// Go HNSW artifact.
	ErrInvalidHNSWFile = errors.New("core: invalid HNSW file")
	// ErrHNSWChecksumMismatch distinguishes detected bit flips from other
	// format violations.
	ErrHNSWChecksumMismatch = errors.New("core: HNSW checksum mismatch")
	// ErrUnsupportedHNSWVersion reports a native Go HNSW artifact whose format
	// version is not supported by this library.
	ErrUnsupportedHNSWVersion = errors.New("core: unsupported HNSW file version")
)

// Save durably publishes the immutable graph as one checksummed native Go
// HNSW file. Replacing an existing file is atomic to concurrent openers.
func (i *HNSWIndex) Save(ctx context.Context, path string) error {
	if ctx == nil {
		return errors.New("core: nil HNSW save context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if path == "" {
		return fmt.Errorf("%w: empty path", ErrInvalidHNSWFile)
	}
	encoded, err := encodeHNSWIndex(ctx, i)
	if err != nil {
		return err
	}
	if err := ailego.WriteFileAtomic(ctx, path, encoded, 0o600); err != nil {
		return fmt.Errorf("core: save HNSW file: %w", err)
	}
	return nil
}

// OpenHNSWIndex reads and fully verifies a native Go HNSW artifact. The
// returned graph owns all decoded memory and does not retain the source file.
func OpenHNSWIndex(ctx context.Context, path string) (*HNSWIndex, error) {
	if ctx == nil {
		return nil, errors.New("core: nil HNSW open context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if path == "" {
		return nil, fmt.Errorf("%w: empty path", ErrInvalidHNSWFile)
	}
	encoded, err := readHNSWFile(ctx, path)
	if err != nil {
		return nil, fmt.Errorf("core: read HNSW file: %w", err)
	}
	index, err := decodeHNSWIndex(ctx, encoded)
	if err != nil {
		return nil, fmt.Errorf("core: open HNSW file: %w", err)
	}
	return index, nil
}

func encodeHNSWIndex(ctx context.Context, index *HNSWIndex) ([]byte, error) {
	if ctx == nil {
		return nil, errors.New("core: nil HNSW encode context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if index == nil {
		return nil, fmt.Errorf("%w: nil index", ErrInvalidHNSWFile)
	}
	if err := validateHNSWIndex(ctx, index); err != nil {
		return nil, err
	}
	payloadSize, err := checkedHNSWPayloadSize(index)
	if err != nil {
		return nil, err
	}
	payload := make([]byte, 0, payloadSize)
	for position, key := range index.keys {
		if position&255 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		payload = binary.LittleEndian.AppendUint64(payload, key)
		payload = binary.LittleEndian.AppendUint32(payload, uint32(index.levels[position]))
		start := position * index.dimension
		for _, value := range index.vectors[start : start+index.dimension] {
			payload = binary.LittleEndian.AppendUint32(payload, math.Float32bits(value))
		}
		for _, neighbors := range index.neighbors[position] {
			payload = binary.LittleEndian.AppendUint32(payload, uint32(len(neighbors)))
			for _, neighbor := range neighbors {
				payload = binary.LittleEndian.AppendUint32(payload, uint32(neighbor))
			}
		}
	}
	if len(payload) != payloadSize {
		return nil, fmt.Errorf("%w: internal payload length", ErrInvalidHNSWFile)
	}

	header := make([]byte, hnswHeaderSize)
	copy(header[:8], hnswFileMagic[:])
	binary.LittleEndian.PutUint16(header[8:10], hnswFileVersion)
	binary.LittleEndian.PutUint16(header[10:12], hnswHeaderSize)
	binary.LittleEndian.PutUint64(header[16:24], uint64(hnswHeaderSize+payloadSize))
	binary.LittleEndian.PutUint64(header[24:32], uint64(payloadSize))
	binary.LittleEndian.PutUint64(header[32:40], uint64(len(index.keys)))
	binary.LittleEndian.PutUint32(header[40:44], uint32(index.dimension))
	binary.LittleEndian.PutUint32(header[44:48], uint32(index.options.M))
	binary.LittleEndian.PutUint32(header[48:52], uint32(index.options.EFConstruction))
	header[52] = byte(index.options.Metric)
	entryPoint := uint64(math.MaxUint64)
	if index.entryPoint >= 0 {
		entryPoint = uint64(index.entryPoint)
	}
	binary.LittleEndian.PutUint64(header[56:64], entryPoint)
	binary.LittleEndian.PutUint32(header[64:68], uint32(int32(index.maxLevel)))
	binary.LittleEndian.PutUint64(header[68:76], index.options.Seed)
	binary.LittleEndian.PutUint64(header[76:84], index.levelRNGState)
	binary.LittleEndian.PutUint32(header[84:88], ailego.CRC32C(payload))
	binary.LittleEndian.PutUint32(header[108:112], ailego.CRC32C(header[:108]))
	return append(header, payload...), nil
}

func decodeHNSWIndex(ctx context.Context, encoded []byte) (*HNSWIndex, error) {
	if ctx == nil {
		return nil, errors.New("core: nil HNSW decode context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(encoded) < hnswHeaderSize {
		return nil, fmt.Errorf("%w: truncated header", ErrInvalidHNSWFile)
	}
	header := encoded[:hnswHeaderSize]
	if !bytes.Equal(header[:8], hnswFileMagic[:]) {
		return nil, fmt.Errorf("%w: bad magic", ErrInvalidHNSWFile)
	}
	version := binary.LittleEndian.Uint16(header[8:10])
	if version != hnswFileVersion {
		return nil, fmt.Errorf("%w: %d", ErrUnsupportedHNSWVersion, version)
	}
	if binary.LittleEndian.Uint16(header[10:12]) != hnswHeaderSize {
		return nil, fmt.Errorf("%w: bad header size", ErrInvalidHNSWFile)
	}
	if binary.LittleEndian.Uint32(header[12:16]) != 0 ||
		!hnswAllZero(header[53:56]) ||
		!hnswAllZero(header[88:108]) {
		return nil, fmt.Errorf("%w: nonzero reserved field", ErrInvalidHNSWFile)
	}
	if got, want := ailego.CRC32C(header[:108]), binary.LittleEndian.Uint32(header[108:112]); got != want {
		return nil, fmt.Errorf("%w: header got %08x, want %08x", ErrHNSWChecksumMismatch, got, want)
	}
	if binary.LittleEndian.Uint64(header[16:24]) != uint64(len(encoded)) ||
		binary.LittleEndian.Uint64(header[24:32]) != uint64(len(encoded)-hnswHeaderSize) {
		return nil, fmt.Errorf("%w: inconsistent file length", ErrInvalidHNSWFile)
	}
	payload := encoded[hnswHeaderSize:]
	if got, want := ailego.CRC32C(payload), binary.LittleEndian.Uint32(header[84:88]); got != want {
		return nil, fmt.Errorf("%w: payload got %08x, want %08x", ErrHNSWChecksumMismatch, got, want)
	}

	count64 := binary.LittleEndian.Uint64(header[32:40])
	dimension64 := uint64(binary.LittleEndian.Uint32(header[40:44]))
	if count64 > uint64(math.MaxUint32) || count64 > uint64(maxPlatformInt()) {
		return nil, fmt.Errorf("%w: node count exceeds format capacity", ErrInvalidHNSWFile)
	}
	if dimension64 == 0 || dimension64 > MaxRotationDimension {
		return nil, fmt.Errorf("%w: invalid dimension %d", ErrInvalidHNSWFile, dimension64)
	}
	count, dimension := int(count64), int(dimension64)
	minimumSize, err := checkedHNSWMinimumPayloadSize(dimension, count)
	if err != nil || minimumSize > len(payload) {
		return nil, fmt.Errorf("%w: invalid payload length", ErrInvalidHNSWFile)
	}
	options, err := decodeHNSWOptions(header)
	if err != nil {
		return nil, err
	}
	entry64 := binary.LittleEndian.Uint64(header[56:64])
	maxLevel := int(int32(binary.LittleEndian.Uint32(header[64:68])))
	if maxLevel < -1 || maxLevel > MaxHNSWLevel {
		return nil, fmt.Errorf("%w: invalid maximum level", ErrInvalidHNSWFile)
	}
	entryPoint := -1
	if entry64 != math.MaxUint64 {
		if entry64 >= count64 {
			return nil, fmt.Errorf("%w: entry point out of range", ErrInvalidHNSWFile)
		}
		entryPoint = int(entry64)
	}
	if (count == 0 && (entryPoint != -1 || maxLevel != -1)) ||
		(count != 0 && (entryPoint < 0 || maxLevel < 0)) {
		return nil, fmt.Errorf("%w: inconsistent entry point", ErrInvalidHNSWFile)
	}
	if count > maxPlatformInt()/dimension {
		return nil, fmt.Errorf("%w: vector storage exceeds platform capacity", ErrInvalidHNSWFile)
	}

	index := &HNSWIndex{
		dimension:     dimension,
		options:       options,
		keys:          make([]uint64, count),
		vectors:       make([]float32, count*dimension),
		positions:     make(map[uint64]int, count),
		levels:        make([]int, count),
		neighbors:     make([][][]int, count),
		entryPoint:    entryPoint,
		maxLevel:      maxLevel,
		levelRNGState: binary.LittleEndian.Uint64(header[76:84]),
	}
	offset := 0
	for position := 0; position < count; position++ {
		if position&255 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		vectorBytes := dimension * 4
		if !hnswPayloadAvailable(payload, offset, hnswRecordFixedBytes+vectorBytes+hnswLevelFixedBytes) {
			return nil, fmt.Errorf("%w: truncated node %d", ErrInvalidHNSWFile, position)
		}
		key := binary.LittleEndian.Uint64(payload[offset : offset+8])
		offset += 8
		if _, duplicate := index.positions[key]; duplicate {
			return nil, fmt.Errorf("%w: duplicate key %d", ErrInvalidHNSWFile, key)
		}
		index.keys[position] = key
		index.positions[key] = position
		level := uint64(binary.LittleEndian.Uint32(payload[offset : offset+4]))
		offset += 4
		if level > MaxHNSWLevel {
			return nil, fmt.Errorf("%w: invalid node level %d", ErrInvalidHNSWFile, level)
		}
		index.levels[position] = int(level)
		index.neighbors[position] = make([][]int, int(level)+1)
		start := position * dimension
		for component := 0; component < dimension; component++ {
			value := math.Float32frombits(binary.LittleEndian.Uint32(payload[offset : offset+4]))
			offset += 4
			if !finiteFloat32(value) {
				return nil, fmt.Errorf("%w: non-finite vector", ErrInvalidHNSWFile)
			}
			index.vectors[start+component] = value
		}
		for currentLevel := 0; currentLevel <= int(level); currentLevel++ {
			if !hnswPayloadAvailable(payload, offset, hnswLevelFixedBytes) {
				return nil, fmt.Errorf("%w: truncated neighbors", ErrInvalidHNSWFile)
			}
			degree64 := uint64(binary.LittleEndian.Uint32(payload[offset : offset+4]))
			offset += 4
			degreeLimit := options.M
			if currentLevel == 0 {
				degreeLimit *= 2
			}
			if degree64 > uint64(degreeLimit) || degree64 > count64 {
				return nil, fmt.Errorf("%w: node %d level %d degree out of range", ErrInvalidHNSWFile, position, currentLevel)
			}
			degree := int(degree64)
			if !hnswPayloadAvailable(payload, offset, degree*4) {
				return nil, fmt.Errorf("%w: truncated neighbor positions", ErrInvalidHNSWFile)
			}
			neighbors := make([]int, degree)
			seen := make(map[int]struct{}, degree)
			for neighborIndex := range neighbors {
				neighbor64 := uint64(binary.LittleEndian.Uint32(payload[offset : offset+4]))
				offset += 4
				if neighbor64 >= count64 || neighbor64 == uint64(position) {
					return nil, fmt.Errorf("%w: invalid neighbor reference", ErrInvalidHNSWFile)
				}
				neighbor := int(neighbor64)
				if _, duplicate := seen[neighbor]; duplicate {
					return nil, fmt.Errorf("%w: duplicate neighbor reference", ErrInvalidHNSWFile)
				}
				seen[neighbor] = struct{}{}
				neighbors[neighborIndex] = neighbor
			}
			index.neighbors[position][currentLevel] = neighbors
		}
	}
	if offset != len(payload) {
		return nil, fmt.Errorf("%w: trailing payload data", ErrInvalidHNSWFile)
	}
	if err := validateHNSWIndex(ctx, index); err != nil {
		return nil, err
	}
	return index, nil
}

func decodeHNSWOptions(header []byte) (HNSWBuildOptions, error) {
	m64 := uint64(binary.LittleEndian.Uint32(header[44:48]))
	ef64 := uint64(binary.LittleEndian.Uint32(header[48:52]))
	if m64 > uint64(maxPlatformInt()) || ef64 > uint64(maxPlatformInt()) {
		return HNSWBuildOptions{}, fmt.Errorf("%w: options exceed platform capacity", ErrInvalidHNSWFile)
	}
	options := HNSWBuildOptions{
		Metric:         Metric(header[52]),
		M:              int(m64),
		EFConstruction: int(ef64),
		Seed:           binary.LittleEndian.Uint64(header[68:76]),
	}
	if err := options.Validate(); err != nil {
		return HNSWBuildOptions{}, fmt.Errorf("%w: %v", ErrInvalidHNSWFile, err)
	}
	return options, nil
}

func validateHNSWIndex(ctx context.Context, index *HNSWIndex) error {
	if ctx == nil {
		return errors.New("core: nil HNSW validation context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if index == nil {
		return fmt.Errorf("%w: nil index", ErrInvalidHNSWFile)
	}
	if index.dimension <= 0 || index.dimension > MaxRotationDimension {
		return fmt.Errorf("%w: invalid dimension", ErrInvalidHNSWFile)
	}
	if err := index.options.Validate(); err != nil {
		return fmt.Errorf("%w: %v", ErrInvalidHNSWFile, err)
	}
	if index.options.M > math.MaxUint32 || index.options.EFConstruction > math.MaxUint32 {
		return fmt.Errorf("%w: options exceed format capacity", ErrInvalidHNSWFile)
	}
	count := len(index.keys)
	if uint64(count) > math.MaxUint32 || count > maxPlatformInt()/index.dimension ||
		len(index.vectors) != count*index.dimension || len(index.positions) != count ||
		len(index.levels) != count || len(index.neighbors) != count {
		return fmt.Errorf("%w: inconsistent graph storage", ErrInvalidHNSWFile)
	}
	if count == 0 {
		if index.entryPoint != -1 || index.maxLevel != -1 {
			return fmt.Errorf("%w: inconsistent empty graph", ErrInvalidHNSWFile)
		}
		return nil
	}
	if index.entryPoint < 0 || index.entryPoint >= count || index.maxLevel < 0 || index.maxLevel > MaxHNSWLevel {
		return fmt.Errorf("%w: invalid graph entry point", ErrInvalidHNSWFile)
	}
	derivedMaxLevel := -1
	for position, key := range index.keys {
		if position&255 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		if mapped, found := index.positions[key]; !found || mapped != position {
			return fmt.Errorf("%w: inconsistent key map", ErrInvalidHNSWFile)
		}
		level := index.levels[position]
		if level < 0 || level > MaxHNSWLevel || len(index.neighbors[position]) != level+1 {
			return fmt.Errorf("%w: invalid node level storage", ErrInvalidHNSWFile)
		}
		derivedMaxLevel = max(derivedMaxLevel, level)
		start := position * index.dimension
		for _, value := range index.vectors[start : start+index.dimension] {
			if !finiteFloat32(value) {
				return fmt.Errorf("%w: non-finite vector", ErrInvalidHNSWFile)
			}
		}
		for currentLevel, neighbors := range index.neighbors[position] {
			degreeLimit := index.options.M
			if currentLevel == 0 {
				degreeLimit *= 2
			}
			if len(neighbors) > degreeLimit {
				return fmt.Errorf("%w: node degree exceeds limit", ErrInvalidHNSWFile)
			}
			seen := make(map[int]struct{}, len(neighbors))
			for _, neighbor := range neighbors {
				if neighbor < 0 || neighbor >= count || neighbor == position || index.levels[neighbor] < currentLevel {
					return fmt.Errorf("%w: invalid neighbor reference", ErrInvalidHNSWFile)
				}
				if _, duplicate := seen[neighbor]; duplicate {
					return fmt.Errorf("%w: duplicate neighbor reference", ErrInvalidHNSWFile)
				}
				seen[neighbor] = struct{}{}
			}
		}
	}
	if derivedMaxLevel != index.maxLevel || index.levels[index.entryPoint] != index.maxLevel {
		return fmt.Errorf("%w: inconsistent maximum level", ErrInvalidHNSWFile)
	}
	return nil
}

func checkedHNSWPayloadSize(index *HNSWIndex) (int, error) {
	minimum, err := checkedHNSWMinimumPayloadSize(index.dimension, len(index.keys))
	if err != nil {
		return 0, err
	}
	total := uint64(minimum)
	for position, level := range index.levels {
		// The minimum already includes one level-count field for every node.
		total += uint64(level) * hnswLevelFixedBytes
		for _, neighbors := range index.neighbors[position] {
			total += uint64(len(neighbors)) * 4
			if total > uint64(maxPlatformInt()-hnswHeaderSize) {
				return 0, fmt.Errorf("%w: payload exceeds platform capacity", ErrInvalidHNSWFile)
			}
		}
	}
	return int(total), nil
}

func checkedHNSWMinimumPayloadSize(dimension, count int) (int, error) {
	if dimension <= 0 || count < 0 {
		return 0, fmt.Errorf("%w: invalid size", ErrInvalidHNSWFile)
	}
	recordBytes := uint64(hnswRecordFixedBytes+hnswLevelFixedBytes) + uint64(dimension)*4
	total := uint64(count) * recordBytes
	if count != 0 && total/recordBytes != uint64(count) {
		return 0, fmt.Errorf("%w: payload size overflow", ErrInvalidHNSWFile)
	}
	if total > uint64(maxPlatformInt()-hnswHeaderSize) {
		return 0, fmt.Errorf("%w: payload exceeds platform capacity", ErrInvalidHNSWFile)
	}
	return int(total), nil
}

func hnswPayloadAvailable(payload []byte, offset, size int) bool {
	return offset >= 0 && size >= 0 && offset <= len(payload) && size <= len(payload)-offset
}

func readHNSWFile(ctx context.Context, path string) ([]byte, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return nil, err
	}
	if info.Size() < 0 || uint64(info.Size()) > uint64(maxPlatformInt()) {
		return nil, fmt.Errorf("%w: file exceeds platform capacity", ErrInvalidHNSWFile)
	}
	encoded := make([]byte, int(info.Size()))
	for offset := 0; offset < len(encoded); {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		end := min(offset+hnswReadChunk, len(encoded))
		if _, err := io.ReadFull(file, encoded[offset:end]); err != nil {
			return nil, err
		}
		offset = end
	}
	return encoded, nil
}

func hnswAllZero(data []byte) bool {
	for _, value := range data {
		if value != 0 {
			return false
		}
	}
	return true
}
