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
	sparseHNSWFileVersion = 1
	sparseHNSWHeaderSize  = 112
	sparseHNSWReadChunk   = 1 << 20

	sparseHNSWRecordFixedBytes = 16 // key, maximum level, and nonzero count
	sparseHNSWLevelFixedBytes  = 4  // neighbor count
	sparseHNSWElementBytes     = 8  // coordinate and FP32 value
)

var (
	sparseHNSWFileMagic = [8]byte{'Z', 'V', 'S', 'P', 'H', 'N', 'S', 'W'}

	// ErrInvalidSparseHNSWFile reports a structurally or semantically invalid
	// native Go sparse HNSW artifact.
	ErrInvalidSparseHNSWFile = errors.New("core: invalid sparse HNSW file")
	// ErrSparseHNSWChecksumMismatch distinguishes bit flips from other format
	// violations.
	ErrSparseHNSWChecksumMismatch = errors.New("core: sparse HNSW checksum mismatch")
	// ErrUnsupportedSparseHNSWVersion reports an unsupported native Go sparse
	// HNSW format version.
	ErrUnsupportedSparseHNSWVersion = errors.New("core: unsupported sparse HNSW file version")
)

// Save durably publishes one complete graph snapshot as a checksummed native
// Go sparse HNSW file.
func (i *SparseHNSWIndex) Save(ctx context.Context, path string) error {
	if ctx == nil {
		return errors.New("core: nil sparse HNSW save context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if path == "" {
		return fmt.Errorf("%w: empty path", ErrInvalidSparseHNSWFile)
	}
	snapshot, err := i.persistenceSnapshot(ctx)
	if err != nil {
		return err
	}
	encoded, err := encodeSparseHNSWIndex(ctx, snapshot)
	if err != nil {
		return err
	}
	if err := ailego.WriteFileAtomic(ctx, path, encoded, 0o600); err != nil {
		return fmt.Errorf("core: save sparse HNSW file: %w", err)
	}
	return nil
}

func (i *SparseHNSWIndex) persistenceSnapshot(ctx context.Context) (*SparseHNSWIndex, error) {
	if i == nil {
		return nil, fmt.Errorf("%w: nil index", ErrInvalidSparseHNSWFile)
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	return cloneSparseHNSWIndex(ctx, i)
}

// OpenSparseHNSWIndex reads and fully verifies a native Go sparse HNSW
// artifact. The returned graph owns its decoded memory.
func OpenSparseHNSWIndex(ctx context.Context, path string) (*SparseHNSWIndex, error) {
	if ctx == nil {
		return nil, errors.New("core: nil sparse HNSW open context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if path == "" {
		return nil, fmt.Errorf("%w: empty path", ErrInvalidSparseHNSWFile)
	}
	encoded, err := readSparseHNSWFile(ctx, path)
	if err != nil {
		return nil, fmt.Errorf("core: read sparse HNSW file: %w", err)
	}
	index, err := decodeSparseHNSWIndex(ctx, encoded)
	if err != nil {
		return nil, fmt.Errorf("core: open sparse HNSW file: %w", err)
	}
	return index, nil
}

func encodeSparseHNSWIndex(ctx context.Context, index *SparseHNSWIndex) ([]byte, error) {
	if ctx == nil {
		return nil, errors.New("core: nil sparse HNSW encode context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := validateSparseHNSWIndex(ctx, index); err != nil {
		return nil, err
	}
	payloadSize, err := checkedSparseHNSWPayloadSize(index)
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
		start, end := index.offsets[position], index.offsets[position+1]
		payload = binary.LittleEndian.AppendUint32(payload, uint32(end-start))
		for element := start; element < end; element++ {
			payload = binary.LittleEndian.AppendUint32(payload, index.indices[element])
			payload = binary.LittleEndian.AppendUint32(payload, math.Float32bits(index.values[element]))
		}
		for _, neighbors := range index.neighbors[position] {
			payload = binary.LittleEndian.AppendUint32(payload, uint32(len(neighbors)))
			for _, neighbor := range neighbors {
				payload = binary.LittleEndian.AppendUint32(payload, uint32(neighbor))
			}
		}
	}
	if len(payload) != payloadSize {
		return nil, fmt.Errorf("%w: internal payload length", ErrInvalidSparseHNSWFile)
	}

	header := make([]byte, sparseHNSWHeaderSize)
	copy(header[:8], sparseHNSWFileMagic[:])
	binary.LittleEndian.PutUint16(header[8:10], sparseHNSWFileVersion)
	binary.LittleEndian.PutUint16(header[10:12], sparseHNSWHeaderSize)
	binary.LittleEndian.PutUint64(header[16:24], uint64(sparseHNSWHeaderSize+payloadSize))
	binary.LittleEndian.PutUint64(header[24:32], uint64(payloadSize))
	binary.LittleEndian.PutUint64(header[32:40], uint64(len(index.keys)))
	binary.LittleEndian.PutUint64(header[40:48], uint64(len(index.indices)))
	binary.LittleEndian.PutUint32(header[48:52], uint32(index.options.M))
	binary.LittleEndian.PutUint32(header[52:56], uint32(index.options.EFConstruction))
	header[56] = byte(index.options.Metric)
	entryPoint := uint64(math.MaxUint64)
	if index.entryPoint >= 0 {
		entryPoint = uint64(index.entryPoint)
	}
	binary.LittleEndian.PutUint64(header[60:68], entryPoint)
	binary.LittleEndian.PutUint32(header[68:72], uint32(int32(index.maxLevel)))
	binary.LittleEndian.PutUint64(header[72:80], index.options.Seed)
	binary.LittleEndian.PutUint64(header[80:88], index.levelRNGState)
	binary.LittleEndian.PutUint32(header[88:92], ailego.CRC32C(payload))
	binary.LittleEndian.PutUint32(header[108:112], ailego.CRC32C(header[:108]))
	return append(header, payload...), nil
}

func decodeSparseHNSWIndex(ctx context.Context, encoded []byte) (*SparseHNSWIndex, error) {
	if ctx == nil {
		return nil, errors.New("core: nil sparse HNSW decode context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(encoded) < sparseHNSWHeaderSize {
		return nil, fmt.Errorf("%w: truncated header", ErrInvalidSparseHNSWFile)
	}
	header := encoded[:sparseHNSWHeaderSize]
	if !bytes.Equal(header[:8], sparseHNSWFileMagic[:]) {
		return nil, fmt.Errorf("%w: bad magic", ErrInvalidSparseHNSWFile)
	}
	version := binary.LittleEndian.Uint16(header[8:10])
	if version != sparseHNSWFileVersion {
		return nil, fmt.Errorf("%w: %d", ErrUnsupportedSparseHNSWVersion, version)
	}
	if binary.LittleEndian.Uint16(header[10:12]) != sparseHNSWHeaderSize {
		return nil, fmt.Errorf("%w: bad header size", ErrInvalidSparseHNSWFile)
	}
	if binary.LittleEndian.Uint32(header[12:16]) != 0 || !hnswAllZero(header[57:60]) || !hnswAllZero(header[92:108]) {
		return nil, fmt.Errorf("%w: nonzero reserved field", ErrInvalidSparseHNSWFile)
	}
	if got, want := ailego.CRC32C(header[:108]), binary.LittleEndian.Uint32(header[108:112]); got != want {
		return nil, fmt.Errorf("%w: header got %08x, want %08x", ErrSparseHNSWChecksumMismatch, got, want)
	}
	if binary.LittleEndian.Uint64(header[16:24]) != uint64(len(encoded)) ||
		binary.LittleEndian.Uint64(header[24:32]) != uint64(len(encoded)-sparseHNSWHeaderSize) {
		return nil, fmt.Errorf("%w: inconsistent file length", ErrInvalidSparseHNSWFile)
	}
	payload := encoded[sparseHNSWHeaderSize:]
	if got, want := ailego.CRC32C(payload), binary.LittleEndian.Uint32(header[88:92]); got != want {
		return nil, fmt.Errorf("%w: payload got %08x, want %08x", ErrSparseHNSWChecksumMismatch, got, want)
	}

	count64 := binary.LittleEndian.Uint64(header[32:40])
	elements64 := binary.LittleEndian.Uint64(header[40:48])
	if count64 > math.MaxUint32 || count64 > uint64(maxPlatformInt()) || elements64 > uint64(maxPlatformInt()) {
		return nil, fmt.Errorf("%w: counts exceed format capacity", ErrInvalidSparseHNSWFile)
	}
	count, elements := int(count64), int(elements64)
	minimum, err := checkedSparseHNSWMinimumPayloadSize(count, elements)
	if err != nil || minimum > len(payload) {
		return nil, fmt.Errorf("%w: invalid payload length", ErrInvalidSparseHNSWFile)
	}
	options, err := decodeSparseHNSWOptions(header)
	if err != nil {
		return nil, err
	}
	entry64 := binary.LittleEndian.Uint64(header[60:68])
	maxLevel := int(int32(binary.LittleEndian.Uint32(header[68:72])))
	if maxLevel < -1 || maxLevel > MaxHNSWLevel {
		return nil, fmt.Errorf("%w: invalid maximum level", ErrInvalidSparseHNSWFile)
	}
	entryPoint := -1
	if entry64 != math.MaxUint64 {
		if entry64 >= count64 {
			return nil, fmt.Errorf("%w: entry point out of range", ErrInvalidSparseHNSWFile)
		}
		entryPoint = int(entry64)
	}
	if (count == 0 && (entryPoint != -1 || maxLevel != -1 || elements != 0)) ||
		(count != 0 && (entryPoint < 0 || maxLevel < 0)) {
		return nil, fmt.Errorf("%w: inconsistent entry point", ErrInvalidSparseHNSWFile)
	}

	index := &SparseHNSWIndex{
		options:       options,
		keys:          make([]uint64, count),
		offsets:       make([]int, count+1),
		indices:       make([]uint32, 0, elements),
		values:        make([]float32, 0, elements),
		positions:     make(map[uint64]int, count),
		levels:        make([]int, count),
		neighbors:     make([][][]int, count),
		entryPoint:    entryPoint,
		maxLevel:      maxLevel,
		levelRNGState: binary.LittleEndian.Uint64(header[80:88]),
	}
	offset := 0
	for position := 0; position < count; position++ {
		if position&255 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		if !sparseHNSWPayloadAvailable(payload, offset, sparseHNSWRecordFixedBytes+sparseHNSWLevelFixedBytes) {
			return nil, fmt.Errorf("%w: truncated node %d", ErrInvalidSparseHNSWFile, position)
		}
		key := binary.LittleEndian.Uint64(payload[offset : offset+8])
		offset += 8
		if _, duplicate := index.positions[key]; duplicate {
			return nil, fmt.Errorf("%w: duplicate key %d", ErrInvalidSparseHNSWFile, key)
		}
		index.keys[position] = key
		index.positions[key] = position
		level64 := uint64(binary.LittleEndian.Uint32(payload[offset : offset+4]))
		offset += 4
		if level64 > MaxHNSWLevel {
			return nil, fmt.Errorf("%w: invalid node level", ErrInvalidSparseHNSWFile)
		}
		level := int(level64)
		index.levels[position] = level
		index.neighbors[position] = make([][]int, level+1)
		nonzero64 := uint64(binary.LittleEndian.Uint32(payload[offset : offset+4]))
		offset += 4
		if nonzero64 > elements64-uint64(len(index.indices)) || nonzero64 > uint64(maxPlatformInt()) {
			return nil, fmt.Errorf("%w: invalid nonzero count", ErrInvalidSparseHNSWFile)
		}
		nonzero := int(nonzero64)
		if !sparseHNSWPayloadAvailable(payload, offset, nonzero*sparseHNSWElementBytes+sparseHNSWLevelFixedBytes) {
			return nil, fmt.Errorf("%w: truncated sparse vector", ErrInvalidSparseHNSWFile)
		}
		var previous uint32
		for element := 0; element < nonzero; element++ {
			coordinate := binary.LittleEndian.Uint32(payload[offset : offset+4])
			value := math.Float32frombits(binary.LittleEndian.Uint32(payload[offset+4 : offset+8]))
			offset += sparseHNSWElementBytes
			if (element != 0 && coordinate <= previous) || !finiteFloat32(value) {
				return nil, fmt.Errorf("%w: invalid sparse vector", ErrInvalidSparseHNSWFile)
			}
			previous = coordinate
			index.indices = append(index.indices, coordinate)
			index.values = append(index.values, value)
		}
		index.offsets[position+1] = len(index.indices)
		for currentLevel := 0; currentLevel <= level; currentLevel++ {
			if !sparseHNSWPayloadAvailable(payload, offset, sparseHNSWLevelFixedBytes) {
				return nil, fmt.Errorf("%w: truncated neighbors", ErrInvalidSparseHNSWFile)
			}
			degree64 := uint64(binary.LittleEndian.Uint32(payload[offset : offset+4]))
			offset += 4
			limit := options.M
			if currentLevel == 0 {
				limit *= 2
			}
			if degree64 > uint64(limit) || degree64 > count64 {
				return nil, fmt.Errorf("%w: node degree out of range", ErrInvalidSparseHNSWFile)
			}
			degree := int(degree64)
			if !sparseHNSWPayloadAvailable(payload, offset, degree*4) {
				return nil, fmt.Errorf("%w: truncated neighbor positions", ErrInvalidSparseHNSWFile)
			}
			var neighbors []int
			if degree != 0 {
				neighbors = make([]int, degree)
			}
			seen := make(map[int]struct{}, degree)
			for neighborIndex := range neighbors {
				neighbor64 := uint64(binary.LittleEndian.Uint32(payload[offset : offset+4]))
				offset += 4
				if neighbor64 >= count64 || neighbor64 == uint64(position) {
					return nil, fmt.Errorf("%w: invalid neighbor reference", ErrInvalidSparseHNSWFile)
				}
				neighbor := int(neighbor64)
				if _, duplicate := seen[neighbor]; duplicate {
					return nil, fmt.Errorf("%w: duplicate neighbor reference", ErrInvalidSparseHNSWFile)
				}
				seen[neighbor] = struct{}{}
				neighbors[neighborIndex] = neighbor
			}
			index.neighbors[position][currentLevel] = neighbors
		}
	}
	if len(index.indices) != elements || offset != len(payload) {
		return nil, fmt.Errorf("%w: inconsistent payload contents", ErrInvalidSparseHNSWFile)
	}
	if err := validateSparseHNSWIndex(ctx, index); err != nil {
		return nil, err
	}
	return index, nil
}

func decodeSparseHNSWOptions(header []byte) (HNSWBuildOptions, error) {
	m64 := uint64(binary.LittleEndian.Uint32(header[48:52]))
	ef64 := uint64(binary.LittleEndian.Uint32(header[52:56]))
	if m64 > uint64(maxPlatformInt()) || ef64 > uint64(maxPlatformInt()) {
		return HNSWBuildOptions{}, fmt.Errorf("%w: options exceed platform capacity", ErrInvalidSparseHNSWFile)
	}
	options := HNSWBuildOptions{
		Metric:         Metric(header[56]),
		M:              int(m64),
		EFConstruction: int(ef64),
		Seed:           binary.LittleEndian.Uint64(header[72:80]),
	}
	if err := options.Validate(); err != nil || options.Metric != MetricIP {
		return HNSWBuildOptions{}, fmt.Errorf("%w: invalid build options", ErrInvalidSparseHNSWFile)
	}
	return options, nil
}

func validateSparseHNSWIndex(ctx context.Context, index *SparseHNSWIndex) error {
	if ctx == nil {
		return errors.New("core: nil sparse HNSW validation context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if index == nil {
		return fmt.Errorf("%w: nil index", ErrInvalidSparseHNSWFile)
	}
	if err := index.options.Validate(); err != nil || index.options.Metric != MetricIP {
		return fmt.Errorf("%w: invalid build options", ErrInvalidSparseHNSWFile)
	}
	if index.options.M > math.MaxUint32 || index.options.EFConstruction > math.MaxUint32 {
		return fmt.Errorf("%w: options exceed format capacity", ErrInvalidSparseHNSWFile)
	}
	count := len(index.keys)
	if uint64(count) > math.MaxUint32 || len(index.offsets) != count+1 || len(index.indices) != len(index.values) ||
		len(index.positions) != count || len(index.levels) != count || len(index.neighbors) != count ||
		index.offsets[0] != 0 || index.offsets[count] != len(index.indices) {
		return fmt.Errorf("%w: inconsistent graph storage", ErrInvalidSparseHNSWFile)
	}
	if count == 0 {
		if index.entryPoint != -1 || index.maxLevel != -1 || len(index.indices) != 0 {
			return fmt.Errorf("%w: inconsistent empty graph", ErrInvalidSparseHNSWFile)
		}
		return nil
	}
	if index.entryPoint < 0 || index.entryPoint >= count || index.maxLevel < 0 || index.maxLevel > MaxHNSWLevel {
		return fmt.Errorf("%w: invalid graph entry point", ErrInvalidSparseHNSWFile)
	}
	derivedMax := -1
	for position, key := range index.keys {
		if position&255 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		start, end := index.offsets[position], index.offsets[position+1]
		if mapped, found := index.positions[key]; !found || mapped != position ||
			start < 0 || end < start || end > len(index.indices) || uint64(end-start) > math.MaxUint32 {
			return fmt.Errorf("%w: invalid key or CSR offsets", ErrInvalidSparseHNSWFile)
		}
		vector := index.sparseVectorAt(position)
		if _, err := sparseHNSWScore(vector, SparseVector{}); err != nil {
			return fmt.Errorf("%w: invalid sparse vector", ErrInvalidSparseHNSWFile)
		}
		level := index.levels[position]
		if level < 0 || level > MaxHNSWLevel || len(index.neighbors[position]) != level+1 {
			return fmt.Errorf("%w: invalid node level storage", ErrInvalidSparseHNSWFile)
		}
		derivedMax = max(derivedMax, level)
		for currentLevel, neighbors := range index.neighbors[position] {
			limit := index.options.M
			if currentLevel == 0 {
				limit *= 2
			}
			if len(neighbors) > limit {
				return fmt.Errorf("%w: node degree exceeds limit", ErrInvalidSparseHNSWFile)
			}
			seen := make(map[int]struct{}, len(neighbors))
			for _, neighbor := range neighbors {
				if neighbor < 0 || neighbor >= count || neighbor == position || index.levels[neighbor] < currentLevel {
					return fmt.Errorf("%w: invalid neighbor reference", ErrInvalidSparseHNSWFile)
				}
				if _, duplicate := seen[neighbor]; duplicate {
					return fmt.Errorf("%w: duplicate neighbor reference", ErrInvalidSparseHNSWFile)
				}
				seen[neighbor] = struct{}{}
			}
		}
	}
	if derivedMax != index.maxLevel || index.levels[index.entryPoint] != index.maxLevel {
		return fmt.Errorf("%w: inconsistent maximum level", ErrInvalidSparseHNSWFile)
	}
	return nil
}

func checkedSparseHNSWPayloadSize(index *SparseHNSWIndex) (int, error) {
	minimum, err := checkedSparseHNSWMinimumPayloadSize(len(index.keys), len(index.indices))
	if err != nil {
		return 0, err
	}
	total := uint64(minimum)
	for position, level := range index.levels {
		total += uint64(level) * sparseHNSWLevelFixedBytes
		for _, neighbors := range index.neighbors[position] {
			total += uint64(len(neighbors)) * 4
			if total > uint64(maxPlatformInt()-sparseHNSWHeaderSize) {
				return 0, fmt.Errorf("%w: payload exceeds platform capacity", ErrInvalidSparseHNSWFile)
			}
		}
	}
	return int(total), nil
}

func checkedSparseHNSWMinimumPayloadSize(count, elements int) (int, error) {
	if count < 0 || elements < 0 {
		return 0, fmt.Errorf("%w: invalid size", ErrInvalidSparseHNSWFile)
	}
	base := uint64(count) * (sparseHNSWRecordFixedBytes + sparseHNSWLevelFixedBytes)
	values := uint64(elements) * sparseHNSWElementBytes
	if (count != 0 && base/(sparseHNSWRecordFixedBytes+sparseHNSWLevelFixedBytes) != uint64(count)) ||
		(elements != 0 && values/sparseHNSWElementBytes != uint64(elements)) || base > math.MaxUint64-values {
		return 0, fmt.Errorf("%w: payload size overflow", ErrInvalidSparseHNSWFile)
	}
	total := base + values
	if total > uint64(maxPlatformInt()-sparseHNSWHeaderSize) {
		return 0, fmt.Errorf("%w: payload exceeds platform capacity", ErrInvalidSparseHNSWFile)
	}
	return int(total), nil
}

func sparseHNSWPayloadAvailable(payload []byte, offset, size int) bool {
	return offset >= 0 && size >= 0 && offset <= len(payload) && size <= len(payload)-offset
}

func readSparseHNSWFile(ctx context.Context, path string) ([]byte, error) {
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
		return nil, fmt.Errorf("%w: file exceeds platform capacity", ErrInvalidSparseHNSWFile)
	}
	encoded := make([]byte, int(info.Size()))
	for offset := 0; offset < len(encoded); {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		end := min(offset+sparseHNSWReadChunk, len(encoded))
		if _, err := io.ReadFull(file, encoded[offset:end]); err != nil {
			return nil, err
		}
		offset = end
	}
	return encoded, nil
}
