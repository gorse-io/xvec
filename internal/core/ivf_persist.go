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
	ivfFileVersion    = 1
	ivfHeaderSize     = 112
	ivfReadChunk      = 1 << 20
	ivfRecordOverhead = 12 // key uint64 plus list uint32
)

var (
	ivfFileMagic = [8]byte{'Z', 'V', 'E', 'C', 'I', 'V', 'F', 0}

	// ErrInvalidIVFFile reports a structurally or semantically invalid native
	// Go IVF artifact.
	ErrInvalidIVFFile = errors.New("core: invalid IVF file")
	// ErrIVFChecksumMismatch distinguishes detected bit flips from other
	// format violations.
	ErrIVFChecksumMismatch = errors.New("core: IVF checksum mismatch")
	// ErrUnsupportedIVFVersion reports a well-identified artifact from an
	// unsupported native Go IVF format version.
	ErrUnsupportedIVFVersion = errors.New("core: unsupported IVF file version")
)

// Save durably publishes the immutable index as one checksummed native Go IVF
// file. Replacing an existing file is atomic to concurrent openers.
func (i *IVFIndex) Save(ctx context.Context, path string) error {
	if ctx == nil {
		return errors.New("core: nil IVF save context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if path == "" {
		return fmt.Errorf("%w: empty path", ErrInvalidIVFFile)
	}
	snapshot, err := i.persistenceSnapshot(ctx)
	if err != nil {
		return err
	}
	encoded, err := encodeIVFIndex(ctx, snapshot)
	if err != nil {
		return err
	}
	if err := ailego.WriteFileAtomic(ctx, path, encoded, 0o600); err != nil {
		return fmt.Errorf("core: save IVF file: %w", err)
	}
	return nil
}

func (i *IVFIndex) persistenceSnapshot(ctx context.Context) (*IVFIndex, error) {
	if i == nil {
		return nil, fmt.Errorf("%w: nil index", ErrInvalidIVFFile)
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	snapshot := &IVFIndex{
		dimension:       i.dimension,
		options:         i.options,
		keys:            append([]uint64(nil), i.keys...),
		vectors:         append([]float32(nil), i.vectors...),
		positions:       make(map[uint64]int, len(i.positions)),
		lists:           make([]ivfList, len(i.lists)),
		listForPosition: append([]int(nil), i.listForPosition...),
	}
	for key, position := range i.positions {
		snapshot.positions[key] = position
	}
	for list := range i.lists {
		if list&255 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		snapshot.lists[list].positions = append([]int(nil), i.lists[list].positions...)
	}
	if i.model != nil {
		centroids, err := cloneVectorsContext(ctx, i.model.centroids)
		if err != nil {
			return nil, err
		}
		snapshot.model = &KMeansModel{
			metric:     i.model.metric,
			dimension:  i.model.dimension,
			centroids:  centroids,
			counts:     append([]int(nil), i.model.counts...),
			cost:       i.model.cost,
			iterations: i.model.iterations,
			converged:  i.model.converged,
		}
	}
	return snapshot, nil
}

// OpenIVFIndex reads and fully verifies a native Go IVF artifact. It never
// returns an index backed by the source file.
func OpenIVFIndex(ctx context.Context, path string) (*IVFIndex, error) {
	if ctx == nil {
		return nil, errors.New("core: nil IVF open context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if path == "" {
		return nil, fmt.Errorf("%w: empty path", ErrInvalidIVFFile)
	}
	encoded, err := readIVFFile(ctx, path)
	if err != nil {
		return nil, fmt.Errorf("core: read IVF file: %w", err)
	}
	index, err := decodeIVFIndex(ctx, encoded)
	if err != nil {
		return nil, fmt.Errorf("core: open IVF file: %w", err)
	}
	return index, nil
}

func encodeIVFIndex(ctx context.Context, index *IVFIndex) ([]byte, error) {
	if index == nil {
		return nil, fmt.Errorf("%w: nil index", ErrInvalidIVFFile)
	}
	if err := validateIVFIndex(ctx, index); err != nil {
		return nil, err
	}
	count := len(index.keys)
	nlist := len(index.lists)
	payloadSize, err := checkedIVFPayloadSize(index.dimension, count, nlist)
	if err != nil {
		return nil, err
	}
	payload := make([]byte, 0, payloadSize)
	if index.model != nil {
		for centroidIndex, centroid := range index.model.centroids {
			if centroidIndex&255 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			for _, value := range centroid {
				payload = binary.LittleEndian.AppendUint32(payload, math.Float32bits(value))
			}
		}
	}
	for position, key := range index.keys {
		if position&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		payload = binary.LittleEndian.AppendUint64(payload, key)
		start := position * index.dimension
		for _, value := range index.vectors[start : start+index.dimension] {
			payload = binary.LittleEndian.AppendUint32(payload, math.Float32bits(value))
		}
		payload = binary.LittleEndian.AppendUint32(payload, uint32(index.listForPosition[position]))
	}
	if len(payload) != payloadSize {
		return nil, fmt.Errorf("%w: internal payload length", ErrInvalidIVFFile)
	}

	header := make([]byte, ivfHeaderSize)
	copy(header[:8], ivfFileMagic[:])
	binary.LittleEndian.PutUint16(header[8:10], ivfFileVersion)
	binary.LittleEndian.PutUint16(header[10:12], ivfHeaderSize)
	binary.LittleEndian.PutUint64(header[16:24], uint64(ivfHeaderSize+payloadSize))
	binary.LittleEndian.PutUint64(header[24:32], uint64(payloadSize))
	binary.LittleEndian.PutUint64(header[32:40], uint64(count))
	binary.LittleEndian.PutUint32(header[40:44], uint32(index.dimension))
	binary.LittleEndian.PutUint32(header[44:48], uint32(nlist))
	header[48] = byte(index.options.Metric)
	binary.LittleEndian.PutUint32(header[52:56], uint32(index.options.NList))
	binary.LittleEndian.PutUint32(header[56:60], uint32(index.options.NIterations))
	binary.LittleEndian.PutUint64(header[60:68], uint64(int64(index.options.Workers)))
	binary.LittleEndian.PutUint64(header[68:76], index.options.Seed)
	binary.LittleEndian.PutUint64(header[76:84], math.Float64bits(index.options.Tolerance))
	if index.model != nil {
		if index.model.converged {
			header[49] = 1
		}
		binary.LittleEndian.PutUint64(header[84:92], math.Float64bits(index.model.cost))
		binary.LittleEndian.PutUint32(header[92:96], uint32(index.model.iterations))
	}
	binary.LittleEndian.PutUint32(header[96:100], ailego.CRC32C(payload))
	binary.LittleEndian.PutUint32(header[108:112], ailego.CRC32C(header[:108]))
	return append(header, payload...), nil
}

func decodeIVFIndex(ctx context.Context, encoded []byte) (*IVFIndex, error) {
	if ctx == nil {
		return nil, errors.New("core: nil IVF decode context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(encoded) < ivfHeaderSize {
		return nil, fmt.Errorf("%w: truncated header", ErrInvalidIVFFile)
	}
	header := encoded[:ivfHeaderSize]
	if !bytes.Equal(header[:8], ivfFileMagic[:]) {
		return nil, fmt.Errorf("%w: bad magic", ErrInvalidIVFFile)
	}
	version := binary.LittleEndian.Uint16(header[8:10])
	if version != ivfFileVersion {
		return nil, fmt.Errorf("%w: %d", ErrUnsupportedIVFVersion, version)
	}
	if binary.LittleEndian.Uint16(header[10:12]) != ivfHeaderSize {
		return nil, fmt.Errorf("%w: bad header size", ErrInvalidIVFFile)
	}
	if binary.LittleEndian.Uint32(header[12:16]) != 0 ||
		binary.LittleEndian.Uint16(header[50:52]) != 0 ||
		binary.LittleEndian.Uint64(header[100:108]) != 0 {
		return nil, fmt.Errorf("%w: nonzero reserved field", ErrInvalidIVFFile)
	}
	if got, want := ailego.CRC32C(header[:108]), binary.LittleEndian.Uint32(header[108:112]); got != want {
		return nil, fmt.Errorf("%w: header got %08x, want %08x", ErrIVFChecksumMismatch, got, want)
	}
	if binary.LittleEndian.Uint64(header[16:24]) != uint64(len(encoded)) ||
		binary.LittleEndian.Uint64(header[24:32]) != uint64(len(encoded)-ivfHeaderSize) {
		return nil, fmt.Errorf("%w: inconsistent file length", ErrInvalidIVFFile)
	}

	count64 := binary.LittleEndian.Uint64(header[32:40])
	dimension64 := uint64(binary.LittleEndian.Uint32(header[40:44]))
	nlist64 := uint64(binary.LittleEndian.Uint32(header[44:48]))
	if dimension64 == 0 || dimension64 > MaxRotationDimension {
		return nil, fmt.Errorf("%w: invalid dimension %d", ErrInvalidIVFFile, dimension64)
	}
	if count64 > uint64(maxPlatformInt()) || nlist64 > uint64(maxPlatformInt()) {
		return nil, fmt.Errorf("%w: count exceeds platform capacity", ErrInvalidIVFFile)
	}
	if (count64 == 0 && nlist64 != 0) || (count64 != 0 && (nlist64 == 0 || nlist64 > count64)) {
		return nil, fmt.Errorf("%w: invalid effective list count", ErrInvalidIVFFile)
	}
	dimension, count, nlist := int(dimension64), int(count64), int(nlist64)
	payloadSize, err := checkedIVFPayloadSize(dimension, count, nlist)
	if err != nil || payloadSize != len(encoded)-ivfHeaderSize {
		return nil, fmt.Errorf("%w: invalid payload length", ErrInvalidIVFFile)
	}
	payload := encoded[ivfHeaderSize:]
	if got, want := ailego.CRC32C(payload), binary.LittleEndian.Uint32(header[96:100]); got != want {
		return nil, fmt.Errorf("%w: payload got %08x, want %08x", ErrIVFChecksumMismatch, got, want)
	}

	options, err := decodeIVFOptions(header)
	if err != nil {
		return nil, err
	}
	trainingCost := math.Float64frombits(binary.LittleEndian.Uint64(header[84:92]))
	trainingIterations := uint64(binary.LittleEndian.Uint32(header[92:96]))
	converged := header[49]
	if converged > 1 || math.IsNaN(trainingCost) || math.IsInf(trainingCost, 0) || trainingIterations > uint64(options.NIterations) {
		return nil, fmt.Errorf("%w: invalid training metadata", ErrInvalidIVFFile)
	}
	if count == 0 && (trainingCost != 0 || trainingIterations != 0 || converged != 0) {
		return nil, fmt.Errorf("%w: empty index has training metadata", ErrInvalidIVFFile)
	}

	index := &IVFIndex{
		dimension:       dimension,
		options:         options,
		keys:            make([]uint64, count),
		vectors:         make([]float32, count*dimension),
		positions:       make(map[uint64]int, count),
		lists:           make([]ivfList, nlist),
		listForPosition: make([]int, count),
	}
	offset := 0
	centroids := make([][]float32, nlist)
	for list := 0; list < nlist; list++ {
		if list&255 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		centroid := make([]float32, dimension)
		for component := range centroid {
			value := math.Float32frombits(binary.LittleEndian.Uint32(payload[offset : offset+4]))
			offset += 4
			if !finiteFloat32(value) {
				return nil, fmt.Errorf("%w: non-finite centroid", ErrInvalidIVFFile)
			}
			centroid[component] = value
		}
		centroids[list] = centroid
	}
	counts := make([]int, nlist)
	for position := 0; position < count; position++ {
		if position&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		key := binary.LittleEndian.Uint64(payload[offset : offset+8])
		offset += 8
		if _, duplicate := index.positions[key]; duplicate {
			return nil, fmt.Errorf("%w: duplicate key %d", ErrInvalidIVFFile, key)
		}
		index.keys[position] = key
		index.positions[key] = position
		start := position * dimension
		for component := 0; component < dimension; component++ {
			value := math.Float32frombits(binary.LittleEndian.Uint32(payload[offset : offset+4]))
			offset += 4
			if !finiteFloat32(value) {
				return nil, fmt.Errorf("%w: non-finite vector", ErrInvalidIVFFile)
			}
			index.vectors[start+component] = value
		}
		list := uint64(binary.LittleEndian.Uint32(payload[offset : offset+4]))
		offset += 4
		if list >= nlist64 {
			return nil, fmt.Errorf("%w: vector list %d out of range", ErrInvalidIVFFile, list)
		}
		index.listForPosition[position] = int(list)
		index.lists[list].positions = append(index.lists[list].positions, position)
		counts[list]++
	}
	if offset != len(payload) {
		return nil, fmt.Errorf("%w: unconsumed payload", ErrInvalidIVFFile)
	}
	if count != 0 {
		index.model = &KMeansModel{
			metric:     options.Metric,
			dimension:  dimension,
			centroids:  centroids,
			counts:     counts,
			cost:       trainingCost,
			iterations: int(trainingIterations),
			converged:  converged == 1,
		}
	}
	return index, nil
}

func decodeIVFOptions(header []byte) (IVFBuildOptions, error) {
	workers64 := int64(binary.LittleEndian.Uint64(header[60:68]))
	if int64(int(workers64)) != workers64 {
		return IVFBuildOptions{}, fmt.Errorf("%w: workers exceed platform capacity", ErrInvalidIVFFile)
	}
	options := IVFBuildOptions{
		Metric:      Metric(header[48]),
		NList:       int(binary.LittleEndian.Uint32(header[52:56])),
		NIterations: int(binary.LittleEndian.Uint32(header[56:60])),
		Workers:     int(workers64),
		Seed:        binary.LittleEndian.Uint64(header[68:76]),
		Tolerance:   math.Float64frombits(binary.LittleEndian.Uint64(header[76:84])),
	}
	if err := options.Validate(); err != nil {
		return IVFBuildOptions{}, fmt.Errorf("%w: %v", ErrInvalidIVFFile, err)
	}
	return options, nil
}

func validateIVFIndex(ctx context.Context, index *IVFIndex) error {
	if index.dimension <= 0 || index.dimension > MaxRotationDimension {
		return fmt.Errorf("%w: invalid dimension", ErrInvalidIVFFile)
	}
	if err := index.options.Validate(); err != nil {
		return fmt.Errorf("%w: %v", ErrInvalidIVFFile, err)
	}
	if index.options.NList > math.MaxUint32 || index.options.NIterations > math.MaxUint32 {
		return fmt.Errorf("%w: options exceed format capacity", ErrInvalidIVFFile)
	}
	count := len(index.keys)
	if _, err := checkedIVFPayloadSize(index.dimension, count, len(index.lists)); err != nil {
		return err
	}
	if count > maxPlatformInt()/index.dimension || len(index.vectors) != count*index.dimension || len(index.positions) != count {
		return fmt.Errorf("%w: inconsistent vector storage", ErrInvalidIVFFile)
	}
	if count == 0 {
		if index.model != nil || len(index.lists) != 0 || len(index.listForPosition) != 0 {
			return fmt.Errorf("%w: inconsistent empty index", ErrInvalidIVFFile)
		}
		return nil
	}
	if index.model == nil || index.model.metric != index.options.Metric || index.model.dimension != index.dimension ||
		len(index.model.centroids) != len(index.lists) || len(index.model.counts) != len(index.lists) ||
		len(index.listForPosition) != count || len(index.lists) == 0 || len(index.lists) > count ||
		index.model.iterations < 0 || index.model.iterations > index.options.NIterations ||
		math.IsNaN(index.model.cost) || math.IsInf(index.model.cost, 0) {
		return fmt.Errorf("%w: inconsistent trained index", ErrInvalidIVFFile)
	}
	seen := make([]bool, count)
	counts := make([]int, len(index.lists))
	for list, centroid := range index.model.centroids {
		if len(centroid) != index.dimension {
			return fmt.Errorf("%w: invalid centroid dimension", ErrInvalidIVFFile)
		}
		for _, value := range centroid {
			if !finiteFloat32(value) {
				return fmt.Errorf("%w: non-finite centroid", ErrInvalidIVFFile)
			}
		}
		for _, position := range index.lists[list].positions {
			if position < 0 || position >= count || seen[position] || index.listForPosition[position] != list {
				return fmt.Errorf("%w: invalid list membership", ErrInvalidIVFFile)
			}
			seen[position] = true
			counts[list]++
		}
		if counts[list] != index.model.counts[list] {
			return fmt.Errorf("%w: inconsistent centroid count", ErrInvalidIVFFile)
		}
	}
	for position, key := range index.keys {
		if position&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		if !seen[position] {
			return fmt.Errorf("%w: incomplete list coverage", ErrInvalidIVFFile)
		}
		if mapped, ok := index.positions[key]; !ok || mapped != position {
			return fmt.Errorf("%w: inconsistent key map", ErrInvalidIVFFile)
		}
		start := position * index.dimension
		for _, value := range index.vectors[start : start+index.dimension] {
			if !finiteFloat32(value) {
				return fmt.Errorf("%w: non-finite vector", ErrInvalidIVFFile)
			}
		}
	}
	return nil
}

func checkedIVFPayloadSize(dimension, count, nlist int) (int, error) {
	if dimension <= 0 || count < 0 || nlist < 0 {
		return 0, fmt.Errorf("%w: invalid size", ErrInvalidIVFFile)
	}
	dim := uint64(dimension)
	centroidBytes := uint64(nlist) * dim * 4
	recordBytes := uint64(count) * (ivfRecordOverhead + dim*4)
	if nlist != 0 && centroidBytes/dim/4 != uint64(nlist) ||
		count != 0 && recordBytes/(ivfRecordOverhead+dim*4) != uint64(count) ||
		centroidBytes > math.MaxUint64-recordBytes {
		return 0, fmt.Errorf("%w: payload size overflow", ErrInvalidIVFFile)
	}
	total := centroidBytes + recordBytes
	if total > uint64(maxPlatformInt()-ivfHeaderSize) {
		return 0, fmt.Errorf("%w: payload exceeds platform capacity", ErrInvalidIVFFile)
	}
	return int(total), nil
}

func readIVFFile(ctx context.Context, path string) ([]byte, error) {
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
		return nil, fmt.Errorf("%w: file exceeds platform capacity", ErrInvalidIVFFile)
	}
	encoded := make([]byte, int(info.Size()))
	for offset := 0; offset < len(encoded); {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		end := min(offset+ivfReadChunk, len(encoded))
		if _, err := io.ReadFull(file, encoded[offset:end]); err != nil {
			return nil, err
		}
		offset = end
	}
	return encoded, nil
}

func maxPlatformInt() int { return int(^uint(0) >> 1) }
