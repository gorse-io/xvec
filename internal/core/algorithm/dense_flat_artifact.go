// Copyright 2026-present the xvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package core

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"math"
	"os"
	"unsafe"

	mmap "github.com/blevesearch/mmap-go"
	iohelper "github.com/gorse-io/xvec/internal/ailego/io"
	mathutil "github.com/gorse-io/xvec/internal/ailego/math"
)

const (
	denseFlatArtifactVersion    = uint32(1)
	denseFlatArtifactHeaderSize = uint64(128)
	denseFlatArtifactCRCOffset  = 96
	denseFlatArtifactWriteChunk = 1 << 20
)

var (
	denseFlatArtifactMagic = [8]byte{'X', 'V', 'F', 'L', 'A', 'T', 0, 1}
	denseFlatCRCTable      = crc32.MakeTable(crc32.Castagnoli)

	// ErrDenseFlatReadOnly is returned by mutation methods on a reopened artifact.
	ErrDenseFlatReadOnly = errors.New("core: dense Flat artifact is read-only")
	// ErrDenseFlatClosed is returned when an operation needs a closed artifact's backing data.
	ErrDenseFlatClosed = errors.New("core: dense Flat artifact is closed")
)

type denseFlatArtifactLayout struct {
	dimension        uint64
	metric           Metric
	count            uint64
	keysOffset       uint64
	keysLength       uint64
	vectorsOffset    uint64
	vectorsLength    uint64
	magnitudesOffset uint64
	magnitudesLength uint64
	totalLength      uint64
	checksum         uint32
}

// Save atomically publishes a complete exact dense Flat artifact.
func (i *DenseFlatIndex) Save(ctx context.Context, path string) error {
	if i == nil {
		return errors.New("core: nil dense Flat index")
	}
	if ctx == nil {
		return errors.New("core: nil dense Flat save context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if !nativeLittleEndian() {
		return errors.New("core: dense Flat artifacts require a little-endian host")
	}

	i.mu.RLock()
	defer i.mu.RUnlock()
	if i.closed {
		return ErrDenseFlatClosed
	}
	layout, err := makeDenseFlatArtifactLayout(i.dimension, i.metric, len(i.keys))
	if err != nil {
		return err
	}
	if layout.totalLength > uint64(maxPlatformInt()) {
		return ErrDenseCapacity
	}
	header := encodeDenseFlatArtifactHeader(layout)
	return iohelper.WriteFileAtomicFunc(ctx, path, 0o600, func(file *os.File) error {
		checksum := crc32.New(denseFlatCRCTable)
		writer := io.MultiWriter(file, checksum)
		if err := writeDenseFlatBytes(ctx, writer, header); err != nil {
			return err
		}
		position := denseFlatArtifactHeaderSize
		if err := writeDenseFlatPadding(ctx, writer, &position, layout.keysOffset); err != nil {
			return err
		}
		if err := writeDenseFlatBytes(ctx, writer, uint64Bytes(i.keys)); err != nil {
			return err
		}
		position += layout.keysLength
		if err := writeDenseFlatPadding(ctx, writer, &position, layout.vectorsOffset); err != nil {
			return err
		}
		if err := writeDenseFlatBytes(ctx, writer, float32Bytes(i.vectors)); err != nil {
			return err
		}
		position += layout.vectorsLength
		if err := writeDenseFlatPadding(ctx, writer, &position, layout.magnitudesOffset); err != nil {
			return err
		}
		if err := writeDenseFlatBytes(ctx, writer, float32Bytes(i.magnitudes)); err != nil {
			return err
		}
		position += layout.magnitudesLength
		if position != layout.totalLength {
			return errors.New("core: inconsistent dense Flat artifact length")
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		var encoded [4]byte
		binary.LittleEndian.PutUint32(encoded[:], checksum.Sum32())
		if _, err := file.WriteAt(encoded[:], denseFlatArtifactCRCOffset); err != nil {
			return fmt.Errorf("core: write dense Flat artifact checksum: %w", err)
		}
		return nil
	})
}

// OpenDenseFlatIndexWithMmap validates and opens an immutable dense Flat artifact.
// With useMmap, keys and FP32 sections directly reference a read-only mapping.
func OpenDenseFlatIndexWithMmap(ctx context.Context, path string, useMmap bool) (*DenseFlatIndex, error) {
	if ctx == nil {
		return nil, errors.New("core: nil dense Flat open context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if !nativeLittleEndian() {
		return nil, errors.New("core: dense Flat artifacts require a little-endian host")
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("core: open dense Flat artifact: %w", err)
	}
	defer func() { _ = file.Close() }()
	info, err := file.Stat()
	if err != nil {
		return nil, fmt.Errorf("core: stat dense Flat artifact: %w", err)
	}
	if !info.Mode().IsRegular() {
		return nil, errors.New("core: dense Flat artifact is not a regular file")
	}
	if info.Size() < int64(denseFlatArtifactHeaderSize) {
		return nil, errors.New("core: truncated dense Flat artifact header")
	}
	var header [denseFlatArtifactHeaderSize]byte
	if _, err := io.ReadFull(file, header[:]); err != nil {
		return nil, fmt.Errorf("core: read dense Flat artifact header: %w", err)
	}
	layout, err := decodeDenseFlatArtifactHeader(header[:], info.Size())
	if err != nil {
		return nil, err
	}
	if layout.totalLength > uint64(maxPlatformInt()) {
		return nil, ErrDenseCapacity
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	var backing []byte
	var mapping mmap.MMap
	if useMmap {
		mapping, err = mmap.MapRegion(file, int(layout.totalLength), mmap.RDONLY, 0, 0)
		if err != nil {
			return nil, fmt.Errorf("core: mmap dense Flat artifact: %w", err)
		}
		backing = mapping
	} else {
		backing = make([]byte, int(layout.totalLength))
		if _, err = file.ReadAt(backing, 0); err != nil {
			return nil, fmt.Errorf("core: read dense Flat artifact: %w", err)
		}
	}
	keep := false
	defer func() {
		if !keep && mapping != nil {
			_ = mapping.Unmap()
		}
	}()
	if err := validateDenseFlatArtifactData(ctx, backing, layout); err != nil {
		return nil, err
	}

	keys := bytesAsUint64(backing[layout.keysOffset : layout.keysOffset+layout.keysLength])
	vectors := bytesAsFloat32(backing[layout.vectorsOffset : layout.vectorsOffset+layout.vectorsLength])
	magnitudes := bytesAsFloat32(backing[layout.magnitudesOffset : layout.magnitudesOffset+layout.magnitudesLength])
	positions := make(map[uint64]int, len(keys))
	for position, key := range keys {
		if position&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		if _, duplicate := positions[key]; duplicate {
			return nil, fmt.Errorf("%w: %d", ErrDuplicateKey, key)
		}
		positions[key] = position
	}
	index := &DenseFlatIndex{
		dimension:  int(layout.dimension),
		metric:     layout.metric,
		keys:       keys,
		vectors:    vectors,
		magnitudes: magnitudes,
		positions:  positions,
		readOnly:   true,
		backing:    backing,
		mapping:    mapping,
	}
	if mapping != nil {
		index.unmap = mapping.Unmap
	}
	keep = true
	return index, nil
}

// Close releases an artifact mapping after all concurrent readers finish.
func (i *DenseFlatIndex) Close() error {
	if i == nil {
		return nil
	}
	i.mu.Lock()
	defer i.mu.Unlock()
	if i.closed {
		return nil
	}
	var unmapErr error
	if i.unmap != nil {
		unmapErr = i.unmap()
	}
	i.keys = nil
	i.vectors = nil
	i.magnitudes = nil
	i.positions = nil
	i.backing = nil
	i.mapping = nil
	i.unmap = nil
	i.closed = true
	if unmapErr != nil {
		return fmt.Errorf("core: unmap dense Flat artifact: %w", unmapErr)
	}
	return nil
}

func makeDenseFlatArtifactLayout(dimension int, metric Metric, count int) (denseFlatArtifactLayout, error) {
	if dimension <= 0 {
		return denseFlatArtifactLayout{}, ErrInvalidDimension
	}
	if !metric.Valid() {
		return denseFlatArtifactLayout{}, errors.New("core: invalid dense Flat artifact metric")
	}
	if count < 0 {
		return denseFlatArtifactLayout{}, ErrDenseCapacity
	}
	vectors, ok := checkedMul64(uint64(count), uint64(dimension))
	if !ok {
		return denseFlatArtifactLayout{}, ErrDenseCapacity
	}
	vectors, ok = checkedMul64(vectors, 4)
	if !ok {
		return denseFlatArtifactLayout{}, ErrDenseCapacity
	}
	keys, ok := checkedMul64(uint64(count), 8)
	if !ok {
		return denseFlatArtifactLayout{}, ErrDenseCapacity
	}
	magnitudes := uint64(0)
	if metric == MetricCosine {
		magnitudes, ok = checkedMul64(uint64(count), 4)
		if !ok {
			return denseFlatArtifactLayout{}, ErrDenseCapacity
		}
	}
	layout := denseFlatArtifactLayout{dimension: uint64(dimension), metric: metric, count: uint64(count)}
	layout.keysOffset = denseFlatArtifactHeaderSize
	layout.keysLength = keys
	layout.vectorsOffset, ok = alignAndAdd(layout.keysOffset, keys, 8)
	if !ok {
		return denseFlatArtifactLayout{}, ErrDenseCapacity
	}
	layout.vectorsLength = vectors
	layout.magnitudesOffset, ok = alignAndAdd(layout.vectorsOffset, vectors, 8)
	if !ok {
		return denseFlatArtifactLayout{}, ErrDenseCapacity
	}
	layout.magnitudesLength = magnitudes
	layout.totalLength, ok = checkedAdd64(layout.magnitudesOffset, magnitudes)
	if !ok {
		return denseFlatArtifactLayout{}, ErrDenseCapacity
	}
	return layout, nil
}

func encodeDenseFlatArtifactHeader(layout denseFlatArtifactLayout) []byte {
	header := make([]byte, denseFlatArtifactHeaderSize)
	copy(header[:8], denseFlatArtifactMagic[:])
	binary.LittleEndian.PutUint32(header[8:12], denseFlatArtifactVersion)
	binary.LittleEndian.PutUint32(header[12:16], uint32(denseFlatArtifactHeaderSize))
	binary.LittleEndian.PutUint64(header[16:24], layout.dimension)
	binary.LittleEndian.PutUint32(header[24:28], uint32(layout.metric))
	binary.LittleEndian.PutUint64(header[32:40], layout.count)
	binary.LittleEndian.PutUint64(header[40:48], layout.keysOffset)
	binary.LittleEndian.PutUint64(header[48:56], layout.keysLength)
	binary.LittleEndian.PutUint64(header[56:64], layout.vectorsOffset)
	binary.LittleEndian.PutUint64(header[64:72], layout.vectorsLength)
	binary.LittleEndian.PutUint64(header[72:80], layout.magnitudesOffset)
	binary.LittleEndian.PutUint64(header[80:88], layout.magnitudesLength)
	binary.LittleEndian.PutUint64(header[88:96], layout.totalLength)
	return header
}

func decodeDenseFlatArtifactHeader(header []byte, fileSize int64) (denseFlatArtifactLayout, error) {
	if len(header) != int(denseFlatArtifactHeaderSize) {
		return denseFlatArtifactLayout{}, errors.New("core: invalid dense Flat artifact header size")
	}
	if string(header[:8]) != string(denseFlatArtifactMagic[:]) {
		return denseFlatArtifactLayout{}, errors.New("core: invalid dense Flat artifact magic")
	}
	if binary.LittleEndian.Uint32(header[8:12]) != denseFlatArtifactVersion {
		return denseFlatArtifactLayout{}, errors.New("core: unsupported dense Flat artifact version")
	}
	if binary.LittleEndian.Uint32(header[12:16]) != uint32(denseFlatArtifactHeaderSize) {
		return denseFlatArtifactLayout{}, errors.New("core: invalid dense Flat artifact header length")
	}
	for _, b := range append(header[28:32:32], header[100:]...) {
		if b != 0 {
			return denseFlatArtifactLayout{}, errors.New("core: nonzero dense Flat artifact reserved field")
		}
	}
	rawMetric := binary.LittleEndian.Uint32(header[24:28])
	if rawMetric > uint32(MetricMIPSL2) {
		return denseFlatArtifactLayout{}, errors.New("core: invalid dense Flat artifact metric")
	}
	layout := denseFlatArtifactLayout{
		dimension:        binary.LittleEndian.Uint64(header[16:24]),
		metric:           Metric(rawMetric),
		count:            binary.LittleEndian.Uint64(header[32:40]),
		keysOffset:       binary.LittleEndian.Uint64(header[40:48]),
		keysLength:       binary.LittleEndian.Uint64(header[48:56]),
		vectorsOffset:    binary.LittleEndian.Uint64(header[56:64]),
		vectorsLength:    binary.LittleEndian.Uint64(header[64:72]),
		magnitudesOffset: binary.LittleEndian.Uint64(header[72:80]),
		magnitudesLength: binary.LittleEndian.Uint64(header[80:88]),
		totalLength:      binary.LittleEndian.Uint64(header[88:96]),
		checksum:         binary.LittleEndian.Uint32(header[96:100]),
	}
	if layout.dimension == 0 || layout.dimension > uint64(maxPlatformInt()) {
		return denseFlatArtifactLayout{}, ErrInvalidDimension
	}
	if !layout.metric.Valid() {
		return denseFlatArtifactLayout{}, errors.New("core: invalid dense Flat artifact metric")
	}
	if layout.count > uint64(maxPlatformInt()) {
		return denseFlatArtifactLayout{}, ErrDenseCapacity
	}
	expected, err := makeDenseFlatArtifactLayout(int(layout.dimension), layout.metric, int(layout.count))
	if err != nil {
		return denseFlatArtifactLayout{}, err
	}
	if layout.keysOffset != expected.keysOffset || layout.keysLength != expected.keysLength ||
		layout.vectorsOffset != expected.vectorsOffset || layout.vectorsLength != expected.vectorsLength ||
		layout.magnitudesOffset != expected.magnitudesOffset || layout.magnitudesLength != expected.magnitudesLength ||
		layout.totalLength != expected.totalLength {
		return denseFlatArtifactLayout{}, errors.New("core: invalid dense Flat artifact section layout")
	}
	if layout.keysOffset%8 != 0 || layout.vectorsOffset%4 != 0 || layout.magnitudesOffset%4 != 0 {
		return denseFlatArtifactLayout{}, errors.New("core: unaligned dense Flat artifact section")
	}
	if layout.totalLength != uint64(fileSize) {
		return denseFlatArtifactLayout{}, errors.New("core: dense Flat artifact length mismatch")
	}
	return layout, nil
}

func validateDenseFlatArtifactData(ctx context.Context, data []byte, layout denseFlatArtifactLayout) error {
	if len(data) != int(layout.totalLength) {
		return errors.New("core: truncated dense Flat artifact")
	}
	checksum := crc32.New(denseFlatCRCTable)
	if err := checksumDenseFlatBytes(ctx, checksum, data[:denseFlatArtifactCRCOffset]); err != nil {
		return err
	}
	_, _ = checksum.Write([]byte{0, 0, 0, 0})
	if err := checksumDenseFlatBytes(ctx, checksum, data[denseFlatArtifactCRCOffset+4:]); err != nil {
		return err
	}
	if checksum.Sum32() != layout.checksum {
		return errors.New("core: dense Flat artifact checksum mismatch")
	}
	vectors := bytesAsFloat32(data[layout.vectorsOffset : layout.vectorsOffset+layout.vectorsLength])
	for offset := 0; offset < len(vectors); offset += int(layout.dimension) {
		if offset&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		if err := mathutil.ValidateDense(vectors[offset:offset+int(layout.dimension)], int(layout.dimension)); err != nil {
			return fmt.Errorf("core: invalid dense Flat artifact vector: %w", err)
		}
	}
	if layout.metric == MetricCosine {
		magnitudes := bytesAsFloat32(data[layout.magnitudesOffset : layout.magnitudesOffset+layout.magnitudesLength])
		for offset, magnitude := range magnitudes {
			if offset&4095 == 0 {
				if err := ctx.Err(); err != nil {
					return err
				}
			}
			if math.IsNaN(float64(magnitude)) || math.IsInf(float64(magnitude), 0) || magnitude < 0 {
				return errors.New("core: invalid dense Flat artifact magnitude")
			}
			start := offset * int(layout.dimension)
			expected := mathutil.L2Magnitude(vectors[start : start+int(layout.dimension)])
			if math.IsNaN(float64(expected)) || math.IsInf(float64(expected), 0) {
				return errors.New("core: invalid dense Flat artifact vector magnitude")
			}
			// Magnitudes may have been produced by a different architecture's
			// accumulation path. Permit only narrow FP32 rounding differences.
			tolerance := 1e-6 + 1e-5*math.Abs(float64(expected))
			if math.Abs(float64(magnitude)-float64(expected)) > tolerance {
				return errors.New("core: inconsistent dense Flat artifact magnitude")
			}
		}
	}
	return ctx.Err()
}

func writeDenseFlatBytes(ctx context.Context, writer io.Writer, data []byte) error {
	for len(data) > 0 {
		if err := ctx.Err(); err != nil {
			return err
		}
		chunk := min(len(data), denseFlatArtifactWriteChunk)
		written, err := writer.Write(data[:chunk])
		if err != nil {
			return fmt.Errorf("core: write dense Flat artifact: %w", err)
		}
		if written != chunk {
			return io.ErrShortWrite
		}
		data = data[chunk:]
	}
	return nil
}

func checksumDenseFlatBytes(ctx context.Context, checksum io.Writer, data []byte) error {
	for len(data) > 0 {
		if err := ctx.Err(); err != nil {
			return err
		}
		chunk := min(len(data), denseFlatArtifactWriteChunk)
		written, err := checksum.Write(data[:chunk])
		if err != nil {
			return fmt.Errorf("core: checksum dense Flat artifact: %w", err)
		}
		if written != chunk {
			return io.ErrShortWrite
		}
		data = data[chunk:]
	}
	return nil
}

func writeDenseFlatPadding(ctx context.Context, writer io.Writer, position *uint64, target uint64) error {
	var padding [8]byte
	if target < *position || target-*position > uint64(len(padding)) {
		return errors.New("core: invalid dense Flat artifact padding")
	}
	if err := writeDenseFlatBytes(ctx, writer, padding[:target-*position]); err != nil {
		return err
	}
	*position = target
	return nil
}

func checkedAdd64(left, right uint64) (uint64, bool) {
	if left > ^uint64(0)-right {
		return 0, false
	}
	return left + right, true
}

func checkedMul64(left, right uint64) (uint64, bool) {
	if left != 0 && right > ^uint64(0)/left {
		return 0, false
	}
	return left * right, true
}

func alignAndAdd(offset, length, alignment uint64) (uint64, bool) {
	end, ok := checkedAdd64(offset, length)
	if !ok {
		return 0, false
	}
	padding := (alignment - end%alignment) % alignment
	return checkedAdd64(end, padding)
}

func nativeLittleEndian() bool {
	value := uint16(1)
	return *(*byte)(unsafe.Pointer(&value)) == 1
}

func uint64Bytes(values []uint64) []byte {
	if len(values) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(values))), len(values)*8)
}

func float32Bytes(values []float32) []byte {
	if len(values) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(values))), len(values)*4)
}

func bytesAsUint64(data []byte) []uint64 {
	if len(data) == 0 {
		return nil
	}
	return unsafe.Slice((*uint64)(unsafe.Pointer(unsafe.SliceData(data))), len(data)/8)
}

func bytesAsFloat32(data []byte) []float32 {
	if len(data) == 0 {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(unsafe.SliceData(data))), len(data)/4)
}
