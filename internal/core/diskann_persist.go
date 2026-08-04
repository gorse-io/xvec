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
	diskANNIndexFileVersion  = 1
	diskANNIndexHeaderSize   = DiskANNSectorSize
	diskANNIndexHeaderCRCPos = diskANNIndexHeaderSize - 4
)

var (
	diskANNIndexMagic = [8]byte{'Z', 'V', 'E', 'C', 'D', 'I', 'D', 'X'}

	ErrInvalidDiskANNFile             = errors.New("core: invalid DiskANN index file")
	ErrDiskANNIndexChecksumMismatch   = errors.New("core: DiskANN index checksum mismatch")
	ErrUnsupportedDiskANNIndexVersion = errors.New("core: unsupported DiskANN index version")
	errDiskANNIndexSectionChecksum    = errors.New("core: DiskANN index section checksum mismatch")
)

type diskANNIndexSections struct {
	totalLength   int64
	keysOffset    int64
	keysLength    int64
	offsetsOffset int64
	offsetsLength int64
	pivotsOffset  int64
	pivotsLength  int64
	codesOffset   int64
	codesLength   int64
	nodesOffset   int64
	nodesLength   int64
}

type diskANNIndexHeader struct {
	count            int
	dimension        int
	metric           Metric
	traversalMetric  Metric
	maxDegree        int
	listSize         int
	configuredChunks int
	actualChunks     int
	entryPoint       int
	sections         diskANNIndexSections
	keysCRC          uint32
	offsetsCRC       uint32
	pivotsCRC        uint32
	codesCRC         uint32
	nodesCRC         uint32
}

// Save atomically publishes a complete native DiskANN artifact.
func (i *DiskANNIndex) Save(ctx context.Context, path string) error {
	if ctx == nil {
		return errors.New("core: nil DiskANN save context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if path == "" {
		return fmt.Errorf("%w: empty path", ErrInvalidDiskANNFile)
	}
	if i == nil {
		return fmt.Errorf("%w: nil index", ErrInvalidDiskANNFile)
	}
	i.closeMu.RLock()
	defer i.closeMu.RUnlock()
	if i.closed {
		return ErrDiskANNClosed
	}
	encoded, err := encodeDiskANNIndex(ctx, i)
	if err != nil {
		return err
	}
	if err := ailego.WriteFileAtomic(ctx, path, encoded, 0o600); err != nil {
		return fmt.Errorf("core: save DiskANN file: %w", err)
	}
	return nil
}

// OpenDiskANNIndex opens and validates a complete artifact. cacheCapacity zero
// disables node caching; workers zero lets the shared parallel helper choose.
func OpenDiskANNIndex(ctx context.Context, path string, cacheCapacity, workers int) (*DiskANNIndex, error) {
	if ctx == nil {
		return nil, errors.New("core: nil DiskANN open context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if path == "" {
		return nil, fmt.Errorf("%w: empty path", ErrInvalidDiskANNFile)
	}
	if cacheCapacity < 0 || workers < 0 {
		return nil, fmt.Errorf("%w: negative runtime option", ErrInvalidDiskANNOptions)
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("core: open DiskANN file: %w", err)
	}
	info, err := file.Stat()
	if err != nil {
		_ = file.Close()
		return nil, fmt.Errorf("core: stat DiskANN file: %w", err)
	}
	index, err := openDiskANNIndexReader(ctx, file, info.Size(), cacheCapacity, workers, file)
	if err != nil {
		_ = file.Close()
		return nil, err
	}
	return index, nil
}

func encodeDiskANNIndex(ctx context.Context, index *DiskANNIndex) ([]byte, error) {
	if ctx == nil {
		return nil, errors.New("core: nil DiskANN encode context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := validateDiskANNIndex(ctx, index); err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return nil, err
		}
		return nil, fmt.Errorf("%w: %v", ErrInvalidDiskANNFile, err)
	}
	nodeLength := index.nodes.layout.TotalLength()
	if nodeLength < 0 || nodeLength > int64(maxPlatformInt()) {
		return nil, fmt.Errorf("%w: node artifact exceeds platform capacity", ErrInvalidDiskANNFile)
	}
	nodeArtifact := make([]byte, int(nodeLength))
	if err := readFullAt(ctx, index.nodes.reader, nodeArtifact, 0); err != nil {
		return nil, fmt.Errorf("core: snapshot DiskANN node artifact: %w", err)
	}
	actualChunks := 0
	if index.pq != nil {
		actualChunks = index.pq.Chunks()
	}
	sections, err := calculateDiskANNIndexSections(len(index.keys), index.dimension, actualChunks, nodeLength)
	if err != nil {
		return nil, err
	}
	if sections.totalLength > int64(maxPlatformInt()) {
		return nil, fmt.Errorf("%w: artifact exceeds platform capacity", ErrInvalidDiskANNFile)
	}
	encoded := make([]byte, int(sections.totalLength))
	for position, key := range index.keys {
		if position&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		binary.LittleEndian.PutUint64(encoded[int(sections.keysOffset)+position*8:], key)
	}
	if index.pq != nil {
		state := index.pq.State()
		for position, offset := range state.ChunkOffsets {
			binary.LittleEndian.PutUint32(encoded[int(sections.offsetsOffset)+position*4:], uint32(offset))
		}
		for position, pivot := range state.Pivots {
			if position&16383 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			binary.LittleEndian.PutUint32(encoded[int(sections.pivotsOffset)+position*4:], math.Float32bits(pivot))
		}
		copy(encoded[sections.codesOffset:sections.codesOffset+sections.codesLength], index.codes)
	}
	copy(encoded[sections.nodesOffset:sections.nodesOffset+sections.nodesLength], nodeArtifact)
	header := diskANNIndexHeader{
		count: len(index.keys), dimension: index.dimension, metric: index.metric,
		traversalMetric: index.traversalMetric, maxDegree: index.options.MaxDegree,
		listSize: index.options.ListSize, configuredChunks: index.options.PQChunks,
		actualChunks: actualChunks, entryPoint: index.entryPoint, sections: sections,
		keysCRC:    ailego.CRC32C(encoded[sections.keysOffset : sections.keysOffset+sections.keysLength]),
		offsetsCRC: ailego.CRC32C(encoded[sections.offsetsOffset : sections.offsetsOffset+sections.offsetsLength]),
		pivotsCRC:  ailego.CRC32C(encoded[sections.pivotsOffset : sections.pivotsOffset+sections.pivotsLength]),
		codesCRC:   ailego.CRC32C(encoded[sections.codesOffset : sections.codesOffset+sections.codesLength]),
		nodesCRC:   ailego.CRC32C(nodeArtifact),
	}
	copy(encoded[:diskANNIndexHeaderSize], encodeDiskANNIndexHeader(header))
	return encoded, nil
}

func encodeDiskANNIndexHeader(meta diskANNIndexHeader) []byte {
	header := make([]byte, diskANNIndexHeaderSize)
	copy(header[:8], diskANNIndexMagic[:])
	binary.LittleEndian.PutUint16(header[8:10], diskANNIndexFileVersion)
	binary.LittleEndian.PutUint16(header[10:12], diskANNIndexHeaderSize)
	binary.LittleEndian.PutUint64(header[16:24], uint64(meta.sections.totalLength))
	binary.LittleEndian.PutUint64(header[24:32], uint64(meta.count))
	binary.LittleEndian.PutUint32(header[32:36], uint32(meta.dimension))
	header[36], header[37] = byte(meta.metric), byte(meta.traversalMetric)
	binary.LittleEndian.PutUint32(header[40:44], uint32(meta.maxDegree))
	binary.LittleEndian.PutUint32(header[44:48], uint32(meta.listSize))
	binary.LittleEndian.PutUint32(header[48:52], uint32(meta.configuredChunks))
	binary.LittleEndian.PutUint32(header[52:56], uint32(meta.actualChunks))
	entry := uint64(math.MaxUint64)
	if meta.entryPoint >= 0 {
		entry = uint64(meta.entryPoint)
	}
	binary.LittleEndian.PutUint64(header[56:64], entry)
	binary.LittleEndian.PutUint64(header[64:72], uint64(meta.sections.keysOffset))
	binary.LittleEndian.PutUint64(header[72:80], uint64(meta.sections.keysLength))
	binary.LittleEndian.PutUint64(header[80:88], uint64(meta.sections.offsetsOffset))
	binary.LittleEndian.PutUint64(header[88:96], uint64(meta.sections.offsetsLength))
	binary.LittleEndian.PutUint64(header[96:104], uint64(meta.sections.pivotsOffset))
	binary.LittleEndian.PutUint64(header[104:112], uint64(meta.sections.pivotsLength))
	binary.LittleEndian.PutUint64(header[112:120], uint64(meta.sections.codesOffset))
	binary.LittleEndian.PutUint64(header[120:128], uint64(meta.sections.codesLength))
	binary.LittleEndian.PutUint64(header[128:136], uint64(meta.sections.nodesOffset))
	binary.LittleEndian.PutUint64(header[136:144], uint64(meta.sections.nodesLength))
	binary.LittleEndian.PutUint32(header[144:148], meta.keysCRC)
	binary.LittleEndian.PutUint32(header[148:152], meta.offsetsCRC)
	binary.LittleEndian.PutUint32(header[152:156], meta.pivotsCRC)
	binary.LittleEndian.PutUint32(header[156:160], meta.codesCRC)
	binary.LittleEndian.PutUint32(header[160:164], meta.nodesCRC)
	binary.LittleEndian.PutUint32(header[diskANNIndexHeaderCRCPos:], ailego.CRC32C(header[:diskANNIndexHeaderCRCPos]))
	return header
}

func openDiskANNIndexReader(
	ctx context.Context,
	reader io.ReaderAt,
	fileSize int64,
	cacheCapacity, workers int,
	closer io.Closer,
) (*DiskANNIndex, error) {
	if ctx == nil {
		return nil, errors.New("core: nil DiskANN reader-open context")
	}
	if reader == nil || fileSize < diskANNIndexHeaderSize || cacheCapacity < 0 || workers < 0 {
		return nil, ErrInvalidDiskANNFile
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	headerBytes := make([]byte, diskANNIndexHeaderSize)
	if err := readFullAt(ctx, reader, headerBytes, 0); err != nil {
		return nil, err
	}
	meta, err := decodeDiskANNIndexHeader(headerBytes, fileSize)
	if err != nil {
		return nil, err
	}
	checks := []struct {
		offset, length int64
		checksum       uint32
		name           string
	}{
		{meta.sections.keysOffset, meta.sections.keysLength, meta.keysCRC, "keys"},
		{meta.sections.offsetsOffset, meta.sections.offsetsLength, meta.offsetsCRC, "chunk offsets"},
		{meta.sections.pivotsOffset, meta.sections.pivotsLength, meta.pivotsCRC, "pivots"},
		{meta.sections.codesOffset, meta.sections.codesLength, meta.codesCRC, "codes"},
		{meta.sections.nodesOffset, meta.sections.nodesLength, meta.nodesCRC, "nodes"},
	}
	for _, check := range checks {
		if err := verifyDiskANNIndexSection(ctx, reader, check.offset, check.length, check.checksum); err != nil {
			if errors.Is(err, errDiskANNIndexSectionChecksum) {
				return nil, fmt.Errorf("%w: %s: %v", ErrDiskANNIndexChecksumMismatch, check.name, err)
			}
			return nil, err
		}
	}
	metadataEnd := meta.sections.codesOffset + meta.sections.codesLength
	if meta.sections.nodesOffset > metadataEnd {
		padding, err := readDiskANNIndexSection(ctx, reader, metadataEnd, meta.sections.nodesOffset-metadataEnd)
		if err != nil {
			return nil, err
		}
		if !allZeroBytes(padding) {
			return nil, fmt.Errorf("%w: non-zero alignment padding", ErrInvalidDiskANNFile)
		}
	}

	keysBytes, err := readDiskANNIndexSection(ctx, reader, meta.sections.keysOffset, meta.sections.keysLength)
	if err != nil {
		return nil, err
	}
	keys := make([]uint64, meta.count)
	positions := make(map[uint64]int, meta.count)
	for position := range keys {
		if position&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		key := binary.LittleEndian.Uint64(keysBytes[position*8:])
		if _, duplicate := positions[key]; duplicate {
			return nil, fmt.Errorf("%w: duplicate key", ErrInvalidDiskANNFile)
		}
		keys[position], positions[key] = key, position
	}

	var model *PQModel
	var codes []byte
	if meta.count != 0 {
		offsetBytes, err := readDiskANNIndexSection(ctx, reader, meta.sections.offsetsOffset, meta.sections.offsetsLength)
		if err != nil {
			return nil, err
		}
		offsets := make([]int, meta.actualChunks+1)
		for position := range offsets {
			value := uint64(binary.LittleEndian.Uint32(offsetBytes[position*4:]))
			if value > uint64(maxPlatformInt()) {
				return nil, fmt.Errorf("%w: chunk offset exceeds platform capacity", ErrInvalidDiskANNFile)
			}
			offsets[position] = int(value)
		}
		pivotBytes, err := readDiskANNIndexSection(ctx, reader, meta.sections.pivotsOffset, meta.sections.pivotsLength)
		if err != nil {
			return nil, err
		}
		pivots := make([]float32, len(pivotBytes)/4)
		for position := range pivots {
			if position&16383 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			pivots[position] = math.Float32frombits(binary.LittleEndian.Uint32(pivotBytes[position*4:]))
		}
		model, err = RestorePQModel(PQModelState{
			Dimension: meta.dimension, Metric: meta.traversalMetric,
			ChunkOffsets: offsets, Pivots: pivots,
		})
		if err != nil {
			return nil, fmt.Errorf("%w: restore PQ model: %v", ErrInvalidDiskANNFile, err)
		}
		codes, err = readDiskANNIndexSection(ctx, reader, meta.sections.codesOffset, meta.sections.codesLength)
		if err != nil {
			return nil, err
		}
	}
	nodeSection := io.NewSectionReader(reader, meta.sections.nodesOffset, meta.sections.nodesLength)
	nodeReader, err := OpenDiskANNNodeReader(ctx, nodeSection, meta.sections.nodesLength, cacheCapacity, workers)
	if err != nil {
		return nil, fmt.Errorf("core: open DiskANN node section: %w", err)
	}
	index := &DiskANNIndex{
		closer: closer, dimension: meta.dimension, metric: meta.metric,
		traversalMetric: meta.traversalMetric,
		options: DiskANNBuildOptions{
			Metric: meta.metric, MaxDegree: meta.maxDegree, ListSize: meta.listSize,
			PQChunks: meta.configuredChunks, Workers: workers, CacheCapacity: cacheCapacity,
		},
		keys: keys, positions: positions, entryPoint: meta.entryPoint,
		pq: model, codes: codes, nodes: nodeReader,
	}
	if meta.metric == MetricMIPSL2 && model != nil {
		index.codeNorms, err = diskANNPQCodeNorms(ctx, model, codes, meta.count)
		if err != nil {
			return nil, err
		}
	}
	if err := validateDiskANNIndex(ctx, index); err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return nil, err
		}
		return nil, fmt.Errorf("%w: %v", ErrInvalidDiskANNFile, err)
	}
	return index, nil
}

func decodeDiskANNIndexHeader(header []byte, fileSize int64) (diskANNIndexHeader, error) {
	if len(header) != diskANNIndexHeaderSize || !bytes.Equal(header[:8], diskANNIndexMagic[:]) {
		return diskANNIndexHeader{}, fmt.Errorf("%w: bad header", ErrInvalidDiskANNFile)
	}
	if version := binary.LittleEndian.Uint16(header[8:10]); version != diskANNIndexFileVersion {
		return diskANNIndexHeader{}, fmt.Errorf("%w: %d", ErrUnsupportedDiskANNIndexVersion, version)
	}
	if binary.LittleEndian.Uint16(header[10:12]) != diskANNIndexHeaderSize || binary.LittleEndian.Uint32(header[12:16]) != 0 ||
		!allZeroBytes(header[38:40]) || !allZeroBytes(header[164:diskANNIndexHeaderCRCPos]) {
		return diskANNIndexHeader{}, fmt.Errorf("%w: invalid reserved header fields", ErrInvalidDiskANNFile)
	}
	if got, want := ailego.CRC32C(header[:diskANNIndexHeaderCRCPos]), binary.LittleEndian.Uint32(header[diskANNIndexHeaderCRCPos:]); got != want {
		return diskANNIndexHeader{}, fmt.Errorf("%w: header got %08x, want %08x", ErrDiskANNIndexChecksumMismatch, got, want)
	}
	total := binary.LittleEndian.Uint64(header[16:24])
	count64 := binary.LittleEndian.Uint64(header[24:32])
	if total > math.MaxInt64 || int64(total) != fileSize || count64 > math.MaxUint32 || count64 > uint64(maxPlatformInt()) {
		return diskANNIndexHeader{}, fmt.Errorf("%w: invalid total length or count", ErrInvalidDiskANNFile)
	}
	values := []uint32{
		binary.LittleEndian.Uint32(header[32:36]), binary.LittleEndian.Uint32(header[40:44]),
		binary.LittleEndian.Uint32(header[44:48]), binary.LittleEndian.Uint32(header[48:52]),
		binary.LittleEndian.Uint32(header[52:56]),
	}
	for _, value := range values {
		if uint64(value) > uint64(maxPlatformInt()) {
			return diskANNIndexHeader{}, fmt.Errorf("%w: header value exceeds platform capacity", ErrInvalidDiskANNFile)
		}
	}
	meta := diskANNIndexHeader{
		count: int(count64), dimension: int(values[0]), metric: Metric(header[36]), traversalMetric: Metric(header[37]),
		maxDegree: int(values[1]), listSize: int(values[2]), configuredChunks: int(values[3]), actualChunks: int(values[4]),
		keysCRC: binary.LittleEndian.Uint32(header[144:148]), offsetsCRC: binary.LittleEndian.Uint32(header[148:152]),
		pivotsCRC: binary.LittleEndian.Uint32(header[152:156]), codesCRC: binary.LittleEndian.Uint32(header[156:160]),
		nodesCRC: binary.LittleEndian.Uint32(header[160:164]),
	}
	options := DiskANNBuildOptions{
		Metric: meta.metric, MaxDegree: meta.maxDegree, ListSize: meta.listSize, PQChunks: meta.configuredChunks,
	}
	if err := options.Validate(); err != nil || meta.dimension <= 0 || meta.dimension > MaxRotationDimension || meta.configuredChunks > meta.dimension {
		return diskANNIndexHeader{}, fmt.Errorf("%w: invalid options", ErrInvalidDiskANNFile)
	}
	expectedTraversal := diskANNTraversalMetric(meta.metric)
	if meta.traversalMetric != expectedTraversal {
		return diskANNIndexHeader{}, fmt.Errorf("%w: invalid traversal metric", ErrInvalidDiskANNFile)
	}
	entry64 := binary.LittleEndian.Uint64(header[56:64])
	if meta.count == 0 {
		if entry64 != math.MaxUint64 || meta.actualChunks != 0 {
			return diskANNIndexHeader{}, fmt.Errorf("%w: invalid empty state", ErrInvalidDiskANNFile)
		}
		meta.entryPoint = -1
	} else {
		if entry64 >= uint64(meta.count) || meta.actualChunks <= 0 || meta.actualChunks > meta.dimension ||
			(meta.configuredChunks != 0 && meta.configuredChunks != meta.actualChunks) {
			return diskANNIndexHeader{}, fmt.Errorf("%w: invalid entry point or PQ chunks", ErrInvalidDiskANNFile)
		}
		meta.entryPoint = int(entry64)
	}
	nodeLength64 := binary.LittleEndian.Uint64(header[136:144])
	if nodeLength64 > math.MaxInt64 {
		return diskANNIndexHeader{}, fmt.Errorf("%w: node length overflow", ErrInvalidDiskANNFile)
	}
	sections, err := calculateDiskANNIndexSections(meta.count, meta.dimension, meta.actualChunks, int64(nodeLength64))
	if err != nil {
		return diskANNIndexHeader{}, err
	}
	serialized := []uint64{
		binary.LittleEndian.Uint64(header[64:72]), binary.LittleEndian.Uint64(header[72:80]),
		binary.LittleEndian.Uint64(header[80:88]), binary.LittleEndian.Uint64(header[88:96]),
		binary.LittleEndian.Uint64(header[96:104]), binary.LittleEndian.Uint64(header[104:112]),
		binary.LittleEndian.Uint64(header[112:120]), binary.LittleEndian.Uint64(header[120:128]),
		binary.LittleEndian.Uint64(header[128:136]), nodeLength64,
	}
	expected := []int64{
		sections.keysOffset, sections.keysLength, sections.offsetsOffset, sections.offsetsLength,
		sections.pivotsOffset, sections.pivotsLength, sections.codesOffset, sections.codesLength,
		sections.nodesOffset, sections.nodesLength,
	}
	for position := range expected {
		if serialized[position] > math.MaxInt64 || int64(serialized[position]) != expected[position] {
			return diskANNIndexHeader{}, fmt.Errorf("%w: inconsistent section layout", ErrInvalidDiskANNFile)
		}
	}
	if sections.totalLength != fileSize {
		return diskANNIndexHeader{}, fmt.Errorf("%w: inconsistent file length", ErrInvalidDiskANNFile)
	}
	meta.sections = sections
	return meta, nil
}

func calculateDiskANNIndexSections(count, dimension, chunks int, nodeLength int64) (diskANNIndexSections, error) {
	if count < 0 || dimension <= 0 || chunks < 0 || nodeLength < diskANNNodeHeaderSize {
		return diskANNIndexSections{}, fmt.Errorf("%w: invalid section inputs", ErrInvalidDiskANNFile)
	}
	if (count == 0 && chunks != 0) || (count > 0 && (chunks <= 0 || chunks > dimension)) {
		return diskANNIndexSections{}, fmt.Errorf("%w: invalid PQ section inputs", ErrInvalidDiskANNFile)
	}
	checkedProduct := func(left, right uint64) (int64, error) {
		if right != 0 && left > math.MaxUint64/right {
			return 0, ErrInvalidDiskANNFile
		}
		value := left * right
		if value > math.MaxInt64 {
			return 0, ErrInvalidDiskANNFile
		}
		return int64(value), nil
	}
	keysLength, err := checkedProduct(uint64(count), 8)
	if err != nil {
		return diskANNIndexSections{}, fmt.Errorf("%w: keys length overflow", ErrInvalidDiskANNFile)
	}
	offsetsLength, pivotsLength, codesLength := int64(0), int64(0), int64(0)
	if count != 0 {
		offsetsLength, err = checkedProduct(uint64(chunks+1), 4)
		if err == nil {
			pivotsLength, err = checkedProduct(uint64(PQCentroidCount)*uint64(dimension), 4)
		}
		if err == nil {
			codesLength, err = checkedProduct(uint64(count), uint64(chunks))
		}
		if err != nil {
			return diskANNIndexSections{}, fmt.Errorf("%w: PQ length overflow", ErrInvalidDiskANNFile)
		}
	}
	sections := diskANNIndexSections{keysOffset: diskANNIndexHeaderSize, keysLength: keysLength}
	sections.offsetsOffset = sections.keysOffset + sections.keysLength
	sections.offsetsLength = offsetsLength
	sections.pivotsOffset = sections.offsetsOffset + sections.offsetsLength
	sections.pivotsLength = pivotsLength
	sections.codesOffset = sections.pivotsOffset + sections.pivotsLength
	sections.codesLength = codesLength
	metadataEnd := sections.codesOffset + sections.codesLength
	if metadataEnd < 0 || metadataEnd > math.MaxInt64-(DiskANNSectorSize-1) {
		return diskANNIndexSections{}, fmt.Errorf("%w: metadata length overflow", ErrInvalidDiskANNFile)
	}
	sections.nodesOffset = (metadataEnd + DiskANNSectorSize - 1) / DiskANNSectorSize * DiskANNSectorSize
	sections.nodesLength = nodeLength
	if nodeLength > math.MaxInt64-sections.nodesOffset {
		return diskANNIndexSections{}, fmt.Errorf("%w: total length overflow", ErrInvalidDiskANNFile)
	}
	sections.totalLength = sections.nodesOffset + nodeLength
	return sections, nil
}

func readDiskANNIndexSection(ctx context.Context, reader io.ReaderAt, offset, length int64) ([]byte, error) {
	if length < 0 || length > int64(maxPlatformInt()) {
		return nil, fmt.Errorf("%w: section exceeds platform capacity", ErrInvalidDiskANNFile)
	}
	buffer := make([]byte, int(length))
	if len(buffer) != 0 {
		if err := readFullAt(ctx, reader, buffer, offset); err != nil {
			return nil, err
		}
	}
	return buffer, nil
}

func verifyDiskANNIndexSection(ctx context.Context, reader io.ReaderAt, offset, length int64, want uint32) error {
	const chunkSize = 1 << 20
	bufferSize := chunkSize
	if length < chunkSize {
		bufferSize = int(length)
	}
	buffer := make([]byte, bufferSize)
	var crc uint32
	for readOffset := int64(0); readOffset < length; {
		if err := ctx.Err(); err != nil {
			return err
		}
		readLength := min(len(buffer), int(length-readOffset))
		if err := readFullAt(ctx, reader, buffer[:readLength], offset+readOffset); err != nil {
			return err
		}
		crc = ailego.UpdateCRC32C(crc, buffer[:readLength])
		readOffset += int64(readLength)
	}
	if crc != want {
		return fmt.Errorf("%w: got %08x, want %08x", errDiskANNIndexSectionChecksum, crc, want)
	}
	return nil
}

func validateDiskANNIndex(ctx context.Context, index *DiskANNIndex) error {
	if ctx == nil {
		return errors.New("core: nil DiskANN validation context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if index == nil || index.dimension <= 0 || index.dimension > MaxRotationDimension || index.nodes == nil {
		return errors.New("core: invalid DiskANN index")
	}
	if err := index.options.Validate(); err != nil {
		return err
	}
	count := len(index.keys)
	if count > math.MaxUint32 || len(index.positions) != count || index.nodes.layout.count != count ||
		index.nodes.layout.dimension != index.dimension || index.nodes.layout.metric != index.metric ||
		index.nodes.layout.maxDegree != index.options.MaxDegree {
		return errors.New("core: inconsistent DiskANN storage")
	}
	seen := make(map[uint64]struct{}, count)
	for position, key := range index.keys {
		if position&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		if _, duplicate := seen[key]; duplicate || index.positions[key] != position {
			return errors.New("core: invalid DiskANN key map")
		}
		seen[key] = struct{}{}
	}
	expectedTraversal := diskANNTraversalMetric(index.metric)
	if index.traversalMetric != expectedTraversal {
		return errors.New("core: invalid DiskANN traversal metric")
	}
	if count == 0 {
		if index.entryPoint != -1 || index.pq != nil || len(index.codes) != 0 || len(index.codeNorms) != 0 {
			return errors.New("core: invalid empty DiskANN state")
		}
		return nil
	}
	if index.entryPoint < 0 || index.entryPoint >= count || index.pq == nil || index.pq.dimension != index.dimension ||
		index.pq.metric != index.traversalMetric || uint64(count)*uint64(index.pq.Chunks()) > uint64(maxPlatformInt()) ||
		len(index.codes) != int(uint64(count)*uint64(index.pq.Chunks())) ||
		(index.options.PQChunks != 0 && index.options.PQChunks != index.pq.Chunks()) {
		return errors.New("core: inconsistent DiskANN PQ state")
	}
	if index.metric == MetricMIPSL2 {
		if len(index.codeNorms) != count {
			return errors.New("core: missing DiskANN MIPSL2 code norms")
		}
		for _, norm := range index.codeNorms {
			if norm < 0 || !finiteFloat32(norm) {
				return errors.New("core: invalid DiskANN MIPSL2 code norm")
			}
		}
	} else if len(index.codeNorms) != 0 {
		return errors.New("core: unexpected DiskANN code norms")
	}
	return nil
}
