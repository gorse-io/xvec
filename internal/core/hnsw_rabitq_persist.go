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
	hnswRaBitQFileVersion = 1
	hnswRaBitQHeaderSize  = 128
	hnswRaBitQFactorBytes = 5 * 8
)

var (
	hnswRaBitQFileMagic = [8]byte{'Z', 'V', 'E', 'C', 'H', 'R', 'B', 'Q'}

	ErrInvalidHNSWRaBitQFile        = errors.New("core: invalid HNSW-RaBitQ file")
	ErrHNSWRaBitQChecksumMismatch   = errors.New("core: HNSW-RaBitQ checksum mismatch")
	ErrUnsupportedHNSWRaBitQVersion = errors.New("core: unsupported HNSW-RaBitQ file version")
)

// Save durably and atomically publishes one complete graph/model/code
// generation in the native Go HNSW-RaBitQ format.
func (i *HNSWRaBitQIndex) Save(ctx context.Context, path string) error {
	if ctx == nil {
		return errors.New("core: nil HNSW-RaBitQ save context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if path == "" {
		return fmt.Errorf("%w: empty path", ErrInvalidHNSWRaBitQFile)
	}
	snapshot, err := i.hnswRaBitQPersistenceSnapshot(ctx)
	if err != nil {
		return err
	}
	encoded, err := encodeHNSWRaBitQIndex(ctx, snapshot)
	if err != nil {
		return err
	}
	if err := ailego.WriteFileAtomic(ctx, path, encoded, 0o600); err != nil {
		return fmt.Errorf("core: save HNSW-RaBitQ file: %w", err)
	}
	return nil
}

func (i *HNSWRaBitQIndex) hnswRaBitQPersistenceSnapshot(ctx context.Context) (*HNSWRaBitQIndex, error) {
	if i == nil {
		return nil, fmt.Errorf("%w: nil index", ErrInvalidHNSWRaBitQFile)
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	if err := validateHNSWRaBitQGeneration(i); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidHNSWRaBitQFile, err)
	}
	base, err := cloneHNSWIndex(ctx, i.base)
	if err != nil {
		return nil, err
	}
	model, err := RestoreRaBitQModel(i.model.State())
	if err != nil {
		return nil, fmt.Errorf("%w: snapshot model: %v", ErrInvalidHNSWRaBitQFile, err)
	}
	return &HNSWRaBitQIndex{
		options: i.options, base: base, model: model, codes: cloneRaBitQCodes(i.codes),
	}, nil
}

// OpenHNSWRaBitQIndex reads and verifies a native Go HNSW-RaBitQ artifact.
func OpenHNSWRaBitQIndex(ctx context.Context, path string) (*HNSWRaBitQIndex, error) {
	if ctx == nil {
		return nil, errors.New("core: nil HNSW-RaBitQ open context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if path == "" {
		return nil, fmt.Errorf("%w: empty path", ErrInvalidHNSWRaBitQFile)
	}
	encoded, err := readHNSWFile(ctx, path)
	if err != nil {
		return nil, fmt.Errorf("core: read HNSW-RaBitQ file: %w", err)
	}
	index, err := decodeHNSWRaBitQIndex(ctx, encoded)
	if err != nil {
		return nil, fmt.Errorf("core: open HNSW-RaBitQ file: %w", err)
	}
	return index, nil
}

func encodeHNSWRaBitQIndex(ctx context.Context, index *HNSWRaBitQIndex) ([]byte, error) {
	if ctx == nil {
		return nil, errors.New("core: nil HNSW-RaBitQ encode context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := validateHNSWRaBitQIndex(ctx, index); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidHNSWRaBitQFile, err)
	}
	if err := validateHNSWRaBitQFormatOptions(index.options); err != nil {
		return nil, err
	}
	base, err := encodeHNSWIndex(ctx, index.base)
	if err != nil {
		return nil, fmt.Errorf("%w: encode graph: %v", ErrInvalidHNSWRaBitQFile, err)
	}
	state := index.model.State()
	payloadSize, recordSize, err := checkedHNSWRaBitQPayloadSize(len(base), state, len(index.codes))
	if err != nil {
		return nil, err
	}
	payload := make([]byte, 0, payloadSize)
	payload = append(payload, base...)
	for centroidIndex, centroid := range state.Centroids {
		if centroidIndex&63 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		for _, value := range centroid {
			payload = binary.LittleEndian.AppendUint32(payload, math.Float32bits(value))
		}
	}
	payload = append(payload, state.RotationSigns...)
	for position, code := range index.codes {
		if position&255 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		start := len(payload)
		payload = binary.LittleEndian.AppendUint32(payload, uint32(code.cluster))
		for _, factor := range [...]float64{code.coarseAdd, code.coarseRescale, code.coarseError, code.fullAdd, code.fullRescale} {
			payload = binary.LittleEndian.AppendUint64(payload, math.Float64bits(factor))
		}
		payload = append(payload, code.binaryCode...)
		payload = append(payload, code.extraCode...)
		if len(payload)-start != recordSize {
			return nil, fmt.Errorf("%w: internal code record length", ErrInvalidHNSWRaBitQFile)
		}
	}
	if len(payload) != payloadSize {
		return nil, fmt.Errorf("%w: internal payload length", ErrInvalidHNSWRaBitQFile)
	}

	header := make([]byte, hnswRaBitQHeaderSize)
	copy(header[:8], hnswRaBitQFileMagic[:])
	binary.LittleEndian.PutUint16(header[8:10], hnswRaBitQFileVersion)
	binary.LittleEndian.PutUint16(header[10:12], hnswRaBitQHeaderSize)
	binary.LittleEndian.PutUint64(header[16:24], uint64(hnswRaBitQHeaderSize+payloadSize))
	binary.LittleEndian.PutUint64(header[24:32], uint64(payloadSize))
	binary.LittleEndian.PutUint64(header[32:40], uint64(len(base)))
	binary.LittleEndian.PutUint64(header[40:48], uint64(len(index.codes)))
	binary.LittleEndian.PutUint32(header[48:52], uint32(index.base.dimension))
	header[52] = byte(index.options.Metric)
	header[53] = byte(index.options.TotalBits)
	binary.LittleEndian.PutUint32(header[56:60], uint32(len(state.Centroids)))
	binary.LittleEndian.PutUint32(header[60:64], uint32(len(state.RotationSigns)))
	binary.LittleEndian.PutUint64(header[64:72], math.Float64bits(state.ExtraScale))
	binary.LittleEndian.PutUint64(header[72:80], index.model.fingerprint)
	binary.LittleEndian.PutUint32(header[80:84], uint32(index.options.M))
	binary.LittleEndian.PutUint32(header[84:88], uint32(index.options.EFConstruction))
	binary.LittleEndian.PutUint32(header[88:92], uint32(index.options.Clusters))
	binary.LittleEndian.PutUint32(header[92:96], uint32(index.options.SampleCount))
	binary.LittleEndian.PutUint32(header[96:100], uint32(index.options.MaxIterations))
	binary.LittleEndian.PutUint32(header[100:104], uint32(index.options.Workers))
	binary.LittleEndian.PutUint64(header[104:112], index.options.Seed)
	binary.LittleEndian.PutUint32(header[112:116], ailego.CRC32C(payload))
	binary.LittleEndian.PutUint32(header[124:128], ailego.CRC32C(header[:124]))
	return append(header, payload...), nil
}

func decodeHNSWRaBitQIndex(ctx context.Context, encoded []byte) (*HNSWRaBitQIndex, error) {
	if ctx == nil {
		return nil, errors.New("core: nil HNSW-RaBitQ decode context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(encoded) < hnswRaBitQHeaderSize {
		return nil, fmt.Errorf("%w: truncated header", ErrInvalidHNSWRaBitQFile)
	}
	header := encoded[:hnswRaBitQHeaderSize]
	if !bytes.Equal(header[:8], hnswRaBitQFileMagic[:]) {
		return nil, fmt.Errorf("%w: bad magic", ErrInvalidHNSWRaBitQFile)
	}
	version := binary.LittleEndian.Uint16(header[8:10])
	if version != hnswRaBitQFileVersion {
		return nil, fmt.Errorf("%w: %d", ErrUnsupportedHNSWRaBitQVersion, version)
	}
	if binary.LittleEndian.Uint16(header[10:12]) != hnswRaBitQHeaderSize ||
		binary.LittleEndian.Uint32(header[12:16]) != 0 || header[54] != 0 || header[55] != 0 ||
		!hnswAllZero(header[116:124]) {
		return nil, fmt.Errorf("%w: invalid header fields", ErrInvalidHNSWRaBitQFile)
	}
	if got, want := ailego.CRC32C(header[:124]), binary.LittleEndian.Uint32(header[124:128]); got != want {
		return nil, fmt.Errorf("%w: header got %08x, want %08x", ErrHNSWRaBitQChecksumMismatch, got, want)
	}
	if binary.LittleEndian.Uint64(header[16:24]) != uint64(len(encoded)) ||
		binary.LittleEndian.Uint64(header[24:32]) != uint64(len(encoded)-hnswRaBitQHeaderSize) {
		return nil, fmt.Errorf("%w: inconsistent file length", ErrInvalidHNSWRaBitQFile)
	}
	payload := encoded[hnswRaBitQHeaderSize:]
	if got, want := ailego.CRC32C(payload), binary.LittleEndian.Uint32(header[112:116]); got != want {
		return nil, fmt.Errorf("%w: payload got %08x, want %08x", ErrHNSWRaBitQChecksumMismatch, got, want)
	}

	baseLength64 := binary.LittleEndian.Uint64(header[32:40])
	count64 := binary.LittleEndian.Uint64(header[40:48])
	dimension64 := uint64(binary.LittleEndian.Uint32(header[48:52]))
	centroidCount64 := uint64(binary.LittleEndian.Uint32(header[56:60]))
	rotationBytes64 := uint64(binary.LittleEndian.Uint32(header[60:64]))
	for _, value := range []uint64{baseLength64, count64, dimension64, centroidCount64, rotationBytes64} {
		if value > uint64(maxPlatformInt()) {
			return nil, fmt.Errorf("%w: field exceeds platform capacity", ErrInvalidHNSWRaBitQFile)
		}
	}
	baseLength, count, dimension := int(baseLength64), int(count64), int(dimension64)
	centroidCount, rotationBytes := int(centroidCount64), int(rotationBytes64)
	if baseLength < hnswHeaderSize || baseLength > len(payload) || uint64(count) > math.MaxUint32 ||
		dimension < MinRaBitQDimension || dimension > MaxRaBitQDimension || centroidCount <= 0 {
		return nil, fmt.Errorf("%w: invalid size fields", ErrInvalidHNSWRaBitQFile)
	}
	options, err := decodeHNSWRaBitQOptions(header)
	if err != nil {
		return nil, err
	}
	state := RaBitQModelState{
		Dimension: dimension, Metric: options.Metric, TotalBits: options.TotalBits,
		Centroids: make([][]float32, centroidCount), ExtraScale: math.Float64frombits(binary.LittleEndian.Uint64(header[64:72])),
	}
	centroidBytes64 := uint64(centroidCount) * uint64(dimension) * 4
	if centroidBytes64 > uint64(maxPlatformInt()) || uint64(baseLength)+centroidBytes64+uint64(rotationBytes) > uint64(len(payload)) {
		return nil, fmt.Errorf("%w: model storage exceeds payload", ErrInvalidHNSWRaBitQFile)
	}
	offset := baseLength
	for centroidIndex := range state.Centroids {
		if centroidIndex&63 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		centroid := make([]float32, dimension)
		for component := range centroid {
			centroid[component] = math.Float32frombits(binary.LittleEndian.Uint32(payload[offset : offset+4]))
			offset += 4
		}
		state.Centroids[centroidIndex] = centroid
	}
	state.RotationSigns = slicesCloneBytes(payload[offset : offset+rotationBytes])
	offset += rotationBytes
	model, err := RestoreRaBitQModel(state)
	if err != nil {
		return nil, fmt.Errorf("%w: restore model: %v", ErrInvalidHNSWRaBitQFile, err)
	}
	if model.fingerprint != binary.LittleEndian.Uint64(header[72:80]) {
		return nil, fmt.Errorf("%w: model fingerprint mismatch", ErrInvalidHNSWRaBitQFile)
	}
	base, err := decodeHNSWIndex(ctx, payload[:baseLength])
	if err != nil {
		return nil, fmt.Errorf("%w: decode graph: %v", ErrInvalidHNSWRaBitQFile, err)
	}
	if len(base.keys) != count || base.dimension != dimension {
		return nil, fmt.Errorf("%w: graph metadata mismatch", ErrInvalidHNSWRaBitQFile)
	}
	recordSize := hnswRaBitQCodeRecordSize(model.paddedDimension, model.totalBits)
	if recordSize <= 0 || count > (len(payload)-offset)/recordSize || offset+count*recordSize != len(payload) {
		return nil, fmt.Errorf("%w: invalid code payload length", ErrInvalidHNSWRaBitQFile)
	}
	codes := make([]RaBitQCode, count)
	binaryBytes := model.paddedDimension / 8
	extraBytes := model.paddedDimension * model.extraBits / 8
	for position := range codes {
		if position&255 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		cluster := uint64(binary.LittleEndian.Uint32(payload[offset : offset+4]))
		offset += 4
		if cluster >= uint64(model.Len()) {
			return nil, fmt.Errorf("%w: code cluster out of range", ErrInvalidHNSWRaBitQFile)
		}
		factors := [5]float64{}
		for factorIndex := range factors {
			factors[factorIndex] = math.Float64frombits(binary.LittleEndian.Uint64(payload[offset : offset+8]))
			offset += 8
		}
		code := RaBitQCode{
			modelFingerprint: model.fingerprint, cluster: int(cluster),
			paddedDimension: model.paddedDimension, totalBits: model.totalBits,
			coarseAdd: factors[0], coarseRescale: factors[1], coarseError: factors[2],
			fullAdd: factors[3], fullRescale: factors[4],
			binaryCode: slicesCloneBytes(payload[offset : offset+binaryBytes]),
		}
		offset += binaryBytes
		code.extraCode = slicesCloneBytes(payload[offset : offset+extraBytes])
		offset += extraBytes
		if err := code.validate(); err != nil {
			return nil, fmt.Errorf("%w: invalid code factors", ErrInvalidHNSWRaBitQFile)
		}
		codes[position] = code
	}
	index := &HNSWRaBitQIndex{options: options, base: base, model: model, codes: codes}
	if err := validateHNSWRaBitQIndex(ctx, index); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidHNSWRaBitQFile, err)
	}
	return index, nil
}

func decodeHNSWRaBitQOptions(header []byte) (HNSWRaBitQBuildOptions, error) {
	values := [6]uint64{
		uint64(binary.LittleEndian.Uint32(header[80:84])),
		uint64(binary.LittleEndian.Uint32(header[84:88])),
		uint64(binary.LittleEndian.Uint32(header[88:92])),
		uint64(binary.LittleEndian.Uint32(header[92:96])),
		uint64(binary.LittleEndian.Uint32(header[96:100])),
		uint64(binary.LittleEndian.Uint32(header[100:104])),
	}
	for _, value := range values {
		if value > uint64(maxPlatformInt()) {
			return HNSWRaBitQBuildOptions{}, fmt.Errorf("%w: options exceed platform capacity", ErrInvalidHNSWRaBitQFile)
		}
	}
	options := HNSWRaBitQBuildOptions{
		Metric: Metric(header[52]), TotalBits: int(header[53]), M: int(values[0]), EFConstruction: int(values[1]),
		Clusters: int(values[2]), SampleCount: int(values[3]), MaxIterations: int(values[4]), Workers: int(values[5]),
		Seed: binary.LittleEndian.Uint64(header[104:112]),
	}
	if err := options.Validate(); err != nil {
		return HNSWRaBitQBuildOptions{}, fmt.Errorf("%w: %v", ErrInvalidHNSWRaBitQFile, err)
	}
	return options, nil
}

func validateHNSWRaBitQFormatOptions(options HNSWRaBitQBuildOptions) error {
	for _, value := range []int{options.M, options.EFConstruction, options.Clusters, options.SampleCount, options.MaxIterations, options.Workers} {
		if value < 0 || uint64(value) > math.MaxUint32 {
			return fmt.Errorf("%w: options exceed format capacity", ErrInvalidHNSWRaBitQFile)
		}
	}
	return nil
}

func checkedHNSWRaBitQPayloadSize(baseLength int, state RaBitQModelState, count int) (int, int, error) {
	if baseLength < 0 || state.Dimension <= 0 || len(state.Centroids) == 0 || count < 0 {
		return 0, 0, fmt.Errorf("%w: invalid payload inputs", ErrInvalidHNSWRaBitQFile)
	}
	recordSize := hnswRaBitQCodeRecordSize(roundUpRaBitQDimension(state.Dimension), state.TotalBits)
	centroidBytes := uint64(len(state.Centroids)) * uint64(state.Dimension) * 4
	total := uint64(baseLength) + centroidBytes + uint64(len(state.RotationSigns)) + uint64(count)*uint64(recordSize)
	if recordSize <= 0 || total > uint64(maxPlatformInt()-hnswRaBitQHeaderSize) {
		return 0, 0, fmt.Errorf("%w: payload exceeds platform capacity", ErrInvalidHNSWRaBitQFile)
	}
	return int(total), recordSize, nil
}

func hnswRaBitQCodeRecordSize(paddedDimension, totalBits int) int {
	if paddedDimension < MinRaBitQDimension || paddedDimension%64 != 0 || totalBits < 1 || totalBits > 9 {
		return 0
	}
	return 4 + hnswRaBitQFactorBytes + paddedDimension/8 + paddedDimension*(totalBits-1)/8
}

func slicesCloneBytes(source []byte) []byte {
	return append([]byte(nil), source...)
}
