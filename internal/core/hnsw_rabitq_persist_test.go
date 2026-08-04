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
	"context"
	"encoding/binary"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/gorse-io/zvec/internal/ailego"
	"github.com/stretchr/testify/require"
)

func TestHNSWRaBitQPersistenceRoundTripReplaceAndIncrement(t *testing.T) {
	index := buildHNSWRaBitQ(t, hnswRaBitQCandidates(180, 70), hnswRaBitQTestOptions(MetricCosine))
	path := filepath.Join(t.TempDir(), "vectors.hrbtq")
	{
		err := index.Save(context.Background(), path)
		require.NoError(t, err)
	}

	info, err := os.Stat(path)
	require.NoError(t, err)
	assertPrivateFileMode(t, info.Mode())
	opened, err := OpenHNSWRaBitQIndex(context.Background(), path)
	require.NoError(t, err)

	assertSameHNSWRaBitQIndex(t, opened, index)
	query := hnswRaBitQCandidates(1, 70)[0].Vector
	options := HNSWRaBitQSearchOptions{SearchOptions: SearchOptions{TopK: 25}, EF: 80}
	for _, refine := range []bool{false, true} {
		options.Refine = refine
		want, err := index.SearchHNSWRaBitQ(context.Background(), query, options)
		require.NoError(t, err)

		got, err := opened.SearchHNSWRaBitQ(context.Background(), query, options)
		require.NoError(t, err)
		require.Equal(t, want, got)
	}

	next := hnswRaBitQCandidates(1, 70)[0]
	next.Key = 999999
	{
		err := opened.Add(context.Background(), next.Key, next.Vector)
		require.NoError(t, err)
	}
	{
		err := opened.Save(context.Background(), path)
		require.NoError(t, err)
	}

	reopened, err := OpenHNSWRaBitQIndex(context.Background(), path)
	require.NoError(t, err)

	assertSameHNSWRaBitQIndex(t, reopened, opened)

	replacement := buildHNSWRaBitQ(t, hnswRaBitQCandidates(40, 64), hnswRaBitQTestOptions(MetricIP))
	{
		err := replacement.Save(context.Background(), path)
		require.NoError(t, err)
	}

	reopened, err = OpenHNSWRaBitQIndex(context.Background(), path)
	require.NoError(t, err)

	assertSameHNSWRaBitQIndex(t, reopened, replacement)
}

func TestHNSWRaBitQPersistenceEmptyCancellationAndErrors(t *testing.T) {
	builder, err := NewHNSWRaBitQBuilder(64, hnswRaBitQTestOptions(MetricL2))
	require.NoError(t, err)

	index, err := builder.Build(context.Background())
	require.NoError(t, err)

	dir := t.TempDir()
	path := filepath.Join(dir, "empty.hrbtq")
	{
		err := index.Save(context.Background(), path)
		require.NoError(t, err)
	}

	original, err := os.ReadFile(path)
	require.NoError(t, err)

	opened, err := OpenHNSWRaBitQIndex(context.Background(), path)
	require.NoError(t, err)

	assertSameHNSWRaBitQIndex(t, opened, index)

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	{
		err := index.Save(canceled, path)
		require.ErrorIs(t, err, context.Canceled)
	}

	after, _ := os.ReadFile(path)
	require.True(t, slices.Equal(after, original),
		"canceled Save changed published artifact")
	{
		_, err := OpenHNSWRaBitQIndex(canceled, path)
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		_, err := encodeHNSWRaBitQIndex(canceled, index)
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		_, err := decodeHNSWRaBitQIndex(canceled, original)
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		err := index.Save(nil, path)
		require.Error(t, err,
			"nil Save context succeeded")
	}
	{
		_, err := OpenHNSWRaBitQIndex(nil, path)
		require.Error(t, err,
			"nil Open context succeeded")
	}
	{
		err := index.Save(context.Background(), "")
		require.ErrorIs(t, err, ErrInvalidHNSWRaBitQFile)
	}
	{
		_, err := OpenHNSWRaBitQIndex(context.Background(), "")
		require.ErrorIs(t, err, ErrInvalidHNSWRaBitQFile)
	}
	{
		_, err := OpenHNSWRaBitQIndex(context.Background(), filepath.Join(dir, "missing"))
		require.ErrorIs(t, err, os.ErrNotExist)
	}

	var nilIndex *HNSWRaBitQIndex
	{
		err := nilIndex.Save(context.Background(), filepath.Join(dir, "nil"))
		require.ErrorIs(t, err, ErrInvalidHNSWRaBitQFile)
	}
}

func TestHNSWRaBitQPersistenceDetectsCorruption(t *testing.T) {
	valid, err := encodeHNSWRaBitQIndex(context.Background(), buildHNSWRaBitQ(t, hnswRaBitQCandidates(32, 64), hnswRaBitQTestOptions(MetricL2)))
	require.NoError(t, err)

	for _, cut := range []int{0, 1, hnswRaBitQHeaderSize - 1, hnswRaBitQHeaderSize, len(valid) - 1} {
		{
			_, err := decodeHNSWRaBitQIndex(context.Background(), valid[:cut])
			require.ErrorIs(t, err, ErrInvalidHNSWRaBitQFile)
		}
	}
	trailing := append(slices.Clone(valid), 0)
	{
		_, err := decodeHNSWRaBitQIndex(context.Background(), trailing)
		require.ErrorIs(t, err, ErrInvalidHNSWRaBitQFile)
	}

	badMagic := slices.Clone(valid)
	badMagic[0] ^= 1
	{
		_, err := decodeHNSWRaBitQIndex(context.Background(), badMagic)
		require.ErrorIs(t, err, ErrInvalidHNSWRaBitQFile)
	}

	badVersion := slices.Clone(valid)
	binary.LittleEndian.PutUint16(badVersion[8:10], hnswRaBitQFileVersion+1)
	{
		_, err := decodeHNSWRaBitQIndex(context.Background(), badVersion)
		require.ErrorIs(t, err, ErrUnsupportedHNSWRaBitQVersion)
	}

	badHeader := slices.Clone(valid)
	badHeader[80] ^= 1
	{
		_, err := decodeHNSWRaBitQIndex(context.Background(), badHeader)
		require.ErrorIs(t, err, ErrHNSWRaBitQChecksumMismatch)
	}

	badPayload := slices.Clone(valid)
	badPayload[len(badPayload)-1] ^= 1
	{
		_, err := decodeHNSWRaBitQIndex(context.Background(), badPayload)
		require.ErrorIs(t, err, ErrHNSWRaBitQChecksumMismatch)
	}

	badFingerprint := slices.Clone(valid)
	badFingerprint[72] ^= 1
	refreshHNSWRaBitQChecksums(badFingerprint)
	{
		_, err := decodeHNSWRaBitQIndex(context.Background(), badFingerprint)
		require.ErrorIs(t, err, ErrInvalidHNSWRaBitQFile)
	}

	badCluster := slices.Clone(valid)
	baseLength := int(binary.LittleEndian.Uint64(badCluster[32:40]))
	dimension := int(binary.LittleEndian.Uint32(badCluster[48:52]))
	centroids := int(binary.LittleEndian.Uint32(badCluster[56:60]))
	rotationBytes := int(binary.LittleEndian.Uint32(badCluster[60:64]))
	codeOffset := hnswRaBitQHeaderSize + baseLength + centroids*dimension*4 + rotationBytes
	binary.LittleEndian.PutUint32(badCluster[codeOffset:codeOffset+4], mathMaxUint32)
	refreshHNSWRaBitQChecksums(badCluster)
	{
		_, err := decodeHNSWRaBitQIndex(context.Background(), badCluster)
		require.ErrorIs(t, err, ErrInvalidHNSWRaBitQFile)
	}

	badGraph := slices.Clone(valid)
	badGraph[hnswRaBitQHeaderSize+hnswHeaderSize] ^= 1
	refreshHNSWRaBitQChecksums(badGraph)
	{
		_, err := decodeHNSWRaBitQIndex(context.Background(), badGraph)
		require.ErrorIs(t, err, ErrInvalidHNSWRaBitQFile)
	}
}

func FuzzHNSWRaBitQDecode(f *testing.F) {
	valid, err := encodeHNSWRaBitQIndex(context.Background(), buildHNSWRaBitQ(f, hnswRaBitQCandidates(8, 64), hnswRaBitQTestOptions(MetricL2)))
	require.NoError(f, err)

	f.Add(valid)
	f.Add([]byte("ZVECHR BQ"))
	f.Fuzz(func(t *testing.T, encoded []byte) {
		index, err := decodeHNSWRaBitQIndex(context.Background(), encoded)
		if err != nil {
			return
		}
		{
			err := validateHNSWRaBitQIndex(context.Background(), index)
			require.NoError(t, err)
		}
	})
}

const mathMaxUint32 = ^uint32(0)

func refreshHNSWRaBitQChecksums(encoded []byte) {
	header := encoded[:hnswRaBitQHeaderSize]
	payload := encoded[hnswRaBitQHeaderSize:]
	binary.LittleEndian.PutUint32(header[112:116], ailego.CRC32C(payload))
	binary.LittleEndian.PutUint32(header[124:128], ailego.CRC32C(header[:124]))
}

func assertSameHNSWRaBitQIndex(t testing.TB, got, want *HNSWRaBitQIndex) {
	t.Helper()
	require.Equal(t, want.options, got.options,

		"HNSW-RaBitQ indexes differ")
	require.Equal(t, want.ModelState(), got.ModelState(),

		"HNSW-RaBitQ indexes differ")
	require.True(t, slices.Equal(got.base.keys, want.base.keys),

		"HNSW-RaBitQ indexes differ")
	require.True(t, slices.Equal(got.base.vectors, want.base.vectors),

		"HNSW-RaBitQ indexes differ")
	require.True(t, slices.Equal(got.base.levels, want.base.levels),

		"HNSW-RaBitQ indexes differ")
	require.Len(t, got.base.neighbors, len(want.base.neighbors),

		"HNSW-RaBitQ indexes differ")
	require.Equal(t, want.base.entryPoint, got.base.entryPoint,

		"HNSW-RaBitQ indexes differ")
	require.Equal(t, want.base.maxLevel, got.base.maxLevel,

		"HNSW-RaBitQ indexes differ")
	require.Equal(t, want.base.levelRNGState, got.base.levelRNGState,
		"HNSW-RaBitQ indexes differ")
	require.Len(t, got.codes, len(want.codes),
		"HNSW-RaBitQ indexes differ")

	for position := range got.base.neighbors {
		require.Len(t, got.base.neighbors[position], len(want.base.neighbors[position]),
			"HNSW-RaBitQ level counts differ")

		for level := range got.base.neighbors[position] {
			require.True(t, slices.Equal(got.base.neighbors[position][level], want.base.neighbors[position][level]),
				"HNSW-RaBitQ neighbors differ")
		}
	}
	for position := range got.codes {
		require.Equal(t, want.codes[position], got.codes[position],
			"HNSW-RaBitQ codes differ")
	}
}
