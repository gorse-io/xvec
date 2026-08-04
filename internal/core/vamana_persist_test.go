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
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/gorse-io/zvec/internal/ailego"
	"github.com/stretchr/testify/require"
)

func TestVamanaPersistenceRoundTripReplaceAndIncrement(t *testing.T) {
	options := DefaultVamanaBuildOptions(MetricCosine)
	options.MaxDegree, options.SearchListSize, options.MaxOcclusionSize = 8, 40, 80
	index := buildVamana(t, hnswRaBitQCandidates(180, 70), options)
	path := filepath.Join(t.TempDir(), "vectors.vamana")
	{
		err := index.Save(context.Background(), path)
		require.NoError(t, err)
	}

	info, err := os.Stat(path)
	require.NoError(t, err)
	assertPrivateFileMode(t, info.Mode())
	opened, err := OpenVamanaIndex(context.Background(), path)
	require.NoError(t, err)

	assertSameVamanaIndex(t, opened, index)
	query := hnswRaBitQCandidates(1, 70)[0].Vector
	search := VamanaSearchOptions{SearchOptions: SearchOptions{TopK: 20}, EFSearch: 80}
	want, err := index.SearchVamana(context.Background(), query, search)
	require.NoError(t, err)

	got, err := opened.SearchVamana(context.Background(), query, search)
	require.NoError(t, err)
	require.Equal(t, want, got)

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

	reopened, err := OpenVamanaIndex(context.Background(), path)
	require.NoError(t, err)

	assertSameVamanaIndex(t, reopened, opened)

	replacementOptions := DefaultVamanaBuildOptions(MetricIP)
	replacementOptions.MaxDegree, replacementOptions.SearchListSize = 4, 16
	replacement := buildVamana(t, hnswBuildInputs(40), replacementOptions)
	{
		err := replacement.Save(context.Background(), path)
		require.NoError(t, err)
	}

	reopened, err = OpenVamanaIndex(context.Background(), path)
	require.NoError(t, err)

	assertSameVamanaIndex(t, reopened, replacement)
}

func TestVamanaPersistenceEmptyCancellationAndErrors(t *testing.T) {
	index := buildVamana(t, nil, DefaultVamanaBuildOptions(MetricL2))
	largeOptions := DefaultVamanaBuildOptions(MetricL2)
	largeOptions.MaxDegree, largeOptions.SearchListSize = 8, 40
	largeIndex := buildVamana(t, hnswRaBitQCandidates(300, 70), largeOptions)
	dir := t.TempDir()
	path := filepath.Join(dir, "empty.vamana")
	{
		err := index.Save(context.Background(), path)
		require.NoError(t, err)
	}

	original, err := os.ReadFile(path)
	require.NoError(t, err)

	opened, err := OpenVamanaIndex(context.Background(), path)
	require.NoError(t, err)

	assertSameVamanaIndex(t, opened, index)
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	{
		err := index.Save(canceled, path)
		require.ErrorIs(t, err, context.Canceled)
	}

	after, err := os.ReadFile(path)
	require.NoError(t, err)
	require.True(t, slices.Equal(after, original),
		"canceled Save changed artifact")
	{
		_, err := OpenVamanaIndex(canceled, path)
		require.ErrorIs(t, err, context.Canceled)
	}

	midSave := newCancelAfterChecks(7)
	{
		err := largeIndex.Save(midSave, path)
		require.ErrorIs(t, err, context.Canceled)
	}

	after, err = os.ReadFile(path)
	require.NoError(t, err)
	require.True(t, slices.Equal(after, original),
		"mid-save cancellation changed artifact")

	midEncode := newCancelAfterChecks(5)
	{
		_, err := encodeVamanaIndex(midEncode, largeIndex)
		require.ErrorIs(t, err, context.Canceled)
	}

	largeEncoded, err := encodeVamanaIndex(context.Background(), largeIndex)
	require.NoError(t, err)

	midDecode := newCancelAfterChecks(3)
	{
		_, err := decodeVamanaIndex(midDecode, largeEncoded)
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		err := index.Save(nil, path)
		require.Error(t, err,
			"nil Save context succeeded")
	}
	{
		_, err := OpenVamanaIndex(nil, path)
		require.Error(t, err,
			"nil Open context succeeded")
	}
	{
		_, err := encodeVamanaIndex(nil, index)
		require.Error(t, err,
			"nil encode context succeeded")
	}
	{
		_, err := decodeVamanaIndex(nil, original)
		require.Error(t, err,
			"nil decode context succeeded")
	}
	{
		err := index.Save(context.Background(), "")
		require.ErrorIs(t, err, ErrInvalidVamanaFile)
	}
	{
		_, err := OpenVamanaIndex(context.Background(), "")
		require.ErrorIs(t, err, ErrInvalidVamanaFile)
	}
	{
		_, err := OpenVamanaIndex(context.Background(), filepath.Join(dir, "missing"))
		require.ErrorIs(t, err, os.ErrNotExist)
	}

	var nilIndex *VamanaIndex
	{
		err := nilIndex.Save(context.Background(), filepath.Join(dir, "nil"))
		require.ErrorIs(t, err, ErrInvalidVamanaFile)
	}
}

func TestVamanaPersistenceDetectsCorruption(t *testing.T) {
	options := DefaultVamanaBuildOptions(MetricL2)
	options.MaxDegree, options.SearchListSize = 4, 16
	valid, err := encodeVamanaIndex(context.Background(), buildVamana(t, hnswBuildInputs(32), options))
	require.NoError(t, err)

	for _, cut := range []int{0, 1, vamanaHeaderSize - 1, vamanaHeaderSize, len(valid) - 1} {
		{
			_, err := decodeVamanaIndex(context.Background(), valid[:cut])
			require.ErrorIs(t, err, ErrInvalidVamanaFile)
		}
	}
	trailing := append(slices.Clone(valid), 0)
	{
		_, err := decodeVamanaIndex(context.Background(), trailing)
		require.ErrorIs(t, err, ErrInvalidVamanaFile)
	}

	badMagic := slices.Clone(valid)
	badMagic[0] ^= 1
	{
		_, err := decodeVamanaIndex(context.Background(), badMagic)
		require.ErrorIs(t, err, ErrInvalidVamanaFile)
	}

	badVersion := slices.Clone(valid)
	binary.LittleEndian.PutUint16(badVersion[8:10], vamanaFileVersion+1)
	{
		_, err := decodeVamanaIndex(context.Background(), badVersion)
		require.ErrorIs(t, err, ErrUnsupportedVamanaVersion)
	}

	badHeader := slices.Clone(valid)
	badHeader[56] ^= 1
	{
		_, err := decodeVamanaIndex(context.Background(), badHeader)
		require.ErrorIs(t, err, ErrVamanaChecksumMismatch)
	}

	badPayload := slices.Clone(valid)
	badPayload[len(badPayload)-1] ^= 1
	{
		_, err := decodeVamanaIndex(context.Background(), badPayload)
		require.ErrorIs(t, err, ErrVamanaChecksumMismatch)
	}

	badSelfLoop := slices.Clone(valid)
	count := int(binary.LittleEndian.Uint64(badSelfLoop[32:40]))
	dimension := int(binary.LittleEndian.Uint32(badSelfLoop[48:52]))
	adjacencyOffset := vamanaHeaderSize + count*8 + count*dimension*4
	degree := int(binary.LittleEndian.Uint32(badSelfLoop[adjacencyOffset : adjacencyOffset+4]))
	require.False(t, degree == 0,
		"fixture entry has no neighbors")

	binary.LittleEndian.PutUint32(badSelfLoop[adjacencyOffset+4:adjacencyOffset+8], 0)
	refreshVamanaChecksums(badSelfLoop)
	{
		_, err := decodeVamanaIndex(context.Background(), badSelfLoop)
		require.ErrorIs(t, err, ErrInvalidVamanaFile)
	}

	badDegree := slices.Clone(valid)
	binary.LittleEndian.PutUint32(badDegree[adjacencyOffset:adjacencyOffset+4], uint32(options.MaxDegree+1))
	refreshVamanaChecksums(badDegree)
	{
		_, err := decodeVamanaIndex(context.Background(), badDegree)
		require.ErrorIs(t, err, ErrInvalidVamanaFile)
	}

	badDuplicateKey := slices.Clone(valid)
	copy(badDuplicateKey[vamanaHeaderSize+8:vamanaHeaderSize+16], badDuplicateKey[vamanaHeaderSize:vamanaHeaderSize+8])
	refreshVamanaChecksums(badDuplicateKey)
	{
		_, err := decodeVamanaIndex(context.Background(), badDuplicateKey)
		require.ErrorIs(t, err, ErrInvalidVamanaFile)
	}

	badVector := slices.Clone(valid)
	vectorOffset := vamanaHeaderSize + count*8
	binary.LittleEndian.PutUint32(badVector[vectorOffset:vectorOffset+4], math.Float32bits(float32(math.NaN())))
	refreshVamanaChecksums(badVector)
	{
		_, err := decodeVamanaIndex(context.Background(), badVector)
		require.ErrorIs(t, err, ErrInvalidVamanaFile)
	}

	badEntry := slices.Clone(valid)
	binary.LittleEndian.PutUint64(badEntry[72:80], uint64(count))
	refreshVamanaChecksums(badEntry)
	{
		_, err := decodeVamanaIndex(context.Background(), badEntry)
		require.ErrorIs(t, err, ErrInvalidVamanaFile)
	}

	badNeighbor := slices.Clone(valid)
	binary.LittleEndian.PutUint32(badNeighbor[adjacencyOffset+4:adjacencyOffset+8], uint32(count))
	refreshVamanaChecksums(badNeighbor)
	{
		_, err := decodeVamanaIndex(context.Background(), badNeighbor)
		require.ErrorIs(t, err, ErrInvalidVamanaFile)
	}

	badDuplicateNeighbor := slices.Clone(valid)
	duplicateOffset, found := vamanaAdjacencyWithDegree(badDuplicateNeighbor, 2)
	require.True(t, found,
		"fixture has no node with two neighbors")

	copy(badDuplicateNeighbor[duplicateOffset+8:duplicateOffset+12], badDuplicateNeighbor[duplicateOffset+4:duplicateOffset+8])
	refreshVamanaChecksums(badDuplicateNeighbor)
	{
		_, err := decodeVamanaIndex(context.Background(), badDuplicateNeighbor)
		require.ErrorIs(t, err, ErrInvalidVamanaFile)
	}

	badCount := slices.Clone(valid)
	binary.LittleEndian.PutUint64(badCount[32:40], uint64(math.MaxUint32)+1)
	refreshVamanaChecksums(badCount)
	{
		_, err := decodeVamanaIndex(context.Background(), badCount)
		require.ErrorIs(t, err, ErrInvalidVamanaFile)
	}
}

func FuzzVamanaDecode(f *testing.F) {
	options := DefaultVamanaBuildOptions(MetricL2)
	options.MaxDegree, options.SearchListSize = 4, 12
	valid, err := encodeVamanaIndex(context.Background(), buildVamana(f, hnswBuildInputs(8), options))
	require.NoError(f, err)

	f.Add(valid)
	f.Add([]byte("not-vamana"))
	f.Fuzz(func(t *testing.T, encoded []byte) {
		index, err := decodeVamanaIndex(context.Background(), encoded)
		if err != nil {
			return
		}
		{
			err := validateVamanaIndex(context.Background(), index)
			require.NoError(t, err)
		}
	})
}

func refreshVamanaChecksums(encoded []byte) {
	header := encoded[:vamanaHeaderSize]
	payload := encoded[vamanaHeaderSize:]
	binary.LittleEndian.PutUint32(header[80:84], ailego.CRC32C(payload))
	binary.LittleEndian.PutUint32(header[124:128], ailego.CRC32C(header[:124]))
}

func vamanaAdjacencyWithDegree(encoded []byte, minimum int) (int, bool) {
	count := int(binary.LittleEndian.Uint64(encoded[32:40]))
	dimension := int(binary.LittleEndian.Uint32(encoded[48:52]))
	offset := vamanaHeaderSize + count*8 + count*dimension*4
	for range count {
		degree := int(binary.LittleEndian.Uint32(encoded[offset : offset+4]))
		if degree >= minimum {
			return offset, true
		}
		offset += 4 + degree*4
	}
	return 0, false
}

func assertSameVamanaIndex(t testing.TB, got, want *VamanaIndex) {
	t.Helper()
	require.Equal(t, want.dimension, got.dimension,

		"Vamana indexes differ")
	require.Equal(t, want.options, got.options,

		"Vamana indexes differ")
	require.Equal(t, want.entryPoint, got.entryPoint,

		"Vamana indexes differ")
	require.True(t, slices.Equal(got.keys, want.keys),

		"Vamana indexes differ")
	require.True(t, slices.Equal(got.vectors, want.vectors),

		"Vamana indexes differ")
	require.Equal(t, want.positions, got.positions,

		"Vamana indexes differ")
	require.Equal(t, want.neighbors, got.neighbors,

		"Vamana indexes differ")
	require.Equal(t, want.neighborDistances, got.neighborDistances,
		"Vamana indexes differ")
}
