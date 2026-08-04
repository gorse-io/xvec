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
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/gorse-io/zvec/internal/ailego"
	"github.com/stretchr/testify/require"
)

func TestDiskANNPersistenceRoundTripSearchCacheAndReplace(t *testing.T) {
	options := DefaultDiskANNBuildOptions(MetricCosine)
	options.MaxDegree, options.ListSize, options.PQChunks, options.CacheCapacity = 8, 24, 4, 20
	candidates := diskANNIndexCandidates(96, 8)
	index := buildDiskANNIndex(t, candidates, options)
	dir := t.TempDir()
	path := filepath.Join(dir, "vectors.diskann")
	{
		err := index.Save(context.Background(), path)
		require.NoError(t, err)
	}

	info, err := os.Stat(path)
	require.NoError(t, err)
	assertPrivateFileMode(t, info.Mode())
	opened, err := OpenDiskANNIndex(context.Background(), path, 24, 4)
	require.NoError(t, err)

	defer opened.Close()
	require.Equal(t, index.Dimension(), opened.Dimension())
	require.Equal(t, index.Metric(), opened.Metric())
	require.Equal(t, index.Len(), opened.Len())
	require.Equal(t, index.PQChunks(), opened.PQChunks())
	require.Equal(t, DiskANNBuildOptions{
		Metric: MetricCosine, MaxDegree: 8, ListSize: 24, PQChunks: 4, Workers: 4, CacheCapacity: 24,
	}, opened.BuildOptions())

	query := candidates[31].Vector
	search := DiskANNSearchOptions{SearchOptions: SearchOptions{TopK: 15}, ListSize: 72}
	want, err := index.SearchDiskANN(context.Background(), query, search)
	require.NoError(t, err)

	got, err := opened.SearchDiskANN(context.Background(), query, search)
	require.NoError(t, err)
	require.Equal(t, want, got)
	{
		warmed, err := opened.WarmCache(context.Background(), 10)
		require.NoError(t, err)
		require.True(t, warmed == 10)
	}

	vector, found := opened.Vector(candidates[40].Key)
	require.True(t, found,
		"opened original vector differs")
	require.True(t, slices.Equal(vector, candidates[40].Vector),
		"opened original vector differs")

	copyPath := filepath.Join(dir, "copy.diskann")
	{
		err := opened.Save(context.Background(), copyPath)
		require.NoError(t, err)
	}

	copyIndex, err := OpenDiskANNIndex(context.Background(), copyPath, 0, 1)
	require.NoError(t, err)

	copyResults, err := copyIndex.SearchDiskANN(context.Background(), query, search)
	require.NoError(t, err)
	require.Equal(t, want, copyResults,
		"saved-opened index changed results")
	{
		err := copyIndex.Close()
		require.NoError(t, err)
	}
	{
		_, err := copyIndex.SearchDiskANN(context.Background(), query, search)
		require.ErrorIs(t, err, ErrDiskANNClosed)
	}

	replacementOptions := DefaultDiskANNBuildOptions(MetricIP)
	replacementOptions.MaxDegree, replacementOptions.ListSize, replacementOptions.PQChunks = 5, 12, 2
	replacement := buildDiskANNIndex(t, diskANNIndexCandidates(40, 6), replacementOptions)
	{
		err := replacement.Save(context.Background(), copyPath)
		require.NoError(t, err)
	}

	reopened, err := OpenDiskANNIndex(context.Background(), copyPath, 0, 2)
	require.NoError(t, err)

	defer reopened.Close()
	require.Equal(t, MetricIP, reopened.Metric())
	require.True(t, reopened.Dimension() == 6)
	require.True(t, reopened.Len() == 40)
}

func TestDiskANNPersistenceMIPSL2TraversalState(t *testing.T) {
	options := DefaultDiskANNBuildOptions(MetricMIPSL2)
	options.MaxDegree, options.ListSize, options.PQChunks = 6, 16, 3
	candidates := diskANNIndexCandidates(48, 6)
	index := buildDiskANNIndex(t, candidates, options)
	path := filepath.Join(t.TempDir(), "mips.diskann")
	{
		err := index.Save(context.Background(), path)
		require.NoError(t, err)
	}

	opened, err := OpenDiskANNIndex(context.Background(), path, 8, 2)
	require.NoError(t, err)

	defer opened.Close()
	require.Len(t, opened.codeNorms, len(candidates))
	require.Equal(t, MetricL2, opened.traversalMetric)

	search := DiskANNSearchOptions{SearchOptions: SearchOptions{TopK: 10}, ListSize: len(candidates)}
	want, err := index.SearchDiskANN(context.Background(), candidates[19].Vector, search)
	require.NoError(t, err)

	got, err := opened.SearchDiskANN(context.Background(), candidates[19].Vector, search)
	require.NoError(t, err)
	require.Equal(t, want, got)
}

func TestDiskANNPersistenceEmptyCancellationAndErrors(t *testing.T) {
	options := DefaultDiskANNBuildOptions(MetricL2)
	options.MaxDegree, options.ListSize = 4, 8
	empty := buildDiskANNIndex(t, nil, options)
	dir := t.TempDir()
	path := filepath.Join(dir, "empty.diskann")
	{
		err := empty.Save(context.Background(), path)
		require.NoError(t, err)
	}

	original, err := os.ReadFile(path)
	require.NoError(t, err)

	opened, err := OpenDiskANNIndex(context.Background(), path, 0, 0)
	require.NoError(t, err)
	require.True(t, opened.Len() == 0,
		"empty metadata differs")
	require.True(t, opened.PQChunks() == 0,
		"empty metadata differs")
	{
		err := opened.Close()
		require.NoError(t, err)
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	{
		err := empty.Save(canceled, path)
		require.ErrorIs(t, err, context.Canceled)
	}

	after, err := os.ReadFile(path)
	require.NoError(t, err)
	require.True(t, slices.Equal(after, original),
		"canceled Save changed artifact")

	nonEmptyOptions := DefaultDiskANNBuildOptions(MetricL2)
	nonEmptyOptions.MaxDegree, nonEmptyOptions.ListSize, nonEmptyOptions.PQChunks = 6, 16, 4
	nonEmpty := buildDiskANNIndex(t, diskANNIndexCandidates(96, 8), nonEmptyOptions)
	midSave := newCancelAfterChecks(5)
	{
		err := nonEmpty.Save(midSave, path)
		require.ErrorIs(t, err, context.Canceled)
	}

	after, err = os.ReadFile(path)
	require.NoError(t, err)
	require.True(t, slices.Equal(after, original),
		"mid-save cancellation changed artifact")

	midOpen := newCancelAfterChecks(5)
	{
		_, err := OpenDiskANNIndex(midOpen, path, 0, 1)
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		_, err := OpenDiskANNIndex(canceled, path, 0, 0)
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		err := empty.Save(nil, path)
		require.Error(t, err,
			"nil Save context succeeded")
	}
	{
		_, err := OpenDiskANNIndex(nil, path, 0, 0)
		require.Error(t, err,
			"nil Open context succeeded")
	}
	{
		err := empty.Save(context.Background(), "")
		require.ErrorIs(t, err, ErrInvalidDiskANNFile)
	}
	{
		_, err := OpenDiskANNIndex(context.Background(), "", 0, 0)
		require.ErrorIs(t, err, ErrInvalidDiskANNFile)
	}
	{
		_, err := OpenDiskANNIndex(context.Background(), filepath.Join(dir, "missing"), 0, 0)
		require.ErrorIs(t, err, os.ErrNotExist)
	}
	{
		_, err := OpenDiskANNIndex(context.Background(), path, -1, 0)
		require.ErrorIs(t, err, ErrInvalidDiskANNOptions)
	}

	var nilIndex *DiskANNIndex
	{
		err := nilIndex.Save(context.Background(), path)
		require.ErrorIs(t, err, ErrInvalidDiskANNFile)
	}
}

func TestDiskANNPersistenceDetectsHeaderSectionAndRecordCorruption(t *testing.T) {
	options := DefaultDiskANNBuildOptions(MetricL2)
	options.MaxDegree, options.ListSize, options.PQChunks = 5, 12, 3
	valid, err := encodeDiskANNIndex(context.Background(), buildDiskANNIndex(t, diskANNIndexCandidates(24, 6), options))
	require.NoError(t, err)

	for _, cut := range []int{0, 1, diskANNIndexHeaderSize - 1, diskANNIndexHeaderSize, len(valid) - 1} {
		{
			_, err := openDiskANNIndexReader(context.Background(), bytes.NewReader(valid[:cut]), int64(cut), 0, 1, nil)
			require.Error(t, err)
		}
	}
	trailing := append(slices.Clone(valid), 0)
	{
		_, err := openDiskANNIndexReader(context.Background(), bytes.NewReader(trailing), int64(len(trailing)), 0, 1, nil)
		require.ErrorIs(t, err, ErrInvalidDiskANNFile)
	}

	badMagic := slices.Clone(valid)
	badMagic[0] ^= 1
	{
		_, err := openDiskANNIndexReader(context.Background(), bytes.NewReader(badMagic), int64(len(badMagic)), 0, 1, nil)
		require.ErrorIs(t, err, ErrInvalidDiskANNFile)
	}

	badVersion := slices.Clone(valid)
	binary.LittleEndian.PutUint16(badVersion[8:10], diskANNIndexFileVersion+1)
	{
		_, err := openDiskANNIndexReader(context.Background(), bytes.NewReader(badVersion), int64(len(badVersion)), 0, 1, nil)
		require.ErrorIs(t, err, ErrUnsupportedDiskANNIndexVersion)
	}

	badHeader := slices.Clone(valid)
	badHeader[40] ^= 1
	{
		_, err := openDiskANNIndexReader(context.Background(), bytes.NewReader(badHeader), int64(len(badHeader)), 0, 1, nil)
		require.ErrorIs(t, err, ErrDiskANNIndexChecksumMismatch)
	}

	for _, offset := range []int{
		int(binary.LittleEndian.Uint64(valid[64:72])),
		int(binary.LittleEndian.Uint64(valid[80:88])),
		int(binary.LittleEndian.Uint64(valid[96:104])),
		int(binary.LittleEndian.Uint64(valid[112:120])),
		int(binary.LittleEndian.Uint64(valid[128:136])),
	} {
		corrupt := slices.Clone(valid)
		corrupt[offset] ^= 1
		{
			_, err := openDiskANNIndexReader(context.Background(), bytes.NewReader(corrupt), int64(len(corrupt)), 0, 1, nil)
			require.ErrorIs(t, err, ErrDiskANNIndexChecksumMismatch)
		}
	}

	// Refresh both whole-section checksums but leave the node record checksum
	// stale. Open accepts the complete sections; the random read detects the
	// damaged record at its own integrity boundary.
	recordCorrupt := slices.Clone(valid)
	nodesOffset := int(binary.LittleEndian.Uint64(recordCorrupt[128:136]))
	nodesLength := int(binary.LittleEndian.Uint64(recordCorrupt[136:144]))
	recordCorrupt[nodesOffset+diskANNNodeHeaderSize+4] ^= 1
	nodeHeader := recordCorrupt[nodesOffset : nodesOffset+diskANNNodeHeaderSize]
	nodeData := recordCorrupt[nodesOffset+diskANNNodeHeaderSize : nodesOffset+nodesLength]
	binary.LittleEndian.PutUint32(nodeHeader[72:76], ailego.CRC32C(nodeData))
	binary.LittleEndian.PutUint32(nodeHeader[diskANNNodeHeaderCRCPos:], ailego.CRC32C(nodeHeader[:diskANNNodeHeaderCRCPos]))
	binary.LittleEndian.PutUint32(recordCorrupt[160:164], ailego.CRC32C(recordCorrupt[nodesOffset:nodesOffset+nodesLength]))
	binary.LittleEndian.PutUint32(recordCorrupt[diskANNIndexHeaderCRCPos:], ailego.CRC32C(recordCorrupt[:diskANNIndexHeaderCRCPos]))
	opened, err := openDiskANNIndexReader(context.Background(), bytes.NewReader(recordCorrupt), int64(len(recordCorrupt)), 0, 1, nil)
	require.NoError(t, err)
	{
		_, err := opened.nodes.ReadNode(context.Background(), 0)
		require.ErrorIs(t, err, ErrDiskANNChecksumMismatch)
	}
}

func FuzzDiskANNIndexFile(f *testing.F) {
	options := DefaultDiskANNBuildOptions(MetricL2)
	options.MaxDegree, options.ListSize, options.PQChunks = 3, 6, 2
	valid, err := encodeDiskANNIndex(context.Background(), buildDiskANNIndex(f, diskANNIndexCandidates(8, 4), options))
	require.NoError(f, err)

	f.Add(valid)
	f.Add([]byte("not-a-diskann-index"))
	f.Fuzz(func(t *testing.T, encoded []byte) {
		if len(encoded) > 4<<20 {
			return
		}
		index, err := openDiskANNIndexReader(context.Background(), bytes.NewReader(encoded), int64(len(encoded)), 0, 1, nil)
		if err == nil {
			_ = index.Close()
		}
	})
}

func BenchmarkDiskANNSearchWarmCache(b *testing.B) {
	options := DefaultDiskANNBuildOptions(MetricL2)
	options.MaxDegree, options.ListSize, options.PQChunks, options.CacheCapacity = 8, 24, 4, 128
	candidates := diskANNIndexCandidates(128, 8)
	index := buildDiskANNIndex(b, candidates, options)
	_, _ = index.WarmCache(context.Background(), 128)
	query := candidates[27].Vector
	search := DiskANNSearchOptions{SearchOptions: SearchOptions{TopK: 10}, ListSize: 48}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		{
			_, err := index.SearchDiskANN(context.Background(), query, search)
			if err != nil {
				require.NoError(b, err)
			}
		}
	}
}
