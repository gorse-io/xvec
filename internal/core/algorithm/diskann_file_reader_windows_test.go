// Copyright 2026-present the xvec project
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

//go:build windows

package core

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestWindowsDiskANNReaderUsesIOCP(t *testing.T) {
	path := filepath.Join(t.TempDir(), "磁盘索引.dat")
	contents := make([]byte, 3*DiskANNSectorSize)
	for sector := range 3 {
		value := byte(sector*37 + 11)
		copy(contents[sector*DiskANNSectorSize:], bytes.Repeat([]byte{value}, DiskANNSectorSize))
	}
	require.NoError(t, os.WriteFile(path, contents, 0o600))

	reader, err := openDiskANNReaderAt(path, false)
	require.NoError(t, err)
	defer func() { require.NoError(t, reader.Close()) }()

	_, ok := reader.(*windowsDiskANNReader)
	require.True(t, ok, "ordinary Windows DiskANN files must use IOCP")

	buffers := [][]byte{make([]byte, DiskANNSectorSize), make([]byte, DiskANNSectorSize)}
	err = reader.(diskANNBatchReader).ReadBatchAt(context.Background(), []DiskANNReadRequest{
		{Offset: 2 * DiskANNSectorSize, Length: DiskANNSectorSize},
		{Offset: 0, Length: DiskANNSectorSize},
	}, buffers)
	require.NoError(t, err)
	require.Equal(t, contents[2*DiskANNSectorSize:], buffers[0])
	require.Equal(t, contents[:DiskANNSectorSize], buffers[1])
}

func TestWindowsDiskANNReaderCancellationAndShortRead(t *testing.T) {
	path := filepath.Join(t.TempDir(), "diskann.dat")
	require.NoError(t, os.WriteFile(path, make([]byte, 2*DiskANNSectorSize), 0o600))

	reader, err := openDiskANNReaderAt(path, false)
	require.NoError(t, err)
	defer func() { require.NoError(t, reader.Close()) }()
	batch := reader.(diskANNBatchReader)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err = batch.ReadBatchAt(ctx, []DiskANNReadRequest{{Offset: 0, Length: DiskANNSectorSize}}, [][]byte{make([]byte, DiskANNSectorSize)})
	require.ErrorIs(t, err, context.Canceled)

	err = batch.ReadBatchAt(context.Background(), []DiskANNReadRequest{{Offset: 2 * DiskANNSectorSize, Length: DiskANNSectorSize}}, [][]byte{make([]byte, DiskANNSectorSize)})
	require.ErrorIs(t, err, ErrDiskANNShortRead)
}
