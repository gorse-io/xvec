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

package core

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/require"
)

type batchOnlyDiskANNReader struct {
	data        []byte
	batchCalled bool
}

func (r *batchOnlyDiskANNReader) ReadAt([]byte, int64) (int, error) {
	return 0, errors.New("unexpected scalar read")
}

func (r *batchOnlyDiskANNReader) ReadBatchAt(
	ctx context.Context, requests []DiskANNReadRequest, buffers [][]byte,
) error {
	r.batchCalled = true
	for index, request := range requests {
		if err := ctx.Err(); err != nil {
			return err
		}
		copy(buffers[index], r.data[request.Offset:request.Offset+int64(request.Length)])
	}
	return nil
}

func TestParallelReadAtUsesBatchReader(t *testing.T) {
	reader := &batchOnlyDiskANNReader{data: []byte("0123456789")}
	got, err := ParallelReadAt(context.Background(), reader, []DiskANNReadRequest{
		{Offset: 6, Length: 4},
		{Offset: 1, Length: 3},
	}, 2)
	require.NoError(t, err)
	require.True(t, reader.batchCalled)
	require.Equal(t, [][]byte{[]byte("6789"), []byte("123")}, got)
}

func TestParallelReadAtUsesBatchSectionReader(t *testing.T) {
	reader := &batchOnlyDiskANNReader{data: []byte("0123456789")}
	section := newDiskANNSectionReader(reader, 2, 6)
	got, err := ParallelReadAt(context.Background(), section, []DiskANNReadRequest{
		{Offset: 1, Length: 3},
	}, 2)
	require.NoError(t, err)
	require.True(t, reader.batchCalled)
	require.Equal(t, [][]byte{[]byte("345")}, got)
}
