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

//go:build linux

package core

import (
	"context"
	"fmt"
	"os"
	"testing"

	"golang.org/x/sys/unix"
)

const benchmarkDiskANNSectors = 1024

func BenchmarkLinuxDiskANNBufferedBatchRead(b *testing.B) {
	path := b.TempDir() + "/index"
	file, err := os.Create(path)
	if err != nil {
		b.Fatal(err)
	}
	if err = file.Truncate(benchmarkDiskANNSectors * DiskANNSectorSize); err != nil {
		_ = file.Close()
		b.Fatal(err)
	}
	if err = file.Close(); err != nil {
		b.Fatal(err)
	}

	for _, batchSize := range []int{1, 8, 32, MaxDiskANNReadSectors} {
		b.Run(fmt.Sprintf("batch=%d", batchSize), func(b *testing.B) {
			b.Run("pread", func(b *testing.B) {
				reader, err := openLinuxDiskANNReader(path, 1, linuxDiskANNOpenConfig{
					openDirect: func(*os.File) (int, error) { return -1, unix.EOPNOTSUPP },
					newRing: func(uint32) (diskANNRing, error) {
						return nil, unix.EPERM
					},
				})
				if err != nil {
					b.Fatal(err)
				}
				defer func() {
					if err := reader.Close(); err != nil {
						b.Error(err)
					}
				}()
				warmLinuxDiskANNReader(b, reader)
				benchmarkLinuxDiskANNReader(b, reader, batchSize)
			})

			b.Run("io_uring", func(b *testing.B) {
				reader, err := openLinuxDiskANNReader(path, 1, linuxDiskANNOpenConfig{
					openDirect: func(*os.File) (int, error) { return -1, unix.EOPNOTSUPP },
				})
				if err != nil {
					if isDiskANNIOUringCapabilityError(err) {
						b.Skipf("io_uring unavailable: %v", err)
					}
					b.Fatal(err)
				}
				defer func() {
					if err := reader.Close(); err != nil {
						b.Error(err)
					}
				}()
				warmLinuxDiskANNReader(b, reader)
				if reader.fallback.Load() {
					b.Skip("io_uring unavailable")
				}
				benchmarkLinuxDiskANNReader(b, reader, batchSize)
			})
		})
	}
}

func warmLinuxDiskANNReader(b *testing.B, reader *linuxDiskANNReader) {
	buffer := [][]byte{make([]byte, DiskANNSectorSize)}
	request := []DiskANNReadRequest{{Length: DiskANNSectorSize}}
	if err := reader.ReadBatchAt(context.Background(), request, buffer); err != nil {
		b.Fatal(err)
	}
}

func benchmarkLinuxDiskANNReader(b *testing.B, reader *linuxDiskANNReader, batchSize int) {
	buffers := make([][]byte, batchSize)
	requests := make([]DiskANNReadRequest, batchSize)
	for i := range buffers {
		buffers[i] = make([]byte, DiskANNSectorSize)
		requests[i].Length = DiskANNSectorSize
	}

	ctx := context.Background()
	b.SetBytes(int64(batchSize * DiskANNSectorSize))
	b.ReportAllocs()
	b.ResetTimer()
	for iteration := 0; iteration < b.N; iteration++ {
		for i := range requests {
			sector := (iteration*batchSize + i) % benchmarkDiskANNSectors
			requests[i].Offset = int64(sector * DiskANNSectorSize)
		}
		if err := reader.ReadBatchAt(ctx, requests, buffers); err != nil {
			b.Fatal(err)
		}
	}
}
