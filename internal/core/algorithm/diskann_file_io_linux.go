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
	"errors"
	"fmt"
	"io"
	"os"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/gorse-io/xvec/internal/ailego/parallel"
	"golang.org/x/sys/unix"
)

type diskANNRing interface {
	Read(context.Context, int, int64, []DiskANNReadRequest, [][]byte) error
	Close() error
}

type diskANNRingFactory func(uint32) (diskANNRing, error)

type linuxDiskANNOpenConfig struct {
	openStable func(string) (*os.File, error)
	openDirect func(*os.File) (int, error)
	newRing    diskANNRingFactory
}

type linuxDiskANNReader struct {
	lifeMu sync.RWMutex
	closed bool

	stable   *os.File
	directFD int
	size     int64
	workers  int
	capacity int
	newRing  diskANNRingFactory

	fallback     atomic.Bool
	fallbackOnce sync.Once
	fallbackCh   chan struct{}
	poolChange   chan struct{}
	poolMu       sync.Mutex
	created      int
	idle         chan diskANNRing
}

var errDiskANNUsePread = errors.New("core: use DiskANN pread fallback")

func openDiskANNDirectReader(path string, workers int) (diskANNReaderAt, error) {
	return openLinuxDiskANNReader(path, workers, linuxDiskANNOpenConfig{})
}

func openLinuxDiskANNReader(path string, workers int, config linuxDiskANNOpenConfig) (_ *linuxDiskANNReader, resultErr error) {
	if config.openStable == nil {
		config.openStable = os.Open
	}
	if config.openDirect == nil {
		config.openDirect = reopenDiskANNDirect
	}
	if config.newRing == nil {
		config.newRing = func(entries uint32) (diskANNRing, error) {
			return newDiskANNIOUring(entries)
		}
	}

	stable, err := config.openStable(path)
	if err != nil {
		return nil, err
	}
	directFD := -1
	defer func() {
		if resultErr == nil {
			return
		}
		if directFD >= 0 {
			_ = unix.Close(directFD)
		}
		_ = stable.Close()
	}()

	info, err := stable.Stat()
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() {
		return nil, errors.New("core: DiskANN reader requires a regular file")
	}

	directFD, err = config.openDirect(stable)
	if err != nil {
		if !isDiskANNDirectCapabilityError(err) {
			return nil, err
		}
		directFD = -1
	}

	capacity := workers
	if capacity == 0 {
		capacity = runtime.GOMAXPROCS(0)
	}
	if capacity < 1 {
		capacity = 1
	}
	reader := &linuxDiskANNReader{
		stable: stable, directFD: directFD, size: info.Size(), workers: workers,
		capacity: capacity, newRing: config.newRing, idle: make(chan diskANNRing, capacity),
		fallbackCh: make(chan struct{}), poolChange: make(chan struct{}),
	}

	return reader, nil
}

func reopenDiskANNDirect(stable *os.File) (int, error) {
	return unix.Open(fmt.Sprintf("/proc/self/fd/%d", stable.Fd()), unix.O_RDONLY|unix.O_CLOEXEC|unix.O_DIRECT, 0)
}

func isDiskANNDirectCapabilityError(err error) bool {
	return errors.Is(err, unix.EINVAL) || errors.Is(err, unix.EOPNOTSUPP) || errors.Is(err, unix.ENOTSUP) ||
		errors.Is(err, unix.EPERM) || errors.Is(err, unix.EACCES) || errors.Is(err, unix.ENOSYS) || errors.Is(err, unix.ENOENT)
}

func isDiskANNIOUringCapabilityError(err error) bool {
	return errors.Is(err, unix.ENOSYS) || errors.Is(err, unix.EPERM) || errors.Is(err, unix.EACCES) ||
		errors.Is(err, unix.EOPNOTSUPP) || isDiskANNIOUringCapabilityEINVAL(err)
}

func isDiskANNIOUringCapabilityEINVAL(err error) bool {
	var capabilityErr *diskANNIOUringCapabilityError
	return errors.Is(err, unix.EINVAL) && errors.As(err, &capabilityErr)
}

func (r *linuxDiskANNReader) Size() int64 { return r.size }

func (r *linuxDiskANNReader) ReadAt(buffer []byte, offset int64) (int, error) {
	r.lifeMu.RLock()
	defer r.lifeMu.RUnlock()
	if r.closed {
		return 0, os.ErrClosed
	}
	n, err := r.stable.ReadAt(buffer, offset)
	if err == nil && n != len(buffer) {
		return n, io.EOF
	}
	return n, err
}

func (r *linuxDiskANNReader) ReadBatchAt(ctx context.Context, requests []DiskANNReadRequest, buffers [][]byte) error {
	if err := r.validateBatch(ctx, requests, buffers); err != nil {
		return err
	}
	if len(requests) == 0 {
		return nil
	}

	r.lifeMu.RLock()
	defer r.lifeMu.RUnlock()
	if r.closed {
		return os.ErrClosed
	}
	if r.fallback.Load() {
		return r.preadBatch(ctx, requests, buffers)
	}

	ring, err := r.acquireRing(ctx)
	if errors.Is(err, errDiskANNUsePread) {
		return r.preadBatch(ctx, requests, buffers)
	}
	if err != nil {
		return err
	}
	if r.fallback.Load() {
		r.releaseRing(ring)
		return r.preadBatch(ctx, requests, buffers)
	}
	fd := r.directFD
	if fd < 0 {
		fd = int(r.stable.Fd())
	}
	err = ring.Read(ctx, fd, r.size, requests, buffers)
	if err != nil && !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) {
		r.disableRings(ring)
	} else {
		r.releaseRing(ring)
	}
	return err
}

func (r *linuxDiskANNReader) validateBatch(ctx context.Context, requests []DiskANNReadRequest, buffers [][]byte) error {
	if ctx == nil {
		return errors.New("core: nil DiskANN Linux read context")
	}
	if len(requests) != len(buffers) {
		return errors.New("core: inconsistent DiskANN Linux batch")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	for index, request := range requests {
		if request.Offset < 0 || request.Length <= 0 || request.Offset > r.size-int64(request.Length) || len(buffers[index]) != request.Length {
			return fmt.Errorf("%w: invalid Linux request %d", ErrDiskANNShortRead, index)
		}
	}
	return nil
}

func (r *linuxDiskANNReader) acquireRing(ctx context.Context) (diskANNRing, error) {
	for {
		r.poolMu.Lock()
		if r.fallback.Load() {
			r.poolMu.Unlock()
			return nil, errDiskANNUsePread
		}
		select {
		case ring := <-r.idle:
			r.poolMu.Unlock()
			return ring, nil
		default:
		}

		if r.created < r.capacity {
			r.created++
			r.poolMu.Unlock()
			ring, err := r.newRing(diskANNIOUringBatchCap)
			if err != nil {
				r.poolMu.Lock()
				r.created--
				close(r.poolChange)
				r.poolChange = make(chan struct{})
				r.poolMu.Unlock()
				if isDiskANNIOUringCapabilityError(err) {
					r.disableRings(nil)
					return nil, errDiskANNUsePread
				}
				return nil, err
			}
			if r.fallback.Load() {
				r.releaseRing(ring)
				return nil, errDiskANNUsePread
			}
			return ring, nil
		}
		poolChange := r.poolChange
		r.poolMu.Unlock()

		select {
		case ring := <-r.idle:
			r.poolMu.Lock()
			if r.fallback.Load() {
				r.created--
				r.poolMu.Unlock()
				_ = ring.Close()
				return nil, errDiskANNUsePread
			}
			r.poolMu.Unlock()
			return ring, nil
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-r.fallbackCh:
			return nil, errDiskANNUsePread
		case <-poolChange:
			continue
		}
	}
}

func (r *linuxDiskANNReader) releaseRing(ring diskANNRing) {
	r.poolMu.Lock()
	if r.fallback.Load() {
		r.created--
		r.poolMu.Unlock()
		_ = ring.Close()
		return
	}
	r.idle <- ring
	r.poolMu.Unlock()
}

func (r *linuxDiskANNReader) setFallback() {
	r.fallback.Store(true)
	r.fallbackOnce.Do(func() { close(r.fallbackCh) })
}

func (r *linuxDiskANNReader) disableRings(current diskANNRing) {
	r.poolMu.Lock()
	r.setFallback()
	rings := make([]diskANNRing, 0, len(r.idle)+1)
	if current != nil {
		rings = append(rings, current)
		r.created--
	}
	for len(r.idle) > 0 {
		rings = append(rings, <-r.idle)
		r.created--
	}
	r.poolMu.Unlock()
	for _, ring := range rings {
		_ = ring.Close()
	}
}

func (r *linuxDiskANNReader) preadBatch(ctx context.Context, requests []DiskANNReadRequest, buffers [][]byte) error {
	return parallel.ParallelFor(ctx, len(requests), r.workers, func(ctx context.Context, index int) error {
		if err := readFullAt(ctx, r.stable, buffers[index], requests[index].Offset); err != nil {
			return fmt.Errorf("core: DiskANN pread request %d: %w", index, err)
		}
		return nil
	})
}

func (r *linuxDiskANNReader) Close() error {
	r.lifeMu.Lock()
	defer r.lifeMu.Unlock()
	if r.closed {
		return nil
	}
	r.closed = true

	var result error
	r.poolMu.Lock()
	rings := make([]diskANNRing, 0, len(r.idle))
	for len(r.idle) > 0 {
		rings = append(rings, <-r.idle)
	}
	r.created = 0
	r.poolMu.Unlock()
	for _, ring := range rings {
		if err := ring.Close(); err != nil && result == nil {
			result = err
		}
	}
	if r.directFD >= 0 {
		if err := unix.Close(r.directFD); err != nil && result == nil {
			result = err
		}
		r.directFD = -1
	}
	if err := r.stable.Close(); err != nil && result == nil {
		result = err
	}
	return result
}
