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
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"golang.org/x/sys/unix"
)

type fakeDiskANNRing struct {
	read  func(context.Context, int, int64, []DiskANNReadRequest, [][]byte) error
	close func() error
}

func (r *fakeDiskANNRing) Read(ctx context.Context, fd int, size int64, requests []DiskANNReadRequest, buffers [][]byte) error {
	if r.read == nil {
		return nil
	}
	return r.read(ctx, fd, size, requests, buffers)
}

func (r *fakeDiskANNRing) Close() error {
	if r.close == nil {
		return nil
	}
	return r.close()
}

func writeLinuxDiskANNTestFile(t *testing.T, name string, sectors int) (string, []byte) {
	t.Helper()
	contents := make([]byte, sectors*DiskANNSectorSize)
	for i := range contents {
		contents[i] = byte(i*31 + 7)
	}
	path := filepath.Join(t.TempDir(), name)
	require.NoError(t, os.WriteFile(path, contents, 0o600))
	return path, contents
}

func preadFakeRing(callCount *atomic.Int32) *fakeDiskANNRing {
	return &fakeDiskANNRing{read: func(_ context.Context, fd int, _ int64, requests []DiskANNReadRequest, buffers [][]byte) error {
		callCount.Add(1)
		for i, request := range requests {
			n, err := unix.Pread(fd, buffers[i], request.Offset)
			if err != nil {
				return err
			}
			if n != request.Length {
				return ErrDiskANNShortRead
			}
		}
		return nil
	}}
}

func TestLinuxDiskANNReaderKeepsStableInodeAfterPathReplacement(t *testing.T) {
	path, original := writeLinuxDiskANNTestFile(t, "index", 2)
	var stableStat, directStat unix.Stat_t
	ring := &fakeDiskANNRing{read: func(_ context.Context, fd int, _ int64, _ []DiskANNReadRequest, buffers [][]byte) error {
		require.NoError(t, unix.Fstat(fd, &directStat))
		copy(buffers[0], original[:len(buffers[0])])
		return nil
	}}
	reader, err := openLinuxDiskANNReader(path, 1, linuxDiskANNOpenConfig{
		newRing: func(uint32) (diskANNRing, error) { return ring, nil },
	})
	require.NoError(t, err)
	defer func() { require.NoError(t, reader.Close()) }()
	require.NoError(t, unix.Fstat(int(reader.stable.Fd()), &stableStat))

	replacement := bytes.Repeat([]byte{0xee}, len(original))
	temporary := path + ".new"
	require.NoError(t, os.WriteFile(temporary, replacement, 0o600))
	require.NoError(t, os.Rename(temporary, path))

	got := make([]byte, 32)
	n, err := reader.ReadAt(got, 0)
	require.NoError(t, err)
	require.Equal(t, len(got), n)
	require.Equal(t, original[:len(got)], got)
	buffers := [][]byte{make([]byte, DiskANNSectorSize)}
	require.NoError(t, reader.ReadBatchAt(context.Background(), []DiskANNReadRequest{{Offset: 0, Length: DiskANNSectorSize}}, buffers))
	require.Equal(t, stableStat.Ino, directStat.Ino)
}

func TestLinuxDiskANNReaderDirectUnavailableUsesStableFDWithRing(t *testing.T) {
	for _, capabilityErr := range []error{unix.EOPNOTSUPP, unix.EINVAL, unix.EPERM, unix.EACCES, unix.ENOSYS} {
		t.Run(capabilityErr.Error(), func(t *testing.T) {
			path, contents := writeLinuxDiskANNTestFile(t, "index", 2)
			var calls atomic.Int32
			var gotFD int
			ring := preadFakeRing(&calls)
			originalRead := ring.read
			ring.read = func(ctx context.Context, fd int, size int64, requests []DiskANNReadRequest, buffers [][]byte) error {
				gotFD = fd
				return originalRead(ctx, fd, size, requests, buffers)
			}
			reader, err := openLinuxDiskANNReader(path, 1, linuxDiskANNOpenConfig{
				openDirect: func(*os.File) (int, error) { return -1, capabilityErr },
				newRing:    func(uint32) (diskANNRing, error) { return ring, nil },
			})
			require.NoError(t, err)
			defer func() { require.NoError(t, reader.Close()) }()

			buffers := [][]byte{make([]byte, DiskANNSectorSize)}
			require.NoError(t, reader.ReadBatchAt(context.Background(), []DiskANNReadRequest{{Offset: DiskANNSectorSize, Length: DiskANNSectorSize}}, buffers))
			require.Equal(t, int(reader.stable.Fd()), gotFD)
			require.Equal(t, contents[DiskANNSectorSize:], buffers[0])
			require.EqualValues(t, 1, calls.Load())
		})
	}
}

func TestLinuxDiskANNReaderMissingProcFDUsesStableFDWithRing(t *testing.T) {
	path, contents := writeLinuxDiskANNTestFile(t, "index", 1)
	var calls atomic.Int32
	var gotFD int
	ring := preadFakeRing(&calls)
	originalRead := ring.read
	ring.read = func(ctx context.Context, fd int, size int64, requests []DiskANNReadRequest, buffers [][]byte) error {
		gotFD = fd
		return originalRead(ctx, fd, size, requests, buffers)
	}
	reader, err := openLinuxDiskANNReader(path, 1, linuxDiskANNOpenConfig{
		openDirect: func(*os.File) (int, error) { return -1, unix.ENOENT },
		newRing:    func(uint32) (diskANNRing, error) { return ring, nil },
	})
	require.NoError(t, err)
	defer func() { require.NoError(t, reader.Close()) }()

	buffers := [][]byte{make([]byte, DiskANNSectorSize)}
	require.NoError(t, reader.ReadBatchAt(context.Background(), []DiskANNReadRequest{{Offset: 0, Length: DiskANNSectorSize}}, buffers))
	require.Equal(t, int(reader.stable.Fd()), gotFD)
	require.Equal(t, contents, buffers[0])
	require.EqualValues(t, 1, calls.Load())
}

func TestLinuxDiskANNReaderRingUnavailableUsesPreadNowAndLater(t *testing.T) {
	path, contents := writeLinuxDiskANNTestFile(t, "index", 2)
	var factories atomic.Int32
	reader, err := openLinuxDiskANNReader(path, 2, linuxDiskANNOpenConfig{
		openDirect: func(*os.File) (int, error) { return -1, unix.EOPNOTSUPP },
		newRing: func(uint32) (diskANNRing, error) {
			factories.Add(1)
			return nil, unix.ENOSYS
		},
	})
	require.NoError(t, err)
	defer func() { require.NoError(t, reader.Close()) }()
	require.Zero(t, factories.Load(), "opening a reader must not create an io_uring")

	for range 2 {
		buffers := [][]byte{make([]byte, 17), make([]byte, 23)}
		err = reader.ReadBatchAt(context.Background(), []DiskANNReadRequest{{Offset: 19, Length: 17}, {Offset: 71, Length: 23}}, buffers)
		require.NoError(t, err)
		require.Equal(t, contents[19:36], buffers[0])
		require.Equal(t, contents[71:94], buffers[1])
	}
	require.EqualValues(t, 1, factories.Load())
}

func TestLinuxDiskANNReaderSurfacesOpenResourceErrors(t *testing.T) {
	path, _ := writeLinuxDiskANNTestFile(t, "index", 1)
	reader, err := openLinuxDiskANNReader(path, 1, linuxDiskANNOpenConfig{
		openDirect: func(*os.File) (int, error) { return -1, unix.ENOMEM },
	})
	require.Nil(t, reader)
	require.ErrorIs(t, err, unix.ENOMEM)

	reader, err = openLinuxDiskANNReader(path, 1, linuxDiskANNOpenConfig{
		openDirect: func(*os.File) (int, error) { return -1, unix.EOPNOTSUPP },
		newRing:    func(uint32) (diskANNRing, error) { return nil, unix.ENOMEM },
	})
	require.NoError(t, err)
	require.NotNil(t, reader)
	err = reader.ReadBatchAt(context.Background(), []DiskANNReadRequest{{Offset: 0, Length: DiskANNSectorSize}}, [][]byte{make([]byte, DiskANNSectorSize)})
	require.ErrorIs(t, err, unix.ENOMEM)
	require.NoError(t, reader.Close())
}

func TestLinuxDiskANNReaderDoesNotTreatArbitraryFactoryEINVALAsCapabilityFailure(t *testing.T) {
	path, _ := writeLinuxDiskANNTestFile(t, "index", 1)
	reader, err := openLinuxDiskANNReader(path, 1, linuxDiskANNOpenConfig{
		openDirect: func(*os.File) (int, error) { return -1, unix.EOPNOTSUPP },
		newRing:    func(uint32) (diskANNRing, error) { return nil, unix.EINVAL },
	})
	require.NoError(t, err)
	defer func() { require.NoError(t, reader.Close()) }()

	err = reader.ReadBatchAt(context.Background(), []DiskANNReadRequest{{Offset: 0, Length: DiskANNSectorSize}}, [][]byte{make([]byte, DiskANNSectorSize)})
	require.ErrorIs(t, err, unix.EINVAL)
	require.False(t, reader.fallback.Load())
}

func TestLinuxDiskANNReaderBoundsConcurrentRings(t *testing.T) {
	path, _ := writeLinuxDiskANNTestFile(t, "index", 1)
	started := make(chan struct{}, 3)
	release := make(chan struct{})
	var active, maximum, factories atomic.Int32
	factory := func(uint32) (diskANNRing, error) {
		factories.Add(1)
		return &fakeDiskANNRing{read: func(context.Context, int, int64, []DiskANNReadRequest, [][]byte) error {
			current := active.Add(1)
			defer active.Add(-1)
			for {
				old := maximum.Load()
				if current <= old || maximum.CompareAndSwap(old, current) {
					break
				}
			}
			started <- struct{}{}
			<-release
			return nil
		}}, nil
	}
	reader, err := openLinuxDiskANNReader(path, 2, linuxDiskANNOpenConfig{
		openDirect: func(*os.File) (int, error) { return -1, unix.EOPNOTSUPP }, newRing: factory,
	})
	require.NoError(t, err)
	defer func() { require.NoError(t, reader.Close()) }()

	var wg sync.WaitGroup
	for range 3 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = reader.ReadBatchAt(context.Background(), []DiskANNReadRequest{{Offset: 0, Length: DiskANNSectorSize}}, [][]byte{make([]byte, DiskANNSectorSize)})
		}()
	}
	<-started
	<-started
	select {
	case <-started:
		t.Fatal("third read started without waiting for a ring")
	case <-time.After(30 * time.Millisecond):
	}
	close(release)
	wg.Wait()
	require.EqualValues(t, 2, factories.Load())
	require.EqualValues(t, 2, maximum.Load())
}

func TestLinuxDiskANNReaderPoolWaitHonorsCancellation(t *testing.T) {
	path, _ := writeLinuxDiskANNTestFile(t, "index", 1)
	started := make(chan struct{})
	release := make(chan struct{})
	ring := &fakeDiskANNRing{read: func(context.Context, int, int64, []DiskANNReadRequest, [][]byte) error {
		close(started)
		<-release
		return nil
	}}
	reader, err := openLinuxDiskANNReader(path, 1, linuxDiskANNOpenConfig{
		openDirect: func(*os.File) (int, error) { return -1, unix.EOPNOTSUPP },
		newRing:    func(uint32) (diskANNRing, error) { return ring, nil },
	})
	require.NoError(t, err)
	defer func() { require.NoError(t, reader.Close()) }()

	firstDone := make(chan error, 1)
	go func() {
		firstDone <- reader.ReadBatchAt(context.Background(), []DiskANNReadRequest{{Offset: 0, Length: DiskANNSectorSize}}, [][]byte{make([]byte, DiskANNSectorSize)})
	}()
	<-started
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	err = reader.ReadBatchAt(ctx, []DiskANNReadRequest{{Offset: 0, Length: DiskANNSectorSize}}, [][]byte{make([]byte, DiskANNSectorSize)})
	require.ErrorIs(t, err, context.DeadlineExceeded)
	close(release)
	require.NoError(t, <-firstDone)
}

type observedDoneContext struct {
	context.Context
	once     sync.Once
	observed chan struct{}
}

func (c *observedDoneContext) Done() <-chan struct{} {
	c.once.Do(func() { close(c.observed) })
	return c.Context.Done()
}

func TestLinuxDiskANNReaderWakesWaiterAfterRingFactoryResourceError(t *testing.T) {
	path, contents := writeLinuxDiskANNTestFile(t, "index", 1)
	firstFactoryStarted := make(chan struct{})
	failFirstFactory := make(chan struct{})
	secondFactoryStarted := make(chan struct{})
	var factories, reads atomic.Int32
	reader, err := openLinuxDiskANNReader(path, 1, linuxDiskANNOpenConfig{
		openDirect: func(*os.File) (int, error) { return -1, unix.EOPNOTSUPP },
		newRing: func(uint32) (diskANNRing, error) {
			if factories.Add(1) == 1 {
				close(firstFactoryStarted)
				<-failFirstFactory
				return nil, unix.ENOMEM
			}
			close(secondFactoryStarted)
			return preadFakeRing(&reads), nil
		},
	})
	require.NoError(t, err)

	request := []DiskANNReadRequest{{Offset: 0, Length: DiskANNSectorSize}}
	firstDone := make(chan error, 1)
	go func() {
		firstDone <- reader.ReadBatchAt(context.Background(), request, [][]byte{make([]byte, DiskANNSectorSize)})
	}()
	<-firstFactoryStarted

	waiterBase, cancelWaiter := context.WithCancel(context.Background())
	waiterContext := &observedDoneContext{Context: waiterBase, observed: make(chan struct{})}
	waiterBuffer := [][]byte{make([]byte, DiskANNSectorSize)}
	waiterDone := make(chan error, 1)
	go func() { waiterDone <- reader.ReadBatchAt(waiterContext, request, waiterBuffer) }()
	<-waiterContext.observed

	close(failFirstFactory)
	require.ErrorIs(t, <-firstDone, unix.ENOMEM)
	select {
	case <-secondFactoryStarted:
	case <-time.After(time.Second):
		cancelWaiter()
		require.ErrorIs(t, <-waiterDone, context.Canceled)
		require.NoError(t, reader.Close())
		t.Fatal("waiter was not woken after the failed factory released capacity")
	}
	require.NoError(t, <-waiterDone)
	cancelWaiter()
	require.Equal(t, contents, waiterBuffer[0])
	require.EqualValues(t, 2, factories.Load())
	require.EqualValues(t, 1, reads.Load())
	require.NoError(t, reader.Close())
}

func TestLinuxDiskANNReaderWakesAllWaitersAfterConcurrentFactoryErrors(t *testing.T) {
	path, _ := writeLinuxDiskANNTestFile(t, "index", 1)
	initialFactoryStarted := make(chan struct{}, 2)
	failInitialFactories := make(chan struct{})
	replacementFactoryStarted := make(chan struct{}, 2)
	releaseReplacementReads := make(chan struct{})
	var factories atomic.Int32
	reader, err := openLinuxDiskANNReader(path, 2, linuxDiskANNOpenConfig{
		openDirect: func(*os.File) (int, error) { return -1, unix.EOPNOTSUPP },
		newRing: func(uint32) (diskANNRing, error) {
			if factories.Add(1) <= 2 {
				initialFactoryStarted <- struct{}{}
				<-failInitialFactories
				return nil, unix.ENOMEM
			}
			replacementFactoryStarted <- struct{}{}
			return &fakeDiskANNRing{read: func(context.Context, int, int64, []DiskANNReadRequest, [][]byte) error {
				<-releaseReplacementReads
				return nil
			}}, nil
		},
	})
	require.NoError(t, err)

	request := []DiskANNReadRequest{{Offset: 0, Length: DiskANNSectorSize}}
	initialDone := make(chan error, 2)
	for range 2 {
		go func() {
			initialDone <- reader.ReadBatchAt(context.Background(), request, [][]byte{make([]byte, DiskANNSectorSize)})
		}()
	}
	<-initialFactoryStarted
	<-initialFactoryStarted

	waiterDone := make(chan error, 2)
	waiterContexts := make([]*observedDoneContext, 2)
	for index := range waiterContexts {
		waiterContexts[index] = &observedDoneContext{Context: context.Background(), observed: make(chan struct{})}
		ctx := waiterContexts[index]
		go func() {
			waiterDone <- reader.ReadBatchAt(ctx, request, [][]byte{make([]byte, DiskANNSectorSize)})
		}()
	}
	for _, ctx := range waiterContexts {
		<-ctx.observed
	}

	close(failInitialFactories)
	for range 2 {
		require.ErrorIs(t, <-initialDone, unix.ENOMEM)
	}
	for range 2 {
		select {
		case <-replacementFactoryStarted:
		case <-time.After(time.Second):
			close(releaseReplacementReads)
			t.Fatal("not every waiter was woken after concurrent factory failures")
		}
	}
	close(releaseReplacementReads)
	for range 2 {
		require.NoError(t, <-waiterDone)
	}
	require.EqualValues(t, 4, factories.Load())
	require.NoError(t, reader.Close())
}

func TestLinuxDiskANNReaderSecondRingCapabilityFailureFallsBackImmediately(t *testing.T) {
	path, contents := writeLinuxDiskANNTestFile(t, "index", 1)
	started := make(chan struct{})
	release := make(chan struct{})
	var factories atomic.Int32
	reader, err := openLinuxDiskANNReader(path, 2, linuxDiskANNOpenConfig{
		openDirect: func(*os.File) (int, error) { return -1, unix.EOPNOTSUPP },
		newRing: func(uint32) (diskANNRing, error) {
			if factories.Add(1) == 1 {
				return &fakeDiskANNRing{read: func(context.Context, int, int64, []DiskANNReadRequest, [][]byte) error {
					close(started)
					<-release
					return nil
				}}, nil
			}
			return nil, unix.EOPNOTSUPP
		},
	})
	require.NoError(t, err)
	defer func() { require.NoError(t, reader.Close()) }()

	firstDone := make(chan error, 1)
	go func() {
		firstDone <- reader.ReadBatchAt(context.Background(), []DiskANNReadRequest{{Offset: 0, Length: DiskANNSectorSize}}, [][]byte{make([]byte, DiskANNSectorSize)})
	}()
	<-started
	for range 2 {
		buffer := [][]byte{make([]byte, 37)}
		err = reader.ReadBatchAt(context.Background(), []DiskANNReadRequest{{Offset: 13, Length: 37}}, buffer)
		require.NoError(t, err)
		require.Equal(t, contents[13:50], buffer[0])
	}
	require.EqualValues(t, 2, factories.Load())
	close(release)
	require.NoError(t, <-firstDone)
}

func TestLinuxDiskANNReaderRuntimeErrorDisablesRingAfterCurrentCall(t *testing.T) {
	path, contents := writeLinuxDiskANNTestFile(t, "index", 1)
	var calls atomic.Int32
	ring := &fakeDiskANNRing{read: func(context.Context, int, int64, []DiskANNReadRequest, [][]byte) error {
		calls.Add(1)
		return unix.EIO
	}}
	reader, err := openLinuxDiskANNReader(path, 1, linuxDiskANNOpenConfig{
		openDirect: func(*os.File) (int, error) { return -1, unix.EOPNOTSUPP },
		newRing:    func(uint32) (diskANNRing, error) { return ring, nil },
	})
	require.NoError(t, err)
	defer func() { require.NoError(t, reader.Close()) }()

	request := []DiskANNReadRequest{{Offset: 9, Length: 31}}
	buffers := [][]byte{make([]byte, 31)}
	err = reader.ReadBatchAt(context.Background(), request, buffers)
	require.ErrorIs(t, err, unix.EIO)
	err = reader.ReadBatchAt(context.Background(), request, buffers)
	require.NoError(t, err)
	require.Equal(t, contents[9:40], buffers[0])
	require.EqualValues(t, 1, calls.Load())
}

func TestLinuxDiskANNReaderRuntimeErrorClosesEveryRingAndCreatesNoMore(t *testing.T) {
	path, contents := writeLinuxDiskANNTestFile(t, "index", 1)
	started := make(chan struct{})
	release := make(chan struct{})
	var factories, closes atomic.Int32
	reader, err := openLinuxDiskANNReader(path, 2, linuxDiskANNOpenConfig{
		openDirect: func(*os.File) (int, error) { return -1, unix.EOPNOTSUPP },
		newRing: func(uint32) (diskANNRing, error) {
			id := factories.Add(1)
			ring := &fakeDiskANNRing{close: func() error { closes.Add(1); return nil }}
			if id == 1 {
				ring.read = func(context.Context, int, int64, []DiskANNReadRequest, [][]byte) error {
					close(started)
					<-release
					return unix.EIO
				}
			}
			return ring, nil
		},
	})
	require.NoError(t, err)

	request := []DiskANNReadRequest{{Offset: 0, Length: DiskANNSectorSize}}
	firstDone := make(chan error, 1)
	go func() {
		firstDone <- reader.ReadBatchAt(context.Background(), request, [][]byte{make([]byte, DiskANNSectorSize)})
	}()
	<-started
	require.NoError(t, reader.ReadBatchAt(context.Background(), request, [][]byte{make([]byte, DiskANNSectorSize)}))
	close(release)
	require.ErrorIs(t, <-firstDone, unix.EIO)
	require.Eventually(t, func() bool { return closes.Load() == 2 }, time.Second, time.Millisecond)

	buffer := [][]byte{make([]byte, 29)}
	require.NoError(t, reader.ReadBatchAt(context.Background(), []DiskANNReadRequest{{Offset: 11, Length: 29}}, buffer))
	require.Equal(t, contents[11:40], buffer[0])
	require.EqualValues(t, 2, factories.Load())
	require.NoError(t, reader.Close())
	require.EqualValues(t, 2, closes.Load())
}

func TestLinuxDiskANNReaderContextErrorDoesNotDisableRing(t *testing.T) {
	path, _ := writeLinuxDiskANNTestFile(t, "index", 1)
	var calls atomic.Int32
	ring := &fakeDiskANNRing{read: func(ctx context.Context, _ int, _ int64, _ []DiskANNReadRequest, _ [][]byte) error {
		if calls.Add(1) == 1 {
			return context.Canceled
		}
		return nil
	}}
	reader, err := openLinuxDiskANNReader(path, 1, linuxDiskANNOpenConfig{
		openDirect: func(*os.File) (int, error) { return -1, unix.EOPNOTSUPP },
		newRing:    func(uint32) (diskANNRing, error) { return ring, nil },
	})
	require.NoError(t, err)
	defer func() { require.NoError(t, reader.Close()) }()

	request := []DiskANNReadRequest{{Offset: 0, Length: DiskANNSectorSize}}
	buffers := [][]byte{make([]byte, DiskANNSectorSize)}
	require.ErrorIs(t, reader.ReadBatchAt(context.Background(), request, buffers), context.Canceled)
	require.NoError(t, reader.ReadBatchAt(context.Background(), request, buffers))
	require.EqualValues(t, 2, calls.Load())
}

func TestLinuxDiskANNReaderCloseWaitsForBorrowerAndIsIdempotent(t *testing.T) {
	path, _ := writeLinuxDiskANNTestFile(t, "index", 1)
	started := make(chan struct{})
	release := make(chan struct{})
	var closes atomic.Int32
	ring := &fakeDiskANNRing{
		read: func(context.Context, int, int64, []DiskANNReadRequest, [][]byte) error {
			close(started)
			<-release
			return nil
		},
		close: func() error { closes.Add(1); return nil },
	}
	reader, err := openLinuxDiskANNReader(path, 1, linuxDiskANNOpenConfig{
		openDirect: func(*os.File) (int, error) { return -1, unix.EOPNOTSUPP },
		newRing:    func(uint32) (diskANNRing, error) { return ring, nil },
	})
	require.NoError(t, err)
	readDone := make(chan error, 1)
	go func() {
		readDone <- reader.ReadBatchAt(context.Background(), []DiskANNReadRequest{{Offset: 0, Length: DiskANNSectorSize}}, [][]byte{make([]byte, DiskANNSectorSize)})
	}()
	<-started
	closeDone := make(chan error, 1)
	go func() { closeDone <- reader.Close() }()
	select {
	case <-closeDone:
		t.Fatal("Close returned while a ring was borrowed")
	case <-time.After(30 * time.Millisecond):
	}
	close(release)
	require.NoError(t, <-readDone)
	require.NoError(t, <-closeDone)
	require.NoError(t, reader.Close())
	require.EqualValues(t, 1, closes.Load())
	_, err = reader.ReadAt(make([]byte, 1), 0)
	require.ErrorIs(t, err, os.ErrClosed)
}

func TestLinuxDiskANNReaderCloseClosesDirectFD(t *testing.T) {
	path, _ := writeLinuxDiskANNTestFile(t, "index", 1)
	var directFD int
	reader, err := openLinuxDiskANNReader(path, 1, linuxDiskANNOpenConfig{
		openDirect: func(file *os.File) (int, error) {
			var duplicateErr error
			directFD, duplicateErr = unix.Dup(int(file.Fd()))
			return directFD, duplicateErr
		},
		newRing: func(uint32) (diskANNRing, error) { return &fakeDiskANNRing{}, nil },
	})
	require.NoError(t, err)
	require.NoError(t, reader.Close())
	_, err = unix.Pread(directFD, make([]byte, 1), 0)
	require.ErrorIs(t, err, unix.EBADF)
	require.NoError(t, reader.Close())
}

func TestLinuxDiskANNReaderPreservesBatchOrderAndBytes(t *testing.T) {
	path, contents := writeLinuxDiskANNTestFile(t, "index", 3)
	var calls atomic.Int32
	reader, err := openLinuxDiskANNReader(path, 1, linuxDiskANNOpenConfig{
		openDirect: func(*os.File) (int, error) { return -1, unix.EOPNOTSUPP },
		newRing:    func(uint32) (diskANNRing, error) { return preadFakeRing(&calls), nil },
	})
	require.NoError(t, err)
	defer func() { require.NoError(t, reader.Close()) }()
	buffers := [][]byte{make([]byte, DiskANNSectorSize), make([]byte, DiskANNSectorSize)}
	err = reader.ReadBatchAt(context.Background(), []DiskANNReadRequest{
		{Offset: 2 * DiskANNSectorSize, Length: DiskANNSectorSize},
		{Offset: 0, Length: DiskANNSectorSize},
	}, buffers)
	require.NoError(t, err)
	require.Equal(t, contents[2*DiskANNSectorSize:], buffers[0])
	require.Equal(t, contents[:DiskANNSectorSize], buffers[1])
}

func TestLinuxDiskANNReaderCloseWithoutReadDoesNotCreateRing(t *testing.T) {
	path, _ := writeLinuxDiskANNTestFile(t, "index", 1)
	var factories atomic.Int32
	reader, err := openLinuxDiskANNReader(path, 1, linuxDiskANNOpenConfig{
		openDirect: func(*os.File) (int, error) { return -1, unix.EOPNOTSUPP },
		newRing: func(uint32) (diskANNRing, error) {
			factories.Add(1)
			return &fakeDiskANNRing{}, nil
		},
	})
	require.NoError(t, err)
	require.Zero(t, factories.Load())
	require.NoError(t, reader.Close())
	require.Zero(t, factories.Load())
}

func TestOpenDiskANNReaderAtLinuxIntegration(t *testing.T) {
	path, contents := writeLinuxDiskANNTestFile(t, "index", 2)
	reader, err := openDiskANNReaderAt(path, false, 2)
	require.NoError(t, err)
	defer func() { require.NoError(t, reader.Close()) }()

	got, err := ParallelReadAt(context.Background(), reader, []DiskANNReadRequest{
		{Offset: DiskANNSectorSize, Length: DiskANNSectorSize},
		{Offset: 0, Length: DiskANNSectorSize},
	}, 2)
	require.NoError(t, err)
	require.Equal(t, contents[DiskANNSectorSize:], got[0])
	require.Equal(t, contents[:DiskANNSectorSize], got[1])
}

func TestLinuxDiskANNReaderValidatesBatch(t *testing.T) {
	path, _ := writeLinuxDiskANNTestFile(t, "index", 1)
	reader, err := openLinuxDiskANNReader(path, 0, linuxDiskANNOpenConfig{
		openDirect: func(*os.File) (int, error) { return -1, unix.EOPNOTSUPP },
		newRing:    func(uint32) (diskANNRing, error) { return &fakeDiskANNRing{}, nil },
	})
	require.NoError(t, err)
	defer func() { require.NoError(t, reader.Close()) }()
	require.GreaterOrEqual(t, reader.capacity, 1)

	err = reader.ReadBatchAt(nil, nil, nil)
	require.Error(t, err)
	err = reader.ReadBatchAt(context.Background(), []DiskANNReadRequest{{Offset: 0, Length: 1}}, nil)
	require.Error(t, err)
	err = reader.ReadBatchAt(context.Background(), []DiskANNReadRequest{{Offset: -1, Length: 1}}, [][]byte{make([]byte, 1)})
	require.ErrorIs(t, err, ErrDiskANNShortRead)
	err = reader.ReadBatchAt(context.Background(), []DiskANNReadRequest{{Offset: 0, Length: 2}}, [][]byte{make([]byte, 1)})
	require.ErrorIs(t, err, ErrDiskANNShortRead)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err = reader.ReadBatchAt(ctx, []DiskANNReadRequest{{Offset: 0, Length: 1}}, [][]byte{make([]byte, 1)})
	require.True(t, errors.Is(err, context.Canceled))
}
