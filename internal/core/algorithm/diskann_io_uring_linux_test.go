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
	"fmt"
	"math"
	"os"
	"sync/atomic"
	"testing"
	"unsafe"

	"github.com/stretchr/testify/require"
	"golang.org/x/sys/unix"
)

func TestDiskANNIOUringABISizes(t *testing.T) {
	require.Equal(t, uintptr(120), unsafe.Sizeof(ioUringParams{}))
	require.Equal(t, uintptr(64), unsafe.Sizeof(ioUringSQE{}))
	require.Equal(t, uintptr(16), unsafe.Sizeof(ioUringCQE{}))
	require.Equal(t, uintptr(40), unsafe.Sizeof(ioUringSQOffsets{}))
	require.Equal(t, uintptr(40), unsafe.Sizeof(ioUringCQOffsets{}))
	require.Equal(t, uintptr(16), unsafe.Sizeof(ioUringProbe{}))
	require.Equal(t, uintptr(8), unsafe.Sizeof(ioUringProbeOp{}))

	require.Equal(t, uintptr(4), unsafe.Offsetof(ioUringParams{}.CQEntries))
	require.Equal(t, uintptr(20), unsafe.Offsetof(ioUringParams{}.Features))
	require.Equal(t, uintptr(40), unsafe.Offsetof(ioUringParams{}.SQOff))
	require.Equal(t, uintptr(80), unsafe.Offsetof(ioUringParams{}.CQOff))
	require.Equal(t, uintptr(4), unsafe.Offsetof(ioUringSQE{}.FD))
	require.Equal(t, uintptr(8), unsafe.Offsetof(ioUringSQE{}.Off))
	require.Equal(t, uintptr(16), unsafe.Offsetof(ioUringSQE{}.Addr))
	require.Equal(t, uintptr(24), unsafe.Offsetof(ioUringSQE{}.Len))
	require.Equal(t, uintptr(32), unsafe.Offsetof(ioUringSQE{}.UserData))
	require.Equal(t, uintptr(48), unsafe.Offsetof(ioUringSQE{}.Addr3))
	require.Equal(t, uintptr(8), unsafe.Offsetof(ioUringCQE{}.Res))
	require.Equal(t, uintptr(12), unsafe.Offsetof(ioUringCQE{}.Flags))
	require.Equal(t, uintptr(20), unsafe.Offsetof(ioUringSQOffsets{}.Dropped))
	require.Equal(t, uintptr(24), unsafe.Offsetof(ioUringSQOffsets{}.Array))
	require.Equal(t, uintptr(32), unsafe.Offsetof(ioUringSQOffsets{}.UserAddr))
	require.Equal(t, uintptr(16), unsafe.Offsetof(ioUringCQOffsets{}.Overflow))
	require.Equal(t, uintptr(20), unsafe.Offsetof(ioUringCQOffsets{}.CQEs))
	require.Equal(t, uintptr(32), unsafe.Offsetof(ioUringCQOffsets{}.UserAddr))
}

func TestDiskANNIOUringRealKernelRead(t *testing.T) {
	ring, err := newDiskANNIOUring(4)
	if errors.Is(err, unix.ENOSYS) || errors.Is(err, unix.EPERM) ||
		errors.Is(err, unix.EACCES) || errors.Is(err, unix.EOPNOTSUPP) || isDiskANNIOUringCapabilityEINVAL(err) {
		t.Skipf("io_uring is unavailable: %v", err)
	}
	require.NoError(t, err)
	defer func() { require.NoError(t, ring.Close()) }()

	contents := make([]byte, 3*DiskANNSectorSize)
	for sector := 0; sector < 3; sector++ {
		for index := 0; index < DiskANNSectorSize; index++ {
			contents[sector*DiskANNSectorSize+index] = byte(sector + 1)
		}
	}
	path := t.TempDir() + "/diskann.dat"
	require.NoError(t, os.WriteFile(path, contents, 0o600))
	file, err := os.Open(path)
	require.NoError(t, err)
	defer func() { require.NoError(t, file.Close()) }()

	requests := []DiskANNReadRequest{
		{Offset: 2 * DiskANNSectorSize, Length: DiskANNSectorSize},
		{Offset: 0, Length: DiskANNSectorSize},
		{Offset: DiskANNSectorSize, Length: DiskANNSectorSize},
	}
	output := buffers(len(requests), DiskANNSectorSize)
	require.NoError(t, ring.Read(context.Background(), int(file.Fd()), int64(len(contents)), requests, output))
	require.True(t, bytes.Equal(bytes.Repeat([]byte{3}, DiskANNSectorSize), output[0]))
	require.True(t, bytes.Equal(bytes.Repeat([]byte{1}, DiskANNSectorSize), output[1]))
	require.True(t, bytes.Equal(bytes.Repeat([]byte{2}, DiskANNSectorSize), output[2]))
}

type fakeDiskANNIOUringSystem struct {
	params      ioUringParams
	setupErr    error
	registerErr error
	mmapErrAt   int
	mmapErr     error
	maps        [][]byte
	mmapLengths []int
	mmapOffsets []int64
	mmapFlags   []int
	munmaps     int
	closes      int
	enterFunc   func(*diskANNIOUring, uint32, uint32, uint32) (uint32, error)
}

func (s *fakeDiskANNIOUringSystem) setup(_ uint32, params *ioUringParams) (int, error) {
	*params = s.params
	return 17, s.setupErr
}
func (s *fakeDiskANNIOUringSystem) register(_ int, _ uint32, probe *ioUringProbeData, _ uint32) error {
	if s.registerErr == nil {
		probe.Header.OpsLen = 1
		probe.Ops[0] = ioUringProbeOp{Op: ioUringOpRead, Flags: ioUringOpSupported}
	}
	return s.registerErr
}
func (s *fakeDiskANNIOUringSystem) mmap(_ int, offset int64, length, _, flags int) ([]byte, error) {
	s.mmapLengths = append(s.mmapLengths, length)
	s.mmapOffsets = append(s.mmapOffsets, offset)
	s.mmapFlags = append(s.mmapFlags, flags)
	if s.mmapErrAt != 0 && len(s.mmapLengths) == s.mmapErrAt {
		if s.mmapErr != nil {
			return nil, s.mmapErr
		}
		return nil, unix.ENOMEM
	}
	mapped := make([]byte, length)
	s.maps = append(s.maps, mapped)
	return mapped, nil
}
func (s *fakeDiskANNIOUringSystem) munmap(_ []byte) error { s.munmaps++; return nil }
func (s *fakeDiskANNIOUringSystem) close(_ int) error     { s.closes++; return nil }
func (s *fakeDiskANNIOUringSystem) enter(r *diskANNIOUring, submit, wait, flags uint32) (uint32, error) {
	if s.enterFunc != nil {
		return s.enterFunc(r, submit, wait, flags)
	}
	return submit, nil
}

func fakeDiskANNIOUringParams(flags uint32) ioUringParams {
	cqBase := uint32(0)
	if flags&ioUringFeatSingleMmap != 0 {
		// SQ and CQ offsets are relative to the same mapping under
		// IORING_FEAT_SINGLE_MMAP, so their control words must not alias.
		cqBase = 128
	}
	return ioUringParams{
		SQEntries: 4,
		CQEntries: 8,
		Features:  flags,
		SQOff:     ioUringSQOffsets{Head: 0, Tail: 4, RingMask: 8, RingEntries: 12, Dropped: 20, Array: 64},
		CQOff: ioUringCQOffsets{
			Head: cqBase, Tail: cqBase + 4, RingMask: cqBase + 8,
			RingEntries: cqBase + 12, Overflow: cqBase + 16, CQEs: cqBase + 64,
		},
	}
}

func TestNewDiskANNIOUringMapsSingleRingAtExactLengthsAndCloses(t *testing.T) {
	system := &fakeDiskANNIOUringSystem{params: fakeDiskANNIOUringParams(ioUringFeatSingleMmap)}
	ring, err := newDiskANNIOUringWithSystem(4, system)
	require.NoError(t, err)
	require.Equal(t, []int{320, 256}, system.mmapLengths)
	require.Equal(t, []int64{ioUringOffSQRing, ioUringOffSQEs}, system.mmapOffsets)
	require.NoError(t, ring.Close())
	require.NoError(t, ring.Close())
	require.Equal(t, 2, system.munmaps)
	require.Equal(t, 1, system.closes)
}

func TestNewDiskANNIOUringCleansUpFailedMapping(t *testing.T) {
	system := &fakeDiskANNIOUringSystem{params: fakeDiskANNIOUringParams(0), mmapErrAt: 2}
	_, err := newDiskANNIOUringWithSystem(4, system)
	require.ErrorIs(t, err, unix.ENOMEM)
	require.Equal(t, 1, system.munmaps)
	require.Equal(t, 1, system.closes)
}

func TestDiskANNIOUringCapabilityEINVALRequiresSetupOrProbeProvenance(t *testing.T) {
	tests := []struct {
		name       string
		system     *fakeDiskANNIOUringSystem
		capability bool
	}{
		{name: "setup", system: &fakeDiskANNIOUringSystem{setupErr: unix.EINVAL}, capability: true},
		{name: "probe", system: &fakeDiskANNIOUringSystem{params: fakeDiskANNIOUringParams(ioUringFeatSingleMmap), registerErr: unix.EINVAL}, capability: true},
		{name: "mmap", system: &fakeDiskANNIOUringSystem{params: fakeDiskANNIOUringParams(ioUringFeatSingleMmap), mmapErrAt: 1, mmapErr: unix.EINVAL}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := newDiskANNIOUringWithSystem(4, tt.system)
			require.ErrorIs(t, err, unix.EINVAL)
			require.Equal(t, tt.capability, isDiskANNIOUringCapabilityError(err))
		})
	}
}

func TestDiskANNIOUringReadPreservesRequestOrderWithPartialUnsortedCompletions(t *testing.T) {
	system := &fakeDiskANNIOUringSystem{params: fakeDiskANNIOUringParams(ioUringFeatSingleMmap)}
	calls := 0
	system.enterFunc = func(r *diskANNIOUring, submit, _, _ uint32) (uint32, error) {
		if submit != 0 {
			calls++
			accepted := submit
			if calls == 1 {
				accepted = 1
			}
			for index := uint32(0); index < accepted; index++ {
				sqe := r.sqes[(atomic.LoadUint32(r.sqHead)+index)&*r.sqMask]
				start := int(sqe.UserData) * r.stagingSlot
				buffer := r.staging[start : start+int(sqe.Len)]
				for i := range buffer {
					buffer[i] = byte(sqe.UserData + 1)
				}
				position := atomic.LoadUint32(r.cqTail)
				r.cqes[position&*r.cqMask] = ioUringCQE{UserData: sqe.UserData, Res: int32(sqe.Len)}
				atomic.StoreUint32(r.cqTail, position+1)
			}
			atomic.AddUint32(r.sqHead, accepted)
			return accepted, nil
		}
		return 0, nil
	}
	ring, err := newDiskANNIOUringWithSystem(4, system)
	require.NoError(t, err)
	defer func() { require.NoError(t, ring.Close()) }()

	requests := []DiskANNReadRequest{{Offset: 8192, Length: 4096}, {Offset: 0, Length: 4096}, {Offset: 4096, Length: 4096}}
	buffers := [][]byte{make([]byte, 4096), make([]byte, 4096), make([]byte, 4096)}
	require.NoError(t, ring.Read(context.Background(), 9, int64(3*4096), requests, buffers))
	require.Equal(t, byte(1), buffers[0][0])
	require.Equal(t, byte(2), buffers[1][0])
	require.Equal(t, byte(3), buffers[2][0])
	require.GreaterOrEqual(t, calls, 2)
}

func TestDiskANNIOUringCopiesReverseOrderCompletionsToRequestBuffers(t *testing.T) {
	system := &fakeDiskANNIOUringSystem{params: fakeDiskANNIOUringParams(ioUringFeatSingleMmap)}
	waits := 0
	system.enterFunc = func(r *diskANNIOUring, submit, wait, flags uint32) (uint32, error) {
		if submit != 0 {
			for offset := uint32(0); offset < submit; offset++ {
				sqe := r.sqes[(atomic.LoadUint32(r.sqHead)+offset)&*r.sqMask]
				slot := r.staging[int(sqe.UserData)*r.stagingSlot:][:sqe.Len]
				for i := range slot {
					slot[i] = byte(sqe.UserData + 1)
				}
			}
			atomic.AddUint32(r.sqHead, submit)
			return submit, nil
		}
		require.Equal(t, uint32(1), wait)
		require.NotZero(t, flags&ioUringEnterGetEvents)
		waits++
		for index := 2; index >= 0; index-- {
			pushCQE(r, ioUringCQE{UserData: uint64(index), Res: 4096})
		}
		return 0, nil
	}
	ring, err := newDiskANNIOUringWithSystem(3, system)
	require.NoError(t, err)
	defer func() { require.NoError(t, ring.Close()) }()

	output := buffers(3, 4096)
	require.NoError(t, ring.Read(context.Background(), 9, 3*4096, requests(3, 4096), output))
	require.Equal(t, 1, waits)
	for index := range output {
		require.Equal(t, byte(index+1), output[index][0])
	}
}

func TestDiskANNIOUringWaitsRepeatedlyForPartialCompletions(t *testing.T) {
	system := &fakeDiskANNIOUringSystem{params: fakeDiskANNIOUringParams(ioUringFeatSingleMmap)}
	submissions, waits := 0, 0
	system.enterFunc = func(r *diskANNIOUring, submit, wait, flags uint32) (uint32, error) {
		if submit != 0 {
			submissions++
			for offset := uint32(0); offset < submit; offset++ {
				sqe := r.sqes[(atomic.LoadUint32(r.sqHead)+offset)&*r.sqMask]
				slot := r.staging[int(sqe.UserData)*r.stagingSlot:][:sqe.Len]
				for i := range slot {
					slot[i] = byte(sqe.UserData + 1)
				}
			}
			atomic.AddUint32(r.sqHead, submit)
			return submit, nil
		}
		require.Equal(t, uint32(1), wait)
		require.NotZero(t, flags&ioUringEnterGetEvents)
		pushCQE(r, ioUringCQE{UserData: uint64(waits), Res: 4096})
		waits++
		return 0, nil
	}
	ring, err := newDiskANNIOUringWithSystem(3, system)
	require.NoError(t, err)
	defer func() { require.NoError(t, ring.Close()) }()

	output := buffers(3, 4096)
	require.NoError(t, ring.Read(context.Background(), 9, 3*4096, requests(3, 4096), output))
	require.Equal(t, 1, submissions, "the full multi-request batch must be accepted together")
	require.Equal(t, 3, waits, "each wait exposes only one completion")
	for index := range output {
		require.Equal(t, byte(index+1), output[index][0])
	}
}

func TestDiskANNIOUringValidatesBeforeSubmitting(t *testing.T) {
	system := &fakeDiskANNIOUringSystem{params: fakeDiskANNIOUringParams(ioUringFeatSingleMmap)}
	submits := 0
	system.enterFunc = func(_ *diskANNIOUring, submit, _, _ uint32) (uint32, error) {
		submits += int(submit)
		return submit, nil
	}
	ring, err := newDiskANNIOUringWithSystem(4, system)
	require.NoError(t, err)
	defer func() { require.NoError(t, ring.Close()) }()

	err = ring.Read(context.Background(), 9, 4096, []DiskANNReadRequest{{Offset: 1, Length: 4096}}, [][]byte{make([]byte, 4096)})
	require.ErrorIs(t, err, ErrDiskANNShortRead)
	require.Zero(t, submits)
}

func TestDiskANNIOUringRejectsNegativeFileSizeBeforeSubmitting(t *testing.T) {
	system := &fakeDiskANNIOUringSystem{params: fakeDiskANNIOUringParams(ioUringFeatSingleMmap)}
	submits := 0
	system.enterFunc = func(_ *diskANNIOUring, submit, _, _ uint32) (uint32, error) {
		submits += int(submit)
		return submit, nil
	}
	ring, err := newDiskANNIOUringWithSystem(1, system)
	require.NoError(t, err)
	defer func() { require.NoError(t, ring.Close()) }()

	err = ring.Read(context.Background(), 9, math.MinInt64, requests(1, 4096), buffers(1, 4096))
	require.ErrorIs(t, err, ErrDiskANNShortRead)
	require.Zero(t, submits)
}

func TestDiskANNIOUringCancellationBeforeSubmit(t *testing.T) {
	system := &fakeDiskANNIOUringSystem{params: fakeDiskANNIOUringParams(ioUringFeatSingleMmap)}
	ring, err := newDiskANNIOUringWithSystem(4, system)
	require.NoError(t, err)
	defer func() { require.NoError(t, ring.Close()) }()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err = ring.Read(ctx, 9, 4096, []DiskANNReadRequest{{Offset: 0, Length: 4096}}, [][]byte{make([]byte, 4096)})
	require.True(t, errors.Is(err, context.Canceled))
}

func TestDiskANNIOUringStagingIsSharedPersistentAndPageAligned(t *testing.T) {
	s := &fakeDiskANNIOUringSystem{params: fakeDiskANNIOUringParams(ioUringFeatSingleMmap)}
	s.enterFunc = successfulEnter(nil)
	r, err := newDiskANNIOUringWithSystem(2, s)
	require.NoError(t, err)
	require.NoError(t, r.Read(context.Background(), 9, 8192, requests(1, 4096), buffers(1, 4096)))
	stagingCalls := len(s.mmapFlags)
	flags := s.mmapFlags[stagingCalls-1]
	require.NotZero(t, flags&unix.MAP_SHARED)
	require.NotZero(t, flags&unix.MAP_ANON)
	require.Zero(t, flags&unix.MAP_PRIVATE)
	require.Equal(t, 4096, r.stagingSlot)
	require.Equal(t, int(*r.sqEntries)*4096, len(r.staging))
	require.NoError(t, r.Read(context.Background(), 9, 8192, requests(1, 4096), buffers(1, 4096)))
	require.Len(t, s.mmapFlags, stagingCalls, "staging mmap must be reused")
	require.NoError(t, r.Close())
	require.Equal(t, systemRingMaps(s.params.Features)+1, s.munmaps)
}

func TestDiskANNIOUringWaitUsesMinCompleteOne(t *testing.T) {
	s := &fakeDiskANNIOUringSystem{params: fakeDiskANNIOUringParams(ioUringFeatSingleMmap)}
	waits := []uint32{}
	s.enterFunc = successfulEnter(func(_, wait, flags uint32) {
		if flags&ioUringEnterGetEvents != 0 {
			waits = append(waits, wait)
		}
	})
	r, err := newDiskANNIOUringWithSystem(1, s)
	require.NoError(t, err)
	defer func() { require.NoError(t, r.Close()) }()
	require.NoError(t, r.Read(context.Background(), 9, 4096, requests(1, 4096), buffers(1, 4096)))
	require.Equal(t, []uint32{1}, waits)
}

func TestDiskANNIOUringPartialSubmitFatalWithdrawsAndDrains(t *testing.T) {
	testPartialSubmitAbort(t, false)
}

func TestDiskANNIOUringPartialSubmitCancellationWithdrawsAndDrains(t *testing.T) {
	testPartialSubmitAbort(t, true)
}

func testPartialSubmitAbort(t *testing.T, cancelAfterFirst bool) {
	t.Helper()
	s := &fakeDiskANNIOUringSystem{params: fakeDiskANNIOUringParams(ioUringFeatSingleMmap)}
	ctx, cancel := context.WithCancel(context.Background())
	submits, waits := 0, 0
	s.enterFunc = func(r *diskANNIOUring, submit, wait, flags uint32) (uint32, error) {
		if submit != 0 {
			submits++
			if submits == 1 {
				atomic.AddUint32(r.sqHead, 1)
				if cancelAfterFirst {
					cancel()
				}
				return 1, nil
			}
			return 0, unix.EIO
		}
		require.Equal(t, uint32(1), wait)
		require.NotZero(t, flags&ioUringEnterGetEvents)
		waits++
		pushCQE(r, ioUringCQE{UserData: 0, Res: 4096})
		return 0, nil
	}
	r, err := newDiskANNIOUringWithSystem(2, s)
	require.NoError(t, err)
	err = r.Read(ctx, 9, 8192, requests(2, 4096), buffers(2, 4096))
	if cancelAfterFirst {
		require.ErrorIs(t, err, context.Canceled)
		require.False(t, r.poisoned.Load())
	} else {
		require.ErrorIs(t, err, unix.EIO)
		require.True(t, r.poisoned.Load())
	}
	require.Equal(t, atomic.LoadUint32(r.sqHead), atomic.LoadUint32(r.sqTail))
	require.Equal(t, 1, waits)
	require.NoError(t, r.Close())
}

func TestDiskANNIOUringBoundsSubmissionRetries(t *testing.T) {
	for _, enterErr := range []error{nil, unix.EINTR, unix.EAGAIN, unix.EBUSY} {
		t.Run(fmt.Sprint(enterErr), func(t *testing.T) {
			s := &fakeDiskANNIOUringSystem{params: fakeDiskANNIOUringParams(ioUringFeatSingleMmap)}
			calls := 0
			s.enterFunc = func(_ *diskANNIOUring, submit, _, _ uint32) (uint32, error) {
				require.NotZero(t, submit)
				calls++
				return 0, enterErr
			}
			r, err := newDiskANNIOUringWithSystem(1, s)
			require.NoError(t, err)
			err = r.Read(context.Background(), 9, 4096, requests(1, 4096), buffers(1, 4096))
			require.Error(t, err)
			require.LessOrEqual(t, calls, diskANNIOUringRetries)
			require.Equal(t, *r.sqHead, *r.sqTail)
			require.NoError(t, r.Close())
		})
	}
}

func TestDiskANNIOUringRetriesTransientCQEPerRequestBoundedly(t *testing.T) {
	for _, result := range []int32{-int32(unix.EINTR), -int32(unix.EAGAIN)} {
		t.Run(fmt.Sprint(result), func(t *testing.T) {
			s := &fakeDiskANNIOUringSystem{params: fakeDiskANNIOUringParams(ioUringFeatSingleMmap)}
			submissions := 0
			s.enterFunc = func(r *diskANNIOUring, submit, _, _ uint32) (uint32, error) {
				if submit != 0 {
					submissions++
					sqe := r.sqes[*r.sqHead&*r.sqMask]
					atomic.AddUint32(r.sqHead, submit)
					res := result
					if submissions == 3 {
						res = int32(sqe.Len)
					}
					pushCQE(r, ioUringCQE{UserData: sqe.UserData, Res: res})
					return submit, nil
				}
				return 0, nil
			}
			r, err := newDiskANNIOUringWithSystem(1, s)
			require.NoError(t, err)
			defer func() { require.NoError(t, r.Close()) }()
			require.NoError(t, r.Read(context.Background(), 9, 4096, requests(1, 4096), buffers(1, 4096)))
			require.Equal(t, 3, submissions)
		})
	}
}

func TestDiskANNIOUringStopsTransientCQERetryAtBound(t *testing.T) {
	s := &fakeDiskANNIOUringSystem{params: fakeDiskANNIOUringParams(ioUringFeatSingleMmap)}
	submissions := 0
	s.enterFunc = func(r *diskANNIOUring, submit, _, _ uint32) (uint32, error) {
		if submit != 0 {
			submissions++
			sqe := r.sqes[*r.sqHead&*r.sqMask]
			atomic.AddUint32(r.sqHead, submit)
			pushCQE(r, ioUringCQE{UserData: sqe.UserData, Res: -int32(unix.EAGAIN)})
			return submit, nil
		}
		return 0, nil
	}
	r, err := newDiskANNIOUringWithSystem(1, s)
	require.NoError(t, err)
	err = r.Read(context.Background(), 9, 4096, requests(1, 4096), buffers(1, 4096))
	require.ErrorIs(t, err, unix.EAGAIN)
	require.LessOrEqual(t, submissions, diskANNIOUringRetries+1)
	require.True(t, r.poisoned.Load())
	require.NoError(t, r.Close())
}

func TestDiskANNIOUringDrainsBadCompletionsPoisonsAndDoesNotCopy(t *testing.T) {
	cases := []struct {
		name string
		cqes []ioUringCQE
	}{
		{"short", []ioUringCQE{{UserData: 0, Res: 2048}}},
		{"negative", []ioUringCQE{{UserData: 0, Res: -int32(unix.EIO)}}},
		{"invalid", []ioUringCQE{{UserData: 99, Res: 4096}}},
		{"duplicate", []ioUringCQE{{UserData: 0, Res: 4096}, {UserData: 0, Res: 4096}}},
		{"extra duplicate", []ioUringCQE{{UserData: 0, Res: 4096}, {UserData: 0, Res: 4096}, {UserData: 1, Res: 4096}}},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			s := &fakeDiskANNIOUringSystem{params: fakeDiskANNIOUringParams(ioUringFeatSingleMmap)}
			s.enterFunc = func(r *diskANNIOUring, submit, _, _ uint32) (uint32, error) {
				if submit != 0 {
					atomic.AddUint32(r.sqHead, submit)
					return submit, nil
				}
				for _, c := range tt.cqes {
					pushCQE(r, c)
				}
				return 0, nil
			}
			requestCount := len(tt.cqes)
			if tt.name == "extra duplicate" {
				requestCount = 2
			}
			r, err := newDiskANNIOUringWithSystem(uint32(requestCount), s)
			require.NoError(t, err)
			out := buffers(requestCount, 4096)
			for _, b := range out {
				b[0] = 0x5a
			}
			err = r.Read(context.Background(), 9, int64(requestCount*4096), requests(requestCount, 4096), out)
			require.Error(t, err)
			require.True(t, r.poisoned.Load())
			for _, b := range out {
				require.Equal(t, byte(0x5a), b[0])
			}
			require.Equal(t, *r.cqHead, *r.cqTail)
			require.NoError(t, r.Close())
		})
	}
}

func TestDiskANNIOUringMalformedCompletionsRetainStaging(t *testing.T) {
	cases := []struct {
		name     string
		requests int
		cqes     []ioUringCQE
	}{
		{"valid-valid-extra-duplicate", 2, []ioUringCQE{{UserData: 0, Res: 4096}, {UserData: 1, Res: 4096}, {UserData: 1, Res: 4096}}},
		{"duplicate-replaces-valid", 2, []ioUringCQE{{UserData: 0, Res: 4096}, {UserData: 0, Res: 4096}}},
		{"unknown", 1, []ioUringCQE{{UserData: 99, Res: 4096}}},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			s := &fakeDiskANNIOUringSystem{params: fakeDiskANNIOUringParams(ioUringFeatSingleMmap)}
			s.enterFunc = func(r *diskANNIOUring, submit, _, _ uint32) (uint32, error) {
				if submit != 0 {
					atomic.AddUint32(r.sqHead, submit)
					return submit, nil
				}
				for _, cqe := range tt.cqes {
					pushCQE(r, cqe)
				}
				return 0, nil
			}
			r, err := newDiskANNIOUringWithSystem(uint32(tt.requests), s)
			require.NoError(t, err)
			err = r.Read(context.Background(), 9, int64(tt.requests*4096), requests(tt.requests, 4096), buffers(tt.requests, 4096))
			require.Error(t, err)
			require.Equal(t, *r.cqTail, *r.cqHead, "the complete tail snapshot must be consumed")
			require.True(t, r.poisoned.Load())
			require.True(t, r.closed)
			require.True(t, r.stagingRetained)
			require.NotNil(t, r.staging)
			require.Equal(t, 1, s.closes)
			require.Equal(t, systemRingMaps(s.params.Features), s.munmaps)
		})
	}
}

func TestDiskANNIOUringQueueDamageRetainsStagingAndPreservesOriginalError(t *testing.T) {
	cases := []struct {
		name    string
		run     func(*diskANNIOUring, context.CancelFunc)
		waitErr error
		want    error
	}{
		{"dropped-after-cancellation", func(r *diskANNIOUring, cancel context.CancelFunc) {
			cancel()
			atomic.AddUint32(r.sqDropped, 1)
			pushCQE(r, ioUringCQE{UserData: 0, Res: 4096})
		}, nil, context.Canceled},
		{"overflow-with-wait-error", func(r *diskANNIOUring, _ context.CancelFunc) {
			atomic.AddUint32(r.cqOverflow, 1)
			pushCQE(r, ioUringCQE{UserData: 0, Res: 4096})
		}, unix.EIO, unix.EIO},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			s := &fakeDiskANNIOUringSystem{params: fakeDiskANNIOUringParams(ioUringFeatSingleMmap)}
			ctx, cancel := context.WithCancel(context.Background())
			s.enterFunc = func(r *diskANNIOUring, submit, _, _ uint32) (uint32, error) {
				if submit != 0 {
					atomic.AddUint32(r.sqHead, submit)
					return submit, nil
				}
				tt.run(r, cancel)
				return 0, tt.waitErr
			}
			r, err := newDiskANNIOUringWithSystem(1, s)
			require.NoError(t, err)
			err = r.Read(ctx, 9, 4096, requests(1, 4096), buffers(1, 4096))
			require.ErrorIs(t, err, tt.want)
			require.True(t, r.poisoned.Load())
			require.True(t, r.closed)
			require.True(t, r.stagingRetained)
			require.NotNil(t, r.staging)
			require.Equal(t, 1, s.closes)
			require.Equal(t, systemRingMaps(s.params.Features), s.munmaps)
		})
	}
}

func TestDiskANNIOUringDetectsSQDroppedAndCQOverflowRelativeToBaseline(t *testing.T) {
	for _, kind := range []string{"dropped", "overflow"} {
		t.Run(kind, func(t *testing.T) {
			s := &fakeDiskANNIOUringSystem{params: fakeDiskANNIOUringParams(ioUringFeatSingleMmap)}
			r, err := newDiskANNIOUringWithSystem(1, s)
			require.NoError(t, err)
			atomic.StoreUint32(r.sqDropped, 7)
			atomic.StoreUint32(r.cqOverflow, 11)
			s.enterFunc = successfulEnter(func(submit, _, _ uint32) {
				if submit != 0 {
					if kind == "dropped" {
						atomic.AddUint32(r.sqDropped, 1)
					} else {
						atomic.AddUint32(r.cqOverflow, 1)
					}
				}
			})
			err = r.Read(context.Background(), 9, 4096, requests(1, 4096), buffers(1, 4096))
			require.Error(t, err)
			require.True(t, r.poisoned.Load())
			require.NoError(t, r.Close())
		})
	}
}

func TestDiskANNIOUringRetainsStagingWhenDrainCannotProveQuiescence(t *testing.T) {
	s := &fakeDiskANNIOUringSystem{params: fakeDiskANNIOUringParams(ioUringFeatSingleMmap)}
	waits := 0
	s.enterFunc = func(r *diskANNIOUring, submit, _, _ uint32) (uint32, error) {
		if submit != 0 {
			atomic.AddUint32(r.sqHead, submit)
			return submit, nil
		}
		waits++
		return 0, unix.EAGAIN
	}
	r, err := newDiskANNIOUringWithSystem(1, s)
	require.NoError(t, err)
	err = r.Read(context.Background(), 9, 4096, requests(1, 4096), buffers(1, 4096))
	require.Error(t, err)
	require.LessOrEqual(t, waits, diskANNIOUringRetries)
	require.True(t, r.poisoned.Load())
	require.True(t, r.closed)
	require.NotNil(t, r.staging)
	require.True(t, r.stagingRetained)
	require.Equal(t, 1, s.closes)
	require.Equal(t, systemRingMaps(s.params.Features), s.munmaps, "staging must be deliberately retained")
}

func TestNewDiskANNIOUringCleansEveryMappingFailureStageExactlyOnce(t *testing.T) {
	cases := []struct {
		name       string
		feature    uint32
		fail, want int
	}{
		{"split-sq", 0, 1, 0}, {"split-cq", 0, 2, 1}, {"split-sqes", 0, 3, 2},
		{"single-ring", ioUringFeatSingleMmap, 1, 0}, {"single-sqes", ioUringFeatSingleMmap, 2, 1},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			s := &fakeDiskANNIOUringSystem{params: fakeDiskANNIOUringParams(tt.feature), mmapErrAt: tt.fail}
			_, err := newDiskANNIOUringWithSystem(4, s)
			require.Error(t, err)
			require.Equal(t, tt.want, s.munmaps)
			require.Equal(t, 1, s.closes)
		})
	}
}

func TestNewDiskANNIOUringRejectsRequestedQueueAboveCapBeforeSetup(t *testing.T) {
	s := &fakeDiskANNIOUringSystem{params: fakeDiskANNIOUringParams(ioUringFeatSingleMmap)}
	_, err := newDiskANNIOUringWithSystem(diskANNIOUringBatchCap+1, s)
	require.Error(t, err)
	require.Zero(t, s.closes, "invalid requested capacity must not create a ring")
}

func TestNewDiskANNIOUringCleansAllMapsAfterBindFailure(t *testing.T) {
	params := fakeDiskANNIOUringParams(0)
	params.SQOff.Head = params.SQOff.Array + params.SQEntries*4
	s := &fakeDiskANNIOUringSystem{params: params}
	_, err := newDiskANNIOUringWithSystem(4, s)
	require.Error(t, err)
	require.Equal(t, 3, s.munmaps)
	require.Equal(t, 1, s.closes)
}

func TestDiskANNIOUringGrowsStagingOnlyBetweenQuiescentBatches(t *testing.T) {
	s := &fakeDiskANNIOUringSystem{params: fakeDiskANNIOUringParams(ioUringFeatSingleMmap)}
	var result int32
	s.enterFunc = func(r *diskANNIOUring, submit, _, _ uint32) (uint32, error) {
		if submit != 0 {
			result = int32(r.sqes[*r.sqHead&*r.sqMask].Len)
			atomic.AddUint32(r.sqHead, submit)
			return submit, nil
		}
		pushCQE(r, ioUringCQE{UserData: 0, Res: result})
		return 0, nil
	}
	r, err := newDiskANNIOUringWithSystem(1, s)
	require.NoError(t, err)
	require.NoError(t, r.Read(context.Background(), 9, 8192, requests(1, 4096), buffers(1, 4096)))
	first := r.staging
	require.NoError(t, r.Read(context.Background(), 9, 8192, requests(1, 8192), buffers(1, 8192)))
	require.Equal(t, 8192, r.stagingSlot)
	require.NotEqual(t, unsafe.Pointer(&first[0]), unsafe.Pointer(&r.staging[0]))
	require.Equal(t, 1, s.munmaps, "old staging is released after the prior batch is quiescent")
	require.NoError(t, r.Close())
}

func requests(n, size int) []DiskANNReadRequest {
	out := make([]DiskANNReadRequest, n)
	for i := range out {
		out[i] = DiskANNReadRequest{Offset: int64(i * size), Length: size}
	}
	return out
}
func buffers(n, size int) [][]byte {
	out := make([][]byte, n)
	for i := range out {
		out[i] = make([]byte, size)
	}
	return out
}
func systemRingMaps(features uint32) int {
	if features&ioUringFeatSingleMmap != 0 {
		return 2
	}
	return 3
}
func pushCQE(r *diskANNIOUring, c ioUringCQE) {
	tail := atomic.LoadUint32(r.cqTail)
	r.cqes[tail&atomic.LoadUint32(r.cqMask)] = c
	atomic.StoreUint32(r.cqTail, tail+1)
}
func successfulEnter(observe func(uint32, uint32, uint32)) func(*diskANNIOUring, uint32, uint32, uint32) (uint32, error) {
	return func(r *diskANNIOUring, submit, wait, flags uint32) (uint32, error) {
		if observe != nil {
			observe(submit, wait, flags)
		}
		if submit != 0 {
			atomic.AddUint32(r.sqHead, submit)
			return submit, nil
		}
		pushCQE(r, ioUringCQE{UserData: 0, Res: 4096})
		return 0, nil
	}
}
