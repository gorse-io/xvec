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
	"math"
	"sync"
	"sync/atomic"
	"unsafe"

	"golang.org/x/sys/unix"
)

const (
	ioUringOffSQRing       = int64(0)
	ioUringOffCQRing       = int64(0x8000000)
	ioUringOffSQEs         = int64(0x10000000)
	ioUringFeatSingleMmap  = uint32(1)
	ioUringEnterGetEvents  = uint32(1)
	ioUringRegisterProbe   = uint32(8)
	ioUringOpRead          = uint8(22)
	ioUringOpSupported     = uint16(1)
	diskANNIOUringRetries  = 64
	diskANNIOUringBatchCap = 128
)

type ioUringSQOffsets struct {
	Head        uint32
	Tail        uint32
	RingMask    uint32
	RingEntries uint32
	Flags       uint32
	Dropped     uint32
	Array       uint32
	Resv1       uint32
	UserAddr    uint64
}

type ioUringCQOffsets struct {
	Head        uint32
	Tail        uint32
	RingMask    uint32
	RingEntries uint32
	Overflow    uint32
	CQEs        uint32
	Flags       uint32
	Resv1       uint32
	UserAddr    uint64
}

type ioUringParams struct {
	SQEntries    uint32
	CQEntries    uint32
	Flags        uint32
	SQThreadCPU  uint32
	SQThreadIdle uint32
	Features     uint32
	WQFd         uint32
	Resv         [3]uint32
	SQOff        ioUringSQOffsets
	CQOff        ioUringCQOffsets
}

type ioUringSQE struct {
	Opcode      uint8
	Flags       uint8
	IOPriority  uint16
	FD          int32
	Off         uint64
	Addr        uint64
	Len         uint32
	OpFlags     uint32
	UserData    uint64
	BufIndex    uint16
	Personality uint16
	SpliceFDIn  int32
	Addr3       uint64
	Pad2        uint64
}

type ioUringCQE struct {
	UserData uint64
	Res      int32
	Flags    uint32
}

type ioUringProbe struct {
	LastOp uint8
	OpsLen uint8
	Resv   uint16
	Resv2  [3]uint32
}

type ioUringProbeOp struct {
	Op    uint8
	Resv  uint8
	Flags uint16
	Resv2 uint32
}

type ioUringProbeData struct {
	Header ioUringProbe
	Ops    [256]ioUringProbeOp
}

type diskANNIOUringSystem interface {
	setup(uint32, *ioUringParams) (int, error)
	register(int, uint32, *ioUringProbeData, uint32) error
	mmap(int, int64, int, int, int) ([]byte, error)
	munmap([]byte) error
	close(int) error
	enter(*diskANNIOUring, uint32, uint32, uint32) (uint32, error)
}

type diskANNIOUringCapabilityError struct {
	stage string
	err   error
}

func (e *diskANNIOUringCapabilityError) Error() string {
	return fmt.Sprintf("core: io_uring %s capability check: %v", e.stage, e.err)
}

func (e *diskANNIOUringCapabilityError) Unwrap() error { return e.err }

type realDiskANNIOUringSystem struct{}

func (realDiskANNIOUringSystem) setup(entries uint32, params *ioUringParams) (int, error) {
	fd, _, errno := unix.Syscall(unix.SYS_IO_URING_SETUP, uintptr(entries), uintptr(unsafe.Pointer(params)), 0)
	if errno != 0 {
		return -1, errno
	}
	return int(fd), nil
}

func (realDiskANNIOUringSystem) register(fd int, opcode uint32, probe *ioUringProbeData, count uint32) error {
	_, _, errno := unix.Syscall6(unix.SYS_IO_URING_REGISTER, uintptr(fd), uintptr(opcode), uintptr(unsafe.Pointer(probe)), uintptr(count), 0, 0)
	if errno != 0 {
		return errno
	}
	return nil
}

func (realDiskANNIOUringSystem) mmap(fd int, offset int64, length, prot, flags int) ([]byte, error) {
	return unix.Mmap(fd, offset, length, prot, flags)
}
func (realDiskANNIOUringSystem) munmap(mapped []byte) error { return unix.Munmap(mapped) }
func (realDiskANNIOUringSystem) close(fd int) error         { return unix.Close(fd) }
func (realDiskANNIOUringSystem) enter(_ *diskANNIOUring, fdSubmit, minComplete, flags uint32) (uint32, error) {
	panic("unreachable")
}

// syscallEnter is separate because the interface's first argument gives fakes
// access to the mapped queues without exposing them outside this package.
func syscallEnter(fd int, submit, minComplete, flags uint32) (uint32, error) {
	result, _, errno := unix.Syscall6(unix.SYS_IO_URING_ENTER, uintptr(fd), uintptr(submit), uintptr(minComplete), uintptr(flags), 0, 0)
	if errno != 0 {
		return 0, errno
	}
	return uint32(result), nil
}

type diskANNIOUring struct {
	mu sync.Mutex

	system  diskANNIOUringSystem
	fd      int
	sqRing  []byte
	cqRing  []byte
	sqesMap []byte

	sqHead          *uint32
	sqTail          *uint32
	sqMask          *uint32
	sqEntries       *uint32
	sqDropped       *uint32
	sqArray         []uint32
	sqes            []ioUringSQE
	cqHead          *uint32
	cqTail          *uint32
	cqMask          *uint32
	cqEntries       *uint32
	cqOverflow      *uint32
	cqes            []ioUringCQE
	staging         []byte
	stagingSlot     int
	stagingRetained bool

	closed   bool
	poisoned atomic.Bool
}

func newDiskANNIOUring(entries uint32) (*diskANNIOUring, error) {
	return newDiskANNIOUringWithSystem(entries, realDiskANNIOUringSystem{})
}

func newDiskANNIOUringWithSystem(entries uint32, system diskANNIOUringSystem) (_ *diskANNIOUring, resultErr error) {
	if entries == 0 || entries > diskANNIOUringBatchCap {
		return nil, errors.New("core: invalid io_uring queue capacity")
	}
	var params ioUringParams
	fd, err := system.setup(entries, &params)
	if err != nil {
		return nil, &diskANNIOUringCapabilityError{stage: "setup", err: err}
	}
	ring := &diskANNIOUring{system: system, fd: fd}
	defer func() {
		if resultErr != nil {
			_ = ring.Close()
		}
	}()
	if params.SQEntries == 0 || params.CQEntries == 0 || params.SQEntries > diskANNIOUringBatchCap {
		return nil, errors.New("core: invalid io_uring setup ABI")
	}
	var probe ioUringProbeData
	if err := system.register(fd, ioUringRegisterProbe, &probe, uint32(len(probe.Ops))); err != nil {
		return nil, &diskANNIOUringCapabilityError{stage: "probe", err: err}
	}
	supported := false
	for index := 0; index < int(probe.Header.OpsLen) && index < len(probe.Ops); index++ {
		if probe.Ops[index].Op == ioUringOpRead && probe.Ops[index].Flags&ioUringOpSupported != 0 {
			supported = true
			break
		}
	}
	if !supported {
		return nil, unix.EOPNOTSUPP
	}

	sqLength, ok := ringMapLength(params.SQOff.Array, params.SQEntries, 4)
	if !ok {
		return nil, errors.New("core: invalid io_uring SQ mapping ABI")
	}
	cqLength, ok := ringMapLength(params.CQOff.CQEs, params.CQEntries, int(unsafe.Sizeof(ioUringCQE{})))
	if !ok {
		return nil, errors.New("core: invalid io_uring CQ mapping ABI")
	}
	mapFlags := unix.MAP_SHARED | unix.MAP_POPULATE
	if params.Features&ioUringFeatSingleMmap != 0 {
		ring.sqRing, err = system.mmap(fd, ioUringOffSQRing, max(sqLength, cqLength), unix.PROT_READ|unix.PROT_WRITE, mapFlags)
		ring.cqRing = ring.sqRing
	} else {
		ring.sqRing, err = system.mmap(fd, ioUringOffSQRing, sqLength, unix.PROT_READ|unix.PROT_WRITE, mapFlags)
		if err == nil {
			ring.cqRing, err = system.mmap(fd, ioUringOffCQRing, cqLength, unix.PROT_READ|unix.PROT_WRITE, mapFlags)
		}
	}
	if err != nil {
		return nil, err
	}
	ring.sqesMap, err = system.mmap(fd, ioUringOffSQEs, int(params.SQEntries)*int(unsafe.Sizeof(ioUringSQE{})), unix.PROT_READ|unix.PROT_WRITE, mapFlags)
	if err != nil {
		return nil, err
	}
	if err := ring.bind(params); err != nil {
		return nil, err
	}
	return ring, nil
}

func ringMapLength(offset, entries uint32, itemSize int) (int, bool) {
	length := uint64(offset) + uint64(entries)*uint64(itemSize)
	return int(length), length <= uint64(math.MaxInt)
}

func mappedPointer(mapped []byte, offset uint32, size uintptr) (unsafe.Pointer, bool) {
	end := uint64(offset) + uint64(size)
	if end > uint64(len(mapped)) || len(mapped) == 0 {
		return nil, false
	}
	return unsafe.Pointer(&mapped[offset]), true
}

func mappedSlice[T any](mapped []byte, offset, count uint32) ([]T, bool) {
	var value T
	pointer, ok := mappedPointer(mapped, offset, uintptr(count)*unsafe.Sizeof(value))
	if !ok {
		return nil, false
	}
	return unsafe.Slice((*T)(pointer), count), true
}

func (r *diskANNIOUring) bind(params ioUringParams) error {
	var ok bool
	if pointer, valid := mappedPointer(r.sqRing, params.SQOff.Head, 4); valid {
		r.sqHead = (*uint32)(pointer)
	} else {
		return errors.New("core: invalid io_uring SQ head")
	}
	if pointer, valid := mappedPointer(r.sqRing, params.SQOff.Tail, 4); valid {
		r.sqTail = (*uint32)(pointer)
	} else {
		return errors.New("core: invalid io_uring SQ tail")
	}
	if pointer, valid := mappedPointer(r.sqRing, params.SQOff.RingMask, 4); valid {
		r.sqMask = (*uint32)(pointer)
	} else {
		return errors.New("core: invalid io_uring SQ mask")
	}
	if pointer, valid := mappedPointer(r.sqRing, params.SQOff.RingEntries, 4); valid {
		r.sqEntries = (*uint32)(pointer)
	} else {
		return errors.New("core: invalid io_uring SQ entries")
	}
	if pointer, valid := mappedPointer(r.sqRing, params.SQOff.Dropped, 4); valid {
		r.sqDropped = (*uint32)(pointer)
	} else {
		return errors.New("core: invalid io_uring SQ dropped")
	}
	if pointer, valid := mappedPointer(r.cqRing, params.CQOff.Head, 4); valid {
		r.cqHead = (*uint32)(pointer)
	} else {
		return errors.New("core: invalid io_uring CQ head")
	}
	if pointer, valid := mappedPointer(r.cqRing, params.CQOff.Tail, 4); valid {
		r.cqTail = (*uint32)(pointer)
	} else {
		return errors.New("core: invalid io_uring CQ tail")
	}
	if pointer, valid := mappedPointer(r.cqRing, params.CQOff.RingMask, 4); valid {
		r.cqMask = (*uint32)(pointer)
	} else {
		return errors.New("core: invalid io_uring CQ mask")
	}
	if pointer, valid := mappedPointer(r.cqRing, params.CQOff.RingEntries, 4); valid {
		r.cqEntries = (*uint32)(pointer)
	} else {
		return errors.New("core: invalid io_uring CQ entries")
	}
	if pointer, valid := mappedPointer(r.cqRing, params.CQOff.Overflow, 4); valid {
		r.cqOverflow = (*uint32)(pointer)
	} else {
		return errors.New("core: invalid io_uring CQ overflow")
	}
	if r.sqArray, ok = mappedSlice[uint32](r.sqRing, params.SQOff.Array, params.SQEntries); !ok {
		return errors.New("core: invalid io_uring SQ array")
	}
	if r.sqes, ok = mappedSlice[ioUringSQE](r.sqesMap, 0, params.SQEntries); !ok {
		return errors.New("core: invalid io_uring SQEs")
	}
	if r.cqes, ok = mappedSlice[ioUringCQE](r.cqRing, params.CQOff.CQEs, params.CQEntries); !ok {
		return errors.New("core: invalid io_uring CQEs")
	}
	// The kernel owns these values. Synthetic systems leave them zero.
	if atomic.LoadUint32(r.sqEntries) == 0 {
		atomic.StoreUint32(r.sqEntries, params.SQEntries)
	}
	if atomic.LoadUint32(r.sqMask) == 0 {
		atomic.StoreUint32(r.sqMask, params.SQEntries-1)
	}
	if atomic.LoadUint32(r.cqEntries) == 0 {
		atomic.StoreUint32(r.cqEntries, params.CQEntries)
	}
	if atomic.LoadUint32(r.cqMask) == 0 {
		atomic.StoreUint32(r.cqMask, params.CQEntries-1)
	}
	return nil
}

func (r *diskANNIOUring) enter(submit, wait, flags uint32) (uint32, error) {
	if _, real := r.system.(realDiskANNIOUringSystem); real {
		return syscallEnter(r.fd, submit, wait, flags)
	}
	return r.system.enter(r, submit, wait, flags)
}

func (r *diskANNIOUring) Read(ctx context.Context, fileFD int, fileSize int64, requests []DiskANNReadRequest, buffers [][]byte) error {
	if ctx == nil {
		return errors.New("core: nil DiskANN io_uring context")
	}
	if len(requests) != len(buffers) || len(requests) > int(atomic.LoadUint32(r.sqEntries)) {
		return errors.New("core: inconsistent DiskANN io_uring batch")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if len(requests) == 0 {
		return nil
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return unix.EBADF
	}
	if r.poisoned.Load() {
		return errors.New("core: poisoned DiskANN io_uring")
	}

	slotSize := 0
	for index, request := range requests {
		if fileSize < 0 || request.Offset < 0 || request.Offset%DiskANNSectorSize != 0 || request.Length <= 0 || request.Length%DiskANNSectorSize != 0 || uint64(request.Length) > uint64(math.MaxUint32) || request.Offset > fileSize || int64(request.Length) > fileSize-request.Offset || len(buffers[index]) != request.Length {
			return fmt.Errorf("%w: invalid io_uring request %d", ErrDiskANNShortRead, index)
		}
		slotSize = max(slotSize, request.Length)
	}
	if slotSize > math.MaxInt-(DiskANNSectorSize-1) {
		return errors.New("core: DiskANN io_uring staging size overflow")
	}
	slotSize = (slotSize + DiskANNSectorSize - 1) / DiskANNSectorSize * DiskANNSectorSize
	if err := r.ensureStaging(slotSize); err != nil {
		return err
	}

	droppedBaseline := atomic.LoadUint32(r.sqDropped)
	overflowBaseline := atomic.LoadUint32(r.cqOverflow)
	queued := make([]int, 0, len(requests))
	prepare := func(index int) {
		tail := atomic.LoadUint32(r.sqTail)
		position := tail & atomic.LoadUint32(r.sqMask)
		request := requests[index]
		slot := r.staging[index*r.stagingSlot : index*r.stagingSlot+request.Length]
		r.sqes[position] = ioUringSQE{Opcode: ioUringOpRead, FD: int32(fileFD), Off: uint64(request.Offset), Addr: uint64(uintptr(unsafe.Pointer(&slot[0]))), Len: uint32(request.Length), UserData: uint64(index)}
		r.sqArray[position] = position
		atomic.StoreUint32(r.sqTail, tail+1)
		queued = append(queued, index)
	}
	for index := range requests {
		prepare(index)
	}

	inFlight := make([]bool, len(requests))
	outstanding := 0
	var resultErr error
	quiescenceUncertain := false
	markUncertain := func(err error) {
		if resultErr == nil {
			resultErr = err
		}
		quiescenceUncertain = true
		r.poisoned.Store(true)
	}
	checkQueueDamage := func() {
		if atomic.LoadUint32(r.sqDropped) != droppedBaseline {
			markUncertain(errors.New("core: DiskANN io_uring submission queue dropped entries"))
		}
		if atomic.LoadUint32(r.cqOverflow) != overflowBaseline {
			markUncertain(errors.New("core: DiskANN io_uring completion overflow"))
		}
	}
	submit := func(want int) bool {
		idle := 0
		for want > 0 {
			if err := ctx.Err(); err != nil {
				if resultErr == nil {
					resultErr = err
				}
				atomic.StoreUint32(r.sqTail, atomic.LoadUint32(r.sqHead))
				queued = queued[:0]
				return true
			}
			count, err := r.enter(uint32(want), 0, 0)
			if count > uint32(want) {
				resultErr = errors.New("core: invalid io_uring submission count")
				r.poisoned.Store(true)
				r.abandonStagingLocked()
				return false
			}
			if count != 0 {
				accepted := int(count)
				if accepted > len(queued) {
					resultErr = errors.New("core: invalid io_uring submission order")
					r.poisoned.Store(true)
					r.abandonStagingLocked()
					return false
				}
				for _, index := range queued[:accepted] {
					if inFlight[index] {
						resultErr = errors.New("core: duplicate io_uring submission")
						r.poisoned.Store(true)
						r.abandonStagingLocked()
						return false
					}
					inFlight[index] = true
				}
				queued = queued[accepted:]
				outstanding += accepted
				want -= accepted
				idle = 0
			}
			if err != nil && !isTransientIOUringError(err) {
				if resultErr == nil {
					resultErr = fmt.Errorf("core: submit DiskANN io_uring: %w", err)
				}
				r.poisoned.Store(true)
				atomic.StoreUint32(r.sqTail, atomic.LoadUint32(r.sqHead))
				queued = queued[:0]
				return true
			}
			if count == 0 {
				idle++
				if idle >= diskANNIOUringRetries {
					if err == nil {
						err = unix.EAGAIN
					}
					resultErr = fmt.Errorf("core: submit DiskANN io_uring made no progress: %w", err)
					atomic.StoreUint32(r.sqTail, atomic.LoadUint32(r.sqHead))
					queued = queued[:0]
					return true
				}
			}
		}
		return true
	}
	if !submit(len(requests)) {
		return resultErr
	}

	retries := make([]int, len(requests))
	waitIdle := 0
	for outstanding > 0 {
		if resultErr == nil {
			if err := ctx.Err(); err != nil {
				resultErr = err
			}
		}
		checkQueueDamage()
		head, tail := atomic.LoadUint32(r.cqHead), atomic.LoadUint32(r.cqTail)
		if head == tail {
			if quiescenceUncertain {
				r.abandonStagingLocked()
				return resultErr
			}
			_, err := r.enter(0, 1, ioUringEnterGetEvents)
			if err != nil && !isTransientIOUringError(err) && resultErr == nil {
				resultErr = fmt.Errorf("core: wait for DiskANN io_uring: %w", err)
				r.poisoned.Store(true)
			}
			if resultErr == nil {
				if err := ctx.Err(); err != nil {
					resultErr = err
				}
			}
			checkQueueDamage()
			head, tail = atomic.LoadUint32(r.cqHead), atomic.LoadUint32(r.cqTail)
			if head == tail {
				if quiescenceUncertain {
					r.abandonStagingLocked()
					return resultErr
				}
				waitIdle++
				if waitIdle >= diskANNIOUringRetries {
					if resultErr == nil {
						resultErr = errors.New("core: DiskANN io_uring completion wait made no progress")
					}
					r.poisoned.Store(true)
					r.abandonStagingLocked()
					return resultErr
				}
			} else {
				waitIdle = 0
			}
			if head == tail {
				continue
			}
		}
		waitIdle = 0
		for head != tail {
			completion := r.cqes[head&atomic.LoadUint32(r.cqMask)]
			head++
			atomic.StoreUint32(r.cqHead, head)
			if completion.UserData >= uint64(len(requests)) || !inFlight[completion.UserData] {
				markUncertain(errors.New("core: invalid DiskANN io_uring completion"))
				continue
			}
			index := int(completion.UserData)
			inFlight[index] = false
			outstanding--
			if completion.Res == -int32(unix.EINTR) || completion.Res == -int32(unix.EAGAIN) {
				if resultErr == nil && ctx.Err() == nil && retries[index] < diskANNIOUringRetries {
					retries[index]++
					prepare(index)
					if !submit(1) {
						return resultErr
					}
					continue
				}
			}
			if completion.Res != int32(requests[index].Length) {
				if resultErr == nil {
					if completion.Res < 0 {
						resultErr = fmt.Errorf("core: DiskANN io_uring request %d: %w", index, unix.Errno(-completion.Res))
					} else {
						resultErr = fmt.Errorf("%w: io_uring request %d read %d of %d bytes", ErrDiskANNShortRead, index, completion.Res, requests[index].Length)
					}
				}
				r.poisoned.Store(true)
			}
		}
		checkQueueDamage()
		if quiescenceUncertain {
			r.abandonStagingLocked()
			return resultErr
		}
	}
	checkQueueDamage()
	if quiescenceUncertain {
		r.abandonStagingLocked()
		return resultErr
	}
	if resultErr != nil {
		return resultErr
	}
	for index, buffer := range buffers {
		copy(buffer, r.staging[index*r.stagingSlot:index*r.stagingSlot+len(buffer)])
	}
	return nil
}

func (r *diskANNIOUring) ensureStaging(slotSize int) error {
	entries := int(atomic.LoadUint32(r.sqEntries))
	if entries <= 0 || slotSize > math.MaxInt/entries {
		return errors.New("core: DiskANN io_uring staging size overflow")
	}
	needed := slotSize * entries
	if r.staging != nil && r.stagingSlot >= slotSize && len(r.staging) >= needed {
		return nil
	}
	mapped, err := r.system.mmap(-1, 0, needed, unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED|unix.MAP_ANON)
	if err != nil {
		return err
	}
	old := r.staging
	r.staging, r.stagingSlot = mapped, slotSize
	if old != nil {
		_ = r.system.munmap(old)
	}
	return nil
}

// abandonStagingLocked closes the ring but intentionally keeps staging mapped:
// an unaccounted kernel request may still hold an address into it.
func (r *diskANNIOUring) abandonStagingLocked() {
	r.poisoned.Store(true)
	r.stagingRetained = true
	_ = r.closeLocked()
}

func isTransientIOUringError(err error) bool {
	return errors.Is(err, unix.EINTR) || errors.Is(err, unix.EAGAIN) || errors.Is(err, unix.EBUSY)
}

func (r *diskANNIOUring) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.closeLocked()
}

func (r *diskANNIOUring) closeLocked() error {
	if r.closed {
		return nil
	}
	r.closed = true
	var result error
	// Stop the kernel from accepting more work before releasing ring mappings.
	if r.fd >= 0 {
		if err := r.system.close(r.fd); err != nil {
			result = err
		}
	}
	r.fd = -1
	if r.sqesMap != nil {
		if err := r.system.munmap(r.sqesMap); err != nil && result == nil {
			result = err
		}
	}
	if len(r.cqRing) != 0 && (len(r.sqRing) == 0 || unsafe.Pointer(&r.cqRing[0]) != unsafe.Pointer(&r.sqRing[0])) {
		if err := r.system.munmap(r.cqRing); err != nil && result == nil {
			result = err
		}
	}
	if r.sqRing != nil {
		if err := r.system.munmap(r.sqRing); err != nil && result == nil {
			result = err
		}
	}
	if r.staging != nil && !r.stagingRetained {
		if err := r.system.munmap(r.staging); err != nil && result == nil {
			result = err
		}
		r.staging = nil
	}
	r.sqRing, r.cqRing, r.sqesMap = nil, nil, nil
	return result
}
