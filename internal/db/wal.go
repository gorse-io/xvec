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

package db

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sync"

	"github.com/gorse-io/zvec/internal/ailego"
)

const (
	walFormatVersion    uint16 = 1
	walFileHeaderSize          = 32
	walRecordHeaderSize        = 32

	// MaxWALRecordSize matches the pinned baseline's per-record safety limit.
	MaxWALRecordSize = 4 << 20
)

var (
	walMagic       = [8]byte{'Z', 'V', 'E', 'C', 'W', 'A', 'L', 0}
	walRecordMagic = [4]byte{'Z', 'R', 'E', 'C'}

	ErrWALNotFound       = errors.New("db: WAL not found")
	ErrWALExists         = errors.New("db: WAL already exists")
	ErrWALCorrupt        = errors.New("db: corrupt WAL")
	ErrWALClosed         = errors.New("db: WAL is closed")
	ErrWALPoisoned       = errors.New("db: WAL append state is poisoned")
	ErrWALRecordTooLarge = errors.New("db: WAL record is too large")
)

// WALOptions controls explicit durability batching. SyncEvery zero disables
// automatic record-count-based syncing; callers can use Sync directly.
type WALOptions struct {
	SyncEvery uint64
}

// WALRecovery describes the valid prefix found while opening a WAL.
type WALRecovery struct {
	Records        uint64
	LastLSN        uint64
	ValidBytes     int64
	TruncatedBytes int64
}

// WALRecord is one replayed opaque operation payload.
type WALRecord struct {
	LSN     uint64
	Payload []byte
}

// WAL is a single-writer, append-only write-ahead log. A sidecar advisory lock
// prevents separate handles or processes from appending concurrently.
type WAL struct {
	path string
	file *os.File
	lock *ailego.FileLock

	mu           sync.Mutex
	options      WALOptions
	recovery     WALRecovery
	size         int64
	lastLSN      uint64
	dirtyRecords uint64
	poisoned     error
	closed       bool
}

// CreateWAL creates a new log, writes and syncs its file header, and keeps an
// exclusive writer lock until Close.
func CreateWAL(ctx context.Context, name string, options WALOptions) (*WAL, error) {
	if ctx == nil {
		return nil, errors.New("db: nil create WAL context")
	}
	if name == "" {
		return nil, errors.New("db: empty WAL path")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if _, err := os.Stat(name); err == nil {
		return nil, ErrWALExists
	} else if !errors.Is(err, os.ErrNotExist) {
		return nil, fmt.Errorf("db: stat WAL before create: %w", err)
	}
	lock, err := ailego.AcquireFileLock(ctx, name+".lock", ailego.LockExclusive)
	if err != nil {
		return nil, fmt.Errorf("db: lock WAL creation: %w", err)
	}
	file, err := os.OpenFile(name, os.O_CREATE|os.O_EXCL|os.O_RDWR, 0o600)
	if err != nil {
		_ = lock.Close()
		if errors.Is(err, os.ErrExist) {
			return nil, ErrWALExists
		}
		return nil, fmt.Errorf("db: create WAL: %w", err)
	}
	keep := false
	defer func() {
		if !keep {
			_ = file.Close()
			_ = lock.Close()
			_ = os.Remove(name)
		}
	}()
	header := encodeWALFileHeader()
	if err := ailego.WriteFullAt(file, header, 0); err != nil {
		return nil, fmt.Errorf("db: write WAL header: %w", err)
	}
	if err := file.Sync(); err != nil {
		return nil, fmt.Errorf("db: sync WAL header: %w", err)
	}
	if err := syncDirectory(filepath.Dir(name)); err != nil {
		return nil, fmt.Errorf("db: sync WAL directory: %w", err)
	}
	keep = true
	recovery := WALRecovery{ValidBytes: walFileHeaderSize}
	return &WAL{
		path: name, file: file, lock: lock, options: options,
		recovery: recovery, size: walFileHeaderSize,
	}, nil
}

// OpenWAL validates an existing log. A partial final header or payload is
// truncated and reported; all other structural or checksum damage is fatal.
func OpenWAL(ctx context.Context, name string, options WALOptions) (*WAL, error) {
	if ctx == nil {
		return nil, errors.New("db: nil open WAL context")
	}
	if name == "" {
		return nil, errors.New("db: empty WAL path")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	lock, err := ailego.AcquireFileLock(ctx, name+".lock", ailego.LockExclusive)
	if err != nil {
		return nil, fmt.Errorf("db: lock WAL open: %w", err)
	}
	file, err := os.OpenFile(name, os.O_RDWR, 0)
	if err != nil {
		_ = lock.Close()
		if errors.Is(err, os.ErrNotExist) {
			return nil, ErrWALNotFound
		}
		return nil, fmt.Errorf("db: open WAL: %w", err)
	}
	fail := func(err error) (*WAL, error) {
		_ = file.Close()
		_ = lock.Close()
		return nil, err
	}
	info, err := file.Stat()
	if err != nil {
		return fail(fmt.Errorf("db: stat WAL: %w", err))
	}
	recovery, err := scanWAL(ctx, file, info.Size())
	if err != nil {
		return fail(err)
	}
	if recovery.TruncatedBytes > 0 {
		if err := file.Truncate(recovery.ValidBytes); err != nil {
			return fail(fmt.Errorf("db: truncate partial WAL tail: %w", err))
		}
		if err := file.Sync(); err != nil {
			return fail(fmt.Errorf("db: sync repaired WAL tail: %w", err))
		}
	}
	return &WAL{
		path: name, file: file, lock: lock, options: options,
		recovery: recovery, size: recovery.ValidBytes, lastLSN: recovery.LastLSN,
	}, nil
}

// Recovery returns the immutable result from opening the log.
func (w *WAL) Recovery() WALRecovery {
	if w == nil {
		return WALRecovery{}
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.recovery
}

// HasRecords reports whether at least one complete record is present.
func (w *WAL) HasRecords() bool {
	if w == nil {
		return false
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.lastLSN != 0
}

// Append writes one opaque payload and returns its monotonically increasing
// LSN. Once a write fails, the handle is poisoned and must be closed and
// reopened so tail recovery can establish a safe append offset.
func (w *WAL) Append(ctx context.Context, payload []byte) (uint64, error) {
	if w == nil {
		return 0, errors.New("db: nil WAL")
	}
	if ctx == nil {
		return 0, errors.New("db: nil WAL append context")
	}
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	if len(payload) == 0 {
		return 0, errors.New("db: empty WAL record")
	}
	if len(payload) > MaxWALRecordSize {
		return 0, fmt.Errorf("%w: got %d bytes, maximum %d", ErrWALRecordTooLarge, len(payload), MaxWALRecordSize)
	}

	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return 0, ErrWALClosed
	}
	if w.poisoned != nil {
		return 0, w.poisoned
	}
	if w.lastLSN == math.MaxUint64 {
		return 0, errors.New("db: WAL LSN exhausted")
	}
	lsn := w.lastLSN + 1
	encoded := encodeWALRecord(lsn, payload)
	if err := ailego.WriteFullAt(w.file, encoded, w.size); err != nil {
		w.poisoned = fmt.Errorf("%w: write record %d: %v", ErrWALPoisoned, lsn, err)
		return 0, w.poisoned
	}
	w.size += int64(len(encoded))
	w.lastLSN = lsn
	w.dirtyRecords++
	if w.options.SyncEvery > 0 && w.dirtyRecords >= w.options.SyncEvery {
		if err := w.file.Sync(); err != nil {
			return lsn, fmt.Errorf("db: sync WAL at LSN %d: %w", lsn, err)
		}
		w.dirtyRecords = 0
	}
	return lsn, nil
}

// Sync makes every successfully appended record durable before returning.
func (w *WAL) Sync(ctx context.Context) error {
	if w == nil {
		return errors.New("db: nil WAL")
	}
	if ctx == nil {
		return errors.New("db: nil WAL sync context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return ErrWALClosed
	}
	if w.poisoned != nil {
		return w.poisoned
	}
	if err := w.file.Sync(); err != nil {
		return fmt.Errorf("db: sync WAL: %w", err)
	}
	w.dirtyRecords = 0
	return nil
}

// NewReader opens an independent reader over the complete-record prefix that
// existed at the time of this call.
func (w *WAL) NewReader() (*WALReader, error) {
	if w == nil {
		return nil, errors.New("db: nil WAL")
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return nil, ErrWALClosed
	}
	if w.poisoned != nil {
		return nil, w.poisoned
	}
	file, err := os.Open(w.path)
	if err != nil {
		return nil, fmt.Errorf("db: open WAL reader: %w", err)
	}
	header := make([]byte, walFileHeaderSize)
	if err := ailego.ReadFullAt(file, header, 0); err != nil {
		_ = file.Close()
		return nil, fmt.Errorf("%w: read WAL reader header: %v", ErrWALCorrupt, err)
	}
	if err := validateWALFileHeader(header); err != nil {
		_ = file.Close()
		return nil, err
	}
	return &WALReader{file: file, offset: walFileHeaderSize, limit: w.size, nextLSN: 1}, nil
}

// Replay invokes apply in LSN order over a stable WAL snapshot.
func (w *WAL) Replay(ctx context.Context, apply func(WALRecord) error) error {
	if ctx == nil {
		return errors.New("db: nil WAL replay context")
	}
	if apply == nil {
		return errors.New("db: nil WAL replay function")
	}
	reader, err := w.NewReader()
	if err != nil {
		return err
	}
	defer reader.Close()
	for {
		record, err := reader.Next(ctx)
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
		if err := apply(record); err != nil {
			return fmt.Errorf("db: replay WAL record %d: %w", record.LSN, err)
		}
	}
}

// Close syncs complete records, closes the file, and releases the writer lock.
// It is idempotent. A poisoned log is closed without syncing its partial tail.
func (w *WAL) Close() error {
	if w == nil {
		return nil
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return nil
	}
	w.closed = true
	var syncErr error
	if w.poisoned == nil && w.dirtyRecords > 0 {
		if err := w.file.Sync(); err != nil {
			syncErr = fmt.Errorf("db: sync WAL on close: %w", err)
		}
	}
	closeErr := w.file.Close()
	lockErr := w.lock.Close()
	return errors.Join(w.poisoned, syncErr, closeErr, lockErr)
}

// WALReader iterates an immutable valid-prefix snapshot. It owns an
// independent file handle and is safe for sequential use.
type WALReader struct {
	mu      sync.Mutex
	file    *os.File
	offset  int64
	limit   int64
	nextLSN uint64
	closed  bool
}

// Next returns the next verified record or io.EOF.
func (r *WALReader) Next(ctx context.Context) (WALRecord, error) {
	if r == nil {
		return WALRecord{}, errors.New("db: nil WAL reader")
	}
	if ctx == nil {
		return WALRecord{}, errors.New("db: nil WAL reader context")
	}
	if err := ctx.Err(); err != nil {
		return WALRecord{}, err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return WALRecord{}, ErrWALClosed
	}
	if r.offset == r.limit {
		return WALRecord{}, io.EOF
	}
	record, size, err := readWALRecordAt(r.file, r.offset, r.limit, r.nextLSN)
	if err != nil {
		return WALRecord{}, err
	}
	r.offset += size
	r.nextLSN++
	return record, nil
}

// Close releases the reader file handle and is idempotent.
func (r *WALReader) Close() error {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil
	}
	r.closed = true
	return r.file.Close()
}

func scanWAL(ctx context.Context, reader io.ReaderAt, size int64) (WALRecovery, error) {
	if size < walFileHeaderSize {
		return WALRecovery{}, fmt.Errorf("%w: file is shorter than its header", ErrWALCorrupt)
	}
	header := make([]byte, walFileHeaderSize)
	if err := ailego.ReadFullAt(reader, header, 0); err != nil {
		return WALRecovery{}, fmt.Errorf("%w: read file header: %v", ErrWALCorrupt, err)
	}
	if err := validateWALFileHeader(header); err != nil {
		return WALRecovery{}, err
	}

	recovery := WALRecovery{ValidBytes: walFileHeaderSize}
	offset := int64(walFileHeaderSize)
	expectedLSN := uint64(1)
	for offset < size {
		if err := ctx.Err(); err != nil {
			return WALRecovery{}, err
		}
		remaining := size - offset
		if remaining < walRecordHeaderSize {
			recovery.TruncatedBytes = remaining
			return recovery, nil
		}
		recordHeader := make([]byte, walRecordHeaderSize)
		if err := ailego.ReadFullAt(reader, recordHeader, offset); err != nil {
			return WALRecovery{}, fmt.Errorf("%w: read record header at %d: %v", ErrWALCorrupt, offset, err)
		}
		metadata, err := decodeWALRecordHeader(recordHeader, expectedLSN)
		if err != nil {
			return WALRecovery{}, fmt.Errorf("%w: record at %d: %v", ErrWALCorrupt, offset, err)
		}
		recordSize := int64(walRecordHeaderSize) + int64(metadata.payloadLength)
		if remaining < recordSize {
			recovery.TruncatedBytes = remaining
			return recovery, nil
		}
		payload := make([]byte, metadata.payloadLength)
		if err := ailego.ReadFullAt(reader, payload, offset+walRecordHeaderSize); err != nil {
			return WALRecovery{}, fmt.Errorf("%w: read record %d payload: %v", ErrWALCorrupt, expectedLSN, err)
		}
		if actual := ailego.CRC32C(payload); actual != metadata.payloadCRC {
			return WALRecovery{}, fmt.Errorf("%w: record %d payload checksum got %08x, want %08x", ErrWALCorrupt, expectedLSN, actual, metadata.payloadCRC)
		}
		offset += recordSize
		recovery.Records++
		recovery.LastLSN = expectedLSN
		recovery.ValidBytes = offset
		expectedLSN++
	}
	return recovery, nil
}

func readWALRecordAt(reader io.ReaderAt, offset, limit int64, expectedLSN uint64) (WALRecord, int64, error) {
	remaining := limit - offset
	if remaining < walRecordHeaderSize {
		return WALRecord{}, 0, fmt.Errorf("%w: partial record header at %d", ErrWALCorrupt, offset)
	}
	header := make([]byte, walRecordHeaderSize)
	if err := ailego.ReadFullAt(reader, header, offset); err != nil {
		return WALRecord{}, 0, fmt.Errorf("%w: read record header: %v", ErrWALCorrupt, err)
	}
	metadata, err := decodeWALRecordHeader(header, expectedLSN)
	if err != nil {
		return WALRecord{}, 0, fmt.Errorf("%w: record at %d: %v", ErrWALCorrupt, offset, err)
	}
	recordSize := int64(walRecordHeaderSize) + int64(metadata.payloadLength)
	if remaining < recordSize {
		return WALRecord{}, 0, fmt.Errorf("%w: partial record %d payload", ErrWALCorrupt, expectedLSN)
	}
	payload := make([]byte, metadata.payloadLength)
	if err := ailego.ReadFullAt(reader, payload, offset+walRecordHeaderSize); err != nil {
		return WALRecord{}, 0, fmt.Errorf("%w: read record payload: %v", ErrWALCorrupt, err)
	}
	if actual := ailego.CRC32C(payload); actual != metadata.payloadCRC {
		return WALRecord{}, 0, fmt.Errorf("%w: record %d payload checksum got %08x, want %08x", ErrWALCorrupt, expectedLSN, actual, metadata.payloadCRC)
	}
	return WALRecord{LSN: metadata.lsn, Payload: payload}, recordSize, nil
}

type walRecordMetadata struct {
	lsn           uint64
	payloadLength uint32
	payloadCRC    uint32
}

func encodeWALFileHeader() []byte {
	header := make([]byte, walFileHeaderSize)
	copy(header[:8], walMagic[:])
	binary.LittleEndian.PutUint16(header[8:10], walFormatVersion)
	binary.LittleEndian.PutUint16(header[10:12], walFileHeaderSize)
	binary.LittleEndian.PutUint32(header[12:16], MaxWALRecordSize)
	binary.LittleEndian.PutUint32(header[24:28], ailego.CRC32C(header[:24]))
	return header
}

func validateWALFileHeader(header []byte) error {
	if len(header) != walFileHeaderSize {
		return fmt.Errorf("%w: invalid file header length %d", ErrWALCorrupt, len(header))
	}
	if !bytes.Equal(header[:8], walMagic[:]) {
		return fmt.Errorf("%w: invalid file magic", ErrWALCorrupt)
	}
	if version := binary.LittleEndian.Uint16(header[8:10]); version != walFormatVersion {
		return fmt.Errorf("%w: unsupported WAL version %d", ErrUnsupportedFormatVersion, version)
	}
	if size := binary.LittleEndian.Uint16(header[10:12]); size != walFileHeaderSize {
		return fmt.Errorf("%w: invalid file header size %d", ErrWALCorrupt, size)
	}
	if maximum := binary.LittleEndian.Uint32(header[12:16]); maximum != MaxWALRecordSize {
		return fmt.Errorf("%w: invalid maximum record size %d", ErrWALCorrupt, maximum)
	}
	if binary.LittleEndian.Uint64(header[16:24]) != 0 || binary.LittleEndian.Uint32(header[28:32]) != 0 {
		return fmt.Errorf("%w: nonzero file header reserved bytes", ErrWALCorrupt)
	}
	if actual, expected := ailego.CRC32C(header[:24]), binary.LittleEndian.Uint32(header[24:28]); actual != expected {
		return fmt.Errorf("%w: file header checksum got %08x, want %08x", ErrWALCorrupt, actual, expected)
	}
	return nil
}

func encodeWALRecord(lsn uint64, payload []byte) []byte {
	encoded := make([]byte, walRecordHeaderSize+len(payload))
	copy(encoded[:4], walRecordMagic[:])
	binary.LittleEndian.PutUint16(encoded[4:6], walFormatVersion)
	binary.LittleEndian.PutUint16(encoded[6:8], walRecordHeaderSize)
	binary.LittleEndian.PutUint64(encoded[8:16], lsn)
	binary.LittleEndian.PutUint32(encoded[16:20], uint32(len(payload)))
	binary.LittleEndian.PutUint32(encoded[20:24], ailego.CRC32C(payload))
	binary.LittleEndian.PutUint32(encoded[24:28], ailego.CRC32C(encoded[:24]))
	copy(encoded[walRecordHeaderSize:], payload)
	return encoded
}

func decodeWALRecordHeader(header []byte, expectedLSN uint64) (walRecordMetadata, error) {
	if len(header) != walRecordHeaderSize {
		return walRecordMetadata{}, fmt.Errorf("invalid record header length %d", len(header))
	}
	if !bytes.Equal(header[:4], walRecordMagic[:]) {
		return walRecordMetadata{}, errors.New("invalid record magic")
	}
	if version := binary.LittleEndian.Uint16(header[4:6]); version != walFormatVersion {
		return walRecordMetadata{}, fmt.Errorf("unsupported record version %d", version)
	}
	if size := binary.LittleEndian.Uint16(header[6:8]); size != walRecordHeaderSize {
		return walRecordMetadata{}, fmt.Errorf("invalid record header size %d", size)
	}
	lsn := binary.LittleEndian.Uint64(header[8:16])
	if lsn != expectedLSN {
		return walRecordMetadata{}, fmt.Errorf("LSN %d, want %d", lsn, expectedLSN)
	}
	payloadLength := binary.LittleEndian.Uint32(header[16:20])
	if payloadLength == 0 || payloadLength > MaxWALRecordSize {
		return walRecordMetadata{}, fmt.Errorf("invalid payload length %d", payloadLength)
	}
	if binary.LittleEndian.Uint32(header[28:32]) != 0 {
		return walRecordMetadata{}, errors.New("nonzero record header reserved bytes")
	}
	if actual, expected := ailego.CRC32C(header[:24]), binary.LittleEndian.Uint32(header[24:28]); actual != expected {
		return walRecordMetadata{}, fmt.Errorf("header checksum got %08x, want %08x", actual, expected)
	}
	return walRecordMetadata{
		lsn: lsn, payloadLength: payloadLength,
		payloadCRC: binary.LittleEndian.Uint32(header[20:24]),
	}, nil
}
