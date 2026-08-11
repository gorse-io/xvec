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

package db

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/gorse-io/xvec/internal/ailego"
)

const (
	snapshotCodecVersion uint16 = 1
	snapshotHeaderSize          = 40
	maxSnapshotPayload          = 256 << 20
)

var ErrSnapshotCorrupt = errors.New("db: corrupt snapshot")

func encodeSnapshot(magic [8]byte, count uint64, payload []byte) ([]byte, error) {
	if len(payload) > maxSnapshotPayload {
		return nil, fmt.Errorf("db: snapshot payload is %d bytes, maximum %d", len(payload), maxSnapshotPayload)
	}
	encoded := make([]byte, snapshotHeaderSize+len(payload))
	copy(encoded[:8], magic[:])
	binary.LittleEndian.PutUint16(encoded[8:10], snapshotCodecVersion)
	binary.LittleEndian.PutUint16(encoded[10:12], snapshotHeaderSize)
	binary.LittleEndian.PutUint64(encoded[16:24], count)
	binary.LittleEndian.PutUint64(encoded[24:32], uint64(len(payload)))
	binary.LittleEndian.PutUint32(encoded[32:36], ailego.CRC32C(payload))
	binary.LittleEndian.PutUint32(encoded[36:40], ailego.CRC32C(encoded[:36]))
	copy(encoded[snapshotHeaderSize:], payload)
	return encoded, nil
}

func decodeSnapshot(encoded []byte, magic [8]byte) (count uint64, payload []byte, err error) {
	if len(encoded) < snapshotHeaderSize {
		return 0, nil, fmt.Errorf("%w: file is shorter than its header", ErrSnapshotCorrupt)
	}
	if !bytes.Equal(encoded[:8], magic[:]) {
		return 0, nil, fmt.Errorf("%w: invalid magic", ErrSnapshotCorrupt)
	}
	if version := binary.LittleEndian.Uint16(encoded[8:10]); version != snapshotCodecVersion {
		return 0, nil, fmt.Errorf("%w: snapshot version %d", ErrUnsupportedFormatVersion, version)
	}
	if size := binary.LittleEndian.Uint16(encoded[10:12]); size != snapshotHeaderSize {
		return 0, nil, fmt.Errorf("%w: invalid header size %d", ErrSnapshotCorrupt, size)
	}
	if binary.LittleEndian.Uint32(encoded[12:16]) != 0 {
		return 0, nil, fmt.Errorf("%w: nonzero reserved bytes", ErrSnapshotCorrupt)
	}
	if actual, expected := ailego.CRC32C(encoded[:36]), binary.LittleEndian.Uint32(encoded[36:40]); actual != expected {
		return 0, nil, fmt.Errorf("%w: header checksum got %08x, want %08x", ErrSnapshotCorrupt, actual, expected)
	}
	payloadLength := binary.LittleEndian.Uint64(encoded[24:32])
	if payloadLength > maxSnapshotPayload || payloadLength != uint64(len(encoded)-snapshotHeaderSize) {
		return 0, nil, fmt.Errorf("%w: invalid payload length %d", ErrSnapshotCorrupt, payloadLength)
	}
	payload = encoded[snapshotHeaderSize:]
	if actual, expected := ailego.CRC32C(payload), binary.LittleEndian.Uint32(encoded[32:36]); actual != expected {
		return 0, nil, fmt.Errorf("%w: payload checksum got %08x, want %08x", ErrSnapshotCorrupt, actual, expected)
	}
	return binary.LittleEndian.Uint64(encoded[16:24]), payload, nil
}

func writeImmutableSnapshot(ctx context.Context, name string, encoded []byte) error {
	if ctx == nil {
		return errors.New("db: nil snapshot write context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if name == "" {
		return errors.New("db: empty snapshot path")
	}
	dir := filepath.Dir(name)
	if err := ensureDirectorySynced(dir); err != nil {
		return fmt.Errorf("db: create snapshot directory: %w", err)
	}
	temp, err := writeTempSynced(ctx, dir, ".snapshot-*.tmp", encoded)
	if err != nil {
		return err
	}
	defer os.Remove(temp)
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := installFileNoReplace(temp, name); err != nil {
		return fmt.Errorf("db: install immutable snapshot: %w", err)
	}
	if err := syncDirectory(dir); err != nil {
		return fmt.Errorf("db: sync snapshot directory: %w", err)
	}
	return nil
}

func ensureDirectorySynced(dir string) error {
	var missing []string
	current := dir
	for {
		info, err := os.Stat(current)
		if err == nil {
			if !info.IsDir() {
				return fmt.Errorf("%s is not a directory", current)
			}
			break
		}
		if !errors.Is(err, os.ErrNotExist) {
			return err
		}
		missing = append(missing, current)
		parent := filepath.Dir(current)
		if parent == current {
			return fmt.Errorf("no existing ancestor for %s", dir)
		}
		current = parent
	}
	if len(missing) == 0 {
		return nil
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	for index := len(missing) - 1; index >= 0; index-- {
		if err := syncDirectory(filepath.Dir(missing[index])); err != nil {
			return err
		}
	}
	return nil
}

func readSnapshotFile(ctx context.Context, name string) ([]byte, error) {
	if ctx == nil {
		return nil, errors.New("db: nil snapshot read context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	file, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return nil, err
	}
	if info.Size() < snapshotHeaderSize || info.Size() > snapshotHeaderSize+maxSnapshotPayload {
		return nil, fmt.Errorf("%w: invalid file size %d", ErrSnapshotCorrupt, info.Size())
	}
	encoded := make([]byte, int(info.Size()))
	if _, err := io.ReadFull(file, encoded); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return encoded, nil
}

func validatePortableRelativePath(name string) error {
	if name == "" || path.IsAbs(name) || path.Clean(name) != name || name == "." || strings.HasPrefix(name, "../") || strings.ContainsRune(name, '\\') {
		return fmt.Errorf("file %q is not a clean portable relative path", name)
	}
	return nil
}
