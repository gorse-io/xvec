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

package ioutil

import (
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

const atomicCrashPayloadSize = 32 << 20

func TestWriteFileAtomicReplaceAndCancel(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	path := filepath.Join(dir, "artifact")
	{
		err := os.WriteFile(path, []byte("old"), 0o644)
		require.NoError(t, err)
	}
	{
		err := WriteFileAtomic(context.Background(), path, []byte("new"), 0o600)
		require.NoError(t, err)
	}

	got, err := os.ReadFile(path)
	require.NoError(t, err)
	require.True(t, string(got) == "new")

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	{
		err := WriteFileAtomic(ctx, path, []byte("bad"), 0o600)
		require.ErrorIs(t, err, context.Canceled)
	}

	got, err = os.ReadFile(path)
	require.NoError(t, err)
	require.True(t, string(got) == "new")
	{
		err := WriteFileAtomic(nil, path, nil, 0o600)
		require.Error(t, err,
			"nil context succeeded")
	}
	{
		err := WriteFileAtomic(context.Background(), "", nil, 0o600)
		require.Error(t, err,
			"empty path succeeded")
	}
}

func TestWriteFileAtomicFuncSeekAndFailure(t *testing.T) {
	path := filepath.Join(t.TempDir(), "streamed")
	require.NoError(t, WriteFileAtomic(context.Background(), path, []byte("old"), 0o600))
	require.NoError(t, WriteFileAtomicFunc(context.Background(), path, 0o600, func(file *os.File) error {
		_, err := file.Write([]byte("placeholder"))
		if err != nil {
			return err
		}
		_, err = file.Seek(0, io.SeekStart)
		if err != nil {
			return err
		}
		_, err = file.Write([]byte("new"))
		return err
	}))
	data, err := os.ReadFile(path)
	require.NoError(t, err)
	require.Equal(t, []byte("newceholder"), data)

	want := errors.New("write failed")
	err = WriteFileAtomicFunc(context.Background(), path, 0o600, func(file *os.File) error {
		_, writeErr := file.Write([]byte("bad"))
		require.NoError(t, writeErr)
		return want
	})
	require.ErrorIs(t, err, want)
	data, err = os.ReadFile(path)
	require.NoError(t, err)
	require.Equal(t, []byte("newceholder"), data)
}

func TestWriteFileAtomicProcessKill(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping subprocess fault injection in short mode")
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "artifact")
	old := []byte("previous-generation")
	{
		err := os.WriteFile(path, old, 0o600)
		require.NoError(t, err)
	}

	command := exec.Command(os.Args[0], "-test.run=^TestWriteFileAtomicCrashHelper$")
	command.Env = append(os.Environ(),
		"ZVEC_ATOMIC_CRASH_HELPER=1",
		"ZVEC_ATOMIC_CRASH_PATH="+path,
	)
	{
		err := command.Start()
		require.NoError(t, err)
	}

	done := make(chan error, 1)
	go func() { done <- command.Wait() }()

	deadline := time.NewTimer(15 * time.Second)
	ticker := time.NewTicker(time.Millisecond)
	defer deadline.Stop()
	defer ticker.Stop()
	waited := false
	for !waited {
		select {
		case <-ticker.C:
			temps, err := filepath.Glob(filepath.Join(dir, ".zvec-atomic-*.tmp"))
			require.NoError(t, err)

			if len(temps) != 0 {
				_ = command.Process.Kill()
				<-done
				waited = true
			}
		case <-done:
			waited = true
		case <-deadline.C:
			_ = command.Process.Kill()
			<-done
			require.FailNow(t, "atomic write helper did not reach a persistence boundary")
		}
	}

	got, err := os.ReadFile(path)
	require.NoError(t, err)

	if bytes.Equal(got, old) {
		return
	}
	require.Len(t, got, atomicCrashPayloadSize)

	for offset, value := range got {
		require.Equal(t, byte(offset), value)
	}
}

func TestWriteFileAtomicCrashHelper(t *testing.T) {
	if os.Getenv("ZVEC_ATOMIC_CRASH_HELPER") != "1" {
		return
	}
	data := make([]byte, atomicCrashPayloadSize)
	for offset := range data {
		data[offset] = byte(offset)
	}
	if err := WriteFileAtomic(context.Background(), os.Getenv("ZVEC_ATOMIC_CRASH_PATH"), data, 0o600); err != nil {
		os.Exit(2)
	}
	os.Exit(0)
}
