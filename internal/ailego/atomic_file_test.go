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

package ailego

import (
	"bytes"
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

const atomicCrashPayloadSize = 32 << 20

func TestWriteFileAtomicReplaceAndCancel(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	path := filepath.Join(dir, "artifact")
	if err := os.WriteFile(path, []byte("old"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := WriteFileAtomic(context.Background(), path, []byte("new"), 0o600); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "new" {
		t.Fatalf("contents = %q", got)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := WriteFileAtomic(ctx, path, []byte("bad"), 0o600); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled error = %v", err)
	}
	got, err = os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "new" {
		t.Fatalf("canceled write changed contents to %q", got)
	}
	if err := WriteFileAtomic(nil, path, nil, 0o600); err == nil {
		t.Fatal("nil context succeeded")
	}
	if err := WriteFileAtomic(context.Background(), "", nil, 0o600); err == nil {
		t.Fatal("empty path succeeded")
	}
}

func TestWriteFileAtomicProcessKill(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping subprocess fault injection in short mode")
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "artifact")
	old := []byte("previous-generation")
	if err := os.WriteFile(path, old, 0o600); err != nil {
		t.Fatal(err)
	}
	command := exec.Command(os.Args[0], "-test.run=^TestWriteFileAtomicCrashHelper$")
	command.Env = append(os.Environ(),
		"ZVEC_ATOMIC_CRASH_HELPER=1",
		"ZVEC_ATOMIC_CRASH_PATH="+path,
	)
	if err := command.Start(); err != nil {
		t.Fatal(err)
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
			if err != nil {
				t.Fatal(err)
			}
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
			t.Fatal("atomic write helper did not reach a persistence boundary")
		}
	}

	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Equal(got, old) {
		return
	}
	if len(got) != atomicCrashPayloadSize {
		t.Fatalf("published torn generation of %d bytes", len(got))
	}
	for offset, value := range got {
		if value != byte(offset) {
			t.Fatalf("published generation differs at byte %d", offset)
		}
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
