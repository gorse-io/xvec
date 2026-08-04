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

package core

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

const (
	diskANNSaveCrashHelperEnv = "ZVEC_DISKANN_SAVE_CRASH_HELPER"
	diskANNSaveSourceEnv      = "ZVEC_DISKANN_SAVE_SOURCE"
	diskANNSaveTargetEnv      = "ZVEC_DISKANN_SAVE_TARGET"
)

// TestV04DiskANNAtomicSaveProcessKill kills a writer only after its temporary
// artifact exists. The published path must still contain one complete,
// checksummed generation—never a prefix of either generation.
func TestV04DiskANNAtomicSaveProcessKill(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping DiskANN subprocess fault injection in short mode")
	}
	dir := t.TempDir()
	target := filepath.Join(dir, "current.diskann")
	oldOptions := DefaultDiskANNBuildOptions(MetricL2)
	oldOptions.MaxDegree, oldOptions.ListSize, oldOptions.PQChunks = 4, 8, 2
	oldIndex := buildDiskANNIndex(t, diskANNIndexCandidates(8, 4), oldOptions)
	{
		err := oldIndex.Save(context.Background(), target)
		require.NoError(t, err)
	}

	// A large fixed-record layout keeps the atomic writer inside its chunked
	// write boundary long enough for the parent to observe and kill it without
	// requiring a production-only fault hook.
	source := filepath.Join(dir, "replacement.diskann")
	newOptions := DefaultDiskANNBuildOptions(MetricL2)
	newOptions.MaxDegree, newOptions.ListSize, newOptions.PQChunks = 32_767, 32_767, 4
	newIndex := buildDiskANNIndex(t, diskANNIndexCandidates(192, 8), newOptions)
	{
		err := newIndex.Save(context.Background(), source)
		require.NoError(t, err)
	}

	command := exec.Command(os.Args[0], "-test.run=^TestV04DiskANNSaveCrashHelper$")
	command.Env = append(os.Environ(),
		diskANNSaveCrashHelperEnv+"=1",
		diskANNSaveSourceEnv+"="+source,
		diskANNSaveTargetEnv+"="+target,
	)
	{
		err := command.Start()
		require.NoError(t, err)
	}

	done := make(chan error, 1)
	go func() { done <- command.Wait() }()

	deadline := time.Now().Add(30 * time.Second)
	killed := false
	for !killed {
		select {
		case err := <-done:
			require.FailNowf(t, "DiskANN crash helper exited before kill boundary", "%v", err)
		default:
		}
		temps, err := filepath.Glob(filepath.Join(dir, ".zvec-atomic-*.tmp"))
		require.NoError(t, err)

		if len(temps) != 0 {
			{
				err := command.Process.Kill()
				require.NoError(t, err)
			}

			<-done
			killed = true
			break
		}
		if time.Now().After(deadline) {
			_ = command.Process.Kill()
			<-done
			require.FailNow(t, "DiskANN helper did not reach the atomic write boundary")
		}
		time.Sleep(time.Millisecond)
	}

	opened, err := OpenDiskANNIndex(context.Background(), target, 0, 1)
	require.NoError(t, err)

	defer opened.Close()
	require.False(t, opened.Len() != 8 && opened.Len() != 192)
}

func TestV04DiskANNSaveCrashHelper(t *testing.T) {
	if os.Getenv(diskANNSaveCrashHelperEnv) != "1" {
		return
	}
	index, err := OpenDiskANNIndex(
		context.Background(), os.Getenv(diskANNSaveSourceEnv), 0, 1,
	)
	if err != nil {
		os.Exit(2)
	}
	if err := index.Save(context.Background(), os.Getenv(diskANNSaveTargetEnv)); err != nil {
		os.Exit(3)
	}
	os.Exit(0)
}
