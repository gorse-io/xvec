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

package main

import (
	"bytes"
	"math"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/gorse-io/xvec"
	"github.com/stretchr/testify/require"
)

func TestParseConfigRequiresDataDir(t *testing.T) {
	_, err := parseConfig(nil, &bytes.Buffer{})
	require.EqualError(t, err, "--data-dir is required")
}

func TestParseConfigDefaults(t *testing.T) {
	got, err := parseConfig([]string{"--data-dir", "/var/lib/xvec"}, &bytes.Buffer{})
	require.NoError(t, err)
	require.Equal(t, ":50051", got.Listen)
	require.Equal(t, "/var/lib/xvec", got.DataDir)
	require.True(t, got.EnableMmap)
	require.Equal(t, xvec.DefaultMaxBufferSize, got.MaxBufferSize)
	require.Equal(t, uint64(0), got.WALSyncEvery)
	require.Equal(t, 128, got.StreamBatchSize)
	require.Equal(t, 15*time.Minute, got.MaxStreamLifetime)
	require.Equal(t, time.Minute, got.StreamIdleTimeout)
	require.Equal(t, 8, got.MaxConcurrentStreamsPerClient)
	require.Equal(t, 30*time.Second, got.ShutdownTimeout)
}

func TestParseConfigOverridesDefaults(t *testing.T) {
	got, err := parseConfig([]string{
		"--data-dir", "data",
		"--listen", "127.0.0.1:0",
		"--enable-mmap=false",
		"--max-buffer-size", "4096",
		"--wal-sync-every", "7",
		"--stream-batch-size", "16",
		"--max-stream-lifetime", "2m",
		"--stream-idle-timeout", "5s",
		"--max-concurrent-streams-client", "3",
		"--shutdown-timeout", "10s",
	}, &bytes.Buffer{})
	require.NoError(t, err)
	require.Equal(t, "127.0.0.1:0", got.Listen)
	require.False(t, got.EnableMmap)
	require.Equal(t, uint32(4096), got.MaxBufferSize)
	require.Equal(t, uint64(7), got.WALSyncEvery)
	require.Equal(t, 16, got.StreamBatchSize)
	require.Equal(t, 2*time.Minute, got.MaxStreamLifetime)
	require.Equal(t, 5*time.Second, got.StreamIdleTimeout)
	require.Equal(t, 3, got.MaxConcurrentStreamsPerClient)
	require.Equal(t, 10*time.Second, got.ShutdownTimeout)
}

func TestParseConfigRejectsInvalidValues(t *testing.T) {
	tests := []struct {
		name string
		args []string
		want string
	}{
		{name: "empty listen", args: []string{"--listen", ""}, want: "--listen must not be empty"},
		{name: "zero max buffer", args: []string{"--max-buffer-size", "0"}, want: "--max-buffer-size must be greater than zero"},
		{name: "max buffer overflow", args: []string{"--max-buffer-size", strconv.FormatUint(uint64(math.MaxUint32)+1, 10)}, want: "--max-buffer-size must not exceed 4294967295"},
		{name: "zero batch", args: []string{"--stream-batch-size", "0"}, want: "--stream-batch-size must be greater than zero"},
		{name: "zero lifetime", args: []string{"--max-stream-lifetime", "0"}, want: "--max-stream-lifetime must be greater than zero"},
		{name: "zero idle timeout", args: []string{"--stream-idle-timeout", "0"}, want: "--stream-idle-timeout must be greater than zero"},
		{name: "zero concurrent streams", args: []string{"--max-concurrent-streams-client", "0"}, want: "--max-concurrent-streams-client must be greater than zero"},
		{name: "zero shutdown timeout", args: []string{"--shutdown-timeout", "0"}, want: "--shutdown-timeout must be greater than zero"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			args := append([]string{"--data-dir", "data"}, tt.args...)
			_, err := parseConfig(args, &bytes.Buffer{})
			require.EqualError(t, err, tt.want)
		})
	}
}

func TestParseConfigRejectsPositionalArguments(t *testing.T) {
	_, err := parseConfig([]string{"--data-dir", "data", "unexpected"}, &bytes.Buffer{})
	require.EqualError(t, err, "unexpected positional arguments: unexpected")
}

func TestParseConfigReportsFlagErrorsToProvidedWriter(t *testing.T) {
	var stderr bytes.Buffer
	_, err := parseConfig([]string{"--not-a-flag"}, &stderr)
	require.Error(t, err)
	require.True(t, strings.Contains(stderr.String(), "flag provided but not defined"))
}
