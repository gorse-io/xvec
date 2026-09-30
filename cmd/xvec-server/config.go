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
	"flag"
	"fmt"
	"io"
	"math"
	"strings"
	"time"

	"github.com/gorse-io/xvec"
)

type config struct {
	Listen                        string
	DataDir                       string
	EnableMmap                    bool
	MaxBufferSize                 uint32
	WALSyncEvery                  uint64
	StreamBatchSize               int
	MaxStreamLifetime             time.Duration
	StreamIdleTimeout             time.Duration
	MaxConcurrentStreamsPerClient int
	ShutdownTimeout               time.Duration
}

func parseConfig(args []string, stderr io.Writer) (config, error) {
	var cfg config
	var maxBufferSize uint64
	flags := flag.NewFlagSet("xvec-server", flag.ContinueOnError)
	flags.SetOutput(stderr)
	flags.StringVar(&cfg.Listen, "listen", ":50051", "address on which to listen")
	flags.StringVar(&cfg.DataDir, "data-dir", "", "directory containing xvec data")
	flags.BoolVar(&cfg.EnableMmap, "enable-mmap", true, "enable memory-mapped collection storage")
	flags.Uint64Var(&maxBufferSize, "max-buffer-size", uint64(xvec.DefaultMaxBufferSize), "maximum collection buffer size in bytes")
	flags.Uint64Var(&cfg.WALSyncEvery, "wal-sync-every", 0, "synchronize the WAL after this many records (zero disables automatic synchronization)")
	flags.IntVar(&cfg.StreamBatchSize, "stream-batch-size", 128, "number of results sent in each stream batch")
	flags.DurationVar(&cfg.MaxStreamLifetime, "max-stream-lifetime", 15*time.Minute, "maximum lifetime of a stream")
	flags.DurationVar(&cfg.StreamIdleTimeout, "stream-idle-timeout", time.Minute, "maximum idle time of a stream")
	flags.IntVar(&cfg.MaxConcurrentStreamsPerClient, "max-concurrent-streams-client", 8, "maximum concurrent streams per client")
	flags.DurationVar(&cfg.ShutdownTimeout, "shutdown-timeout", 30*time.Second, "maximum graceful shutdown duration")
	if err := flags.Parse(args); err != nil {
		return config{}, err
	}
	if flags.NArg() != 0 {
		return config{}, fmt.Errorf("unexpected positional arguments: %s", strings.Join(flags.Args(), " "))
	}
	if cfg.DataDir == "" {
		return config{}, fmt.Errorf("--data-dir is required")
	}
	if cfg.Listen == "" {
		return config{}, fmt.Errorf("--listen must not be empty")
	}
	if maxBufferSize == 0 {
		return config{}, fmt.Errorf("--max-buffer-size must be greater than zero")
	}
	if maxBufferSize > math.MaxUint32 {
		return config{}, fmt.Errorf("--max-buffer-size must not exceed %d", uint64(math.MaxUint32))
	}
	if cfg.StreamBatchSize <= 0 {
		return config{}, fmt.Errorf("--stream-batch-size must be greater than zero")
	}
	if cfg.MaxStreamLifetime <= 0 {
		return config{}, fmt.Errorf("--max-stream-lifetime must be greater than zero")
	}
	if cfg.StreamIdleTimeout <= 0 {
		return config{}, fmt.Errorf("--stream-idle-timeout must be greater than zero")
	}
	if cfg.MaxConcurrentStreamsPerClient <= 0 {
		return config{}, fmt.Errorf("--max-concurrent-streams-client must be greater than zero")
	}
	if cfg.ShutdownTimeout <= 0 {
		return config{}, fmt.Errorf("--shutdown-timeout must be greater than zero")
	}
	cfg.MaxBufferSize = uint32(maxBufferSize)
	return cfg, nil
}
