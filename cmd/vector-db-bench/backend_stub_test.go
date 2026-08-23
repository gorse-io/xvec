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

//go:build cgo && !purego

package main

import (
	"bytes"
	"context"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestBenchmarkBackendDispatchAndZvecStub(t *testing.T) {
	shutdown, err := initializeBenchmarkBackend(benchConfig{Backend: backendXvec})
	require.NoError(t, err)
	require.NotNil(t, shutdown)
	shutdown()

	_, err = initializeBenchmarkBackend(benchConfig{Backend: backendZvec})
	require.ErrorIs(t, err, errZvecBackendUnavailable)

	config := benchConfig{Backend: backendZvec}
	_, err = loadBenchmarkDataset(context.Background(), config, &bytes.Buffer{})
	require.ErrorIs(t, err, errZvecBackendUnavailable)
	_, _, err = openBenchmarkQueryEngine(context.Background(), config)
	require.ErrorIs(t, err, errZvecBackendUnavailable)

	_, err = loadBenchmarkDataset(context.Background(), benchConfig{Backend: "unknown"}, &bytes.Buffer{})
	require.ErrorContains(t, err, "unsupported backend")
	_, _, err = openBenchmarkQueryEngine(context.Background(), benchConfig{Backend: "unknown"})
	require.ErrorContains(t, err, "unsupported backend")

	_, _, err = openBenchmarkQueryEngine(context.Background(), benchConfig{
		Backend: backendXvec,
		Path:    filepath.Join(t.TempDir(), "missing"),
	})
	require.ErrorContains(t, err, "open xvec benchmark collection")
}
