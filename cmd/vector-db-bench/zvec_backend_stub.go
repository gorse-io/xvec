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
	"context"
	"errors"
	"io"
)

var errZvecBackendUnavailable = errors.New("zvec backend requires a pure-Go benchmark build: CGO_ENABLED=0 go build -o vector-db-bench ./cmd/vector-db-bench")

func initializeZvecBackend() (func(), error) {
	return nil, errZvecBackendUnavailable
}

func loadZvecDataset(context.Context, benchConfig, io.Writer) (loadMetrics, error) {
	return loadMetrics{}, errZvecBackendUnavailable
}

func openZvecQueryEngine(benchConfig) (benchmarkQueryEngine, io.Closer, error) {
	return nil, nil, errZvecBackendUnavailable
}
