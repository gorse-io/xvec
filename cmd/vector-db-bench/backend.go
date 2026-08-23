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
	"context"
	"fmt"
	"io"

	"github.com/gorse-io/xvec"
)

func initializeBenchmarkBackend(config benchConfig) (func(), error) {
	if config.Backend == backendZvec {
		return initializeZvecBackend(config)
	}
	return func() {}, nil
}

func loadBenchmarkDataset(ctx context.Context, config benchConfig, log io.Writer) (loadMetrics, error) {
	switch config.Backend {
	case backendXvec:
		return loadXvecDataset(ctx, config, log)
	case backendZvec:
		return loadZvecDataset(ctx, config, log)
	default:
		return loadMetrics{}, fmt.Errorf("unsupported backend %q", config.Backend)
	}
}

func openBenchmarkQueryEngine(ctx context.Context, config benchConfig) (benchmarkQueryEngine, io.Closer, error) {
	switch config.Backend {
	case backendXvec:
		collection, err := xvec.Open(ctx, config.Path, xvec.CollectionOptions{
			ReadOnly: true, EnableMmap: config.EnableMmap, MaxBufferSize: uint32(config.MaxBufferSize),
		})
		if err != nil {
			return nil, nil, fmt.Errorf("open xvec benchmark collection for search: %w", err)
		}
		return newXvecQueryEngine(collection, config), collection, nil
	case backendZvec:
		return openZvecQueryEngine(config)
	default:
		return nil, nil, fmt.Errorf("unsupported backend %q", config.Backend)
	}
}
