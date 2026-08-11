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

package ailego

import (
	"context"
	"errors"
	"runtime"
	"sync/atomic"

	"golang.org/x/sync/errgroup"
)

// ParallelFor calls fn once for each integer in [0, n), using at most workers
// goroutines. A non-positive workers value uses GOMAXPROCS. The first error
// cancels outstanding work.
func ParallelFor(
	ctx context.Context,
	n int,
	workers int,
	fn func(context.Context, int) error,
) error {
	if n < 0 {
		return errors.New("ailego: negative parallel range")
	}
	if n == 0 {
		return nil
	}
	if fn == nil {
		return errors.New("ailego: nil parallel function")
	}
	if workers <= 0 {
		workers = runtime.GOMAXPROCS(0)
	}
	if workers > n {
		workers = n
	}
	if ctx == nil {
		ctx = context.Background()
	}

	group, groupCtx := errgroup.WithContext(ctx)
	var next atomic.Int64
	for range workers {
		group.Go(func() error {
			for {
				if err := groupCtx.Err(); err != nil {
					return err
				}
				index := int(next.Add(1) - 1)
				if index >= n {
					return nil
				}
				if err := fn(groupCtx, index); err != nil {
					return err
				}
			}
		})
	}
	return group.Wait()
}
