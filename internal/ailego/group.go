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
	"context"
	"errors"
	"runtime"
	"sync"
	"sync/atomic"
)

// Group runs related functions under a context canceled by the first error.
type Group struct {
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
	once   sync.Once
	err    error
}

// NewGroup returns a group derived from ctx. A nil context is treated as
// context.Background.
func NewGroup(ctx context.Context) *Group {
	if ctx == nil {
		ctx = context.Background()
	}
	groupCtx, cancel := context.WithCancel(ctx)
	return &Group{ctx: groupCtx, cancel: cancel}
}

// Go starts fn. The first non-nil error cancels the contexts observed by all
// group functions.
func (g *Group) Go(fn func(context.Context) error) {
	if fn == nil {
		panic("ailego: nil group function")
	}
	g.wg.Add(1)
	go func() {
		defer g.wg.Done()
		if err := fn(g.ctx); err != nil {
			g.once.Do(func() {
				g.err = err
				g.cancel()
			})
		}
	}()
}

// Wait blocks until every function returns, releases context resources, and
// returns the first function error.
func (g *Group) Wait() error {
	g.wg.Wait()
	g.cancel()
	return g.err
}

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

	group := NewGroup(ctx)
	var next atomic.Int64
	for range workers {
		group.Go(func(ctx context.Context) error {
			for {
				if err := ctx.Err(); err != nil {
					return err
				}
				index := int(next.Add(1) - 1)
				if index >= n {
					return nil
				}
				if err := fn(ctx, index); err != nil {
					return err
				}
			}
		})
	}
	return group.Wait()
}
