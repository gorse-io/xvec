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
	"sync/atomic"
	"testing"
)

func TestParallelFor(t *testing.T) {
	const count = 200
	seen := make([]atomic.Int32, count)
	if err := ParallelFor(context.Background(), count, 8, func(_ context.Context, index int) error {
		seen[index].Add(1)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	for index := range count {
		if got := seen[index].Load(); got != 1 {
			t.Fatalf("index %d called %d times", index, got)
		}
	}
}

func TestParallelForCancelsOnError(t *testing.T) {
	want := errors.New("failed work item")
	err := ParallelFor(context.Background(), 1000, 1, func(_ context.Context, index int) error {
		if index == 5 {
			return want
		}
		return nil
	})
	if !errors.Is(err, want) {
		t.Fatalf("ParallelFor error = %v", err)
	}
}

func TestGroupCancelsPeers(t *testing.T) {
	want := errors.New("first error")
	group := NewGroup(context.Background())
	group.Go(func(context.Context) error { return want })
	group.Go(func(ctx context.Context) error {
		<-ctx.Done()
		return ctx.Err()
	})
	if err := group.Wait(); !errors.Is(err, want) {
		t.Fatalf("Wait error = %v", err)
	}
}
