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
	"time"

	"github.com/stretchr/testify/require"
)

func TestParallelFor(t *testing.T) {
	const count = 200
	seen := make([]atomic.Int32, count)
	{
		err := ParallelFor(context.Background(), count, 8, func(_ context.Context, index int) error {
			seen[index].Add(1)
			return nil
		})
		require.NoError(t, err)
	}

	for index := range count {
		{
			got := seen[index].Load()
			require.True(t, got == 1)
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
	require.ErrorIs(t, err, want)
}

func TestParallelForCancelsPeers(t *testing.T) {
	want := errors.New("first error")
	ready := make(chan struct{})
	peerCanceled := make(chan bool, 1)
	err := ParallelFor(context.Background(), 2, 2, func(ctx context.Context, index int) error {
		switch index {
		case 0:
			<-ready
			return want
		case 1:
			close(ready)
			select {
			case <-ctx.Done():
				peerCanceled <- true
				return ctx.Err()
			case <-time.After(time.Second):
				peerCanceled <- false
				return errors.New("peer was not canceled")
			}
		default:
			panic("unexpected work item")
		}
	})
	require.ErrorIs(t, err, want)
	require.True(t, <-peerCanceled)
}
