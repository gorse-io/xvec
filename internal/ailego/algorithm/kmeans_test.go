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

package algorithm

import (
	"context"
	"errors"
	"testing"

	mathutil "github.com/gorse-io/xvec/internal/ailego/math"
)

func TestKMC2AcceptsDistanceWeightedProposalAndOwnsCenters(t *testing.T) {
	vectors := [][]float32{{0}, {1}, {10}}
	next := 0
	centers, err := InitializeKMC2(context.Background(), vectors, 2, 2,
		func(int) int { value := next; next++; return value },
		func() float64 { return 0.5 }, mathutil.L2Squared)
	if err != nil {
		t.Fatal(err)
	}
	if len(centers) != 2 || centers[0][0] != 0 || centers[1][0] != 10 {
		t.Fatalf("expected distant proposal, got %v", centers)
	}
	vectors[2][0] = -1
	if centers[1][0] != 10 {
		t.Fatal("centroid aliases training storage")
	}
}

func TestKMC2Cancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	calls := 0
	_, err := InitializeKMC2(ctx, [][]float32{{1}, {2}}, 2, 32,
		func(int) int {
			calls++
			if calls == 2 {
				cancel()
			}
			return 0
		},
		func() float64 { return 0 }, mathutil.L2Squared)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want cancellation", err)
	}
}
