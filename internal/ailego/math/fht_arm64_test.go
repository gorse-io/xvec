//go:build !noasm && arm64

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

package mathutil

import (
	"testing"

	"golang.org/x/sys/cpu"
)

func TestFHTNEON(t *testing.T) {
	if !cpu.ARM64.HasASIMD {
		t.Skip("NEON is not supported")
	}
	testFHTKernels(t, fhtKernels{
		flipSigns:      fhtFlipSignsNEON,
		kacWalk:        fhtKacWalkNEON,
		inverseKacWalk: fhtInverseKacWalkNEON,
		inPlace:        fhtInPlaceNEON,
	})
}
