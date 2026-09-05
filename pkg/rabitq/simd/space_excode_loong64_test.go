//go:build !noasm && loong64

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

package simd

import (
	"testing"

	"golang.org/x/sys/cpu"
)

func TestSpaceExcodeLASX(t *testing.T) {
	if !cpu.Loong64.HasLASX {
		t.Skip("LASX is not supported")
	}
	testSpaceExcode(t, []excodeIPFunc{
		ip16_fxu1_lasx,
		ip64_fxu2_lasx,
		ip64_fxu3_lasx,
		ip16_fxu4_lasx,
		ip64_fxu5_lasx,
		ip64_fxu6_lasx,
		ip64_fxu7_lasx,
		ip16_fxu8_lasx,
	})
}
