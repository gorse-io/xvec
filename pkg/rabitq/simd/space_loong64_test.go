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

func TestSpaceLASX(t *testing.T) {
	if !cpu.Loong64.HasLASX {
		t.Skip("LASX is not supported")
	}
	testSpace(t, scalar_quantize_uint8_lasx, scalar_quantize_uint16_lasx,
		new_transpose_bin_lasx, new_transpose_bin_512_lasx, mask_ip_x0_q_lasx)
}
