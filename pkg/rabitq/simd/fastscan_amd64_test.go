//go:build !noasm && amd64

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
	"reflect"
	"testing"

	"golang.org/x/sys/cpu"
)

func TestFastScanAVX2(t *testing.T) {
	if !cpu.X86.HasAVX2 {
		t.Skip("AVX2 is not supported")
	}
	testFastScan(t, accumulate_avx2, transfer_lut_hacc_avx2, accumulate_hacc_avx2)
}

func TestFastScanAVX512(t *testing.T) {
	if !cpu.X86.HasAVX512F || !cpu.X86.HasAVX512DQ || !cpu.X86.HasAVX512BW {
		t.Skip("AVX-512F, AVX-512DQ, and AVX-512BW are not supported")
	}
	testFastScan(t, accumulate_avx512, transfer_lut_hacc_avx512, accumulate_hacc_avx512)
}

func TestFastScanAVX512Dispatch(t *testing.T) {
	hasAVX512F, hasAVX512DQ, hasAVX512BW := cpu.X86.HasAVX512F, cpu.X86.HasAVX512DQ, cpu.X86.HasAVX512BW
	oldAccumulate, oldTransferLUTHACC, oldAccumulateHACC := accumulate, transferLUTHACC, accumulateHACC
	t.Cleanup(func() {
		cpu.X86.HasAVX512F, cpu.X86.HasAVX512DQ, cpu.X86.HasAVX512BW = hasAVX512F, hasAVX512DQ, hasAVX512BW
		accumulate, transferLUTHACC, accumulateHACC = oldAccumulate, oldTransferLUTHACC, oldAccumulateHACC
	})

	cpu.X86.HasAVX512F, cpu.X86.HasAVX512DQ, cpu.X86.HasAVX512BW = true, true, true
	initFastScan()
	requireFunctionEqual(t, accumulate, accumulate_avx512)
	requireFunctionEqual(t, transferLUTHACC, transfer_lut_hacc_avx512)
	requireFunctionEqual(t, accumulateHACC, accumulate_hacc_avx512)
}

func requireFunctionEqual(t *testing.T, got, want any) {
	t.Helper()
	if reflect.ValueOf(got).Pointer() != reflect.ValueOf(want).Pointer() {
		t.Fatalf("function pointers differ: got %x, want %x", reflect.ValueOf(got).Pointer(), reflect.ValueOf(want).Pointer())
	}
}
