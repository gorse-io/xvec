// Copyright 2026-present the xvec project
// Licensed under the Apache License, Version 2.0.

//go:build !noasm && amd64

package rabitq

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
	"golang.org/x/sys/cpu"
)

func TestAVX512Kernels(t *testing.T) {
	if !cpu.X86.HasAVX2 || !cpu.X86.HasAVX512F || !cpu.X86.HasAVX512BW || !cpu.X86.HasAVX512VL || !cpu.X86.HasAVX512VPOPCNTDQ || !cpu.X86.HasFMA {
		t.Skip("AVX-512 kernels are not supported by this CPU")
	}
	installAVX512Kernels()
	defer installBestKernels()
	testInstalledExcodeKernels(t)
	testInstalledRotatorBoundaries(t)
	testInstalledQuantizers(t)
}

func TestAVX2Kernels(t *testing.T) {
	if !cpu.X86.HasAVX2 || !cpu.X86.HasFMA {
		t.Skip("AVX2 kernels are not supported by this CPU")
	}
	installAVX2Kernels()
	defer installBestKernels()
	testInstalledExcodeKernels(t)
	testInstalledRotatorBoundaries(t)
	testInstalledQuantizers(t)
}

func testInstalledQuantizers(t *testing.T) {
	const (
		lo    = float32(-321.125)
		delta = float32(.1)
	)
	values := make([]float32, 20003)
	state := uint32(0x12345678)
	for i := range values {
		state = state*1664525 + 1013904223
		values[i] = float32(int32(state%20000001)-10000000) / 10000
	}
	want8 := make([]uint8, len(values))
	want16 := make([]uint16, len(values))
	quantizeUint8Scalar(want8, values, lo, delta)
	quantizeUint16Scalar(want16, values, lo, delta)
	got8, err := ScalarQuantizeUint8(values, lo, delta)
	require.NoError(t, err)
	got16, err := ScalarQuantizeUint16(values, lo, delta)
	require.NoError(t, err)
	require.Equal(t, want8, got8)
	require.Equal(t, want16, got16)
}

func testInstalledExcodeKernels(t *testing.T) {
	values := make([]uint8, 64)
	query := make([]float32, 64)
	for i := range values {
		values[i] = uint8((i*7 + 3) & 31)
		query[i] = float32(i%13) - 6
	}
	packed, err := packExCodeSIMD(values, 5)
	require.NoError(t, err)
	var want float32
	for i := range values {
		want += query[i] * float32(values[i])
	}
	got, err := excodeIPSIMD(query, packed, 5)
	require.NoError(t, err)
	require.Equal(t, want, got)
}

func testInstalledRotatorBoundaries(t *testing.T) {
	for _, size := range []int{1, 16, 31, 33, 48, 63, 65} {
		t.Run(fmt.Sprintf("flip/%d", size), func(t *testing.T) {
			signs := make([]byte, (size+7)/8)
			for i := range signs {
				signs[i] = byte(i*73 + 0x5a)
			}
			want := make([]float32, size)
			for i := range want {
				want[i] = float32(i + 1)
			}
			backing := make([]float32, size+64)
			copy(backing, want)
			for i := size; i < len(backing); i++ {
				backing[i] = 12345
			}
			got := backing[:size]
			flipSignScalar(signs, want)
			require.NoError(t, FlipSign(signs, got))
			require.Equal(t, want, got)
			for i := size; i < len(backing); i++ {
				require.Equal(t, float32(12345), backing[i], "canary=%d", i-size)
			}
		})
	}
	for _, size := range []int{16, 32, 48, 64} {
		t.Run(fmt.Sprintf("kac/%d", size), func(t *testing.T) {
			want := make([]float32, size)
			for i := range want {
				want[i] = float32(i) - 7
			}
			backing := make([]float32, size+32)
			copy(backing, want)
			for i := size; i < len(backing); i++ {
				backing[i] = 12345
			}
			got := backing[:size]
			kacsWalkScalar(want)
			require.NoError(t, KacsWalk(got))
			require.Equal(t, want, got)
			for i := size; i < len(backing); i++ {
				require.Equal(t, float32(12345), backing[i], "canary=%d", i-size)
			}
		})
	}
}
