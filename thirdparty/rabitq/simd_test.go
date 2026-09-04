// Copyright 2026-present the xvec project
// Licensed under the Apache License, Version 2.0.

package rabitq

import (
	"math"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestSIMDRotatorKernels(t *testing.T) {
	require.NoError(t, FlipSign(nil, nil))

	data := make([]float32, 64)
	want := make([]float32, len(data))
	signs := make([]byte, len(data)/8)
	for i := range data {
		data[i] = float32(i) - 19.5
		want[i] = data[i]
		if i%3 == 1 {
			signs[i/8] |= 1 << uint(i%8)
			want[i] = -want[i]
		}
	}
	require.NoError(t, FlipSign(signs, data))
	require.Equal(t, want, data)

	for i := range data {
		data[i] = float32(i) - 23
		want[i] = data[i]
	}
	kacWalk(want)
	require.NoError(t, KacsWalk(data))
	require.Equal(t, want, data)
}

func TestSIMDScalarQuantize(t *testing.T) {
	values := make([]float32, 67)
	for i := range values {
		values[i] = float32(i%23)/3 - 2
	}
	values[65] = -math.MaxFloat32
	values[66] = math.MaxFloat32
	got8, err := ScalarQuantizeUint8(values, -2, .25)
	require.NoError(t, err)
	got16, err := ScalarQuantizeUint16(values, -2, .0001)
	require.NoError(t, err)
	inv8 := float32(1) / float32(.25)
	inv16 := float32(1) / float32(.0001)
	for i, value := range values {
		want8 := min(max(math.RoundToEven(float64((value+2)*inv8)), 0), float64(math.MaxUint8))
		want16 := min(max(math.RoundToEven(float64((value+2)*inv16)), 0), float64(math.MaxUint16))
		require.Equal(t, uint8(want8), got8[i])
		require.Equal(t, uint16(want16), got16[i])
	}
}

func TestSIMDTransposeAndMaskKernels(t *testing.T) {
	const dim = 128
	q16 := make([]uint16, dim)
	q8 := make([]uint8, dim)
	for i := range q16 {
		q16[i] = uint16((i*37 + 11) & 15)
		q8[i] = uint8(q16[i])
	}
	got, err := NewTransposeBin(q16, 4)
	require.NoError(t, err)
	got512, err := NewTransposeBin512(q8, 4)
	require.NoError(t, err)
	require.Equal(t, transposeBinScalar(q16, 4), got)
	require.Equal(t, transposeBin512Scalar(q8, 4), got512)

	query := make([]float32, dim)
	code := make([]byte, dim/8)
	var want float32
	for i := range query {
		query[i] = float32(i%17) - 8
		if i%5 < 2 {
			setBinaryBit(code, i)
			want += query[i]
		}
	}
	gotIP, err := MaskIPX0Q(query, code)
	require.NoError(t, err)
	require.Equal(t, want, gotIP)
}

func TestSIMDFastScanKernels(t *testing.T) {
	const dim = 128
	codes := make([]byte, dim*4)
	lut8 := make([]byte, dim*4)
	lut16 := make([]uint16, dim*4)
	for i := range codes {
		codes[i] = byte(i*29 + 7)
		lut8[i] = byte(i*13 + 3)
		lut16[i] = uint16(i*257 + 31)
	}
	got, err := FastScanAccumulate(codes, lut8, dim)
	require.NoError(t, err)
	require.Equal(t, fastScanAccumulateScalar(codes, lut8, dim), got)

	hc, err := FastScanTransferLUTHACC(lut16, dim)
	require.NoError(t, err)
	gotHACC, err := FastScanAccumulateHACC(codes, hc, dim)
	require.NoError(t, err)
	require.Equal(t, fastScanAccumulateHACCScalar(codes, hc, dim), gotHACC)
}

func TestSplitBatchEstimateChunksHACC(t *testing.T) {
	const dim = fastScanHACCChunkSize + 64
	query := make([]float32, dim)
	for i := range query {
		query[i] = float32(i%29) - 14
	}
	q, err := NewSplitBatchQuery(query, 1, MetricIP)
	require.NoError(t, err)

	batch := make([]byte, BatchDataBytes(dim))
	codes := batch[:dim*4]
	for i := range codes {
		codes[i] = byte(i*37 + 11)
	}

	haccLUT := fastScanTransferLUTHACCScalar(q.lut, dim)
	want := fastScanAccumulateHACCScalar(codes, haccLUT, dim)
	got, err := SplitBatchEstimate(batch, q)
	require.NoError(t, err)
	for lane := range got {
		require.Equal(t, q.delta*float32(want[lane])+q.sumVL, got[lane].IPX0QR,
			"lane=%d", lane)
	}
}

func TestSIMDWarmupKernel(t *testing.T) {
	const (
		dim   = 576
		width = 4
	)
	data := make([]uint64, dim/64)
	quantized := make([]uint8, dim)
	var ip, pop uint64
	for i := range quantized {
		quantized[i] = uint8((i*7 + 3) & 15)
		if i%5 < 2 {
			data[i/64] |= uint64(1) << uint(63-i%64)
			ip += uint64(quantized[i])
			pop++
		}
	}
	query := transposeBin512Scalar(quantized, width)
	got, err := WarmupIPX0Q512(data, query, .25, -.5, dim, width)
	require.NoError(t, err)
	require.Equal(t, float32(.25)*float32(ip)-float32(.5)*float32(pop), got)
}

func TestSIMDExcodeAndPackingMatchScalar(t *testing.T) {
	query := make([]float32, 128)
	for i := range query {
		query[i] = float32(i%19) - 9
	}
	for exBits := 1; exBits <= 8; exBits++ {
		values := make([]uint8, len(query))
		for i := range values {
			values[i] = uint8((i*11 + 5) & ((1 << exBits) - 1))
		}
		gotPacked, err := packExCodeSIMD(values, exBits)
		require.NoError(t, err)
		wantPacked := make([]byte, len(values)/8*exBits)
		for base := 0; base < len(values); base += 64 {
			packExBlock(wantPacked[base*exBits/8:], values[base:base+64], exBits)
		}
		require.Equal(t, wantPacked, gotPacked, "bits=%d", exBits)
		gotIP, err := excodeIPSIMD(query, gotPacked, exBits)
		require.NoError(t, err)
		var wantIP float32
		for i, value := range values {
			wantIP += query[i] * float32(value)
		}
		require.Equal(t, wantIP, gotIP, "bits=%d", exBits)
	}
}
