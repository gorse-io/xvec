// Copyright 2026-present the xvec project
// Licensed under the Apache License, Version 2.0.

//go:build !noasm && amd64

package rabitq

import (
	"unsafe"

	"golang.org/x/sys/cpu"
)

func init() {
	installBestKernels()
}

func installBestKernels() {
	if cpu.X86.HasAVX2 && cpu.X86.HasAVX512F && cpu.X86.HasAVX512BW && cpu.X86.HasAVX512VL && cpu.X86.HasAVX512VPOPCNTDQ && cpu.X86.HasFMA {
		installAVX512Kernels()
	} else if cpu.X86.HasAVX2 && cpu.X86.HasFMA {
		installAVX2Kernels()
	}
}

func installAVX2Kernels() {
	flipSignKernel = func(signs []byte, data []float32) {
		n := len(data) &^ 31
		if n != 0 {
			rabitq_flip_sign_avx2(unsafe.Pointer(&signs[0]), unsafe.Pointer(&data[0]), int64(n))
		}
		flipSignScalar(signs[n/8:], data[n:])
	}
	kacsWalkKernel = func(data []float32) {
		rabitq_kacs_walk_avx2(unsafe.Pointer(&data[0]), int64(len(data)))
	}
	quantizeUint8Kernel = func(out []uint8, values []float32, lo, delta float32) {
		if len(values) != 0 {
			rabitq_quantize_u8_avx2(unsafe.Pointer(&out[0]), unsafe.Pointer(&values[0]), int64(len(values)), lo, delta)
		}
	}
	quantizeUint16Kernel = func(out []uint16, values []float32, lo, delta float32) {
		if len(values) != 0 {
			rabitq_quantize_u16_avx2(unsafe.Pointer(&out[0]), unsafe.Pointer(&values[0]), int64(len(values)), lo, delta)
		}
	}
	transposeBinKernel = func(query []uint16, out []uint64, width int) {
		rabitq_transpose_bin_avx2(unsafe.Pointer(&query[0]), unsafe.Pointer(&out[0]), int64(len(query)), int64(width))
	}
	transposeBin512Kernel = func(query []uint8, out []uint64, width int) {
		rabitq_transpose_bin_512_avx2(unsafe.Pointer(&query[0]), unsafe.Pointer(&out[0]), int64(len(query)), int64(width))
	}
	maskIPKernel = func(query []float32, code []byte) float32 {
		return rabitq_mask_ip_avx2(unsafe.Pointer(&query[0]), unsafe.Pointer(&code[0]), int64(len(query)))
	}
	fastScanAccumulateKernel = func(codes, lut []byte, out []uint16, dim int) {
		rabitq_fastscan_accumulate_avx2(unsafe.Pointer(&codes[0]), unsafe.Pointer(&lut[0]), unsafe.Pointer(&out[0]), int64(dim))
	}
	fastScanTransferLUTHACCKernel = func(lut []uint16, out []byte, dim int) {
		rabitq_fastscan_transfer_hacc_avx2(unsafe.Pointer(&lut[0]), unsafe.Pointer(&out[0]), int64(dim))
	}
	fastScanAccumulateHACCKernel = func(codes, lut []byte, out []int32, dim int) {
		rabitq_fastscan_accumulate_hacc_avx2(unsafe.Pointer(&codes[0]), unsafe.Pointer(&lut[0]), unsafe.Pointer(&out[0]), int64(dim))
	}
	warmupIPKernel = func(data, query []uint64, delta, vl float32, width int) float32 {
		return rabitq_warmup_ip_avx2(unsafe.Pointer(&data[0]), unsafe.Pointer(&query[0]), int64(len(data)), int64(width), delta, vl)
	}
	packExCodeKernel = func(values []uint8, bits int) []byte {
		out := make([]byte, len(values)/8*bits)
		rabitq_pack_excode_avx2(unsafe.Pointer(&values[0]), unsafe.Pointer(&out[0]), int64(len(values)), int64(bits))
		return out
	}
	excodeIPKernel = func(query []float32, packed []byte, bits int) float32 {
		return rabitq_excode_ip_avx2(unsafe.Pointer(&query[0]), unsafe.Pointer(&packed[0]), int64(len(query)), int64(bits))
	}
}

func installAVX512Kernels() {
	flipSignKernel = func(signs []byte, data []float32) {
		n := len(data) &^ 63
		if n != 0 {
			rabitq_flip_sign_avx512(unsafe.Pointer(&signs[0]), unsafe.Pointer(&data[0]), int64(n))
		}
		flipSignScalar(signs[n/8:], data[n:])
	}
	kacsWalkKernel = func(data []float32) {
		if len(data)%32 == 0 {
			rabitq_kacs_walk_avx512(unsafe.Pointer(&data[0]), int64(len(data)))
		} else {
			kacsWalkScalar(data)
		}
	}
	quantizeUint8Kernel = func(out []uint8, values []float32, lo, delta float32) {
		if len(values) != 0 {
			rabitq_quantize_u8_avx512(unsafe.Pointer(&out[0]), unsafe.Pointer(&values[0]), int64(len(values)), lo, delta)
		}
	}
	quantizeUint16Kernel = func(out []uint16, values []float32, lo, delta float32) {
		if len(values) != 0 {
			rabitq_quantize_u16_avx512(unsafe.Pointer(&out[0]), unsafe.Pointer(&values[0]), int64(len(values)), lo, delta)
		}
	}
	transposeBinKernel = func(query []uint16, out []uint64, width int) {
		rabitq_transpose_bin_avx512(unsafe.Pointer(&query[0]), unsafe.Pointer(&out[0]), int64(len(query)), int64(width))
	}
	transposeBin512Kernel = func(query []uint8, out []uint64, width int) {
		rabitq_transpose_bin_512_avx512(unsafe.Pointer(&query[0]), unsafe.Pointer(&out[0]), int64(len(query)), int64(width))
	}
	maskIPKernel = func(query []float32, code []byte) float32 {
		return rabitq_mask_ip_avx512(unsafe.Pointer(&query[0]), unsafe.Pointer(&code[0]), int64(len(query)))
	}
	fastScanAccumulateKernel = func(codes, lut []byte, out []uint16, dim int) {
		rabitq_fastscan_accumulate_avx512(unsafe.Pointer(&codes[0]), unsafe.Pointer(&lut[0]), unsafe.Pointer(&out[0]), int64(dim))
	}
	fastScanTransferLUTHACCKernel = func(lut []uint16, out []byte, dim int) {
		rabitq_fastscan_transfer_hacc_avx512(unsafe.Pointer(&lut[0]), unsafe.Pointer(&out[0]), int64(dim))
	}
	fastScanAccumulateHACCKernel = func(codes, lut []byte, out []int32, dim int) {
		rabitq_fastscan_accumulate_hacc_avx512(unsafe.Pointer(&codes[0]), unsafe.Pointer(&lut[0]), unsafe.Pointer(&out[0]), int64(dim))
	}
	warmupIPKernel = func(data, query []uint64, delta, vl float32, width int) float32 {
		return rabitq_warmup_ip_avx512(unsafe.Pointer(&data[0]), unsafe.Pointer(&query[0]), int64(len(data)), int64(width), delta, vl)
	}
	packExCodeKernel = func(values []uint8, bits int) []byte {
		out := make([]byte, len(values)/8*bits)
		rabitq_pack_excode_avx512(unsafe.Pointer(&values[0]), unsafe.Pointer(&out[0]), int64(len(values)), int64(bits))
		return out
	}
	excodeIPKernel = func(query []float32, packed []byte, bits int) float32 {
		return rabitq_excode_ip_avx512(unsafe.Pointer(&query[0]), unsafe.Pointer(&packed[0]), int64(len(query)), int64(bits))
	}
}
