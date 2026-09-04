// Copyright 2026-present the xvec project
// Licensed under the Apache License, Version 2.0.

package rabitq

import (
	"math"
	"math/bits"
)

const fastScanHACCChunkSize = 1024

// FlipSign negates data elements selected by the little-endian bits in signs.
func FlipSign(signs []byte, data []float32) error {
	if len(signs) < (len(data)+7)/8 {
		return ErrInvalidArgument
	}
	if len(data) == 0 {
		return nil
	}
	flipSignKernel(signs, data)
	return nil
}

// KacsWalk applies one unnormalised Kac butterfly to two equal halves.
func KacsWalk(data []float32) error {
	if len(data) == 0 || len(data)%16 != 0 {
		return ErrInvalidArgument
	}
	kacsWalkKernel(data)
	return nil
}

// ScalarQuantizeUint8 quantizes values with round-to-nearest-even and saturation.
func ScalarQuantizeUint8(values []float32, lo, delta float32) ([]uint8, error) {
	if !isFinite(lo) || !isFinite(delta) || delta <= 0 {
		return nil, ErrInvalidArgument
	}
	out := make([]uint8, len(values))
	for _, value := range values {
		if !isFinite(value) {
			return nil, ErrNonFinite
		}
	}
	if len(values) == 0 {
		return out, nil
	}
	quantizeUint8Kernel(out, values, lo, delta)
	return out, nil
}

// ScalarQuantizeUint16 quantizes values with round-to-nearest-even and saturation.
func ScalarQuantizeUint16(values []float32, lo, delta float32) ([]uint16, error) {
	if !isFinite(lo) || !isFinite(delta) || delta <= 0 {
		return nil, ErrInvalidArgument
	}
	out := make([]uint16, len(values))
	for _, value := range values {
		if !isFinite(value) {
			return nil, ErrNonFinite
		}
	}
	if len(values) == 0 {
		return out, nil
	}
	quantizeUint16Kernel(out, values, lo, delta)
	return out, nil
}

// NewTransposeBin transposes bQuery low bits of each uint16 into 64-bit planes.
func NewTransposeBin(query []uint16, bQuery int) ([]uint64, error) {
	if len(query) == 0 || len(query)%64 != 0 || bQuery < 1 || bQuery > 16 {
		return nil, ErrInvalidArgument
	}
	out := make([]uint64, len(query)/64*bQuery)
	transposeBinKernel(query, out, bQuery)
	return out, nil
}

// NewTransposeBin512 transposes bQuery low bits in 512-dimension macro blocks.
func NewTransposeBin512(query []uint8, bQuery int) ([]uint64, error) {
	if len(query) == 0 || len(query)%64 != 0 || bQuery < 1 || bQuery > 8 {
		return nil, ErrInvalidArgument
	}
	out := make([]uint64, len(query)/64*bQuery)
	transposeBin512Kernel(query, out, bQuery)
	return out, nil
}

// MaskIPX0Q sums query coordinates selected by a RaBitQ one-bit code.
func MaskIPX0Q(query []float32, code []byte) (float32, error) {
	if !validDimension(len(query)) || len(code) != len(query)/8 {
		return 0, ErrInvalidLayout
	}
	for _, value := range query {
		if !isFinite(value) {
			return 0, ErrNonFinite
		}
	}
	return maskIPKernel(query, code), nil
}

// FastScanAccumulate evaluates 32 lanes using 8-bit lookup tables.
func FastScanAccumulate(codes, lut []byte, dim int) ([]uint16, error) {
	if !validDimension(dim) || len(codes) != dim*4 || len(lut) != dim*4 {
		return nil, ErrInvalidLayout
	}
	out := make([]uint16, BatchSize)
	fastScanAccumulateKernel(codes, lut, out, dim)
	return out, nil
}

// FastScanTransferLUTHACC splits uint16 lookup tables into the upstream SIMD layout.
func FastScanTransferLUTHACC(lut []uint16, dim int) ([]byte, error) {
	if !validDimension(dim) || len(lut) != dim*4 {
		return nil, ErrInvalidLayout
	}
	out := make([]byte, dim*8)
	fastScanTransferLUTHACCKernel(lut, out, dim)
	return out, nil
}

// FastScanAccumulateHACC evaluates 32 lanes using transferred uint16 lookup tables.
func FastScanAccumulateHACC(codes, lut []byte, dim int) ([]int32, error) {
	if !validDimension(dim) || dim > fastScanHACCChunkSize || len(codes) != dim*4 || len(lut) != dim*8 {
		return nil, ErrInvalidLayout
	}
	out := make([]int32, BatchSize)
	fastScanAccumulateHACCKernel(codes, lut, out, dim)
	return out, nil
}

// WarmupIPX0Q512 computes the weighted bit-plane intersection and data popcount.
func WarmupIPX0Q512(data, query []uint64, delta, vl float32, paddedDim, bQuery int) (float32, error) {
	words := paddedDim / 64
	if paddedDim == 0 || paddedDim%64 != 0 || bQuery < 1 || bQuery > 16 || len(data) != words || len(query) != words*bQuery || !isFinite(delta) || !isFinite(vl) {
		return 0, ErrInvalidArgument
	}
	return warmupIPKernel(data, query, delta, vl, bQuery), nil
}

func flipSignScalar(signs []byte, data []float32) {
	for i := range data {
		if signs[i/8]&(1<<uint(i%8)) != 0 {
			data[i] = -data[i]
		}
	}
}

func kacsWalkScalar(data []float32) {
	half := len(data) / 2
	for i := 0; i < half; i++ {
		a, b := data[i], data[i+half]
		data[i], data[i+half] = a+b, a-b
	}
}

func quantized(value, lo, delta float32, maximum uint64) uint64 {
	// Match the upstream SIMD operation sequence: compute the reciprocal once,
	// then multiply in float32 before rounding.
	x := math.RoundToEven(float64((value - lo) * (1 / delta)))
	if x <= 0 {
		return 0
	}
	if x >= float64(maximum) {
		return maximum
	}
	return uint64(x)
}

func quantizeUint8Scalar(out []uint8, values []float32, lo, delta float32) {
	for i, value := range values {
		out[i] = uint8(quantized(value, lo, delta, math.MaxUint8))
	}
}

func quantizeUint16Scalar(out []uint16, values []float32, lo, delta float32) {
	for i, value := range values {
		out[i] = uint16(quantized(value, lo, delta, math.MaxUint16))
	}
}

func transposeBinScalar(query []uint16, bQuery int) []uint64 {
	out := make([]uint64, len(query)/64*bQuery)
	for block := 0; block < len(query)/64; block++ {
		for bit := 0; bit < bQuery; bit++ {
			var word uint64
			for i := 0; i < 64; i++ {
				word |= uint64((query[block*64+i]>>uint(bit))&1) << uint(63-i)
			}
			out[block*bQuery+bit] = word
		}
	}
	return out
}

func transposeBin512Scalar(query []uint8, bQuery int) []uint64 {
	out := make([]uint64, len(query)/64*bQuery)
	outBase := 0
	for base := 0; base < len(query); base += 512 {
		end := min(base+512, len(query))
		chunks := (end - base) / 64
		for bit := 0; bit < bQuery; bit++ {
			for chunk := 0; chunk < chunks; chunk++ {
				var word uint64
				for i := 0; i < 64; i++ {
					word |= uint64((query[base+chunk*64+i]>>uint(bit))&1) << uint(63-i)
				}
				out[outBase+bit*chunks+chunk] = word
			}
		}
		outBase += chunks * bQuery
	}
	return out
}

func maskIPScalar(query []float32, code []byte) float32 {
	var sum float32
	for i, value := range query {
		if binaryBit(code, i) {
			sum += value
		}
	}
	return sum
}

func fastScanAccumulateScalar(codes, lut []byte, dim int) []uint16 {
	out := make([]uint16, BatchSize)
	for lane := 0; lane < BatchSize; lane++ {
		var sum uint16
		for table := 0; table < dim/4; table++ {
			code := batchCode(codes, dim, lane, table)
			sum += uint16(lut[table*16+code])
		}
		out[lane] = sum
	}
	return out
}

func batchCode(codes []byte, dim, lane, table int) int {
	code := 0
	for bit := 0; bit < 4; bit++ {
		if batchBinaryBit(codes, dim, lane, table*4+bit) {
			code |= 1 << uint(3-bit)
		}
	}
	return code
}

func fastScanTransferLUTHACCScalar(lut []uint16, dim int) []byte {
	out := make([]byte, dim*8)
	for table := 0; table < dim/4; table++ {
		base := table/2*64 + table%2*16
		for entry := 0; entry < 16; entry++ {
			value := lut[table*16+entry]
			out[base+entry] = byte(value)
			out[base+32+entry] = byte(value >> 8)
		}
	}
	return out
}

func fastScanAccumulateHACCScalar(codes, lut []byte, dim int) []int32 {
	out := make([]int32, BatchSize)
	for lane := 0; lane < BatchSize; lane++ {
		var sum int32
		for table := 0; table < dim/4; table++ {
			base := table/2*64 + table%2*16
			code := batchCode(codes, dim, lane, table)
			value := uint16(lut[base+code]) | uint16(lut[base+32+code])<<8
			sum += int32(value)
		}
		out[lane] = sum
	}
	return out
}

func warmupIPScalar(data, query []uint64, delta, vl float32, bQuery int) float32 {
	var ip, pop uint64
	queryBase := 0
	for dataBase := 0; dataBase < len(data); dataBase += 8 {
		chunks := min(8, len(data)-dataBase)
		for chunk := 0; chunk < chunks; chunk++ {
			word := data[dataBase+chunk]
			pop += uint64(bits.OnesCount64(word))
			for bit := 0; bit < bQuery; bit++ {
				ip += uint64(bits.OnesCount64(word&query[queryBase+bit*chunks+chunk])) << uint(bit)
			}
		}
		queryBase += chunks * bQuery
	}
	return delta*float32(ip) + vl*float32(pop)
}

func packExCodeScalar(values []uint8, exBits int) []byte {
	out := make([]byte, len(values)/8*exBits)
	for base := 0; base < len(values); base += 64 {
		packExBlock(out[base*exBits/8:], values[base:base+64], exBits)
	}
	return out
}

func excodeIPScalar(query []float32, packed []byte, exBits int) float32 {
	values := make([]uint8, len(query))
	for base := 0; base < len(query); base += 64 {
		unpackExBlock(values[base:base+64], packed[base*exBits/8:], exBits)
	}
	var sum float32
	for i, value := range values {
		sum += query[i] * float32(value)
	}
	return sum
}

var (
	flipSignKernel                = flipSignScalar
	kacsWalkKernel                = kacsWalkScalar
	quantizeUint8Kernel           = quantizeUint8Scalar
	quantizeUint16Kernel          = quantizeUint16Scalar
	transposeBinKernel            = func(q []uint16, out []uint64, b int) { copy(out, transposeBinScalar(q, b)) }
	transposeBin512Kernel         = func(q []uint8, out []uint64, b int) { copy(out, transposeBin512Scalar(q, b)) }
	maskIPKernel                  = maskIPScalar
	fastScanAccumulateKernel      = func(c, l []byte, out []uint16, d int) { copy(out, fastScanAccumulateScalar(c, l, d)) }
	fastScanTransferLUTHACCKernel = func(l []uint16, out []byte, d int) { copy(out, fastScanTransferLUTHACCScalar(l, d)) }
	fastScanAccumulateHACCKernel  = func(c, l []byte, out []int32, d int) { copy(out, fastScanAccumulateHACCScalar(c, l, d)) }
	warmupIPKernel                = warmupIPScalar
	packExCodeKernel              = packExCodeScalar
	excodeIPKernel                = excodeIPScalar
)

func packExCodeSIMD(values []uint8, exBits int) ([]byte, error) {
	if exBits < 1 || exBits > 8 || len(values) == 0 || len(values)%64 != 0 {
		return nil, ErrInvalidArgument
	}
	return packExCodeKernel(values, exBits), nil
}

func excodeIPSIMD(query []float32, packed []byte, exBits int) (float32, error) {
	if !validDimension(len(query)) || exBits < 1 || exBits > 8 || len(packed) != len(query)/8*exBits {
		return 0, ErrInvalidLayout
	}
	return excodeIPKernel(query, packed, exBits), nil
}
