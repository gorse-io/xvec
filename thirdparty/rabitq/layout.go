// Copyright 2026-present the xvec project
// Licensed under the Apache License, Version 2.0.

package rabitq

import (
	"encoding/binary"
	"fmt"
	"math"
)

// BatchSize is the fixed FastScan-compatible number of lanes.
const BatchSize = 32

const maxInt = int(^uint(0) >> 1)

func validDimension(dim int) bool { return dim >= 64 && dim%64 == 0 }

// BinDataBytes returns the single-vector one-bit layout size.
func BinDataBytes(paddedDim int) int {
	if !validDimension(paddedDim) {
		return 0
	}
	return paddedDim/8 + 3*4
}

// BatchDataBytes returns the fixed 32-vector one-bit layout size.
func BatchDataBytes(paddedDim int) int {
	if !validDimension(paddedDim) || paddedDim > (maxInt-3*BatchSize*4)/(BatchSize/8) {
		return 0
	}
	return paddedDim*(BatchSize/8) + 3*BatchSize*4
}

// ExDataBytes returns the per-vector extra-bit layout size.
func ExDataBytes(paddedDim, exBits int) int {
	if !validDimension(paddedDim) || exBits < 1 || exBits > 8 || paddedDim/8 > (maxInt-2*4)/exBits {
		return 0
	}
	return paddedDim/8*exBits + 2*4
}

func getFloat(data []byte, off int) float32 {
	return math.Float32frombits(binary.LittleEndian.Uint32(data[off:]))
}
func putFloat(data []byte, off int, value float32) {
	binary.LittleEndian.PutUint32(data[off:], math.Float32bits(value))
}

// BinData is a checked view of a compact one-bit code and its factors.
type BinData struct {
	data []byte
	dim  int
}

func NewBinData(data []byte, paddedDim int) (BinData, error) {
	if want := BinDataBytes(paddedDim); want == 0 || len(data) != want {
		return BinData{}, ErrInvalidLayout
	}
	return BinData{data: data, dim: paddedDim}, nil
}
func (m BinData) Code() []byte      { return m.data[:m.dim/8] }
func (m BinData) FAdd() float32     { return getFloat(m.data, m.dim/8) }
func (m BinData) FRescale() float32 { return getFloat(m.data, m.dim/8+4) }
func (m BinData) FError() float32   { return getFloat(m.data, m.dim/8+8) }
func (m BinData) setFactors(a, r, e float32) {
	off := m.dim / 8
	putFloat(m.data, off, a)
	putFloat(m.data, off+4, r)
	putFloat(m.data, off+8, e)
}

// ExData is a checked view of a packed extra-bit code and its factors.
type ExData struct {
	data      []byte
	dim, bits int
}

func NewExData(data []byte, paddedDim, exBits int) (ExData, error) {
	if want := ExDataBytes(paddedDim, exBits); want == 0 || len(data) != want {
		return ExData{}, ErrInvalidLayout
	}
	return ExData{data: data, dim: paddedDim, bits: exBits}, nil
}
func (m ExData) Code() []byte      { return m.data[:m.dim/8*m.bits] }
func (m ExData) FAdd() float32     { return getFloat(m.data, len(m.Code())) }
func (m ExData) FRescale() float32 { return getFloat(m.data, len(m.Code())+4) }
func (m ExData) setFactors(a, r float32) {
	off := len(m.Code())
	putFloat(m.data, off, a)
	putFloat(m.data, off+4, r)
}
func (m ExData) Values() ([]uint8, error) { return UnpackExCode(m.Code(), m.dim, m.bits) }

// BatchData is a checked view of 32 interleaved binary codes and factor arrays.
type BatchData struct {
	data []byte
	dim  int
}

func NewBatchData(data []byte, paddedDim int) (BatchData, error) {
	if want := BatchDataBytes(paddedDim); want == 0 || len(data) != want {
		return BatchData{}, ErrInvalidLayout
	}
	return BatchData{data: data, dim: paddedDim}, nil
}
func (m BatchData) Code() []byte { return m.data[:m.dim*(BatchSize/8)] }
func (m BatchData) factor(kind, lane int) float32 {
	return getFloat(m.data, len(m.Code())+(kind*BatchSize+lane)*4)
}
func (m BatchData) FAdd(lane int) float32     { return m.factor(0, lane) }
func (m BatchData) FRescale(lane int) float32 { return m.factor(1, lane) }
func (m BatchData) FError(lane int) float32   { return m.factor(2, lane) }
func (m BatchData) setFactor(kind, lane int, v float32) {
	putFloat(m.data, len(m.Code())+(kind*BatchSize+lane)*4, v)
}

// PackExCode packs codes in the RaBitQ-Library layout for exBits 0..8.
func PackExCode(values []uint8, exBits int) ([]byte, error) {
	if exBits < 0 || exBits > 8 || len(values)%64 != 0 {
		return nil, ErrInvalidArgument
	}
	if exBits == 0 {
		for _, v := range values {
			if v != 0 {
				return nil, ErrInvalidArgument
			}
		}
		return []byte{}, nil
	}
	if len(values) == 0 {
		return []byte{}, nil
	}
	maxv := uint8((1 << exBits) - 1)
	for _, v := range values {
		if v > maxv {
			return nil, ErrInvalidArgument
		}
	}
	out := packExCodeKernel(values, exBits)
	return out, nil
}

func packExBlock(out, in []byte, bits int) {
	switch bits {
	case 1:
		for i, v := range in {
			out[i/8] |= (v & 1) << uint(i%8)
		}
	case 2:
		for i := 0; i < 16; i++ {
			out[i] = in[i]&3 | (in[i+16]&3)<<2 | (in[i+32]&3)<<4 | (in[i+48]&3)<<6
		}
	case 3, 5, 7:
		low := bits - 1
		packExBlock(out, in, low)
		off := 64 * low / 8
		for i, v := range in {
			out[off+i/8] |= ((v >> uint(low)) & 1) << uint(i%8)
		}
	case 4:
		for group := 0; group < 4; group++ {
			for i := 0; i < 8; i++ {
				out[group*8+i] = in[group*16+i]&15 | (in[group*16+i+8]&15)<<4
			}
		}
	case 6:
		for part := 0; part < 3; part++ {
			for i := 0; i < 16; i++ {
				out[part*16+i] = (in[part*16+i] & 0x3f) | ((in[48+i]>>uint(part*2))&3)<<6
			}
		}
	case 8:
		copy(out, in)
	}
}

// UnpackExCode reverses PackExCode.
func UnpackExCode(packed []byte, dim, exBits int) ([]uint8, error) {
	if !validDimension(dim) || exBits < 0 || exBits > 8 || len(packed) != dim/8*exBits {
		return nil, ErrInvalidLayout
	}
	out := make([]byte, dim)
	if exBits == 0 {
		return out, nil
	}
	for base := 0; base < dim; base += 64 {
		unpackExBlock(out[base:base+64], packed[base*exBits/8:], exBits)
	}
	return out, nil
}
func unpackExBlock(out, in []byte, bits int) {
	switch bits {
	case 1:
		for i := range out {
			out[i] = (in[i/8] >> uint(i%8)) & 1
		}
	case 2:
		for i := 0; i < 16; i++ {
			v := in[i]
			out[i] = v & 3
			out[i+16] = (v >> 2) & 3
			out[i+32] = (v >> 4) & 3
			out[i+48] = (v >> 6) & 3
		}
	case 3, 5, 7:
		low := bits - 1
		unpackExBlock(out, in, low)
		off := 64 * low / 8
		for i := range out {
			out[i] |= ((in[off+i/8] >> uint(i%8)) & 1) << uint(low)
		}
	case 4:
		for g := 0; g < 4; g++ {
			for i := 0; i < 8; i++ {
				v := in[g*8+i]
				out[g*16+i] = v & 15
				out[g*16+i+8] = v >> 4
			}
		}
	case 6:
		for p := 0; p < 3; p++ {
			for i := 0; i < 16; i++ {
				v := in[p*16+i]
				out[p*16+i] = v & 0x3f
				out[48+i] |= ((v >> 6) & 3) << uint(p*2)
			}
		}
	case 8:
		copy(out, in)
	default:
		panic(fmt.Sprintf("unreachable ex bits %d", bits))
	}
}
