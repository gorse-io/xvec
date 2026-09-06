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

// Package rabitq implements the RaBitQ APIs used by zvec.
package rabitq

import (
	"encoding/binary"
	"math"
)

// BatchSize is the number of vectors in a FastScan block.
const BatchSize = 32

// BatchDataBytes returns the packed byte size of a 32-vector one-bit block.
func BatchDataBytes(paddedDim int) int { return paddedDim*BatchSize/8 + 3*BatchSize*4 }

// BinDataBytes returns the packed byte size of one compact one-bit vector.
func BinDataBytes(paddedDim int) int { return paddedDim/8 + 3*4 }

// ExDataBytes returns the packed byte size of one extra-code vector.
func ExDataBytes(paddedDim, exBits int) int {
	if exBits == 0 {
		return 0
	}
	return paddedDim*exBits/8 + 2*4
}

// BatchDataMap provides endian-stable access to the persisted batch layout.
type BatchDataMap struct {
	data      []byte
	paddedDim int
}

func NewBatchDataMap(data []byte, paddedDim int) BatchDataMap {
	if paddedDim <= 0 || paddedDim%64 != 0 || len(data) < BatchDataBytes(paddedDim) {
		panic("rabitq: invalid batch data")
	}
	return BatchDataMap{data: data[:BatchDataBytes(paddedDim)], paddedDim: paddedDim}
}
func (m BatchDataMap) BinCode() []byte { return m.data[:m.paddedDim*BatchSize/8] }
func (m BatchDataMap) factor(n, index int) float32 {
	off := m.paddedDim*BatchSize/8 + (n*BatchSize+index)*4
	return math.Float32frombits(binary.LittleEndian.Uint32(m.data[off:]))
}
func (m BatchDataMap) setFactor(n, index int, value float32) {
	off := m.paddedDim*BatchSize/8 + (n*BatchSize+index)*4
	binary.LittleEndian.PutUint32(m.data[off:], math.Float32bits(value))
}
func (m BatchDataMap) FAdd(index int) float32     { return m.factor(0, index) }
func (m BatchDataMap) FRescale(index int) float32 { return m.factor(1, index) }
func (m BatchDataMap) FError(index int) float32   { return m.factor(2, index) }
func (m BatchDataMap) SetFAdd(index int, value float32) {
	m.setFactor(0, index, value)
}
func (m BatchDataMap) SetFRescale(index int, value float32) {
	m.setFactor(1, index, value)
}
func (m BatchDataMap) SetFError(index int, value float32) {
	m.setFactor(2, index, value)
}

// BinDataMap provides views and factors for one compact one-bit vector.
type BinDataMap struct {
	data      []byte
	paddedDim int
}

func NewBinDataMap(data []byte, paddedDim int) BinDataMap {
	if paddedDim <= 0 || paddedDim%64 != 0 || len(data) < BinDataBytes(paddedDim) {
		panic("rabitq: invalid binary data")
	}
	return BinDataMap{data: data[:BinDataBytes(paddedDim)], paddedDim: paddedDim}
}
func (m BinDataMap) BinCode() []byte { return m.data[:m.paddedDim/8] }
func (m BinDataMap) decodedBinCode() []uint64 {
	code := make([]uint64, m.paddedDim/64)
	for i := range code {
		code[i] = binary.LittleEndian.Uint64(m.data[i*8:])
	}
	return code
}
func (m BinDataMap) factor(n int) float32 {
	return math.Float32frombits(binary.LittleEndian.Uint32(m.data[m.paddedDim/8+n*4:]))
}
func (m BinDataMap) setFactor(n int, v float32) {
	binary.LittleEndian.PutUint32(m.data[m.paddedDim/8+n*4:], math.Float32bits(v))
}
func (m BinDataMap) FAdd() float32         { return m.factor(0) }
func (m BinDataMap) FRescale() float32     { return m.factor(1) }
func (m BinDataMap) FError() float32       { return m.factor(2) }
func (m BinDataMap) SetFAdd(v float32)     { m.setFactor(0, v) }
func (m BinDataMap) SetFRescale(v float32) { m.setFactor(1, v) }
func (m BinDataMap) SetFError(v float32)   { m.setFactor(2, v) }

// ExDataMap provides views and factors for one compact extra-code vector.
type ExDataMap struct {
	data              []byte
	paddedDim, exBits int
}

func NewExDataMap(data []byte, paddedDim, exBits int) ExDataMap {
	if paddedDim <= 0 || paddedDim%64 != 0 || exBits < 1 || exBits > 8 || len(data) < ExDataBytes(paddedDim, exBits) {
		panic("rabitq: invalid extra data")
	}
	return ExDataMap{data: data[:ExDataBytes(paddedDim, exBits)], paddedDim: paddedDim, exBits: exBits}
}
func (m ExDataMap) ExCode() []byte { return m.data[:m.paddedDim*m.exBits/8] }
func (m ExDataMap) factor(n int) float32 {
	return math.Float32frombits(binary.LittleEndian.Uint32(m.data[m.paddedDim*m.exBits/8+n*4:]))
}
func (m ExDataMap) setFactor(n int, v float32) {
	binary.LittleEndian.PutUint32(m.data[m.paddedDim*m.exBits/8+n*4:], math.Float32bits(v))
}
func (m ExDataMap) FAddEx() float32         { return m.factor(0) }
func (m ExDataMap) FRescaleEx() float32     { return m.factor(1) }
func (m ExDataMap) SetFAddEx(v float32)     { m.setFactor(0, v) }
func (m ExDataMap) SetFRescaleEx(v float32) { m.setFactor(1, v) }
