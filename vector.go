// Copyright 2026-present the zvec-go project
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

package zvec

import (
	"math"
	"slices"
	"sort"
)

// Binary is an arbitrary byte string. It is distinct from a UTF-8 String
// field and from binary vectors.
type Binary []byte

// Explicit array types keep document values unambiguous when their element
// type is also used by a vector field.
type (
	BinaryArray  []Binary
	StringArray  []string
	BoolArray    []bool
	Int32Array   []int32
	Int64Array   []int64
	Uint32Array  []uint32
	Uint64Array  []uint64
	Float32Array []float32
	Float64Array []float64
)

// Float16 stores the IEEE 754 binary16 bit representation of a number.
type Float16 uint16

// Float16FromFloat32 converts value to binary16 using round-to-nearest-even.
func Float16FromFloat32(value float32) Float16 {
	bits := math.Float32bits(value)
	sign := uint16(bits>>16) & 0x8000
	exponent32 := int((bits >> 23) & 0xff)
	mantissa := bits & 0x7fffff

	if exponent32 == 0xff {
		if mantissa == 0 {
			return Float16(sign | 0x7c00)
		}
		payload := uint16(mantissa >> 13)
		if payload == 0 {
			payload = 1
		}
		return Float16(sign | 0x7c00 | payload)
	}

	exponent16 := exponent32 - 127 + 15
	if exponent16 >= 31 {
		return Float16(sign | 0x7c00)
	}
	if exponent16 <= 0 {
		if exponent16 < -10 {
			return Float16(sign)
		}
		mantissa |= 0x800000
		shift := uint32(14 - exponent16)
		rounded := mantissa >> shift
		remainder := mantissa & ((uint32(1) << shift) - 1)
		halfway := uint32(1) << (shift - 1)
		if remainder > halfway || (remainder == halfway && rounded&1 != 0) {
			rounded++
		}
		return Float16(sign | uint16(rounded))
	}

	rounded := mantissa >> 13
	remainder := mantissa & 0x1fff
	if remainder > 0x1000 || (remainder == 0x1000 && rounded&1 != 0) {
		rounded++
		if rounded == 0x400 {
			rounded = 0
			exponent16++
			if exponent16 >= 31 {
				return Float16(sign | 0x7c00)
			}
		}
	}
	return Float16(sign | uint16(exponent16<<10) | uint16(rounded))
}

// Float32 converts f to the exactly represented float32 value.
func (f Float16) Float32() float32 {
	bits := uint16(f)
	sign := uint32(bits&0x8000) << 16
	exponent := uint32(bits>>10) & 0x1f
	mantissa := uint32(bits & 0x03ff)

	switch exponent {
	case 0:
		if mantissa == 0 {
			return math.Float32frombits(sign)
		}
		exponent32 := int32(127 - 14)
		for mantissa&0x400 == 0 {
			mantissa <<= 1
			exponent32--
		}
		mantissa &= 0x3ff
		return math.Float32frombits(sign | uint32(exponent32)<<23 | mantissa<<13)
	case 0x1f:
		return math.Float32frombits(sign | 0x7f800000 | mantissa<<13)
	default:
		exponent32 := exponent + (127 - 15)
		return math.Float32frombits(sign | exponent32<<23 | mantissa<<13)
	}
}

// DenseVector is implemented by every explicit dense-vector value.
type DenseVector interface {
	DataType() DataType
	Dimension() int
	denseVector()
}

type (
	VectorBinary32 []uint32
	VectorBinary64 []uint64
	VectorFP16     []Float16
	VectorFP32     []float32
	VectorFP64     []float64
	// VectorInt4 stores one signed value per element, in the range [-8, 7].
	// Packing is a disk-codec concern and is not exposed in the Go API.
	VectorInt4  []int8
	VectorInt8  []int8
	VectorInt16 []int16
)

func (VectorBinary32) DataType() DataType { return DataTypeVectorBinary32 }
func (VectorBinary64) DataType() DataType { return DataTypeVectorBinary64 }
func (VectorFP16) DataType() DataType     { return DataTypeVectorFP16 }
func (VectorFP32) DataType() DataType     { return DataTypeVectorFP32 }
func (VectorFP64) DataType() DataType     { return DataTypeVectorFP64 }
func (VectorInt4) DataType() DataType     { return DataTypeVectorInt4 }
func (VectorInt8) DataType() DataType     { return DataTypeVectorInt8 }
func (VectorInt16) DataType() DataType    { return DataTypeVectorInt16 }

func (v VectorBinary32) Dimension() int { return len(v) }
func (v VectorBinary64) Dimension() int { return len(v) }
func (v VectorFP16) Dimension() int     { return len(v) }
func (v VectorFP32) Dimension() int     { return len(v) }
func (v VectorFP64) Dimension() int     { return len(v) }
func (v VectorInt4) Dimension() int     { return len(v) }
func (v VectorInt8) Dimension() int     { return len(v) }
func (v VectorInt16) Dimension() int    { return len(v) }

func (VectorBinary32) denseVector() {}
func (VectorBinary64) denseVector() {}
func (VectorFP16) denseVector()     {}
func (VectorFP32) denseVector()     {}
func (VectorFP64) denseVector()     {}
func (VectorInt4) denseVector()     {}
func (VectorInt8) denseVector()     {}
func (VectorInt16) denseVector()    {}

// Validate checks that every explicit INT4 element is representable.
func (v VectorInt4) Validate() error {
	for index, value := range v {
		if value < -8 || value > 7 {
			return invalidArgument("validate vector", "INT4 value at index %d is outside [-8, 7]", index)
		}
	}
	return nil
}

// SparseVector is implemented by both supported sparse-vector value types.
type SparseVector interface {
	DataType() DataType
	Len() int
	sparseVector()
}

// SparseVectorFP16 stores matching coordinate and binary16 value slices.
type SparseVectorFP16 struct {
	Indices []uint32
	Values  []Float16
}

// SparseVectorFP32 stores matching coordinate and float32 value slices.
type SparseVectorFP32 struct {
	Indices []uint32
	Values  []float32
}

func (SparseVectorFP16) DataType() DataType { return DataTypeSparseVectorFP16 }
func (SparseVectorFP32) DataType() DataType { return DataTypeSparseVectorFP32 }
func (v SparseVectorFP16) Len() int         { return len(v.Indices) }
func (v SparseVectorFP32) Len() int         { return len(v.Indices) }
func (SparseVectorFP16) sparseVector()      {}
func (SparseVectorFP32) sparseVector()      {}

// Validate checks lengths, the coordinate-count limit, and uniqueness. Input
// coordinates may be unsorted; Canonical returns a sorted copy.
func (v SparseVectorFP16) Validate() error {
	return validateSparseIndices(len(v.Indices), len(v.Values), v.Indices)
}

// Validate checks lengths, the coordinate-count limit, and uniqueness. Input
// coordinates may be unsorted; Canonical returns a sorted copy.
func (v SparseVectorFP32) Validate() error {
	return validateSparseIndices(len(v.Indices), len(v.Values), v.Indices)
}

// Canonical returns an independent copy sorted by coordinate.
func (v SparseVectorFP16) Canonical() (SparseVectorFP16, error) {
	if err := v.Validate(); err != nil {
		return SparseVectorFP16{}, err
	}
	order := sparseOrder(v.Indices)
	result := SparseVectorFP16{
		Indices: make([]uint32, len(order)),
		Values:  make([]Float16, len(order)),
	}
	for output, input := range order {
		result.Indices[output] = v.Indices[input]
		result.Values[output] = v.Values[input]
	}
	return result, nil
}

// Canonical returns an independent copy sorted by coordinate.
func (v SparseVectorFP32) Canonical() (SparseVectorFP32, error) {
	if err := v.Validate(); err != nil {
		return SparseVectorFP32{}, err
	}
	order := sparseOrder(v.Indices)
	result := SparseVectorFP32{
		Indices: make([]uint32, len(order)),
		Values:  make([]float32, len(order)),
	}
	for output, input := range order {
		result.Indices[output] = v.Indices[input]
		result.Values[output] = v.Values[input]
	}
	return result, nil
}

func validateSparseIndices(indexCount, valueCount int, indices []uint32) error {
	if indexCount != valueCount {
		return invalidArgument("validate sparse vector", "indices and values have different lengths")
	}
	if indexCount > MaxSparseDimensions {
		return invalidArgument("validate sparse vector", "coordinate count %d exceeds %d", indexCount, MaxSparseDimensions)
	}
	if len(indices) < 2 {
		return nil
	}
	copyOfIndices := slices.Clone(indices)
	slices.Sort(copyOfIndices)
	for index := 1; index < len(copyOfIndices); index++ {
		if copyOfIndices[index] == copyOfIndices[index-1] {
			return invalidArgument("validate sparse vector", "duplicate coordinate %d", copyOfIndices[index])
		}
	}
	return nil
}

func sparseOrder(indices []uint32) []int {
	order := make([]int, len(indices))
	for index := range order {
		order[index] = index
	}
	sort.Slice(order, func(i, j int) bool { return indices[order[i]] < indices[order[j]] })
	return order
}
