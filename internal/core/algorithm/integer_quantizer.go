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

package core

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/gorse-io/xvec/internal/ailego/math"
	"github.com/gorse-io/xvec/internal/ailego/parallel"
	"github.com/gorse-io/xvec/internal/ailego/utility"
)

// Quantization identifies one scalar vector encoding. The values are internal
// and deliberately independent of the public and on-disk enum assignments.
type Quantization uint8

const (
	QuantizationFP16 Quantization = iota + 1
	QuantizationInt8
	QuantizationInt4
)

var (
	ErrInvalidQuantization    = errors.New("core: invalid scalar quantization")
	ErrInvalidQuantizedVector = errors.New("core: invalid quantized vector")
	ErrQuantizationOverflow   = errors.New("core: value overflows quantized representation")
	ErrOddInt4Dimension       = errors.New("core: INT4 quantization requires an even dimension")
)

// QuantizedVector is an immutable scalar-quantized dense vector. Integer
// encodings reconstruct element i as inverseScale*code[i]+offset. The integer
// moments allow distance kernels to avoid materializing decoded vectors.
type QuantizedVector struct {
	kind         Quantization
	dimension    int
	codes        []byte
	inverseScale float32
	offset       float32
	codeSum      float64
	codeSquare   float64
}

// Kind returns the vector encoding.
func (v QuantizedVector) Kind() Quantization { return v.kind }

// Dimension returns the number of logical vector elements.
func (v QuantizedVector) Dimension() int { return v.dimension }

// Codes returns an independent copy of the packed encoded data.
func (v QuantizedVector) Codes() []byte { return slices.Clone(v.codes) }

// InverseScale returns the integer reconstruction multiplier. It is zero for
// FP16 and for constant integer-quantized vectors.
func (v QuantizedVector) InverseScale() float32 { return v.inverseScale }

// Offset returns the integer reconstruction offset. It is zero for FP16.
func (v QuantizedVector) Offset() float32 { return v.offset }

// QuantizeVector scalar-quantizes one finite, non-empty FP32 vector. FP16 uses
// IEEE binary16. INT8 and INT4 use baseline-compatible per-vector affine
// ranges with signed codes; INT4 requires an even logical dimension.
func QuantizeVector(kind Quantization, vector []float32) (QuantizedVector, error) {
	if !kind.valid() {
		return QuantizedVector{}, ErrInvalidQuantization
	}
	if len(vector) == 0 {
		return QuantizedVector{}, mathutil.ErrEmptyVector
	}
	for _, value := range vector {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return QuantizedVector{}, mathutil.ErrNonFiniteVector
		}
	}
	if kind == QuantizationInt4 && len(vector)%2 != 0 {
		return QuantizedVector{}, fmt.Errorf("%w: got %d", ErrOddInt4Dimension, len(vector))
	}

	switch kind {
	case QuantizationFP16:
		return quantizeFP16(vector)
	case QuantizationInt8:
		return quantizeInteger(kind, vector, -127, 127)
	case QuantizationInt4:
		return quantizeInteger(kind, vector, -8, 7)
	default:
		panic("unreachable scalar quantization")
	}
}

// QuantizeBatch converts vectors concurrently while preserving input order.
// No output aliases an input or another output.
func QuantizeBatch(ctx context.Context, kind Quantization, vectors [][]float32, workers int) ([]QuantizedVector, error) {
	if ctx == nil {
		return nil, errors.New("core: nil quantization context")
	}
	if !kind.valid() {
		return nil, ErrInvalidQuantization
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	result := make([]QuantizedVector, len(vectors))
	err := parallel.ParallelFor(ctx, len(vectors), workers, func(ctx context.Context, index int) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		quantized, err := QuantizeVector(kind, vectors[index])
		if err != nil {
			return fmt.Errorf("core: quantize vector %d: %w", index, err)
		}
		result[index] = quantized
		return nil
	})
	if err != nil {
		return nil, err
	}
	return result, nil
}

// Decode returns the independently allocated FP32 vector represented by v.
func (v QuantizedVector) Decode() ([]float32, error) {
	if err := v.validate(); err != nil {
		return nil, err
	}
	decoded := make([]float32, v.dimension)
	switch v.kind {
	case QuantizationFP16:
		for index := range decoded {
			bits := binary.LittleEndian.Uint16(v.codes[index*2:])
			decoded[index] = utility.Float16BitsToFloat32(bits)
		}
	case QuantizationInt8:
		for index, code := range v.codes {
			decoded[index] = v.inverseScale*float32(int8(code)) + v.offset
		}
	case QuantizationInt4:
		for index := range decoded {
			packed := v.codes[index/2]
			code := int8(packed & 0x0f)
			if index%2 != 0 {
				code = int8(packed >> 4)
			}
			if code&0x08 != 0 {
				code -= 16
			}
			decoded[index] = v.inverseScale*float32(code) + v.offset
		}
	}
	return decoded, nil
}

// QuantizedDistance calculates a metric directly from scalar codes. Both
// vectors must use the same encoding and logical dimension.
func QuantizedDistance(metric Metric, left, right QuantizedVector) (float32, error) {
	if !metric.Valid() {
		return 0, errors.New("core: invalid metric")
	}
	if err := left.validate(); err != nil {
		return 0, fmt.Errorf("core: validate left quantized vector: %w", err)
	}
	if err := right.validate(); err != nil {
		return 0, fmt.Errorf("core: validate right quantized vector: %w", err)
	}
	if left.kind != right.kind {
		return 0, fmt.Errorf("%w: encoding mismatch", ErrInvalidQuantizedVector)
	}
	if left.dimension != right.dimension {
		return 0, mathutil.ErrDimensionMismatch
	}
	if left.kind == QuantizationFP16 {
		leftDecoded, _ := left.Decode()
		rightDecoded, _ := right.Decode()
		return metric.Compute(leftDecoded, rightDecoded)
	}

	dotCodes := integerCodeDot(left, right)
	leftScale, rightScale := float64(left.inverseScale), float64(right.inverseScale)
	leftOffset, rightOffset := float64(left.offset), float64(right.offset)
	dimension := float64(left.dimension)
	inner := leftScale*rightScale*dotCodes +
		leftOffset*rightScale*right.codeSum +
		rightOffset*leftScale*left.codeSum +
		dimension*leftOffset*rightOffset
	leftNorm := leftScale*leftScale*left.codeSquare +
		2*leftScale*leftOffset*left.codeSum + dimension*leftOffset*leftOffset
	rightNorm := rightScale*rightScale*right.codeSquare +
		2*rightScale*rightOffset*right.codeSum + dimension*rightOffset*rightOffset

	var score float64
	switch metric {
	case MetricL2:
		score = leftNorm + rightNorm - 2*inner
		if score < 0 && score > -1e-6*max(1, leftNorm+rightNorm) {
			score = 0
		}
	case MetricIP:
		score = inner
	case MetricCosine:
		switch {
		case leftNorm == 0 && rightNorm == 0:
			score = 0
		case leftNorm == 0 || rightNorm == 0:
			score = 1
		default:
			cosine := inner / math.Sqrt(leftNorm*rightNorm)
			cosine = min(1, max(-1, cosine))
			score = 1 - cosine
		}
	case MetricMIPSL2:
		denominator := max(leftNorm, rightNorm)
		if denominator == 0 {
			score = 0
		} else {
			score = 2 - 2*inner/denominator
		}
	}
	return finiteQuantizedScore(score)
}

// QuantizedDistanceToFloat converts query with candidate's encoding and then
// scores the two quantized vectors. This matches streaming-query conversion in
// the baseline scalar-quantized indexes.
func QuantizedDistanceToFloat(metric Metric, candidate QuantizedVector, query []float32) (float32, error) {
	if err := candidate.validate(); err != nil {
		return 0, fmt.Errorf("core: validate candidate quantized vector: %w", err)
	}
	if len(query) != candidate.dimension {
		return 0, mathutil.ErrDimensionMismatch
	}
	quantizedQuery, err := QuantizeVector(candidate.kind, query)
	if err != nil {
		return 0, fmt.Errorf("core: quantize query: %w", err)
	}
	return QuantizedDistance(metric, candidate, quantizedQuery)
}

func quantizeFP16(vector []float32) (QuantizedVector, error) {
	codes := make([]byte, len(vector)*2)
	for index, value := range vector {
		bits := utility.Float32ToFloat16Bits(value)
		decoded := utility.Float16BitsToFloat32(bits)
		if math.IsInf(float64(decoded), 0) {
			return QuantizedVector{}, fmt.Errorf("%w at element %d", ErrQuantizationOverflow, index)
		}
		binary.LittleEndian.PutUint16(codes[index*2:], bits)
	}
	return QuantizedVector{kind: QuantizationFP16, dimension: len(vector), codes: codes}, nil
}

func quantizeInteger(kind Quantization, vector []float32, lower, upper int) (QuantizedVector, error) {
	minimum, maximum := vector[0], vector[0]
	for _, value := range vector[1:] {
		minimum = min(minimum, value)
		maximum = max(maximum, value)
	}
	if minimum == maximum {
		codesLength := len(vector)
		if kind == QuantizationInt4 {
			codesLength /= 2
		}
		return QuantizedVector{
			kind:      kind,
			dimension: len(vector),
			codes:     make([]byte, codesLength),
			offset:    minimum,
		}, nil
	}

	const float32Epsilon = 1.1920928955078125e-7
	span := maximum - minimum
	scale := float32(upper-lower) / max(span, float32Epsilon)
	bias := -minimum*scale + float32(lower)
	inverseScale := 1 / scale
	offset := -bias / scale
	if !finiteFloat32(inverseScale) || !finiteFloat32(offset) {
		return QuantizedVector{}, ErrQuantizationOverflow
	}

	codesLength := len(vector)
	if kind == QuantizationInt4 {
		codesLength /= 2
	}
	result := QuantizedVector{
		kind:         kind,
		dimension:    len(vector),
		codes:        make([]byte, codesLength),
		inverseScale: inverseScale,
		offset:       offset,
	}
	for index, value := range vector {
		code := int(math.Round(float64(value*scale + bias)))
		code = min(upper, max(lower, code))
		result.codeSum += float64(code)
		result.codeSquare += float64(code * code)
		if kind == QuantizationInt8 {
			result.codes[index] = byte(int8(code))
		} else if index%2 == 0 {
			result.codes[index/2] = byte(code) & 0x0f
		} else {
			result.codes[index/2] |= (byte(code) & 0x0f) << 4
		}
	}
	return result, nil
}

func (v QuantizedVector) validate() error {
	if !v.kind.valid() || v.dimension <= 0 {
		return ErrInvalidQuantizedVector
	}
	wantCodes := v.dimension
	switch v.kind {
	case QuantizationFP16:
		wantCodes *= 2
	case QuantizationInt4:
		if v.dimension%2 != 0 {
			return fmt.Errorf("%w: %w", ErrInvalidQuantizedVector, ErrOddInt4Dimension)
		}
		wantCodes /= 2
	}
	if len(v.codes) != wantCodes {
		return fmt.Errorf("%w: got %d code bytes, want %d", ErrInvalidQuantizedVector, len(v.codes), wantCodes)
	}
	if !finiteFloat32(v.inverseScale) || !finiteFloat32(v.offset) {
		return fmt.Errorf("%w: non-finite reconstruction parameters", ErrInvalidQuantizedVector)
	}
	if v.kind == QuantizationFP16 {
		for index := 0; index < v.dimension; index++ {
			decoded := utility.Float16BitsToFloat32(binary.LittleEndian.Uint16(v.codes[index*2:]))
			if !finiteFloat32(decoded) {
				return fmt.Errorf("%w: non-finite FP16 code at element %d", ErrInvalidQuantizedVector, index)
			}
		}
	}
	return nil
}

func (q Quantization) valid() bool {
	return q >= QuantizationFP16 && q <= QuantizationInt4
}

func integerCodeDot(left, right QuantizedVector) float64 {
	var dot float64
	for index := 0; index < left.dimension; index++ {
		leftCode := left.integerCode(index)
		rightCode := right.integerCode(index)
		dot += float64(leftCode * rightCode)
	}
	return dot
}

func (v QuantizedVector) integerCode(index int) int {
	if v.kind == QuantizationInt8 {
		return int(int8(v.codes[index]))
	}
	packed := v.codes[index/2]
	code := int8(packed & 0x0f)
	if index%2 != 0 {
		code = int8(packed >> 4)
	}
	if code&0x08 != 0 {
		code -= 16
	}
	return int(code)
}

func finiteFloat32(value float32) bool {
	return !math.IsNaN(float64(value)) && !math.IsInf(float64(value), 0)
}

func finiteQuantizedScore(value float64) (float32, error) {
	result := float32(value)
	if math.IsNaN(value) || math.IsInf(value, 0) || !finiteFloat32(result) {
		return 0, mathutil.ErrNonFiniteVector
	}
	return result, nil
}
