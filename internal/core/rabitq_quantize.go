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

package core

import (
	"container/heap"
	"context"
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/gorse-io/xvec/internal/ailego"
)

const raBitQErrorConfidence = 1.9

// RaBitQCode is one immutable split code. BinaryCode stores one sign bit per
// padded coordinate; ExtraCode stores the remaining bits in a portable
// least-significant-bit-first stream.
type RaBitQCode struct {
	modelFingerprint uint64
	cluster          int
	paddedDimension  int
	totalBits        int
	binaryCode       []byte
	extraCode        []byte
	coarseAdd        float64
	coarseRescale    float64
	coarseError      float64
	fullAdd          float64
	fullRescale      float64
}

func (c RaBitQCode) Cluster() int { return c.cluster }

func (c RaBitQCode) PaddedDimension() int { return c.paddedDimension }

func (c RaBitQCode) TotalBits() int { return c.totalBits }

func (c RaBitQCode) BinaryCode() []byte { return slices.Clone(c.binaryCode) }

func (c RaBitQCode) ExtraCode() []byte { return slices.Clone(c.extraCode) }

// QuantizedValues expands the portable code into one unsigned total-bit value
// per padded coordinate. It is intended for diagnostics and fixtures.
func (c RaBitQCode) QuantizedValues() ([]uint16, error) {
	if err := c.validate(); err != nil {
		return nil, err
	}
	extraBits := c.totalBits - 1
	values := make([]uint16, c.paddedDimension)
	for index := range values {
		value := uint16(0)
		if extraBits > 0 {
			value = unpackRaBitQValue(c.extraCode, index, extraBits)
		}
		if raBitQBit(c.binaryCode, index) {
			value |= uint16(1) << extraBits
		}
		values[index] = value
	}
	return values, nil
}

func (c RaBitQCode) validate() error {
	if c.modelFingerprint == 0 || c.cluster < 0 || c.paddedDimension < MinRaBitQDimension || c.paddedDimension%64 != 0 ||
		c.totalBits < MinRaBitQTotalBits || c.totalBits > MaxRaBitQTotalBits || len(c.binaryCode) != c.paddedDimension/8 {
		return ErrInvalidRaBitQCode
	}
	extraBits := c.totalBits - 1
	if len(c.extraCode) != c.paddedDimension*extraBits/8 {
		return ErrInvalidRaBitQCode
	}
	for _, factor := range []float64{c.coarseAdd, c.coarseRescale, c.coarseError, c.fullAdd, c.fullRescale} {
		if math.IsNaN(factor) || math.IsInf(factor, 0) {
			return ErrInvalidRaBitQCode
		}
	}
	if c.coarseError < 0 {
		return ErrInvalidRaBitQCode
	}
	return nil
}

func quantizeRaBitQVector(
	vector, centroid []float32,
	cluster, totalBits int,
	extraScale float64,
	innerProduct bool,
) (RaBitQCode, error) {
	if len(vector) == 0 || len(vector) != len(centroid) || len(vector)%64 != 0 {
		return RaBitQCode{}, fmt.Errorf("%w: invalid vector dimensions", ErrInvalidRaBitQCode)
	}
	if totalBits < MinRaBitQTotalBits || totalBits > MaxRaBitQTotalBits || cluster < 0 {
		return RaBitQCode{}, ErrInvalidRaBitQCode
	}
	extraBits := totalBits - 1
	if extraBits > 0 && (extraScale <= 0 || math.IsNaN(extraScale) || math.IsInf(extraScale, 0)) {
		return RaBitQCode{}, fmt.Errorf("%w: invalid extra-code scale", ErrInvalidRaBitQCode)
	}

	residual := make([]float64, len(vector))
	binaryValues := make([]uint16, len(vector))
	binaryCode := make([]byte, len(vector)/8)
	var residualNormSquared float64
	for index := range vector {
		if !finiteFloat32(vector[index]) || !finiteFloat32(centroid[index]) {
			return RaBitQCode{}, ailego.ErrNonFiniteVector
		}
		value := float64(vector[index]) - float64(centroid[index])
		residual[index] = value
		residualNormSquared += value * value
		if value > 0 {
			binaryValues[index] = 1
			setRaBitQBit(binaryCode, index)
		}
	}

	coarseAdd, coarseRescale, coarseError, err := raBitQFactors(
		residual, centroid, binaryValues, 1, innerProduct,
	)
	if err != nil {
		return RaBitQCode{}, err
	}
	code := RaBitQCode{
		cluster: cluster, paddedDimension: len(vector), totalBits: totalBits,
		binaryCode: binaryCode, coarseAdd: coarseAdd,
		coarseRescale: coarseRescale, coarseError: coarseError,
	}
	if extraBits == 0 {
		code.fullAdd = coarseAdd
		code.fullRescale = coarseRescale
		return code, nil
	}

	norm := math.Sqrt(residualNormSquared)
	extraValues := make([]uint16, len(vector))
	factorValues := make([]uint16, len(vector))
	maxExtra := uint16((uint64(1) << extraBits) - 1)
	var inverseIPNorm float64 = 1
	if norm > 0 {
		var ipNorm float64
		for index, value := range residual {
			absoluteNormalized := math.Abs(value) / norm
			quantized := int(extraScale*absoluteNormalized + 1e-5)
			if quantized < 0 {
				quantized = 0
			}
			if quantized > int(maxExtra) {
				quantized = int(maxExtra)
			}
			ipNorm += (float64(quantized) + .5) * absoluteNormalized
			extra := uint16(quantized)
			if value < 0 {
				extra = maxExtra - extra
			}
			extraValues[index] = extra
			// The pinned implementation deliberately differs from the compact
			// one-bit code at zero: factors use >= 0, while binaryCode uses > 0.
			factorSign := uint16(0)
			if value >= 0 {
				factorSign = 1
			}
			factorValues[index] = extra | (factorSign << extraBits)
		}
		if ipNorm > 0 && !math.IsNaN(ipNorm) && !math.IsInf(ipNorm, 0) {
			inverseIPNorm = 1 / ipNorm
		}
	}
	code.extraCode = packRaBitQValues(extraValues, extraBits)
	code.fullAdd, code.fullRescale, _, err = raBitQFactors(
		residual, centroid, factorValues, totalBits, innerProduct,
	)
	if err != nil {
		return RaBitQCode{}, err
	}
	if innerProduct {
		code.fullRescale = -norm * inverseIPNorm
	} else {
		code.fullRescale = -2 * norm * inverseIPNorm
	}
	if err := code.validateWithoutFingerprint(); err != nil {
		return RaBitQCode{}, err
	}
	return code, nil
}

func (c RaBitQCode) validateWithoutFingerprint() error {
	copy := c
	copy.modelFingerprint = 1
	return copy.validate()
}

// raBitQFactors implements the pinned RaBitQ factor equations. levelsBits is
// one for coarse signs and TotalBits for the full unsigned code.
func raBitQFactors(
	residual []float64,
	centroid []float32,
	values []uint16,
	levelsBits int,
	innerProduct bool,
) (add, rescale, errorFactor float64, err error) {
	if len(residual) < 2 || len(residual) != len(centroid) || len(residual) != len(values) {
		return 0, 0, 0, ErrInvalidRaBitQCode
	}
	center := -(float64(uint64(1)<<levelsBits) - 1) / 2
	var normSquared, codeNormSquared, residualCodeDot, centroidCodeDot, residualCentroidDot float64
	for index, residualValue := range residual {
		centeredCode := float64(values[index]) + center
		normSquared += residualValue * residualValue
		codeNormSquared += centeredCode * centeredCode
		residualCodeDot += residualValue * centeredCode
		centroidCodeDot += float64(centroid[index]) * centeredCode
		residualCentroidDot += residualValue * float64(centroid[index])
	}
	if normSquared == 0 {
		if innerProduct {
			return 1, 0, 0, nil
		}
		return 0, 0, 0, nil
	}
	if residualCodeDot <= 0 || math.IsNaN(residualCodeDot) || math.IsInf(residualCodeDot, 0) {
		return 0, 0, 0, ErrInvalidRaBitQCode
	}
	ratio := normSquared*codeNormSquared/(residualCodeDot*residualCodeDot) - 1
	if ratio < 0 && ratio > -1e-12 {
		ratio = 0
	}
	if ratio < 0 || math.IsNaN(ratio) || math.IsInf(ratio, 0) {
		return 0, 0, 0, ErrInvalidRaBitQCode
	}
	norm := math.Sqrt(normSquared)
	temporaryError := norm * raBitQErrorConfidence * math.Sqrt(ratio/float64(len(residual)-1))
	if innerProduct {
		add = 1 - residualCentroidDot + normSquared*centroidCodeDot/residualCodeDot
		rescale = -normSquared / residualCodeDot
		errorFactor = temporaryError
	} else {
		add = normSquared + 2*normSquared*centroidCodeDot/residualCodeDot
		rescale = -2 * normSquared / residualCodeDot
		errorFactor = 2 * temporaryError
	}
	for _, factor := range []float64{add, rescale, errorFactor} {
		if math.IsNaN(factor) || math.IsInf(factor, 0) {
			return 0, 0, 0, ErrQuantizationOverflow
		}
	}
	return add, rescale, errorFactor, nil
}

func packRaBitQValues(values []uint16, bits int) []byte {
	if bits == 0 {
		return []byte{}
	}
	packed := make([]byte, (len(values)*bits+7)/8)
	for index, value := range values {
		bitOffset := index * bits
		for bit := 0; bit < bits; bit++ {
			if value&(uint16(1)<<bit) != 0 {
				position := bitOffset + bit
				packed[position/8] |= byte(1 << (position % 8))
			}
		}
	}
	return packed
}

func unpackRaBitQValue(packed []byte, index, bits int) uint16 {
	var value uint16
	bitOffset := index * bits
	for bit := 0; bit < bits; bit++ {
		position := bitOffset + bit
		if packed[position/8]&(1<<uint(position%8)) != 0 {
			value |= uint16(1) << bit
		}
	}
	return value
}

func setRaBitQBit(packed []byte, index int) {
	packed[index/8] |= 1 << uint(index%8)
}

func raBitQBit(packed []byte, index int) bool {
	return packed[index/8]&(1<<uint(index%8)) != 0
}

func raBitQBinaryDot(packed []byte, query []float32) float64 {
	var result float64
	for index, value := range query {
		if raBitQBit(packed, index) {
			result += float64(value)
		}
	}
	return result
}

func raBitQFullCodeDot(code RaBitQCode, query []float32) float64 {
	extraBits := code.totalBits - 1
	var result float64
	for index, value := range query {
		quantized := unpackRaBitQValue(code.extraCode, index, extraBits)
		if raBitQBit(code.binaryCode, index) {
			quantized |= uint16(1) << extraBits
		}
		result += float64(quantized) * float64(value)
	}
	return result
}

func trainRaBitQExtraScale(ctx context.Context, dimension, extraBits, workers int, seed uint64) (float64, error) {
	if ctx == nil {
		return 0, errors.New("core: nil RaBitQ scale-training context")
	}
	if dimension < MinRaBitQDimension || dimension%64 != 0 || extraBits < 1 || extraBits >= MaxRaBitQTotalBits {
		return 0, ErrInvalidRaBitQOptions
	}
	scales := make([]float64, raBitQScalingSampleSize)
	err := ailego.ParallelFor(ctx, len(scales), workers, func(ctx context.Context, sample int) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		random := splitMix64{state: seed + uint64(sample)*0x9e3779b97f4a7c15}
		vector := make([]float64, dimension)
		var normSquared float64
		for index := 0; index < dimension; index += 2 {
			left, right := raBitQNormalPair(&random)
			vector[index] = left
			normSquared += left * left
			if index+1 < dimension {
				vector[index+1] = right
				normSquared += right * right
			}
		}
		if normSquared == 0 {
			return ErrInvalidRaBitQModel
		}
		inverse := 1 / math.Sqrt(normSquared)
		for index := range vector {
			vector[index] = math.Abs(vector[index] * inverse)
		}
		scale, err := bestRaBitQScale(ctx, vector, extraBits)
		if err != nil {
			return err
		}
		scales[sample] = scale
		return nil
	})
	if err != nil {
		return 0, err
	}
	var result float64
	for _, scale := range scales {
		result += scale
	}
	result /= float64(len(scales))
	if result <= 0 || math.IsNaN(result) || math.IsInf(result, 0) {
		return 0, ErrInvalidRaBitQModel
	}
	return result, nil
}

func raBitQNormalPair(random *splitMix64) (float64, float64) {
	left := random.float64()
	if left <= 0 {
		left = math.SmallestNonzeroFloat64
	}
	right := random.float64()
	radius := math.Sqrt(-2 * math.Log(left))
	angle := 2 * math.Pi * right
	return radius * math.Cos(angle), radius * math.Sin(angle)
}

func bestRaBitQScale(ctx context.Context, absoluteUnitVector []float64, extraBits int) (float64, error) {
	if ctx == nil {
		return 0, errors.New("core: nil RaBitQ scale context")
	}
	if len(absoluteUnitVector) == 0 || extraBits < 1 || extraBits > 8 {
		return 0, ErrInvalidRaBitQOptions
	}
	tightStarts := [...]float64{0, .15, .20, .52, .59, .71, .75, .77, .81}
	maximum := float64(0)
	for _, value := range absoluteUnitVector {
		if value < 0 || math.IsNaN(value) || math.IsInf(value, 0) {
			return 0, ErrInvalidRaBitQCode
		}
		maximum = max(maximum, value)
	}
	if maximum == 0 {
		return 1, nil
	}
	maxCode := (1 << extraBits) - 1
	end := float64(maxCode+10) / maximum
	start := end * tightStarts[extraBits]
	codes := make([]int, len(absoluteUnitVector))
	squaredDenominator := float64(len(absoluteUnitVector)) * .25
	numerator := float64(0)
	events := make(raBitQScaleHeap, 0, len(absoluteUnitVector))
	for index, value := range absoluteUnitVector {
		code := int(start*value + 1e-5)
		codes[index] = code
		squaredDenominator += float64(code*code + code)
		numerator += (float64(code) + .5) * value
		if value > 0 {
			threshold := float64(code+1) / value
			if threshold < end {
				events = append(events, raBitQScaleEvent{threshold: threshold, index: index, code: code + 1})
			}
		}
	}
	heap.Init(&events)
	bestInnerProduct := float64(0)
	bestScale := float64(0)
	iterations := 0
	for events.Len() > 0 {
		if iterations&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return 0, err
			}
		}
		iterations++
		event := heap.Pop(&events).(raBitQScaleEvent)
		codes[event.index] = event.code
		squaredDenominator += float64(2 * event.code)
		numerator += absoluteUnitVector[event.index]
		innerProduct := numerator / math.Sqrt(squaredDenominator)
		if innerProduct > bestInnerProduct {
			bestInnerProduct = innerProduct
			bestScale = event.threshold
		}
		if event.code < maxCode {
			next := float64(event.code+1) / absoluteUnitVector[event.index]
			if next < end {
				heap.Push(&events, raBitQScaleEvent{threshold: next, index: event.index, code: event.code + 1})
			}
		}
	}
	if bestScale <= 0 || math.IsNaN(bestScale) || math.IsInf(bestScale, 0) {
		return 0, ErrInvalidRaBitQCode
	}
	return bestScale, nil
}

type raBitQScaleEvent struct {
	threshold float64
	index     int
	code      int
}

type raBitQScaleHeap []raBitQScaleEvent

func (h raBitQScaleHeap) Len() int { return len(h) }
func (h raBitQScaleHeap) Less(left, right int) bool {
	if h[left].threshold == h[right].threshold {
		return h[left].index < h[right].index
	}
	return h[left].threshold < h[right].threshold
}
func (h raBitQScaleHeap) Swap(left, right int) { h[left], h[right] = h[right], h[left] }
func (h *raBitQScaleHeap) Push(value any)      { *h = append(*h, value.(raBitQScaleEvent)) }
func (h *raBitQScaleHeap) Pop() any {
	old := *h
	last := old[len(old)-1]
	*h = old[:len(old)-1]
	return last
}
