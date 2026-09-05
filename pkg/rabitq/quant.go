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

package rabitq

import (
	"container/heap"
	"fmt"
	"math"

	"github.com/gorse-io/xvec/pkg/rabitq/simd"
)

const constEpsilon float32 = 1.9

// RaBitQConfig caches the scale used by faster extra-bit quantization. A
// non-positive TConst requests the exact scale search.
type RaBitQConfig struct{ TConst float64 }

// FasterConfig deterministically estimates a reusable extra-bit scale.
func FasterConfig(dim, totalBits int) RaBitQConfig {
	config := RaBitQConfig{TConst: -1}
	if totalBits <= 1 {
		return config
	}
	if dim <= 0 || totalBits > 9 {
		return config
	}
	rng := newMT19937(42)
	row := make([]float64, dim)
	var sum float64
	for sample := 0; sample < 100; sample++ {
		var norm2 float64
		for i := range row {
			v := rng.normal()
			row[i] = math.Abs(v)
			norm2 += v * v
		}
		norm := math.Sqrt(norm2)
		for i := range row {
			row[i] /= norm
		}
		sum += bestRescaleFactor64(row, totalBits-1)
	}
	config.TConst = sum / 100
	return config
}

func validateQuantArgs(data, centroid []float32, dim, exBits int, metric MetricType) error {
	if dim <= 1 || dim%64 != 0 {
		return fmt.Errorf("rabitq: padded dimension must be a multiple of 64 greater than 1")
	}
	if len(data) < dim || len(centroid) < dim {
		return fmt.Errorf("rabitq: vector is shorter than padded dimension")
	}
	if exBits < 0 || exBits > 8 {
		return fmt.Errorf("rabitq: extra bits must be in [0, 8]")
	}
	if !validMetric(metric) {
		return fmt.Errorf("rabitq: invalid metric %d", metric)
	}
	return nil
}

// QuantizeSplitSingle writes one-bit and extra-bit codes in the RaBitQ layout.
func QuantizeSplitSingle(data, centroid []float32, paddedDim, exBits int, binData, exData []byte, metric MetricType, config RaBitQConfig) error {
	if err := validateQuantArgs(data, centroid, paddedDim, exBits, metric); err != nil {
		return err
	}
	if len(binData) < BinDataBytes(paddedDim) {
		return fmt.Errorf("rabitq: binary output is too short")
	}
	if exBits > 0 && len(exData) < ExDataBytes(paddedDim, exBits) {
		return fmt.Errorf("rabitq: extra output is too short")
	}
	scratch := newQuantizeScratch(paddedDim)
	scratch.reset(data[:paddedDim], centroid[:paddedDim])
	bin := NewBinDataMap(binData, paddedDim)
	fAdd, fRescale, fError := oneBitFactors(scratch.residual, centroid[:paddedDim], scratch.signs, metric)
	bin.SetFAdd(fAdd)
	bin.SetFRescale(fRescale)
	bin.SetFError(fError)
	packBinary(scratch.signs, bin.BinCode())
	if exBits > 0 {
		quantizeExtra(scratch, centroid[:paddedDim], exBits, NewExDataMap(exData, paddedDim, exBits), metric, config)
	}
	return nil
}

// QuantizeSplitBatch writes up to BatchSize vectors in FastScan layout.
func QuantizeSplitBatch(data, centroid []float32, numPoints, paddedDim, exBits int, batchData, exData []byte, metric MetricType, config RaBitQConfig) error {
	if numPoints < 0 || numPoints > BatchSize {
		return fmt.Errorf("rabitq: batch point count must be in [0, %d]", BatchSize)
	}
	if err := validateQuantArgs(data, centroid, paddedDim, exBits, metric); err != nil {
		if numPoints == 0 && len(data) == 0 { /* validate below */
		} else {
			return err
		}
	}
	if paddedDim <= 1 || paddedDim%64 != 0 || !validMetric(metric) || exBits < 0 || exBits > 8 {
		return fmt.Errorf("rabitq: invalid batch quantization arguments")
	}
	if len(data) < numPoints*paddedDim || len(centroid) < paddedDim {
		return fmt.Errorf("rabitq: batch input is too short")
	}
	if len(batchData) < BatchDataBytes(paddedDim) || len(exData) < numPoints*ExDataBytes(paddedDim, exBits) {
		return fmt.Errorf("rabitq: batch output is too short")
	}
	batch := NewBatchDataMap(batchData, paddedDim)
	clear(batchData[:BatchDataBytes(paddedDim)])
	compact := make([]byte, numPoints*paddedDim/8)
	scratch := newQuantizeScratch(paddedDim)
	for row := 0; row < numPoints; row++ {
		vec := data[row*paddedDim : (row+1)*paddedDim]
		scratch.reset(vec, centroid[:paddedDim])
		fAdd, fRescale, fError := oneBitFactors(scratch.residual, centroid[:paddedDim], scratch.signs, metric)
		batch.SetFAdd(row, fAdd)
		batch.SetFRescale(row, fRescale)
		batch.SetFError(row, fError)
		packBinary8(scratch.signs, compact[row*paddedDim/8:])
		if exBits > 0 {
			off := row * ExDataBytes(paddedDim, exBits)
			quantizeExtra(scratch, centroid[:paddedDim], exBits, NewExDataMap(exData[off:], paddedDim, exBits), metric, config)
		}
	}
	packFastScan(compact, numPoints, paddedDim, batch.BinCode())
	return nil
}

type quantizeScratch struct {
	residual, magnitudes []float32
	signs, raw, combined []byte
}

func newQuantizeScratch(dim int) *quantizeScratch {
	return &quantizeScratch{
		residual: make([]float32, dim), magnitudes: make([]float32, dim),
		signs: make([]byte, dim), raw: make([]byte, dim), combined: make([]byte, dim),
	}
}

func (s *quantizeScratch) reset(data, centroid []float32) {
	for i := range data {
		s.residual[i] = data[i] - centroid[i]
		if s.residual[i] > 0 {
			s.signs[i] = 1
		} else {
			s.signs[i] = 0
		}
	}
}
func normSqr(v []float32) float32 {
	var s float32
	for _, x := range v {
		s += x * x
	}
	return s
}

func oneBitFactors(residual, centroid []float32, signs []byte, metric MetricType) (float32, float32, float32) {
	l2s := normSqr(residual)
	if l2s == 0 {
		if metric == MetricIP {
			return 1, 0, 0
		}
		return 0, 0, 0
	}
	var ipResidual, ipCentroid, codeNorm float32
	for i, code := range signs {
		x := float32(code) - 0.5
		ipResidual += residual[i] * x
		ipCentroid += centroid[i] * x
		codeNorm += x * x
	}
	l2 := float32(math.Sqrt(float64(l2s)))
	errArg := ((l2s*codeNorm)/(ipResidual*ipResidual) - 1) / float32(len(residual)-1)
	if errArg < 0 && errArg > -1e-5 {
		errArg = 0
	}
	tmpError := l2 * constEpsilon * float32(math.Sqrt(float64(errArg)))
	if metric == MetricL2 {
		return l2s + 2*l2s*ipCentroid/ipResidual, -2 * l2s / ipResidual, 2 * tmpError
	}
	return 1 - DotProduct(residual, centroid) + l2s*ipCentroid/ipResidual, -l2s / ipResidual, tmpError
}

func quantizeExtra(scratch *quantizeScratch, centroid []float32, exBits int, out ExDataMap, metric MetricType, config RaBitQConfig) {
	residual, raw := scratch.residual, scratch.raw
	l2s := normSqr(residual)
	if l2s == 0 {
		clear(raw)
		out.SetFAddEx(map[bool]float32{true: 1}[metric == MetricIP])
		out.SetFRescaleEx(0)
		packExcode(raw, out.ExCode(), exBits)
		return
	}
	l2 := float32(math.Sqrt(float64(l2s)))
	magnitudes := scratch.magnitudes
	for i, v := range residual {
		magnitudes[i] = float32(math.Abs(float64(v))) / l2
	}
	t := config.TConst
	if t <= 0 {
		t = bestRescaleFactor(magnitudes, exBits)
	}
	maxCode := (1 << exBits) - 1
	var ipnorm float64
	for i, mag := range magnitudes {
		code := int(t*float64(mag) + 1e-5)
		if config.TConst <= 0 {
			code = quantizedLevel(float64(mag), t, maxCode)
		}
		if code > maxCode {
			code = maxCode
		}
		ipnorm += (float64(code) + 0.5) * float64(mag)
		if residual[i] <= 0 {
			code = (^code) & maxCode
		}
		raw[i] = byte(code)
	}
	ipnormInv := float32(1 / ipnorm)
	if math.IsInf(float64(ipnormInv), 0) || math.IsNaN(float64(ipnormInv)) || ipnormInv == 0 {
		ipnormInv = 1
	}
	combined := scratch.combined
	for i := range raw {
		combined[i] = raw[i]
		if residual[i] > 0 {
			combined[i] += byte(1 << exBits)
		}
	}
	cb := -float32((uint32(1)<<uint(exBits))-1) - 0.5
	var ipResidual, ipCentroid, codeNorm float32
	for i, c := range combined {
		x := float32(c) + cb
		ipResidual += residual[i] * x
		ipCentroid += centroid[i] * x
		codeNorm += x * x
	}
	_ = codeNorm
	if metric == MetricL2 {
		out.SetFAddEx(l2s + 2*l2s*ipCentroid/ipResidual)
		out.SetFRescaleEx(ipnormInv * -2 * l2)
	} else {
		out.SetFAddEx(1 - DotProduct(residual, centroid) + l2s*ipCentroid/ipResidual)
		out.SetFRescaleEx(ipnormInv * -l2)
	}
	packExcode(raw, out.ExCode(), exBits)
}

func quantizedLevel(magnitude, t float64, maxCode int) int {
	if magnitude == 0 {
		return 0
	}
	c := min(int(t*magnitude), maxCode)
	if c < maxCode && float64(c+1)/magnitude <= t {
		c++
	} else if c > 0 && float64(c)/magnitude > t {
		c--
	}
	return c
}

type scaleEvent struct {
	t float64
	i int
}

type scaleEventHeap []scaleEvent

func (h scaleEventHeap) Len() int { return len(h) }
func (h scaleEventHeap) Less(i, j int) bool {
	if h[i].t == h[j].t {
		return h[i].i < h[j].i
	}
	return h[i].t < h[j].t
}
func (h scaleEventHeap) Swap(i, j int)   { h[i], h[j] = h[j], h[i] }
func (h *scaleEventHeap) Push(value any) { *h = append(*h, value.(scaleEvent)) }
func (h *scaleEventHeap) Pop() any {
	old := *h
	last := len(old) - 1
	value := old[last]
	*h = old[:last]
	return value
}

type magnitude interface {
	~float32 | ~float64
}

func bestRescaleFactor(m []float32, bits int) float64 {
	return bestRescaleFactorValues(m, bits)
}

func bestRescaleFactorValues[T magnitude](m []T, bits int) float64 {
	if len(m) == 0 {
		return 0
	}
	var maxMag float64
	for _, value := range m {
		maxMag = max(maxMag, float64(value))
	}
	if maxMag == 0 {
		return 0
	}
	maxCode := (1 << bits) - 1
	end := float64(maxCode+10) / maxMag
	starts := [9]float64{0, .15, .20, .52, .59, .71, .75, .77, .81}
	start := end * starts[bits]
	codes := make([]int, len(m))
	denom := float64(len(m)) * .25
	var numerator float64
	events := make(scaleEventHeap, 0, len(m))
	enqueueNext := func(i int) {
		mag := float64(m[i])
		if mag > 0 && codes[i] < maxCode {
			next := float64(codes[i]+1) / mag
			if next < end {
				heap.Push(&events, scaleEvent{t: next, i: i})
			}
		}
	}
	for i, value := range m {
		mag := float64(value)
		code := quantizedLevel(mag, start, maxCode)
		codes[i] = code
		denom += float64(code*code + code)
		numerator += (float64(code) + .5) * mag
		enqueueNext(i)
	}
	heap.Init(&events)
	bestIP, best := numerator/math.Sqrt(denom), start
	for len(events) > 0 {
		threshold := events[0].t
		for len(events) > 0 && events[0].t == threshold {
			event := heap.Pop(&events).(scaleEvent)
			codes[event.i]++
			denom += 2 * float64(codes[event.i])
			numerator += float64(m[event.i])
			enqueueNext(event.i)
		}
		ip := numerator / math.Sqrt(denom)
		if ip > bestIP {
			bestIP, best = ip, threshold
		}
	}
	return best
}
func packBinary(raw, out []byte) {
	clear(out)
	for i, v := range raw {
		if v != 0 {
			out[i/8] |= 1 << uint(7-i%8)
		}
	}
}
func packBinary8(raw, out []byte) {
	clear(out)
	for i, v := range raw {
		out[i/8] |= v << uint(7-i%8)
	}
}
func packFastScan(compact []byte, num, dim int, out []byte) {
	perm := [16]int{0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15}
	cols := dim / 8
	for col := 0; col < cols; col++ {
		base := col * 32
		for j, p := range perm {
			var a, b, c, d byte
			if p < num {
				a = compact[p*cols+col]
			}
			if p+16 < num {
				b = compact[(p+16)*cols+col]
			}
			if p < num {
				c = a & 15
				a >>= 4
			}
			if p+16 < num {
				d = b & 15
				b >>= 4
			}
			out[base+j] = a | b<<4
			out[base+16+j] = c | d<<4
		}
	}
}
func packExcode(raw, out []byte, bits int) {
	clear(out)
	switch bits {
	case 1:
		for i, v := range raw {
			out[i/8] |= (v & 1) << uint(i%8)
		}
	case 2:
		simd.Pack2BitExcode(raw, out)
	case 3:
		simd.Pack3BitExcode(raw, out)
	case 4:
		simd.Pack4BitExcode(raw, out)
	case 5:
		simd.Pack5BitExcode(raw, out)
	case 6:
		simd.Pack6BitExcode(raw, out)
	case 7:
		simd.Pack7BitExcode(raw, out)
	case 8:
		copy(out, raw)
	}
}

func bestRescaleFactor64(m []float64, bits int) float64 {
	return bestRescaleFactorValues(m, bits)
}
