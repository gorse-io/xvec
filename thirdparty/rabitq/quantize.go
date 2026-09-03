// Copyright 2026-present the xvec project
// Licensed under the Apache License, Version 2.0.

package rabitq

import (
	"container/heap"
	"fmt"
	"math"
)

const errorConfidence float32 = 1.9

// FasterConfig contains the reusable extra-bit scale. A negative TConst selects exact scaling.
type FasterConfig struct{ TConst float64 }

func validConfig(config FasterConfig) bool {
	return !math.IsNaN(config.TConst) && !math.IsInf(config.TConst, 0)
}

// NewFasterConfig computes the reusable RaBitQ+ scale for totalBits in [1,9].
func NewFasterConfig(dim, totalBits int) (FasterConfig, error) {
	if !validDimension(dim) || totalBits < 1 || totalBits > 9 {
		return FasterConfig{}, ErrInvalidArgument
	}
	if totalBits == 1 {
		return FasterConfig{TConst: -1}, nil
	}
	// RaBitQ-Library averages exact scales for 100 seeded Gaussian unit vectors.
	rng := newMT19937(42)
	normal := normalGenerator{rng: rng}
	var sum float64
	for sample := 0; sample < 100; sample++ {
		v := make([]float64, dim)
		var n float64
		for i := range v {
			v[i] = normal.next()
			n += v[i] * v[i]
		}
		inv := 1 / math.Sqrt(n)
		for i := range v {
			v[i] = math.Abs(v[i] * inv)
		}
		sum += bestScale64(v, totalBits-1)
	}
	return FasterConfig{TConst: sum / 100}, nil
}

type mt19937 struct {
	state [624]uint32
	index int
}

func newMT19937(seed uint32) *mt19937 {
	rng := &mt19937{index: 624}
	rng.state[0] = seed
	for i := 1; i < len(rng.state); i++ {
		rng.state[i] = 1812433253*(rng.state[i-1]^(rng.state[i-1]>>30)) + uint32(i)
	}
	return rng
}

func (r *mt19937) next() uint32 {
	if r.index >= len(r.state) {
		for i := range r.state {
			y := (r.state[i] & 0x80000000) | (r.state[(i+1)%624] & 0x7fffffff)
			r.state[i] = r.state[(i+397)%624] ^ (y >> 1)
			if y&1 != 0 {
				r.state[i] ^= 0x9908b0df
			}
		}
		r.index = 0
	}
	y := r.state[r.index]
	r.index++
	y ^= y >> 11
	y ^= (y << 7) & 0x9d2c5680
	y ^= (y << 15) & 0xefc60000
	y ^= y >> 18
	return y
}

type normalGenerator struct {
	rng      *mt19937
	saved    float64
	hasSaved bool
}

func (g *normalGenerator) uniform() float64 {
	const r = 4294967296.0
	return (float64(g.rng.next()) + float64(g.rng.next())*r) / (r * r)
}

func (g *normalGenerator) next() float64 {
	if g.hasSaved {
		g.hasSaved = false
		return g.saved
	}
	var x, y, radiusSquared float64
	for radiusSquared > 1 || radiusSquared == 0 {
		x = 2*g.uniform() - 1
		y = 2*g.uniform() - 1
		radiusSquared = x*x + y*y
	}
	multiplier := math.Sqrt(-2 * math.Log(radiusSquared) / radiusSquared)
	g.saved = x * multiplier
	g.hasSaved = true
	return y * multiplier
}

type scaleEvent struct {
	at          float64
	index, code int
}
type scaleHeap []scaleEvent

func (h scaleHeap) Len() int { return len(h) }
func (h scaleHeap) Less(i, j int) bool {
	if h[i].at == h[j].at {
		return h[i].index < h[j].index
	}
	return h[i].at < h[j].at
}
func (h scaleHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *scaleHeap) Push(x any)   { *h = append(*h, x.(scaleEvent)) }
func (h *scaleHeap) Pop() any     { o := *h; x := o[len(o)-1]; *h = o[:len(o)-1]; return x }
func bestScale(absUnit []float32, bits int) float64 {
	values := make([]float64, len(absUnit))
	for i, value := range absUnit {
		values[i] = float64(value)
	}
	return bestScale64(values, bits)
}

func bestScale64(absUnit []float64, bits int) float64 {
	tight := [9]float64{0, .15, .20, .52, .59, .71, .75, .77, .81}
	var mx float64
	for _, v := range absUnit {
		mx = max(mx, v)
	}
	if mx == 0 {
		return 1
	}
	maxCode := (1 << bits) - 1
	end := float64(maxCode+10) / mx
	start := end * tight[bits]
	den := float64(len(absUnit)) * .25
	num := 0.
	events := make(scaleHeap, 0, len(absUnit))
	for i, v0 := range absUnit {
		v := v0
		code := int(start*v + 1e-5)
		den += float64(code*code + code)
		num += (float64(code) + .5) * v
		if v > 0 {
			at := float64(code+1) / v
			events = append(events, scaleEvent{at, i, code + 1})
		}
	}
	heap.Init(&events)
	bestIP, best := 0., 0.
	for len(events) > 0 {
		e := heap.Pop(&events).(scaleEvent)
		den += float64(2 * e.code)
		num += absUnit[e.index]
		ip := num / math.Sqrt(den)
		if ip > bestIP {
			bestIP, best = ip, e.at
		}
		if e.code < maxCode {
			at := float64(e.code+1) / absUnit[e.index]
			if at < end {
				heap.Push(&events, scaleEvent{at, e.index, e.code + 1})
			}
		}
	}
	if best <= 0 {
		return 1
	}
	return best
}

func residualOf(data, centroid []float32) ([]float32, error) {
	if err := validatePair(data, centroid); err != nil {
		return nil, err
	}
	if !validDimension(len(data)) {
		return nil, ErrInvalidArgument
	}
	r := make([]float32, len(data))
	for i := range r {
		r[i] = data[i] - centroid[i]
	}
	return r, nil
}
func factorValues(residual, centroid []float32, values []uint16, bits int, metric Metric) (add, rescale, errFactor float32, err error) {
	center := -float32((uint32(1)<<uint(bits))-1) / 2
	var norm2, code2, rc, cc, rcent float32
	for i, r := range residual {
		c := float32(values[i]) + center
		norm2 += r * r
		code2 += c * c
		rc += r * c
		cc += centroid[i] * c
		rcent += r * centroid[i]
	}
	if norm2 == 0 {
		if metric == MetricIP {
			return 1, 0, 0, nil
		}
		return 0, 0, 0, nil
	}
	if rc <= 0 {
		return 0, 0, 0, ErrInvalidArgument
	}
	ratio := norm2*code2/(rc*rc) - 1
	if ratio < 0 && ratio > -.00001 {
		ratio = 0
	}
	if ratio < 0 {
		return 0, 0, 0, ErrInvalidArgument
	}
	tmp := float32(math.Sqrt(float64(norm2))) * errorConfidence * float32(math.Sqrt(float64(ratio/float32(len(residual)-1))))
	if metric == MetricL2 {
		add = norm2 + 2*norm2*cc/rc
		rescale = -2 * norm2 / rc
		errFactor = 2 * tmp
	} else {
		add = 1 - rcent + norm2*cc/rc
		rescale = -norm2 / rc
		errFactor = tmp
	}
	if !isFinite(add) || !isFinite(rescale) || !isFinite(errFactor) {
		return 0, 0, 0, ErrNonFinite
	}
	return
}

// QuantizeSplitSingle emits BinData and ExData byte layouts.
func QuantizeSplitSingle(data, centroid []float32, exBits int, metric Metric, config FasterConfig) ([]byte, []byte, error) {
	if !validMetric(metric) || !validConfig(config) || exBits < 0 || exBits > 8 {
		return nil, nil, ErrInvalidArgument
	}
	r, err := residualOf(data, centroid)
	if err != nil {
		return nil, nil, err
	}
	dim := len(r)
	bin := make([]byte, BinDataBytes(dim))
	bm, _ := NewBinData(bin, dim)
	binaryValues := make([]uint16, dim)
	for i, v := range r {
		if v > 0 {
			binaryValues[i] = 1
			setBinaryBit(bm.Code(), i)
		}
	}
	a, s, e, err := factorValues(r, centroid, binaryValues, 1, metric)
	if err != nil {
		return nil, nil, err
	}
	bm.setFactors(a, s, e)
	if exBits == 0 {
		return bin, []byte{}, nil
	}
	norm := float32(math.Sqrt(float64(normSquared(r))))
	absUnit := make([]float32, dim)
	if norm > 0 {
		for i, v := range r {
			absUnit[i] = float32(math.Abs(float64(v / norm)))
		}
	}
	t := config.TConst
	if t <= 0 {
		t = bestScale(absUnit, exBits)
	}
	raw := make([]uint8, dim)
	factor := make([]uint16, dim)
	maxCode := (1 << exBits) - 1
	var ipnorm float64
	for i, v := range absUnit {
		scaled := t*float64(v) + 1e-5
		code := 0
		if scaled >= float64(maxCode) {
			code = maxCode
		} else if scaled > 0 {
			code = int(scaled)
		}
		ipnorm += (float64(code) + .5) * float64(v)
		stored := code
		if r[i] < 0 {
			stored = maxCode - code
		}
		raw[i] = uint8(stored)
		factor[i] = uint16(stored)
		if r[i] > 0 {
			factor[i] |= uint16(1) << uint(exBits)
		}
	}
	packed, _ := PackExCode(raw, exBits)
	ex := make([]byte, ExDataBytes(dim, exBits))
	em, _ := NewExData(ex, dim, exBits)
	copy(em.Code(), packed)
	fa, _, _, err := factorValues(r, centroid, factor, exBits+1, metric)
	if err != nil {
		return nil, nil, err
	}
	inv := float64(1)
	if ipnorm > 0 && !math.IsInf(ipnorm, 0) && !math.IsNaN(ipnorm) {
		inv = 1 / ipnorm
	}
	fr := -norm * float32(inv)
	if metric == MetricL2 {
		fr *= 2
	}
	em.setFactors(fa, fr)
	return bin, ex, nil
}
func setBinaryBit(code []byte, index int) {
	block := (index / 64) * 8
	within := index % 64
	code[block+7-within/8] |= 1 << uint(7-within%8)
}
func binaryBit(code []byte, index int) bool {
	block := (index / 64) * 8
	within := index % 64
	return code[block+7-within/8]&(1<<uint(7-within%8)) != 0
}

// QuantizeSplitBatch quantizes at most 32 vectors and zero-fills unused lanes.
func QuantizeSplitBatch(vectors [][]float32, centroid []float32, exBits int, metric Metric, config FasterConfig) ([]byte, [][]byte, error) {
	if len(vectors) == 0 || len(vectors) > BatchSize {
		return nil, nil, ErrInvalidArgument
	}
	dim := len(centroid)
	if !validDimension(dim) {
		return nil, nil, ErrInvalidArgument
	}
	compact := make([][]byte, BatchSize)
	batchBytes := BatchDataBytes(dim)
	if batchBytes == 0 {
		return nil, nil, ErrInvalidArgument
	}
	batch := make([]byte, batchBytes)
	bm, _ := NewBatchData(batch, dim)
	extras := make([][]byte, len(vectors))
	for lane, v := range vectors {
		bin, ex, err := QuantizeSplitSingle(v, centroid, exBits, metric, config)
		if err != nil {
			return nil, nil, fmt.Errorf("rabitq: quantize lane %d failed, %w", lane, err)
		}
		m, _ := NewBinData(bin, dim)
		compact[lane] = make([]byte, dim/8)
		for i := 0; i < dim; i++ {
			if binaryBit(m.Code(), i) {
				compact[lane][i/8] |= 1 << uint(7-i%8)
			}
		}
		bm.setFactor(0, lane, m.FAdd())
		bm.setFactor(1, lane, m.FRescale())
		bm.setFactor(2, lane, m.FError())
		extras[lane] = ex
	}
	cols := dim / 8
	perm := [16]int{0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15}
	for col := 0; col < cols; col++ {
		dst := bm.Code()[col*32:]
		for j, p := range perm {
			var a, b byte
			if compact[p] != nil {
				a = compact[p][col]
			}
			if compact[p+16] != nil {
				b = compact[p+16][col]
			}
			dst[j] = (a >> 4) | ((b >> 4) << 4)
			dst[j+16] = (a & 15) | ((b & 15) << 4)
		}
	}
	return batch, extras, nil
}

// QuantizeSplitBatchFlat quantizes contiguous row-major vectors and returns contiguous extra records.
func QuantizeSplitBatchFlat(data []float32, numPoints int, centroid []float32, exBits int, metric Metric, config FasterConfig) ([]byte, []byte, error) {
	dim := len(centroid)
	if numPoints < 1 || numPoints > BatchSize || !validDimension(dim) || dim > maxInt/numPoints || len(data) != numPoints*dim {
		return nil, nil, ErrInvalidArgument
	}
	vectors := make([][]float32, numPoints)
	for i := range vectors {
		vectors[i] = data[i*dim : (i+1)*dim]
	}
	batch, records, err := QuantizeSplitBatch(vectors, centroid, exBits, metric, config)
	if err != nil {
		return nil, nil, err
	}
	stride := ExDataBytes(dim, exBits)
	extra := make([]byte, numPoints*stride)
	for i, record := range records {
		copy(extra[i*stride:], record)
	}
	return batch, extra, nil
}

func batchBinaryBit(code []byte, dim, lane, index int) bool {
	col := index / 8
	bit := 7 - index%8
	perm := [16]int{0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15}
	half := lane / 16
	local := lane % 16
	j := 0
	for ; j < 16; j++ {
		if perm[j] == local {
			break
		}
	}
	base := col * 32
	var nib byte
	if bit >= 4 {
		nib = (code[base+j] >> uint(half*4)) & 15
		return nib&(1<<uint(bit-4)) != 0
	}
	nib = (code[base+16+j] >> uint(half*4)) & 15
	return nib&(1<<uint(bit)) != 0
}
