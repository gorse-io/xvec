// Copyright 2026-present the xvec project
// Licensed under the Apache License, Version 2.0.

package rabitq

import (
	"math"
)

// SplitSingleQueryBits is the upstream query quantizer width.
const SplitSingleQueryBits = 4

// DistanceResult contains an estimate, probabilistic lower bound, and reusable one-bit dot product.
type DistanceResult struct{ Distance, LowerBound, IPX0QR float32 }

// ExcodeIPFunc computes query dot unpacked extra-code values.
type ExcodeIPFunc func(query []float32, packed []byte) (float32, error)

// SelectExcodeIPFunc returns the portable scalar kernel for exBits 0..8.
func SelectExcodeIPFunc(exBits int) (ExcodeIPFunc, error) {
	if exBits < 0 || exBits > 8 {
		return nil, ErrInvalidArgument
	}
	return func(query []float32, packed []byte) (float32, error) {
		if !validDimension(len(query)) || len(packed) != len(query)/8*exBits {
			return 0, ErrInvalidLayout
		}
		for _, v := range query {
			if !isFinite(v) {
				return 0, ErrNonFinite
			}
		}
		values, err := UnpackExCode(packed, len(query), exBits)
		if err != nil {
			return 0, err
		}
		var sum float32
		for i, v := range values {
			sum += query[i] * float32(v)
		}
		if !isFinite(sum) {
			return 0, ErrNonFinite
		}
		return sum, nil
	}, nil
}

type queryCore struct {
	query                     []float32
	exBits                    int
	metric                    Metric
	sum, k1, kb, gAdd, gError float32
}

func (c *queryCore) reset(query []float32) error {
	if !validDimension(len(query)) {
		return ErrInvalidArgument
	}
	for _, v := range query {
		if !isFinite(v) {
			return ErrNonFinite
		}
	}
	c.query = append(c.query[:0], query...)
	c.sum = 0
	for _, v := range query {
		c.sum += v
	}
	c.k1 = -.5 * c.sum
	c.kb = -(float32(uint32(1)<<uint(c.exBits)) - .5) * c.sum
	c.gAdd = 0
	c.gError = 0
	return nil
}
func (c *queryCore) setGAdd(norm, ip float32) {
	if c.metric == MetricL2 {
		c.gAdd = norm * norm
	} else {
		c.gAdd = -ip
	}
	c.gError = norm
}

// SplitSingleQuery owns scalar single-vector query state.
type SplitSingleQuery struct {
	queryCore
	config    FasterConfig
	delta, vl float32
	quantized []uint8
}

func NewSplitSingleQuery(query []float32, exBits int, config FasterConfig, metric Metric) (*SplitSingleQuery, error) {
	if exBits < 0 || exBits > 8 || !validMetric(metric) || !validConfig(config) {
		return nil, ErrInvalidArgument
	}
	q := &SplitSingleQuery{queryCore: queryCore{exBits: exBits, metric: metric}, config: config}
	if err := q.Reset(query); err != nil {
		return nil, err
	}
	return q, nil
}
func (q *SplitSingleQuery) Reset(query []float32) error {
	if q == nil {
		return ErrInvalidArgument
	}
	if err := q.queryCore.reset(query); err != nil {
		return err
	}
	q.quantized, q.delta, q.vl = quantizeQuery4(q.query, q.config)
	return nil
}
func (q *SplitSingleQuery) SetGAdd(norm float32, ip ...float32) {
	v := float32(0)
	if len(ip) > 0 {
		v = ip[0]
	}
	q.queryCore.setGAdd(norm, v)
}
func (q *SplitSingleQuery) SetGError(norm float32)  { q.gError = norm }
func (q *SplitSingleQuery) GAdd() float32           { return q.gAdd }
func (q *SplitSingleQuery) GError() float32         { return q.gError }
func (q *SplitSingleQuery) RotatedQuery() []float32 { return append([]float32(nil), q.query...) }
func (q *SplitSingleQuery) Delta() float32          { return q.delta }
func (q *SplitSingleQuery) VL() float32             { return q.vl }

func quantizeQuery4(query []float32, config FasterConfig) ([]uint8, float32, float32) {
	norm := float32(math.Sqrt(float64(normSquared(query))))
	abs := make([]float32, len(query))
	if norm > 0 {
		for i, v := range query {
			abs[i] = float32(math.Abs(float64(v / norm)))
		}
	}
	t := config.TConst
	if t <= 0 {
		t = bestScale(abs, 3)
	}
	codes := make([]uint8, len(query))
	for i, v := range abs {
		scaled := t*float64(v) + 1e-5
		c := 0
		if scaled >= 7 {
			c = 7
		} else if scaled > 0 {
			c = int(scaled)
		}
		if query[i] < 0 {
			c = 7 - c
		} else if query[i] > 0 {
			c += 8
		}
		codes[i] = uint8(c)
	}
	center := float32(-7.5)
	var qn, dot float32
	for i, c := range codes {
		x := float32(c) + center
		qn += x * x
		dot += query[i] * x
	}
	if norm == 0 || qn == 0 {
		return codes, 0, 0
	}
	// Keep the pinned C++ float32 operation sequence. Algebraically reducing this
	// to dot/qn changes rounding and therefore the serialized query factors.
	normQuan := float32(math.Sqrt(float64(qn)))
	cosSimilarity := dot / (norm * normQuan)
	delta := norm / normQuan * cosSimilarity
	return codes, delta, delta * center
}

// SplitBatchQuery owns scalar batch-query state and can be reset for reuse.
type SplitBatchQuery struct {
	queryCore
	lut          []uint16
	delta, sumVL float32
}

func NewSplitBatchQuery(query []float32, exBits int, metric Metric) (*SplitBatchQuery, error) {
	if exBits < 0 || exBits > 8 || !validMetric(metric) {
		return nil, ErrInvalidArgument
	}
	q := &SplitBatchQuery{queryCore: queryCore{exBits: exBits, metric: metric}}
	if err := q.ResetWithOptions(query, exBits, metric, true); err != nil {
		return nil, err
	}
	return q, nil
}
func (q *SplitBatchQuery) Reset(query []float32) error {
	if q == nil {
		return ErrInvalidArgument
	}
	return q.ResetWithOptions(query, q.exBits, q.metric, true)
}

// ResetWithOptions implements zvec's reusable SplitBatchQuery reset operation.
// The current scalar implementation supports the high-accuracy path used by zvec.
func (q *SplitBatchQuery) ResetWithOptions(query []float32, exBits int, metric Metric, highAccuracy bool) error {
	if q == nil {
		return ErrInvalidArgument
	}
	if exBits < 0 || exBits > 8 || !validMetric(metric) || !highAccuracy {
		return ErrInvalidArgument
	}
	oldExBits, oldMetric := q.exBits, q.metric
	q.exBits, q.metric = exBits, metric
	if err := q.queryCore.reset(query); err != nil {
		q.exBits, q.metric = oldExBits, oldMetric
		return err
	}
	q.buildHighAccuracyLUT()
	return nil
}
func (q *SplitBatchQuery) SetGAdd(norm float32, ip ...float32) {
	v := float32(0)
	if len(ip) > 0 {
		v = ip[0]
	}
	q.queryCore.setGAdd(norm, v)
}
func (q *SplitBatchQuery) GAdd() float32           { return q.gAdd }
func (q *SplitBatchQuery) GError() float32         { return q.gError }
func (q *SplitBatchQuery) RotatedQuery() []float32 { return append([]float32(nil), q.query...) }

func (q *SplitBatchQuery) buildHighAccuracyLUT() {
	const entries = 16
	tables := len(q.query) / 4
	floatLUT := make([]float32, tables*entries)
	lo := float32(math.Inf(1))
	hi := float32(math.Inf(-1))
	for table := 0; table < tables; table++ {
		values := floatLUT[table*entries : (table+1)*entries]
		query := q.query[table*4 : table*4+4]
		for code := 1; code < entries; code++ {
			lowBit := code & -code
			position := 0
			for (1 << position) != lowBit {
				position++
			}
			values[code] = values[code-lowBit] + query[3-position]
		}
		for _, value := range values {
			lo = min(lo, value)
			hi = max(hi, value)
		}
	}

	q.lut = resizeUint16(q.lut, len(floatLUT))
	q.delta = (hi - lo) / float32(math.MaxUint16)
	if q.delta == 0 {
		clear(q.lut)
	} else {
		inverseDelta := 1 / q.delta
		for i, value := range floatLUT {
			quantized := math.RoundToEven(float64((value - lo) * inverseDelta))
			quantized = min(max(quantized, 0), float64(math.MaxUint16))
			q.lut[i] = uint16(quantized)
		}
	}
	q.sumVL = lo * float32(tables)
}

func resizeUint16(values []uint16, size int) []uint16 {
	if cap(values) < size {
		return make([]uint16, size)
	}
	values = values[:size]
	clear(values)
	return values
}

func binaryDot(code []byte, query []float32) float32 {
	var sum float32
	for i, v := range query {
		if binaryBit(code, i) {
			sum += v
		}
	}
	return sum
}

// SplitSingleEstimate evaluates one compact one-bit code.
func SplitSingleEstimate(bin []byte, q *SplitSingleQuery) (DistanceResult, error) {
	if q == nil {
		return DistanceResult{}, ErrInvalidArgument
	}
	return SplitSingleEstimateWithFactors(bin, q, q.gAdd, q.gError)
}

// SplitSingleEstimateWithFactors evaluates one code with immutable cluster-local factors.
func SplitSingleEstimateWithFactors(bin []byte, q *SplitSingleQuery, gAdd, gError float32) (DistanceResult, error) {
	if q == nil || !isFinite(gAdd) || !isFinite(gError) {
		return DistanceResult{}, ErrInvalidArgument
	}
	m, err := NewBinData(bin, len(q.query))
	if err != nil {
		return DistanceResult{}, err
	}
	var sumCode, count uint32
	for i, code := range q.quantized {
		if binaryBit(m.Code(), i) {
			sumCode += uint32(code)
			count++
		}
	}
	ip := q.delta*float32(sumCode) + q.vl*float32(count)
	dist := m.FAdd() + gAdd + m.FRescale()*(ip+q.k1)
	low := dist - m.FError()*gError
	if !isFinite(dist) || !isFinite(low) {
		return DistanceResult{}, ErrNonFinite
	}
	return DistanceResult{Distance: dist, LowerBound: low, IPX0QR: ip}, nil
}

// SplitSingleFullDistance evaluates all bits and scales the one-bit error bound.
func SplitSingleFullDistance(bin, ex []byte, q *SplitSingleQuery, exBits int) (DistanceResult, error) {
	if q == nil || q.exBits != exBits {
		return DistanceResult{}, ErrInvalidArgument
	}
	return SplitSingleFullDistanceWithFactors(bin, ex, q, exBits, q.gAdd, q.gError)
}

// SplitSingleFullDistanceWithFactors evaluates all bits with immutable cluster-local factors.
func SplitSingleFullDistanceWithFactors(bin, ex []byte, q *SplitSingleQuery, exBits int, gAdd, gError float32) (DistanceResult, error) {
	if q == nil || q.exBits != exBits || !isFinite(gAdd) || !isFinite(gError) {
		return DistanceResult{}, ErrInvalidArgument
	}
	bm, err := NewBinData(bin, len(q.query))
	if err != nil {
		return DistanceResult{}, err
	}
	ip := binaryDot(bm.Code(), q.query)
	dist, err := splitDistanceBoosting(ex, &q.queryCore, exBits, ip, gAdd)
	if err != nil {
		return DistanceResult{}, err
	}
	low := dist - bm.FError()*gError/float32(uint32(1)<<uint(exBits))
	if !isFinite(dist) || !isFinite(low) {
		return DistanceResult{}, ErrNonFinite
	}
	return DistanceResult{Distance: dist, LowerBound: low, IPX0QR: ip}, nil
}

// SplitBatchEstimate evaluates all 32 lanes in a BatchData layout.
func SplitBatchEstimate(batch []byte, q *SplitBatchQuery) ([]DistanceResult, error) {
	if q == nil {
		return nil, ErrInvalidArgument
	}
	m, err := NewBatchData(batch, len(q.query))
	if err != nil {
		return nil, err
	}
	out := make([]DistanceResult, BatchSize)
	for lane := 0; lane < BatchSize; lane++ {
		var accumulated uint64
		for table := 0; table < len(q.query)/4; table++ {
			var code int
			for bit := 0; bit < 4; bit++ {
				if batchBinaryBit(m.Code(), len(q.query), lane, table*4+bit) {
					code |= 1 << (3 - bit)
				}
			}
			accumulated += uint64(q.lut[table*16+code])
		}
		ip := q.delta*float32(accumulated) + q.sumVL
		dist := m.FAdd(lane) + q.gAdd + m.FRescale(lane)*(ip+q.k1)
		low := dist - m.FError(lane)*q.gError
		if !isFinite(dist) || !isFinite(low) {
			return nil, ErrNonFinite
		}
		out[lane] = DistanceResult{Distance: dist, LowerBound: low, IPX0QR: ip}
	}
	return out, nil
}

// SplitDistanceBoosting refines a one-bit result using packed extra bits.
func SplitDistanceBoosting(ex []byte, q *SplitBatchQuery, exBits int, ipX0QR float32) (float32, error) {
	if q == nil {
		return 0, ErrInvalidArgument
	}
	return splitDistanceBoosting(ex, &q.queryCore, exBits, ipX0QR, q.gAdd)
}

func splitDistanceBoosting(ex []byte, c *queryCore, exBits int, ipX0QR, gAdd float32) (float32, error) {
	if c == nil || c.exBits != exBits || exBits < 1 || exBits > 8 {
		return 0, ErrInvalidArgument
	}
	m, err := NewExData(ex, len(c.query), exBits)
	if err != nil {
		return 0, err
	}
	fn, _ := SelectExcodeIPFunc(exBits)
	ip, err := fn(c.query, m.Code())
	if err != nil {
		return 0, err
	}
	dist := m.FAdd() + gAdd + m.FRescale()*(float32(uint32(1)<<uint(exBits))*ipX0QR+ip+c.kb)
	if !isFinite(dist) {
		return 0, ErrNonFinite
	}
	return dist, nil
}
