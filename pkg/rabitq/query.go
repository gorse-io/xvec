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
	"fmt"
	"math"

	"github.com/gorse-io/xvec/pkg/rabitq/simd"
)

const SplitSingleQueryNumBits = 4

type lookupTable struct {
	lut          []byte
	floats       []float32
	words        []uint16
	delta, sumVL float32
}

func (l *lookupTable) reset(query []float32, dim int, hacc bool) {
	n := dim * 4
	bytes := n
	if hacc {
		bytes *= 2
	}
	if cap(l.lut) < bytes {
		l.lut = make([]byte, bytes)
	} else {
		l.lut = l.lut[:bytes]
		clear(l.lut)
	}
	if cap(l.floats) < n {
		l.floats = make([]float32, n)
	} else {
		l.floats = l.floats[:n]
	}
	pos := [16]int{3, 3, 2, 3, 1, 3, 2, 3, 0, 3, 2, 3, 1, 3, 2, 3}
	for cb := 0; cb < dim/4; cb++ {
		dst := l.floats[cb*16:]
		dst[0] = 0
		for j := 1; j < 16; j++ {
			low := j & -j
			dst[j] = dst[j-low] + query[cb*4+pos[j]]
		}
	}
	lo, hi := l.floats[0], l.floats[0]
	for _, v := range l.floats[1:] {
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	levels := float32(255)
	if hacc {
		levels = 65535
	}
	l.delta = (hi - lo) / levels
	l.sumVL = lo * float32(n/16)
	if l.delta == 0 {
		return
	}
	if hacc {
		if cap(l.words) < n {
			l.words = make([]uint16, n)
		} else {
			l.words = l.words[:n]
		}
		simd.ScalarQuantizeUint16(l.words, l.floats, lo, l.delta)
		simd.TransferLUTHACC(l.words, l.lut, dim)
	} else {
		simd.ScalarQuantizeUint8(l.lut, l.floats, lo, l.delta)
	}
}

// SplitBatchQuery stores a reusable FastScan query LUT.
type SplitBatchQuery struct {
	rotatedQuery                   []float32
	table                          lookupTable
	gAdd, gError, k1xsumq, kbxsumq float32
	paddedDim, exBits              int
	metric                         MetricType
	hacc                           bool
}

func NewSplitBatchQuery(query []float32, paddedDim, exBits int, metric MetricType, useHACC bool) (*SplitBatchQuery, error) {
	q := new(SplitBatchQuery)
	if err := q.Reset(query, paddedDim, exBits, metric, useHACC); err != nil {
		return nil, err
	}
	return q, nil
}

// Reset replaces all query-dependent state while retaining allocated LUT buffers.
func (q *SplitBatchQuery) Reset(query []float32, paddedDim, exBits int, metric MetricType, useHACC bool) error {
	if paddedDim <= 0 || paddedDim%64 != 0 || len(query) < paddedDim || exBits < 0 || exBits > 8 || !validMetric(metric) {
		return fmt.Errorf("rabitq: invalid split batch query")
	}
	if cap(q.rotatedQuery) < paddedDim {
		q.rotatedQuery = make([]float32, paddedDim)
	} else {
		q.rotatedQuery = q.rotatedQuery[:paddedDim]
	}
	copy(q.rotatedQuery, query[:paddedDim])
	q.table.reset(q.rotatedQuery, paddedDim, useHACC)
	q.metric = metric
	q.hacc = useHACC
	q.paddedDim = paddedDim
	q.exBits = exBits
	q.gAdd = 0
	q.gError = 0
	var sum float32
	for _, v := range q.rotatedQuery {
		sum += v
	}
	q.k1xsumq = sum * -.5
	q.kbxsumq = sum * (-float32((uint32(1)<<uint(exBits+1))-1) / 2)
	return nil
}
func (q *SplitBatchQuery) SetGAdd(norm float32, ip ...float32) {
	q.gError = norm
	if q.metric == MetricL2 {
		q.gAdd = norm * norm
	} else {
		q.gAdd = 0
		if len(ip) > 0 {
			q.gAdd = -ip[0]
		}
	}
}
func (q *SplitBatchQuery) RotatedQuery() []float32 { return append([]float32(nil), q.rotatedQuery...) }
func (q *SplitBatchQuery) Delta() float32          { return q.table.delta }
func (q *SplitBatchQuery) SumVLLUT() float32       { return q.table.sumVL }
func (q *SplitBatchQuery) K1XSumQ() float32        { return q.k1xsumq }
func (q *SplitBatchQuery) KBXSumQ() float32        { return q.kbxsumq }
func (q *SplitBatchQuery) GAdd() float32           { return q.gAdd }
func (q *SplitBatchQuery) GError() float32         { return q.gError }
func (q *SplitBatchQuery) LUT() []byte             { return append([]byte(nil), q.table.lut...) }
func (q *SplitBatchQuery) LUTCapacity() int        { return cap(q.table.lut) }

// SplitSingleQuery stores a transposed four-bit query.
type SplitSingleQuery struct {
	rotatedQuery                              []float32
	queryBin                                  []uint64
	delta, vl, gAdd, gError, k1xsumq, kbxsumq float32
	paddedDim, exBits                         int
	metric                                    MetricType
}

func NewSplitSingleQuery(query []float32, paddedDim, exBits int, config RaBitQConfig, metric MetricType) (*SplitSingleQuery, error) {
	if paddedDim <= 0 || paddedDim%64 != 0 || len(query) < paddedDim || exBits < 0 || exBits > 8 || exBits+1 > 9 || !validMetric(metric) {
		return nil, fmt.Errorf("rabitq: invalid split single query")
	}
	q := &SplitSingleQuery{
		rotatedQuery: append([]float32(nil), query[:paddedDim]...), queryBin: make([]uint64, paddedDim*SplitSingleQueryNumBits/64),
		paddedDim: paddedDim, exBits: exBits, metric: metric,
	}
	var sum float32
	for _, v := range q.rotatedQuery {
		sum += v
	}
	q.k1xsumq = sum * -.5
	q.kbxsumq = sum * (-float32((uint32(1)<<uint(exBits+1))-1) / 2)
	raw := quantizeScalarQuery(q.rotatedQuery, SplitSingleQueryNumBits, config, &q.delta, &q.vl)
	simd.TransposeBin512(raw, q.queryBin, SplitSingleQueryNumBits)
	return q, nil
}
func quantizeScalarQuery(data []float32, totalBits int, config RaBitQConfig, delta, vl *float32) []byte {
	raw := make([]byte, len(data))
	l2 := normSqr(data)
	if l2 == 0 {
		return raw
	}
	exBits := totalBits - 1
	norm := float32(math.Sqrt(float64(l2)))
	mags := make([]float32, len(data))
	for i, v := range data {
		mags[i] = float32(math.Abs(float64(v))) / norm
	}
	t := config.TConst
	if t <= 0 {
		t = bestRescaleFactor(mags, exBits)
	}
	maxCode := (1 << exBits) - 1
	for i, m := range mags {
		c := int(t*float64(m) + 1e-5)
		if config.TConst <= 0 {
			c = quantizedLevel(float64(m), t, maxCode)
		}
		if c > maxCode {
			c = maxCode
		}
		if data[i] <= 0 {
			c = (^c) & maxCode
		}
		if data[i] > 0 {
			c += 1 << exBits
		}
		raw[i] = byte(c)
	}
	cb := -float32((uint32(1)<<uint(exBits))-1) - .5
	var dot, normCode float32
	for i, c := range raw {
		x := float32(c) + cb
		dot += data[i] * x
		normCode += x * x
	}
	*delta = dot / normCode
	*vl = *delta * cb
	return raw
}
func (q *SplitSingleQuery) SetGAdd(norm float32, ip ...float32) {
	q.gError = norm
	if q.metric == MetricL2 {
		q.gAdd = norm * norm
	} else {
		q.gAdd = 0
		if len(ip) > 0 {
			q.gAdd = -ip[0]
		}
	}
}
func (q *SplitSingleQuery) SetGError(norm float32)  { q.gError = norm }
func (q *SplitSingleQuery) RotatedQuery() []float32 { return append([]float32(nil), q.rotatedQuery...) }
func (q *SplitSingleQuery) QueryBin() []uint64      { return append([]uint64(nil), q.queryBin...) }
func (q *SplitSingleQuery) NumBits() int            { return SplitSingleQueryNumBits }
func (q *SplitSingleQuery) Delta() float32          { return q.delta }
func (q *SplitSingleQuery) VL() float32             { return q.vl }
func (q *SplitSingleQuery) K1XSumQ() float32        { return q.k1xsumq }
func (q *SplitSingleQuery) KBXSumQ() float32        { return q.kbxsumq }
func (q *SplitSingleQuery) GAdd() float32           { return q.gAdd }
func (q *SplitSingleQuery) GError() float32         { return q.gError }

func (q *SplitBatchQuery) configuredPaddedDim() int     { return q.paddedDim }
func (q *SplitBatchQuery) configuredExtraBits() int     { return q.exBits }
func (q *SplitSingleQuery) configuredPaddedDim() int    { return q.paddedDim }
func (q *SplitSingleQuery) configuredExtraBits() int    { return q.exBits }
func (q *SplitBatchQuery) rotatedQueryData() []float32  { return q.rotatedQuery }
func (q *SplitSingleQuery) rotatedQueryData() []float32 { return q.rotatedQuery }
