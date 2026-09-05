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

import "github.com/gorse-io/xvec/pkg/rabitq/simd"

// SplitQuery is implemented by single and batch query wrappers.
type SplitQuery interface {
	rotatedQueryData() []float32
	KBXSumQ() float32
	GAdd() float32
	configuredPaddedDim() int
	configuredExtraBits() int
}

// SplitBatchEstDist estimates all 32 lanes in one FastScan block.
func SplitBatchEstDist(batchData []byte, q *SplitBatchQuery, paddedDim int, estDistance, lowDistance, ipX0QR []float32, useHACC bool) {
	if q == nil || paddedDim != q.paddedDim || useHACC != q.hacc {
		panic("rabitq: batch estimator configuration does not match query")
	}
	if len(estDistance) < BatchSize || len(lowDistance) < BatchSize || len(ipX0QR) < BatchSize {
		panic("rabitq: batch result is too short")
	}
	batch := NewBatchDataMap(batchData, paddedDim)
	acc := make([]int32, BatchSize)
	codes, lut := batch.BinCode(), q.table.lut
	remaining := paddedDim
	codeOff, lutOff := 0, 0
	for remaining > 0 {
		chunk := min(remaining, 1024)
		if useHACC {
			part := make([]int32, BatchSize)
			simd.AccumulateHACC(codes[codeOff:], lut[lutOff:], part, chunk)
			for i := range acc {
				acc[i] += part[i]
			}
			lutOff += chunk * 8
		} else {
			part := make([]uint16, BatchSize)
			simd.Accumulate(codes[codeOff:], lut[lutOff:], part, chunk)
			for i := range acc {
				acc[i] += int32(part[i])
			}
			lutOff += chunk * 4
		}
		codeOff += chunk * 4
		remaining -= chunk
	}
	for i := 0; i < BatchSize; i++ {
		ipX0QR[i] = q.Delta()*float32(acc[i]) + q.SumVLLUT()
		estDistance[i] = batch.FAdd(i) + q.GAdd() + batch.FRescale(i)*(ipX0QR[i]+q.K1XSumQ())
		lowDistance[i] = estDistance[i] - batch.FError(i)*q.GError()
	}
}

// SplitDistanceBoosting refines a one-bit result with packed extra bits.
func SplitDistanceBoosting(exData []byte, ipFunc ExcodeIPFunc, q SplitQuery, paddedDim, exBits int, ipX0QR float32) float32 {
	if q == nil || paddedDim != q.configuredPaddedDim() || exBits != q.configuredExtraBits() {
		panic("rabitq: boosting configuration does not match query")
	}
	ex := NewExDataMap(exData, paddedDim, exBits)
	return ex.FAddEx() + q.GAdd() + ex.FRescaleEx()*(float32(uint32(1)<<uint(exBits))*ipX0QR+ipFunc(q.rotatedQueryData(), ex.ExCode())+q.KBXSumQ())
}

// SplitSingleEstDist computes the one-bit estimate and lower bound.
func SplitSingleEstDist(binData []byte, q *SplitSingleQuery, paddedDim int, gAdd, gError float32) (ipX0QR, estDistance, lowDistance float32) {
	if q == nil || paddedDim != q.paddedDim {
		panic("rabitq: single estimator configuration does not match query")
	}
	bin := NewBinDataMap(binData, paddedDim)
	ipX0QR = simd.WarmupIPX0Q512(bin.decodedBinCode(), q.queryBin, q.Delta(), q.VL(), paddedDim, q.NumBits())
	estDistance = bin.FAdd() + gAdd + bin.FRescale()*(ipX0QR+q.K1XSumQ())
	lowDistance = estDistance - bin.FError()*gError
	return
}

// SplitSingleFullDist computes the extra-bit estimate and lower bound.
func SplitSingleFullDist(binData, exData []byte, ipFunc ExcodeIPFunc, q *SplitSingleQuery, paddedDim, exBits int, gAdd, gError float32) (estDistance, lowDistance, ipX0QR float32) {
	if q == nil || paddedDim != q.paddedDim || exBits != q.exBits {
		panic("rabitq: full estimator configuration does not match query")
	}
	if exBits == 0 {
		ipX0QR, estDistance, lowDistance = SplitSingleEstDist(binData, q, paddedDim, gAdd, gError)
		return
	}
	bin := NewBinDataMap(binData, paddedDim)
	ex := NewExDataMap(exData, paddedDim, exBits)
	ipX0QR = simd.MaskIPX0Q(q.rotatedQuery, bin.decodedBinCode())
	estDistance = ex.FAddEx() + gAdd + ex.FRescaleEx()*(float32(uint32(1)<<uint(exBits))*ipX0QR+ipFunc(q.rotatedQuery, ex.ExCode())+q.KBXSumQ())
	lowDistance = estDistance - bin.FError()*gError/float32(uint32(1)<<uint(exBits))
	return
}
