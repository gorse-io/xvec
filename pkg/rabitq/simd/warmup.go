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

package simd

import (
	"math/bits"
	"unsafe"
)

var warmupIPX0Q512 = warmupIPX0Q512Scalar

// WarmupIPX0Q512 computes the inner-product warmup term for binary data and a
// bit-transposed quantized query. Padded dimension must be a positive multiple
// of 64. Query bits must be between 1 and 8. Data must contain at least
// paddedDimension/64 elements, and query must contain at least
// paddedDimension/64*queryBits elements.
func WarmupIPX0Q512(data, query []uint64, delta, vl float32, paddedDimension, queryBits int) float32 {
	return warmupIPX0Q512(
		unsafe.Pointer(&data[0]),
		unsafe.Pointer(&query[0]),
		delta,
		vl,
		int64(paddedDimension),
		int64(queryBits),
	)
}

func warmupIPX0Q512Scalar(data, query unsafe.Pointer, delta, vl float32, paddedDimension, queryBits int64) float32 {
	chunks := paddedDimension / 64
	dataValues := unsafe.Slice((*uint64)(data), chunks)
	queryValues := unsafe.Slice((*uint64)(query), chunks*queryBits)
	var ip, ppc uint64
	var queryBase int64
	for block := int64(0); block < chunks; {
		blockChunks := min(int64(8), chunks-block)
		for chunk := int64(0); chunk < blockChunks; chunk++ {
			x := dataValues[block+chunk]
			ppc += uint64(bits.OnesCount64(x))
			for bit := int64(0); bit < queryBits; bit++ {
				y := queryValues[queryBase+bit*blockChunks+chunk]
				ip += uint64(bits.OnesCount64(x&y)) << uint(bit)
			}
		}
		block += blockChunks
		queryBase += blockChunks * queryBits
	}
	return delta*float32(ip) + vl*float32(ppc)
}
