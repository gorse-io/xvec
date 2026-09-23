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

package mathbatch

import mathutil "github.com/gorse-io/xvec/internal/ailego/math"

var innerProductsInt8Kernel4 = innerProductsInt8Scalar4

// InnerProductsInt8 computes exact signed INT8 dot products for one query and
// many candidates. Codes are stored as bytes. Every candidate must have at
// least len(query) bytes; output must have room for all candidates.
func InnerProductsInt8(query []byte, candidates [][]byte, output []int64) {
	i := 0
	for ; i+4 <= len(candidates); i += 4 {
		innerProductsInt8Kernel4(query, candidates[i], candidates[i+1], candidates[i+2], candidates[i+3], output[i:i+4])
	}
	for ; i < len(candidates); i++ {
		output[i] = mathutil.InnerProductInt8(query, candidates[i][:len(query)])
	}
}

func innerProductsInt8Scalar4(query, first, second, third, fourth []byte, output []int64) {
	var a, b, c, d int64
	for i, code := range query {
		q := int64(int8(code))
		a += q * int64(int8(first[i]))
		b += q * int64(int8(second[i]))
		c += q * int64(int8(third[i]))
		d += q * int64(int8(fourth[i]))
	}
	output[0], output[1], output[2], output[3] = a, b, c, d
}
