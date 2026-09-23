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

package mathutil

var innerProductInt8Kernel = innerProductInt8Scalar

// InnerProductInt8 computes the unchecked dot product of signed INT8 codes
// stored in byte slices. Callers must guarantee equal lengths. Accumulation
// uses int64 so dimensions whose dot product exceeds int32 remain exact.
func InnerProductInt8(left, right []byte) int64 {
	return innerProductInt8Kernel(left, right)
}

func innerProductInt8Scalar(left, right []byte) (sum int64) {
	for i, value := range left {
		sum += int64(int8(value)) * int64(int8(right[i]))
	}
	return sum
}
