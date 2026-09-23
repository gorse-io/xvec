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
	"fmt"
	"testing"
)

func BenchmarkIntegerCodeDot(b *testing.B) {
	for _, dimension := range []int{128, 768, 1536} {
		b.Run(fmt.Sprintf("INT8/%d", dimension), func(b *testing.B) {
			left := make([]float32, dimension)
			right := make([]float32, dimension)
			for i := range left {
				left[i] = float32(i%251) - 125
				right[i] = float32(i%239) - 119
			}
			l, err := QuantizeVector(QuantizationInt8, left)
			if err != nil {
				b.Fatal(err)
			}
			r, err := QuantizeVector(QuantizationInt8, right)
			if err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.SetBytes(int64(2 * dimension))
			for b.Loop() {
				integerCodeDot(l, r)
			}
		})
	}
}
