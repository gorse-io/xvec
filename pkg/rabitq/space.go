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

	"github.com/gorse-io/xvec/pkg/rabitq/simd"
)

// MetricType selects the distance estimator algebra.
type MetricType uint8

const (
	MetricL2 MetricType = iota
	MetricIP
)

func validMetric(metric MetricType) bool { return metric == MetricL2 || metric == MetricIP }

// DotProduct computes a float32 inner product.
func DotProduct(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("rabitq: vector lengths differ")
	}
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// EuclideanSqr computes squared Euclidean distance using float32 arithmetic.
func EuclideanSqr(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("rabitq: vector lengths differ")
	}
	var sum float32
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum
}

// ExcodeIPFunc computes an inner product with a packed unsigned code.
type ExcodeIPFunc func(query []float32, compactCode []byte) float32

// SelectExcodeIPFunc selects the packed-code kernel for widths 0 through 8.
func SelectExcodeIPFunc(exBits int) (ExcodeIPFunc, error) {
	switch exBits {
	case 0:
		return func([]float32, []byte) float32 { return 0 }, nil
	case 1:
		return simd.IP16FxU1, nil
	case 2:
		return simd.IP64FxU2, nil
	case 3:
		return simd.IP64FxU3, nil
	case 4:
		return simd.IP16FxU4, nil
	case 5:
		return simd.IP64FxU5, nil
	case 6:
		return simd.IP64FxU6, nil
	case 7:
		return simd.IP64FxU7, nil
	case 8:
		return simd.IP16FxU8, nil
	default:
		return nil, fmt.Errorf("rabitq: unsupported extra-code width %d", exBits)
	}
}
