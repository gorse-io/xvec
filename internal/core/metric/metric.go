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

// Package metric provides dense-vector score computation and ordering.
package metric

import (
	"errors"

	"github.com/gorse-io/xvec/internal/ailego/math"
)

// Metric selects score computation and ordering for vector search.
type Metric uint8

const (
	L2 Metric = iota + 1
	IP
	Cosine
	MIPSL2
)

// Distance selects the unchecked dense-distance kernel for this metric.
func (m Metric) Distance() (mathutil.DenseDistance, error) {
	switch m {
	case L2:
		return mathutil.L2Squared, nil
	case IP:
		return mathutil.InnerProduct, nil
	case Cosine:
		return mathutil.CosineDistance, nil
	case MIPSL2:
		return mathutil.MIPSL2Squared, nil
	default:
		return nil, errors.New("core: invalid metric")
	}
}

// Better reports whether left should rank before right.
func (m Metric) Better(left, right float32) bool {
	if m == IP {
		return left > right
	}
	return left < right
}

// Valid reports whether m identifies a supported metric.
func (m Metric) Valid() bool {
	return m >= L2 && m <= MIPSL2
}
