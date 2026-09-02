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

import "github.com/gorse-io/xvec/internal/ailego/math_batch"

func denseDistances2(metric Metric, query, first, second []float32) (float32, float32) {
	if metric == MetricIP {
		return mathbatch.InnerProducts2(query, first, second)
	}
	return mathbatch.SquaredEuclideanDistances2(query, first, second)
}

func denseDistances4(metric Metric, query, first, second, third, fourth []float32) (float32, float32, float32, float32) {
	if metric == MetricIP {
		return mathbatch.InnerProducts4(query, first, second, third, fourth)
	}
	return mathbatch.SquaredEuclideanDistances4(query, first, second, third, fourth)
}
