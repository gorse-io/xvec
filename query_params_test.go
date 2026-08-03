// Copyright 2026-present the zvec-go project
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

package zvec

import (
	"errors"
	"math"
	"testing"
)

func TestQueryParamsDefaultsValidate(t *testing.T) {
	params := []QueryParams{
		NewFlatQueryParams(), NewHNSWQueryParams(), NewHNSWRaBitQQueryParams(),
		NewIVFQueryParams(), NewDiskANNQueryParams(), NewVamanaQueryParams(),
		NewFTSQueryParams(),
	}
	for _, params := range params {
		if err := params.Validate(); err != nil {
			t.Errorf("%T defaults: %v", params, err)
		}
	}
}

func TestQueryParamsRejectInvalidValues(t *testing.T) {
	hnsw := NewHNSWQueryParams()
	hnsw.EF = 0
	ivf := NewIVFQueryParams()
	ivf.NProbe = 0
	flat := NewFlatQueryParams()
	flat.Radius = -1
	diskANN := NewDiskANNQueryParams()
	diskANN.ListSize = 0
	vamana := NewVamanaQueryParams()
	vamana.EFSearch = 0
	nan := NewFlatQueryParams()
	nan.ScaleFactor = float32(math.NaN())

	params := []QueryParams{
		hnsw, ivf, flat, diskANN, vamana, nan, FTSQueryParams{DefaultOperator: "XOR"},
	}
	for _, params := range params {
		if err := params.Validate(); !errors.Is(err, ErrInvalidArgument) {
			t.Errorf("%T invalid error = %v", params, err)
		}
	}
}
