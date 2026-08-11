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

package xvec

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestQueryParamsDefaultsValidate(t *testing.T) {
	params := []QueryParams{
		NewFlatQueryParams(), NewHNSWQueryParams(), NewHNSWRaBitQQueryParams(),
		NewIVFQueryParams(), NewDiskANNQueryParams(), NewVamanaQueryParams(),
		NewFTSQueryParams(),
	}
	for _, params := range params {
		{
			err := params.Validate()
			assert.NoError(t, err)
		}
	}
}

func TestQueryParamsRejectInvalidValues(t *testing.T) {
	hnsw := NewHNSWQueryParams()
	hnsw.EF = 0
	hnswLarge := NewHNSWQueryParams()
	hnswLarge.EF = MaxGraphEFSearch + 1
	rabitqLarge := NewHNSWRaBitQQueryParams()
	rabitqLarge.EF = MaxGraphEFSearch + 1
	ivf := NewIVFQueryParams()
	ivf.NProbe = 0
	flat := NewFlatQueryParams()
	flat.Radius = -1
	diskANN := NewDiskANNQueryParams()
	diskANN.ListSize = 0
	diskANNLarge := NewDiskANNQueryParams()
	diskANNLarge.ListSize = int(uint64(math.MaxUint32) + 1)
	vamana := NewVamanaQueryParams()
	vamana.EFSearch = 0
	vamanaLarge := NewVamanaQueryParams()
	vamanaLarge.EFSearch = MaxGraphEFSearch + 1
	nan := NewFlatQueryParams()
	nan.ScaleFactor = float32(math.NaN())

	params := []QueryParams{
		hnsw, hnswLarge, rabitqLarge, ivf, flat, diskANN, diskANNLarge, vamana, vamanaLarge, nan,
		FTSQueryParams{DefaultOperator: "XOR"},
	}
	for _, params := range params {
		{
			err := params.Validate()
			assert.ErrorIs(t, err, ErrInvalidArgument)
		}
	}
}
