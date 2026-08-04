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
	"math"
	"strings"
)

const (
	DefaultIVFNProbe            = 10
	DefaultRefinerScaleFactor   = 10
	DefaultDiskANNQueryListSize = 300
)

// QueryParams is the common, sealed interface for index-specific search
// controls.
type QueryParams interface {
	IndexType() IndexType
	Validate() error
	cloneQueryParams() QueryParams
}

// QueryOptions contains controls shared by vector query parameter types. A
// zero Radius disables radius filtering.
type QueryOptions struct {
	Radius     float32
	Linear     bool
	UseRefiner bool
}

// FlatQueryParams configures exact vector search.
type FlatQueryParams struct {
	QueryOptions
	ScaleFactor float32
}

func NewFlatQueryParams() FlatQueryParams {
	return FlatQueryParams{ScaleFactor: DefaultRefinerScaleFactor}
}

func (FlatQueryParams) IndexType() IndexType { return IndexTypeFlat }
func (p FlatQueryParams) Validate() error {
	if err := p.QueryOptions.validate("validate Flat query params"); err != nil {
		return err
	}
	return validateScaleFactor("validate Flat query params", p.ScaleFactor)
}
func (p FlatQueryParams) cloneQueryParams() QueryParams { return p }

// HNSWQueryParams configures HNSW graph traversal.
type HNSWQueryParams struct {
	QueryOptions
	EF             int
	PrefetchOffset uint32
	PrefetchLines  uint32
}

func NewHNSWQueryParams() HNSWQueryParams {
	return HNSWQueryParams{
		EF:             DefaultHNSWEFSearch,
		PrefetchOffset: DefaultPrefetchOffset,
		PrefetchLines:  DefaultPrefetchLines,
	}
}

func (HNSWQueryParams) IndexType() IndexType { return IndexTypeHNSW }
func (p HNSWQueryParams) Validate() error {
	if err := p.QueryOptions.validate("validate HNSW query params"); err != nil {
		return err
	}
	if p.EF <= 0 || p.EF > MaxGraphEFSearch {
		return invalidArgument("validate HNSW query params", "EF must be in [1, %d]", MaxGraphEFSearch)
	}
	return nil
}
func (p HNSWQueryParams) cloneQueryParams() QueryParams { return p }

// HNSWRaBitQQueryParams configures HNSW traversal over RaBitQ codes.
type HNSWRaBitQQueryParams struct {
	QueryOptions
	EF int
}

func NewHNSWRaBitQQueryParams() HNSWRaBitQQueryParams {
	return HNSWRaBitQQueryParams{EF: DefaultHNSWEFSearch}
}

func (HNSWRaBitQQueryParams) IndexType() IndexType { return IndexTypeHNSWRaBitQ }
func (p HNSWRaBitQQueryParams) Validate() error {
	if err := p.QueryOptions.validate("validate HNSW RaBitQ query params"); err != nil {
		return err
	}
	if p.EF <= 0 || p.EF > MaxGraphEFSearch {
		return invalidArgument("validate HNSW RaBitQ query params", "EF must be in [1, %d]", MaxGraphEFSearch)
	}
	return nil
}
func (p HNSWRaBitQQueryParams) cloneQueryParams() QueryParams { return p }

// IVFQueryParams configures inverted-list probing and optional refinement.
type IVFQueryParams struct {
	QueryOptions
	NProbe      int
	ScaleFactor float32
}

func NewIVFQueryParams() IVFQueryParams {
	return IVFQueryParams{NProbe: DefaultIVFNProbe, ScaleFactor: DefaultRefinerScaleFactor}
}

func (IVFQueryParams) IndexType() IndexType { return IndexTypeIVF }
func (p IVFQueryParams) Validate() error {
	if err := p.QueryOptions.validate("validate IVF query params"); err != nil {
		return err
	}
	if p.NProbe <= 0 {
		return invalidArgument("validate IVF query params", "NProbe must be positive")
	}
	return validateScaleFactor("validate IVF query params", p.ScaleFactor)
}
func (p IVFQueryParams) cloneQueryParams() QueryParams { return p }

// DiskANNQueryParams configures the disk graph search frontier.
type DiskANNQueryParams struct {
	QueryOptions
	ListSize int
}

func NewDiskANNQueryParams() DiskANNQueryParams {
	return DiskANNQueryParams{ListSize: DefaultDiskANNQueryListSize}
}

func (DiskANNQueryParams) IndexType() IndexType { return IndexTypeDiskANN }
func (p DiskANNQueryParams) Validate() error {
	if err := p.QueryOptions.validate("validate DiskANN query params"); err != nil {
		return err
	}
	if p.ListSize <= 0 {
		return invalidArgument("validate DiskANN query params", "ListSize must be positive")
	}
	return nil
}
func (p DiskANNQueryParams) cloneQueryParams() QueryParams { return p }

// VamanaQueryParams configures Vamana graph traversal.
type VamanaQueryParams struct {
	QueryOptions
	EFSearch       int
	PrefetchOffset uint32
	PrefetchLines  uint32
}

func NewVamanaQueryParams() VamanaQueryParams {
	return VamanaQueryParams{
		EFSearch:       DefaultVamanaEFSearch,
		PrefetchOffset: DefaultPrefetchOffset,
		PrefetchLines:  DefaultPrefetchLines,
	}
}

func (VamanaQueryParams) IndexType() IndexType { return IndexTypeVamana }
func (p VamanaQueryParams) Validate() error {
	if err := p.QueryOptions.validate("validate Vamana query params"); err != nil {
		return err
	}
	if p.EFSearch <= 0 || p.EFSearch > MaxGraphEFSearch {
		return invalidArgument("validate Vamana query params", "EFSearch must be in [1, %d]", MaxGraphEFSearch)
	}
	return nil
}
func (p VamanaQueryParams) cloneQueryParams() QueryParams { return p }

// FTSQueryParams configures parsing of adjacent bare terms. DefaultOperator is
// case-insensitive and may be empty, OR, or AND; empty means OR.
type FTSQueryParams struct {
	DefaultOperator string
}

func NewFTSQueryParams() FTSQueryParams { return FTSQueryParams{} }

func (FTSQueryParams) IndexType() IndexType { return IndexTypeFTS }
func (p FTSQueryParams) Validate() error {
	switch strings.ToUpper(p.DefaultOperator) {
	case "", "OR", "AND":
		return nil
	default:
		return invalidArgument("validate FTS query params", "DefaultOperator must be OR or AND")
	}
}
func (p FTSQueryParams) cloneQueryParams() QueryParams { return p }

func (o QueryOptions) validate(op string) error {
	radius := float64(o.Radius)
	if math.IsNaN(radius) || math.IsInf(radius, 0) || o.Radius < 0 {
		return invalidArgument(op, "Radius must be finite and non-negative")
	}
	return nil
}

func validateScaleFactor(op string, scale float32) error {
	value := float64(scale)
	if math.IsNaN(value) || math.IsInf(value, 0) || scale <= 0 {
		return invalidArgument(op, "ScaleFactor must be finite and positive")
	}
	return nil
}
