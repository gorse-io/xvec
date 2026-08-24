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
	"context"
	cryptorand "crypto/rand"
	"errors"
	"fmt"
	"io"
	"math"
	"slices"

	"github.com/gorse-io/xvec/internal/ailego/math"
	"github.com/gorse-io/xvec/internal/ailego/parallel"
)

var (
	ErrInvalidRotator = errors.New("core: invalid vector rotator")
	ErrInvalidSigns   = errors.New("core: invalid rotation sign state")
)

// MaxRotationDimension matches the public dense-vector dimension ceiling.
const MaxRotationDimension = 65535

// DenseReformer transforms vectors into and out of an index representation.
// Implementations must be safe for concurrent calls.
type DenseReformer interface {
	Dimension() int
	Transform(vector []float32) ([]float32, error)
	Revert(vector []float32) ([]float32, error)
}

// Rotator is a dimension-preserving orthogonal dense-vector transform.
type Rotator interface {
	Dimension() int
	Rotate(vector []float32) ([]float32, error)
	Unrotate(vector []float32) ([]float32, error)
}

// FHTRotator implements the baseline four-round random-sign FHT/Kac rotation.
// Its immutable sign state makes concurrent transforms safe.
type FHTRotator struct {
	dimension       int
	truncated       int
	bytesPerRound   int
	inverseSqrtSize float32
	signs           []byte
}

// NewFHTRotator creates a random rotator using crypto/rand. Persist Signs with
// an index so queries after reopen use the identical transform.
func NewFHTRotator(dimension int) (*FHTRotator, error) {
	return NewFHTRotatorWithReader(dimension, cryptorand.Reader)
}

// NewFHTRotatorWithReader creates a rotator from caller-provided randomness.
func NewFHTRotatorWithReader(dimension int, random io.Reader) (*FHTRotator, error) {
	if dimension <= 0 || dimension > MaxRotationDimension {
		return nil, fmt.Errorf("%w: dimension %d", ErrInvalidRotator, dimension)
	}
	if random == nil {
		return nil, fmt.Errorf("%w: nil randomness source", ErrInvalidRotator)
	}
	signs := make([]byte, 4*((dimension+7)/8))
	if _, err := io.ReadFull(random, signs); err != nil {
		return nil, fmt.Errorf("core: generate FHT rotation signs: %w", err)
	}
	return NewFHTRotatorFromSigns(dimension, signs)
}

// NewFHTRotatorFromSigns restores a rotator from its exact four-round sign
// state. Extra and missing bytes are rejected to make persisted state
// canonical.
func NewFHTRotatorFromSigns(dimension int, signs []byte) (*FHTRotator, error) {
	if dimension <= 0 || dimension > MaxRotationDimension {
		return nil, fmt.Errorf("%w: dimension %d", ErrInvalidRotator, dimension)
	}
	bytesPerRound := (dimension + 7) / 8
	want := 4 * bytesPerRound
	if len(signs) != want {
		return nil, fmt.Errorf("%w: got %d bytes, want %d", ErrInvalidSigns, len(signs), want)
	}
	truncated := floorPowerOfTwo(dimension)
	return &FHTRotator{
		dimension:       dimension,
		truncated:       truncated,
		bytesPerRound:   bytesPerRound,
		inverseSqrtSize: 1 / float32(math.Sqrt(float64(float32(truncated)))),
		signs:           slices.Clone(signs),
	}, nil
}

// Dimension returns the unchanged input and output dimension.
func (r *FHTRotator) Dimension() int {
	if r == nil {
		return 0
	}
	return r.dimension
}

// Signs returns an independent copy of the canonical sign state.
func (r *FHTRotator) Signs() []byte {
	if r == nil {
		return nil
	}
	return slices.Clone(r.signs)
}

// Rotate applies four random-sign FHT rounds and returns a new vector.
func (r *FHTRotator) Rotate(vector []float32) ([]float32, error) {
	if err := r.validateVector(vector); err != nil {
		return nil, err
	}
	result := slices.Clone(vector)
	if r.truncated == r.dimension {
		for round := 0; round < 4; round++ {
			roundSigns := r.signs[round*r.bytesPerRound : (round+1)*r.bytesPerRound]
			_ = mathutil.FHTFlipSigns(roundSigns, result)
			_ = mathutil.FHTInPlace(result)
			mathutil.ScaleFloat32(result, r.inverseSqrtSize)
		}
		return validateTransformedVector(result)
	}

	start := r.dimension - r.truncated
	for round := 0; round < 4; round++ {
		roundSigns := r.signs[round*r.bytesPerRound : (round+1)*r.bytesPerRound]
		_ = mathutil.FHTFlipSigns(roundSigns, result)
		window := result[:r.truncated]
		if round%2 != 0 {
			window = result[start:]
		}
		_ = mathutil.FHTInPlace(window)
		mathutil.ScaleFloat32(window, r.inverseSqrtSize)
		_ = mathutil.FHTKacWalk(result)
	}
	mathutil.ScaleFloat32(result, .25)
	return validateTransformedVector(result)
}

// Unrotate reverses Rotate and returns a new vector.
func (r *FHTRotator) Unrotate(vector []float32) ([]float32, error) {
	if err := r.validateVector(vector); err != nil {
		return nil, err
	}
	result := slices.Clone(vector)
	if r.truncated == r.dimension {
		for round := 3; round >= 0; round-- {
			_ = mathutil.FHTInPlace(result)
			mathutil.ScaleFloat32(result, r.inverseSqrtSize)
			roundSigns := r.signs[round*r.bytesPerRound : (round+1)*r.bytesPerRound]
			_ = mathutil.FHTFlipSigns(roundSigns, result)
		}
		return validateTransformedVector(result)
	}

	mathutil.ScaleFloat32(result, 4)
	start := r.dimension - r.truncated
	for round := 3; round >= 0; round-- {
		_ = mathutil.FHTInverseKacWalk(result)
		window := result[:r.truncated]
		if round%2 != 0 {
			window = result[start:]
		}
		_ = mathutil.FHTInPlace(window)
		mathutil.ScaleFloat32(window, r.inverseSqrtSize)
		roundSigns := r.signs[round*r.bytesPerRound : (round+1)*r.bytesPerRound]
		_ = mathutil.FHTFlipSigns(roundSigns, result)
	}
	return validateTransformedVector(result)
}

// RotateBatch rotates vectors concurrently while preserving their order.
func (r *FHTRotator) RotateBatch(ctx context.Context, vectors [][]float32, workers int) ([][]float32, error) {
	if ctx == nil {
		return nil, errors.New("core: nil rotation context")
	}
	if r == nil {
		return nil, ErrInvalidRotator
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	result := make([][]float32, len(vectors))
	err := parallel.ParallelFor(ctx, len(vectors), workers, func(ctx context.Context, index int) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		rotated, err := r.Rotate(vectors[index])
		if err != nil {
			return fmt.Errorf("core: rotate vector %d: %w", index, err)
		}
		result[index] = rotated
		return nil
	})
	if err != nil {
		return nil, err
	}
	return result, nil
}

func (r *FHTRotator) validateVector(vector []float32) error {
	if r == nil || r.dimension <= 0 || len(r.signs) != 4*r.bytesPerRound {
		return ErrInvalidRotator
	}
	if len(vector) != r.dimension {
		return fmt.Errorf("%w: got %d, want %d", mathutil.ErrDimensionMismatch, len(vector), r.dimension)
	}
	for _, value := range vector {
		if !finiteFloat32(value) {
			return mathutil.ErrNonFiniteVector
		}
	}
	return nil
}

func validateTransformedVector(vector []float32) ([]float32, error) {
	for _, value := range vector {
		if !finiteFloat32(value) {
			return nil, mathutil.ErrNonFiniteVector
		}
	}
	return vector, nil
}

func floorPowerOfTwo(value int) int {
	result := 1
	for result <= value/2 {
		result *= 2
	}
	return result
}

// RotationReformer adapts any Rotator to the general DenseReformer contract.
type RotationReformer struct{ rotator Rotator }

// NewRotationReformer constructs a reversible rotation preprocessor.
func NewRotationReformer(rotator Rotator) (*RotationReformer, error) {
	if rotator == nil || rotator.Dimension() <= 0 {
		return nil, ErrInvalidRotator
	}
	return &RotationReformer{rotator: rotator}, nil
}

func (r *RotationReformer) Dimension() int {
	if r == nil || r.rotator == nil {
		return 0
	}
	return r.rotator.Dimension()
}

// Transform rotates a vector into index space.
func (r *RotationReformer) Transform(vector []float32) ([]float32, error) {
	if r == nil || r.rotator == nil {
		return nil, ErrInvalidRotator
	}
	return r.rotator.Rotate(vector)
}

// Revert inverse-rotates a vector into original space.
func (r *RotationReformer) Revert(vector []float32) ([]float32, error) {
	if r == nil || r.rotator == nil {
		return nil, ErrInvalidRotator
	}
	return r.rotator.Unrotate(vector)
}

var _ Rotator = (*FHTRotator)(nil)
var _ DenseReformer = (*RotationReformer)(nil)
