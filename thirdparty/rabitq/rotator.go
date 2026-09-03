// Copyright 2026-present the xvec project
// Licensed under the Apache License, Version 2.0.

package rabitq

import (
	cryptorand "crypto/rand"
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

// RotatorType identifies an upstream-compatible rotator implementation.
type RotatorType uint8

const (
	RotatorMatrix RotatorType = iota
	RotatorFHTKac
)

// ChooseRotator creates a rotator using cryptographic randomness.
func ChooseRotator(dim, padded int, typ RotatorType) (Rotator, error) {
	return NewRotator(dim, padded, typ, cryptorand.Reader)
}

// NewRotator creates either supported rotator with caller-supplied randomness.
// A zero padded dimension selects the upstream padding requirement.
func NewRotator(dim, padded int, typ RotatorType, random io.Reader) (Rotator, error) {
	if padded == 0 {
		var err error
		padded, err = PaddedDimension(dim, typ)
		if err != nil {
			return nil, err
		}
	}
	switch typ {
	case RotatorFHTKac:
		return NewFHTKacRotatorWithReader(dim, padded, random)
	case RotatorMatrix:
		return NewMatrixRotator(dim, padded, random)
	default:
		return nil, ErrInvalidArgument
	}
}

// Rotator maps an input dimension into a padded orthogonal space.
type Rotator interface {
	Type() RotatorType
	Dimension() int
	PaddedDimension() int
	BinarySize() int
	DumpBytes() int
	Rotate([]float32) ([]float32, error)
	MarshalBinary() ([]byte, error)
	Save(io.Writer) error
}

// PaddedDimension returns the storage dimension used by the selected rotator.
func PaddedDimension(dim int, typ RotatorType) (int, error) {
	if dim < 1 || dim > 4095 {
		return 0, ErrInvalidArgument
	}
	switch typ {
	case RotatorMatrix:
		return dim, nil
	case RotatorFHTKac:
		if dim < 64 || dim > 4095 {
			return 0, ErrInvalidArgument
		}
		return (dim + 63) &^ 63, nil
	default:
		return 0, ErrInvalidArgument
	}
}

// LoadRotator constructs a rotator from the raw payload written by RaBitQ-Library.
func LoadRotator(dim, padded int, typ RotatorType, state []byte) (Rotator, error) {
	switch typ {
	case RotatorFHTKac:
		return NewFHTKacRotatorFromState(dim, padded, state)
	case RotatorMatrix:
		return NewMatrixRotatorFromState(dim, padded, state)
	default:
		return nil, ErrInvalidArgument
	}
}

// FHTKacRotator implements the four-round random-sign FHT/Kac transform.
type FHTKacRotator struct {
	dim, padded, truncated int
	signs                  []byte
}

func NewFHTKacRotator(dim, padded int) (*FHTKacRotator, error) {
	return NewFHTKacRotatorWithReader(dim, padded, cryptorand.Reader)
}
func NewFHTKacRotatorWithReader(dim, padded int, random io.Reader) (*FHTKacRotator, error) {
	if random == nil {
		return nil, ErrInvalidArgument
	}
	if err := validateFHTDims(dim, padded); err != nil {
		return nil, err
	}
	s := make([]byte, 4*padded/8)
	if _, err := io.ReadFull(random, s); err != nil {
		return nil, fmt.Errorf("rabitq: read FHT signs: %w", err)
	}
	return NewFHTKacRotatorFromState(dim, padded, s)
}
func validateFHTDims(dim, padded int) error {
	if dim < 64 || dim > 4095 || padded < dim || padded%64 != 0 {
		return ErrInvalidArgument
	}
	return nil
}
func NewFHTKacRotatorFromState(dim, padded int, state []byte) (*FHTKacRotator, error) {
	if err := validateFHTDims(dim, padded); err != nil {
		return nil, err
	}
	if len(state) != 4*padded/8 {
		return nil, ErrInvalidLayout
	}
	tr := 1
	for tr <= dim/2 {
		tr *= 2
	}
	return &FHTKacRotator{dim: dim, padded: padded, truncated: tr, signs: append([]byte(nil), state...)}, nil
}
func (r *FHTKacRotator) Dimension() int {
	if r == nil {
		return 0
	}
	return r.dim
}
func (r *FHTKacRotator) PaddedDimension() int {
	if r == nil {
		return 0
	}
	return r.padded
}
func (r *FHTKacRotator) State() []byte {
	if r == nil {
		return nil
	}
	return append([]byte(nil), r.signs...)
}
func (r *FHTKacRotator) MarshalBinary() ([]byte, error) {
	if r == nil {
		return nil, ErrInvalidArgument
	}
	return r.State(), nil
}
func (r *FHTKacRotator) Type() RotatorType { return RotatorFHTKac }
func (r *FHTKacRotator) BinarySize() int {
	if r == nil {
		return 0
	}
	return len(r.signs)
}
func (r *FHTKacRotator) DumpBytes() int { return r.BinarySize() }
func (r *FHTKacRotator) Save(dst io.Writer) error {
	if r == nil || dst == nil {
		return ErrInvalidArgument
	}
	return writeState(dst, r.signs)
}
func (r *FHTKacRotator) Rotate(src []float32) ([]float32, error) {
	if r == nil || len(src) != r.dim {
		return nil, ErrDimensionMismatch
	}
	for _, v := range src {
		if !isFinite(v) {
			return nil, ErrNonFinite
		}
	}
	dst := make([]float32, r.padded)
	copy(dst, src)
	fac := float32(1 / math.Sqrt(float64(r.truncated)))
	stride := r.padded / 8
	if r.truncated == r.padded {
		for round := 0; round < 4; round++ {
			flipSigns(dst, r.signs[round*stride:])
			fht(dst)
			scale(dst, fac)
		}
		return dst, nil
	}
	start := r.padded - r.truncated
	for round := 0; round < 4; round++ {
		flipSigns(dst, r.signs[round*stride:])
		window := dst[:r.truncated]
		if round&1 != 0 {
			window = dst[start:]
		}
		fht(window)
		scale(window, fac)
		kacWalk(dst)
	}
	scale(dst, .25)
	return dst, nil
}
func flipSigns(data []float32, signs []byte) {
	for i := range data {
		if signs[i/8]&(1<<uint(i%8)) != 0 {
			data[i] = -data[i]
		}
	}
}
func fht(data []float32) {
	for w := 1; w < len(data); w <<= 1 {
		for b := 0; b < len(data); b += 2 * w {
			for i := b; i < b+w; i++ {
				a, c := data[i], data[i+w]
				data[i] = a + c
				data[i+w] = a - c
			}
		}
	}
}
func scale(data []float32, f float32) {
	for i := range data {
		data[i] *= f
	}
}
func kacWalk(data []float32) {
	half := len(data) / 2
	base := len(data) % 2
	off := base + half
	for i := 0; i < half; i++ {
		a, b := data[i], data[i+off]
		data[i] = a + b
		data[i+off] = a - b
	}
	if base != 0 {
		data[half] *= float32(math.Sqrt2)
	}
}

// MatrixRotator multiplies row vectors by a matrix with orthonormal rows.
type MatrixRotator struct {
	dim, padded int
	matrix      []float32
}

// NewMatrixRotator generates deterministic Gaussian rows from random and orthonormalizes them.
func NewMatrixRotator(dim, padded int, random io.Reader) (*MatrixRotator, error) {
	if random == nil || !validMatrixDims(dim, padded) {
		return nil, ErrInvalidArgument
	}
	raw := make([]float32, dim*padded)
	buf := make([]byte, 8)
	for i := range raw {
		if _, err := io.ReadFull(random, buf); err != nil {
			return nil, fmt.Errorf("rabitq: read matrix randomness: %w", err)
		}
		u1 := (float64(binary.LittleEndian.Uint32(buf[:4])) + .5) / (1 << 32)
		u2 := (float64(binary.LittleEndian.Uint32(buf[4:])) + .5) / (1 << 32)
		raw[i] = float32(math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2))
	}
	for row := 0; row < dim; row++ {
		r := raw[row*padded : (row+1)*padded]
		for prev := 0; prev < row; prev++ {
			p := raw[prev*padded : (prev+1)*padded]
			var dot float32
			for i := range r {
				dot += r[i] * p[i]
			}
			for i := range r {
				r[i] -= dot * p[i]
			}
		}
		n := float32(math.Sqrt(float64(normSquared(r))))
		if n == 0 {
			return nil, ErrInvalidArgument
		}
		scale(r, 1/n)
	}
	return NewMatrixRotatorFromMatrix(dim, padded, raw)
}
func NewMatrixRotatorFromMatrix(dim, padded int, matrix []float32) (*MatrixRotator, error) {
	if !validMatrixDims(dim, padded) || len(matrix) != dim*padded {
		return nil, ErrInvalidArgument
	}
	m := append([]float32(nil), matrix...)
	const tol = .0002
	for r := 0; r < dim; r++ {
		row := m[r*padded : (r+1)*padded]
		for _, v := range row {
			if !isFinite(v) {
				return nil, ErrNonFinite
			}
		}
		for p := 0; p <= r; p++ {
			other := m[p*padded : (p+1)*padded]
			var d float32
			for i := range row {
				d += row[i] * other[i]
			}
			want := float32(0)
			if p == r {
				want = 1
			}
			if float32(math.Abs(float64(d-want))) > tol {
				return nil, fmt.Errorf("%w: matrix rows are not orthonormal", ErrInvalidArgument)
			}
		}
	}
	return &MatrixRotator{dim: dim, padded: padded, matrix: m}, nil
}
func NewMatrixRotatorFromState(dim, padded int, state []byte) (*MatrixRotator, error) {
	if !validMatrixDims(dim, padded) || len(state) != dim*padded*4 {
		return nil, ErrInvalidLayout
	}
	m := make([]float32, dim*padded)
	for i := range m {
		m[i] = math.Float32frombits(binary.LittleEndian.Uint32(state[i*4:]))
		if !isFinite(m[i]) {
			return nil, ErrNonFinite
		}
	}
	return &MatrixRotator{dim: dim, padded: padded, matrix: m}, nil
}
func (r *MatrixRotator) Type() RotatorType { return RotatorMatrix }
func (r *MatrixRotator) Dimension() int {
	if r == nil {
		return 0
	}
	return r.dim
}
func (r *MatrixRotator) PaddedDimension() int {
	if r == nil {
		return 0
	}
	return r.padded
}
func (r *MatrixRotator) Rotate(src []float32) ([]float32, error) {
	if r == nil || len(src) != r.dim {
		return nil, ErrDimensionMismatch
	}
	for _, v := range src {
		if !isFinite(v) {
			return nil, ErrNonFinite
		}
	}
	out := make([]float32, r.padded)
	for i, v := range src {
		row := r.matrix[i*r.padded : (i+1)*r.padded]
		for j := range out {
			out[j] += v * row[j]
		}
	}
	return out, nil
}
func (r *MatrixRotator) MarshalBinary() ([]byte, error) {
	if r == nil {
		return nil, ErrInvalidArgument
	}
	out := make([]byte, len(r.matrix)*4)
	for i, v := range r.matrix {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out, nil
}
func (r *MatrixRotator) BinarySize() int {
	if r == nil {
		return 0
	}
	return len(r.matrix) * 4
}
func (r *MatrixRotator) DumpBytes() int { return r.BinarySize() }
func (r *MatrixRotator) Save(dst io.Writer) error {
	if r == nil || dst == nil {
		return ErrInvalidArgument
	}
	state, err := r.MarshalBinary()
	if err != nil {
		return err
	}
	return writeState(dst, state)
}
func (r *MatrixRotator) State() []byte { out, _ := r.MarshalBinary(); return out }

func validMatrixDims(dim, padded int) bool {
	return dim >= 1 && dim <= 4096 && padded >= dim && padded <= 4096
}

func writeState(dst io.Writer, state []byte) error {
	written, err := dst.Write(state)
	if err == nil && written != len(state) {
		return io.ErrShortWrite
	}
	return err
}

var _ Rotator = (*FHTKacRotator)(nil)
var _ Rotator = (*MatrixRotator)(nil)
