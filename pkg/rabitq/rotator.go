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
	"crypto/rand"
	"encoding/binary"
	"fmt"
	"math"
	"unsafe"

	"github.com/gorse-io/xvec/pkg/rabitq/simd"
)

// RotatorType selects the persisted rotation representation.
type RotatorType uint8

const (
	MatrixRotator RotatorType = iota
	FhtKacRotator
)

// Rotator transforms vectors and supports the byte persistence used by zvec.
type Rotator interface {
	Rotate(src, dst []float32) error
	Load(data []byte) error
	Save(data []byte) error
	DumpBytes() int
	Size() int
}

// ChooseRotator constructs a rotator with the upstream padding rules.
func ChooseRotator(dim int, typ RotatorType, paddedDim int) (Rotator, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("rabitq: dimension must be positive")
	}
	if paddedDim == 0 {
		if typ == FhtKacRotator {
			paddedDim = (dim + 63) &^ 63
		} else {
			paddedDim = dim
		}
	}
	if paddedDim < dim {
		return nil, fmt.Errorf("rabitq: padded dimension is smaller than dimension")
	}
	switch typ {
	case FhtKacRotator:
		if paddedDim%64 != 0 {
			return nil, fmt.Errorf("rabitq: FHT padded dimension must be a multiple of 64")
		}
		log := floorLog2(dim)
		if log < 6 || log > 11 {
			return nil, fmt.Errorf("rabitq: unsupported FHT dimension %d", dim)
		}
		r := &fhtKacRotator{dim: dim, paddedDim: paddedDim, truncDim: 1 << log, flip: make([]byte, paddedDim/2)}
		if _, err := rand.Read(r.flip); err != nil {
			return nil, err
		}
		return r, nil
	case MatrixRotator:
		matrix, err := randomOrthogonalMatrix(dim, paddedDim)
		if err != nil {
			return nil, err
		}
		return &matrixRotator{dim: dim, paddedDim: paddedDim, matrix: matrix}, nil
	default:
		return nil, fmt.Errorf("rabitq: invalid rotator type %d", typ)
	}
}

func randomOrthogonalMatrix(dim, paddedDim int) ([]float32, error) {
	rows := make([]float64, dim*paddedDim)
	random := make([]byte, paddedDim*8)
	for row := 0; row < dim; row++ {
		current := rows[row*paddedDim : (row+1)*paddedDim]
		for {
			if _, err := rand.Read(random); err != nil {
				return nil, err
			}
			for i := 0; i < paddedDim; i += 2 {
				u1 := (float64(binary.LittleEndian.Uint64(random[i*8:])>>11) + 1) / (1<<53 + 1)
				u2 := u1
				if i+1 < paddedDim {
					u2 = (float64(binary.LittleEndian.Uint64(random[(i+1)*8:])>>11) + 1) / (1<<53 + 1)
				}
				radius, angle := math.Sqrt(-2*math.Log(u1)), 2*math.Pi*u2
				current[i] = radius * math.Cos(angle)
				if i+1 < paddedDim {
					current[i+1] = radius * math.Sin(angle)
				}
			}
			for pass := 0; pass < 2; pass++ {
				for previous := 0; previous < row; previous++ {
					basis := rows[previous*paddedDim : (previous+1)*paddedDim]
					var dot float64
					for i := range current {
						dot += current[i] * basis[i]
					}
					for i := range current {
						current[i] -= dot * basis[i]
					}
				}
			}
			var normSquared float64
			for _, value := range current {
				normSquared += value * value
			}
			if normSquared > 1e-24 {
				inverseNorm := 1 / math.Sqrt(normSquared)
				for i := range current {
					current[i] *= inverseNorm
				}
				break
			}
		}
	}
	matrix := make([]float32, len(rows))
	for i := range rows {
		matrix[i] = float32(rows[i])
	}
	return matrix, nil
}

func floorLog2(v int) int {
	n := -1
	for v > 0 {
		v >>= 1
		n++
	}
	return n
}

type matrixRotator struct {
	dim, paddedDim int
	matrix         []float32
}

func (r *matrixRotator) Rotate(src, dst []float32) error {
	if len(src) < r.dim || len(dst) < r.paddedDim {
		return fmt.Errorf("rabitq: rotator buffer is too short")
	}
	src = src[:r.dim]
	dst = dst[:r.paddedDim]
	if float32SlicesOverlap(src, dst) {
		src = append([]float32(nil), src...)
	}
	for j := 0; j < r.paddedDim; j++ {
		var sum float32
		for i := 0; i < r.dim; i++ {
			sum += src[i] * r.matrix[i*r.paddedDim+j]
		}
		dst[j] = sum
	}
	return nil
}

func float32SlicesOverlap(a, b []float32) bool {
	if len(a) == 0 || len(b) == 0 {
		return false
	}
	aStart := uintptr(unsafe.Pointer(unsafe.SliceData(a)))
	bStart := uintptr(unsafe.Pointer(unsafe.SliceData(b)))
	aEnd := aStart + uintptr(len(a))*unsafe.Sizeof(a[0])
	bEnd := bStart + uintptr(len(b))*unsafe.Sizeof(b[0])
	return aStart < bEnd && bStart < aEnd
}

func (r *matrixRotator) Load(data []byte) error {
	if len(data) < r.DumpBytes() {
		return fmt.Errorf("rabitq: matrix state is too short")
	}
	for i := range r.matrix {
		r.matrix[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return nil
}
func (r *matrixRotator) Save(data []byte) error {
	if len(data) < r.DumpBytes() {
		return fmt.Errorf("rabitq: matrix output is too short")
	}
	for i, v := range r.matrix {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(v))
	}
	return nil
}
func (r *matrixRotator) DumpBytes() int { return len(r.matrix) * 4 }
func (r *matrixRotator) Size() int      { return r.paddedDim }

type fhtKacRotator struct {
	dim, paddedDim, truncDim int
	flip                     []byte
}

func (r *fhtKacRotator) Rotate(src, dst []float32) error {
	if len(src) < r.dim || len(dst) < r.paddedDim {
		return fmt.Errorf("rabitq: rotator buffer is too short")
	}
	copy(dst, src[:r.dim])
	clear(dst[r.dim:r.paddedDim])
	factor := float32(1 / math.Sqrt(float64(r.truncDim)))
	stride := r.paddedDim / 8
	if r.truncDim == r.paddedDim {
		for walk := 0; walk < 4; walk++ {
			simd.FlipSign(r.flip[walk*stride:], dst[:r.paddedDim])
			hadamard(dst[:r.truncDim])
			scale(dst[:r.truncDim], factor)
		}
		return nil
	}
	start := r.paddedDim - r.truncDim
	for walk := 0; walk < 4; walk++ {
		simd.FlipSign(r.flip[walk*stride:], dst[:r.paddedDim])
		if walk%2 == 0 {
			hadamard(dst[:r.truncDim])
			scale(dst[:r.truncDim], factor)
		} else {
			hadamard(dst[start:])
			scale(dst[start:], factor)
		}
		simd.KacsWalk(dst[:r.paddedDim])
	}
	scale(dst[:r.paddedDim], .25)
	return nil
}
func hadamard(v []float32) {
	for width := 1; width < len(v); width *= 2 {
		for base := 0; base < len(v); base += 2 * width {
			for i := 0; i < width; i++ {
				x, y := v[base+i], v[base+width+i]
				v[base+i] = x + y
				v[base+width+i] = x - y
			}
		}
	}
}
func scale(v []float32, f float32) {
	for i := range v {
		v[i] *= f
	}
}
func (r *fhtKacRotator) Load(data []byte) error {
	if len(data) < len(r.flip) {
		return fmt.Errorf("rabitq: FHT state is too short")
	}
	copy(r.flip, data)
	return nil
}
func (r *fhtKacRotator) Save(data []byte) error {
	if len(data) < len(r.flip) {
		return fmt.Errorf("rabitq: FHT output is too short")
	}
	copy(data, r.flip)
	return nil
}
func (r *fhtKacRotator) DumpBytes() int { return len(r.flip) }
func (r *fhtKacRotator) Size() int      { return r.paddedDim }
