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

package ailego

import (
	"errors"
	"math"
)

var (
	ErrInvalidFHTLength = errors.New("ailego: FHT length must be a non-zero power of two")
	ErrShortSignBits    = errors.New("ailego: sign-bit buffer is too short")
)

// FHTFlipSigns negates elements selected by the little-endian bits in signs.
func FHTFlipSigns(signs []byte, data []float32) error {
	if len(signs) < (len(data)+7)/8 {
		return ErrShortSignBits
	}
	for index := range data {
		if signs[index/8]&(1<<uint(index%8)) != 0 {
			data[index] = -data[index]
		}
	}
	return nil
}

// FHTInPlace applies the unnormalized Walsh-Hadamard transform.
func FHTInPlace(data []float32) error {
	if len(data) == 0 || len(data)&(len(data)-1) != 0 {
		return ErrInvalidFHTLength
	}
	for width := 1; width < len(data); width <<= 1 {
		for block := 0; block < len(data); block += width << 1 {
			for index := block; index < block+width; index++ {
				left, right := data[index], data[index+width]
				data[index] = left + right
				data[index+width] = left - right
			}
		}
	}
	return nil
}

// FHTKacWalk performs the baseline pairwise Kac reduction for an arbitrary
// non-empty length.
func FHTKacWalk(data []float32) error {
	if len(data) == 0 {
		return ErrInvalidFHTLength
	}
	half := len(data) / 2
	base := len(data) % 2
	offset := base + half
	for index := 0; index < half; index++ {
		left, right := data[index], data[index+offset]
		data[index] = left + right
		data[index+offset] = left - right
	}
	if base != 0 {
		data[half] *= float32(math.Sqrt(2))
	}
	return nil
}

// FHTInverseKacWalk reverses FHTKacWalk.
func FHTInverseKacWalk(data []float32) error {
	if len(data) == 0 {
		return ErrInvalidFHTLength
	}
	half := len(data) / 2
	base := len(data) % 2
	offset := base + half
	if base != 0 {
		data[half] *= float32(math.Sqrt(.5))
	}
	for index := 0; index < half; index++ {
		left, right := data[index], data[index+offset]
		data[index] = (left + right) * .5
		data[index+offset] = (left - right) * .5
	}
	return nil
}

// ScaleFloat32 multiplies data in place.
func ScaleFloat32(data []float32, factor float32) {
	for index := range data {
		data[index] *= factor
	}
}
