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

import "math"

// mt19937 and normal reproduce libstdc++'s std::mt19937 plus
// std::normal_distribution<double>. FasterConfig depends on this deterministic
// stream to match RaBitQ-Library's cached scaling factors.
type mt19937 struct {
	state  [624]uint32
	index  int
	spare  float64
	cached bool
}

func newMT19937(seed uint32) *mt19937 {
	r := &mt19937{index: 624}
	r.state[0] = seed
	for i := 1; i < len(r.state); i++ {
		x := r.state[i-1]
		r.state[i] = 1812433253*(x^(x>>30)) + uint32(i)
	}
	return r
}

func (r *mt19937) uint32() uint32 {
	if r.index >= len(r.state) {
		for i := range r.state {
			y := (r.state[i] & 0x80000000) | (r.state[(i+1)%624] & 0x7fffffff)
			r.state[i] = r.state[(i+397)%624] ^ (y >> 1)
			if y&1 != 0 {
				r.state[i] ^= 0x9908b0df
			}
		}
		r.index = 0
	}
	y := r.state[r.index]
	r.index++
	y ^= y >> 11
	y ^= (y << 7) & 0x9d2c5680
	y ^= (y << 15) & 0xefc60000
	y ^= y >> 18
	return y
}

func (r *mt19937) uniform() float64 {
	const scale = 1.0 / 18446744073709551616.0
	return (float64(r.uint32()) + float64(r.uint32())*4294967296.0) * scale
}

func (r *mt19937) normal() float64 {
	if r.cached {
		r.cached = false
		return r.spare
	}
	for {
		x := 2*r.uniform() - 1
		y := 2*r.uniform() - 1
		r2 := x*x + y*y
		if r2 == 0 || r2 > 1 {
			continue
		}
		multiplier := math.Sqrt(-2 * math.Log(r2) / r2)
		r.spare = y * multiplier
		r.cached = true
		return x * multiplier
	}
}
