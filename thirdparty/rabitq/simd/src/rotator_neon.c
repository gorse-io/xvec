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

#include <arm_neon.h>
#include <stdint.h>

// NEON port of VectorDB-NTU/RaBitQ-Library src/simd/rotator_avx2.cpp at
// revision 540242ea0a68926f1b827bf1f9add844f07a427b.

void flip_sign_neon(const uint8_t *flip, float *data, int64_t dim) {
    for (int64_t i = 0; i < dim; i += 4) {
        uint16_t bits = flip[i / 8];
        if (i / 8 + 1 < (dim + 7) / 8) {
            bits |= (uint16_t)flip[i / 8 + 1] << 8;
        }
        bits >>= i % 8;

        volatile uint64_t mask0 = ((uint64_t)(bits & 0x01U) << 31) |
                                  ((uint64_t)(bits & 0x02U) << 62);
        volatile uint64_t mask1 = ((uint64_t)(bits & 0x04U) << 29) |
                                  ((uint64_t)(bits & 0x08U) << 60);
        const uint64x2_t mask = vcombine_u64(vcreate_u64(mask0), vcreate_u64(mask1));
        const uint32x4_t values = vreinterpretq_u32_f32(vld1q_f32(data + i));
        vst1q_f32(
            data + i,
            vreinterpretq_f32_u32(veorq_u32(values, vreinterpretq_u32_u64(mask)))
        );
    }
}

void kacs_walk_neon(float *data, int64_t len) {
    const int64_t half = len / 2;
    for (int64_t i = 0; i < half; i += 4) {
        const float32x4_t x = vld1q_f32(data + i);
        const float32x4_t y = vld1q_f32(data + i + half);

        vst1q_f32(data + i, vaddq_f32(x, y));
        vst1q_f32(data + i + half, vsubq_f32(x, y));
    }
}
