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

// NEON port of VectorDB-NTU/RaBitQ-Library src/simd/space_avx2.cpp and
// space_avx512.cpp at revision 540242ea0a68926f1b827bf1f9add844f07a427b.

void scalar_quantize_uint8_neon(
    uint8_t *result, const float *data, int64_t dimension, float lo, float delta
) {
    const int64_t vectorized = dimension & ~7LL;
    volatile float one = 1.0F;
    volatile float half = 0.5F;
    const float one_over_delta = one / delta;
    const float32x4_t lo4 = vdupq_n_f32(lo);
    const float32x4_t reciprocal4 = vdupq_n_f32(one_over_delta);
    const float32x4_t half4 = vdupq_n_f32(half);
    int64_t i = 0;
    for (; i < vectorized; i += 8) {
        const float32x4_t a = vmulq_f32(vsubq_f32(vld1q_f32(data + i), lo4), reciprocal4);
        const float32x4_t b = vmulq_f32(vsubq_f32(vld1q_f32(data + i + 4), lo4), reciprocal4);
        const uint16x8_t u16 = vcombine_u16(
            vqmovun_s32(vcvtq_s32_f32(vaddq_f32(a, half4))),
            vqmovun_s32(vcvtq_s32_f32(vaddq_f32(b, half4)))
        );
        vst1_u8(result + i, vqmovn_u16(u16));
    }
    for (; i < dimension; ++i) {
        result[i] = (uint8_t)((data[i] - lo) * one_over_delta + half);
    }
}

void scalar_quantize_uint16_neon(
    uint16_t *result, const float *data, int64_t dimension, float lo, float delta
) {
    const int64_t vectorized = dimension & ~3LL;
    volatile float one = 1.0F;
    volatile float half = 0.5F;
    const float one_over_delta = one / delta;
    const float32x4_t lo4 = vdupq_n_f32(lo);
    const float32x4_t reciprocal4 = vdupq_n_f32(one_over_delta);
    const float32x4_t half4 = vdupq_n_f32(half);
    int64_t i = 0;
    for (; i < vectorized; i += 4) {
        const float32x4_t current = vmulq_f32(
            vsubq_f32(vld1q_f32(data + i), lo4), reciprocal4
        );
        vst1_u16(result + i, vqmovun_s32(vcvtq_s32_f32(vaddq_f32(current, half4))));
    }
    for (; i < dimension; ++i) {
        result[i] = (uint16_t)((data[i] - lo) * one_over_delta + half);
    }
}

void new_transpose_bin_neon(
    const uint16_t *data, uint64_t *result, int64_t padded_dimension, int64_t bits
) {
    for (int64_t block = 0; block < padded_dimension / 64; ++block) {
        for (int64_t bit = 0; bit < bits; ++bit) {
            volatile uint64_t plane = 0;
            for (int64_t i = 0; i < 64; ++i) {
                plane |= (uint64_t)((data[block * 64 + i] >> bit) & 1U) << (63 - i);
            }
            result[block * bits + bit] = plane;
        }
    }
}

void new_transpose_bin_512_neon(
    const uint8_t *data, uint64_t *result, int64_t padded_dimension, int64_t bits
) {
    for (int64_t block_start = 0; block_start < padded_dimension; block_start += 512) {
        const int64_t block_size = padded_dimension - block_start < 512
            ? padded_dimension - block_start
            : 512;
        const int64_t chunks = block_size / 64;
        const int64_t output_base = block_start / 64 * bits;
        for (int64_t bit = 0; bit < bits; ++bit) {
            for (int64_t chunk = 0; chunk < chunks; ++chunk) {
                volatile uint64_t plane = 0;
                for (int64_t i = 0; i < 64; ++i) {
                    plane |= (uint64_t)((data[block_start + chunk * 64 + i] >> bit) & 1U)
                        << (63 - i);
                }
                result[output_base + bit * chunks + chunk] = plane;
            }
        }
    }
}

float mask_ip_x0_q_neon(
    const float *query, const uint64_t *data, int64_t padded_dimension
) {
    float32x4_t sum = vdupq_n_f32(0);
    for (int64_t i = 0; i < padded_dimension; i += 4) {
        const uint64_t bits = data[i / 64];
        const int shift = 63 - (int)(i % 64);
        volatile uint32_t lanes[4];
        lanes[0] = (bits & (1ULL << shift)) ? UINT32_MAX : 0;
        lanes[1] = (bits & (1ULL << (shift - 1))) ? UINT32_MAX : 0;
        lanes[2] = (bits & (1ULL << (shift - 2))) ? UINT32_MAX : 0;
        lanes[3] = (bits & (1ULL << (shift - 3))) ? UINT32_MAX : 0;
        sum = vaddq_f32(
            sum,
            vreinterpretq_f32_u32(
                vandq_u32(vreinterpretq_u32_f32(vld1q_f32(query + i)), vld1q_u32(lanes))
            )
        );
    }
    return vaddvq_f32(sum);
}
