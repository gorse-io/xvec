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

// NEON port of VectorDB-NTU/RaBitQ-Library src/simd/space_excode_avx2.cpp and
// space_excode_avx512.cpp at revision 540242ea0a68926f1b827bf1f9add844f07a427b.

static inline uint8x8_t bit_checker_neon(void) {
    volatile uint8_t lanes[8];
    lanes[0] = 1;
    lanes[1] = 2;
    lanes[2] = 4;
    lanes[3] = 8;
    lanes[4] = 16;
    lanes[5] = 32;
    lanes[6] = 64;
    lanes[7] = 128;
    return vld1_u8((const uint8_t *)lanes);
}

static inline uint8x16_t expand_bits_neon(
    uint8_t low, uint8_t high, uint8x8_t checker
) {
    const uint8x8_t low_bits = vshr_n_u8(vtst_u8(vdup_n_u8(low), checker), 7);
    const uint8x8_t high_bits = vshr_n_u8(vtst_u8(vdup_n_u8(high), checker), 7);
    return vcombine_u8(low_bits, high_bits);
}

static inline uint8x16_t add_top_bits_neon(
    uint8x16_t values, const uint8_t *top_bits, int group, int shift, uint8x8_t checker
) {
    const uint8x16_t top = expand_bits_neon(
        top_bits[group * 2], top_bits[group * 2 + 1], checker
    );
    return vorrq_u8(values, vshlq_u8(top, vdupq_n_s8((int8_t)shift)));
}

static inline void contribute_ip_neon(
    uint8x16_t values, const float *query, float32x4_t *sum
) {
    const uint16x8_t low = vmovl_u8(vget_low_u8(values));
    const uint16x8_t high = vmovl_high_u8(values);
    *sum = vfmaq_f32(
        *sum, vld1q_f32(query), vcvtq_f32_u32(vmovl_u16(vget_low_u16(low)))
    );
    *sum = vfmaq_f32(
        *sum, vld1q_f32(query + 4), vcvtq_f32_u32(vmovl_high_u16(low))
    );
    *sum = vfmaq_f32(
        *sum, vld1q_f32(query + 8), vcvtq_f32_u32(vmovl_u16(vget_low_u16(high)))
    );
    *sum = vfmaq_f32(
        *sum, vld1q_f32(query + 12), vcvtq_f32_u32(vmovl_high_u16(high))
    );
}

float ip16_fxu1_neon(const float *query, const uint8_t *compact_code, int64_t dim) {
    const uint8x8_t checker = bit_checker_neon();
    float32x4_t sum = vdupq_n_f32(0);
    for (int64_t i = 0; i < dim; i += 16) {
        contribute_ip_neon(
            expand_bits_neon(compact_code[0], compact_code[1], checker), query + i, &sum
        );
        compact_code += 2;
    }
    return vaddvq_f32(sum);
}

float ip64_fxu2_neon(const float *query, const uint8_t *compact_code, int64_t dim) {
    const uint8x16_t mask = vdupq_n_u8(3);
    float32x4_t sum = vdupq_n_f32(0);
    for (int64_t i = 0; i < dim; i += 64) {
        const uint8x16_t compact = vld1q_u8(compact_code);
        contribute_ip_neon(vandq_u8(compact, mask), query + i, &sum);
        contribute_ip_neon(vandq_u8(vshrq_n_u8(compact, 2), mask), query + i + 16, &sum);
        contribute_ip_neon(vandq_u8(vshrq_n_u8(compact, 4), mask), query + i + 32, &sum);
        contribute_ip_neon(vandq_u8(vshrq_n_u8(compact, 6), mask), query + i + 48, &sum);
        compact_code += 16;
    }
    return vaddvq_f32(sum);
}

float ip64_fxu3_neon(const float *query, const uint8_t *compact_code, int64_t dim) {
    const uint8x16_t mask = vdupq_n_u8(3);
    const uint8x8_t checker = bit_checker_neon();
    float32x4_t sum = vdupq_n_f32(0);
    for (int64_t i = 0; i < dim; i += 64) {
        const uint8x16_t compact = vld1q_u8(compact_code);
        const uint8_t *top = compact_code + 16;
        contribute_ip_neon(add_top_bits_neon(vandq_u8(compact, mask), top, 0, 2, checker), query + i, &sum);
        contribute_ip_neon(add_top_bits_neon(vandq_u8(vshrq_n_u8(compact, 2), mask), top, 1, 2, checker), query + i + 16, &sum);
        contribute_ip_neon(add_top_bits_neon(vandq_u8(vshrq_n_u8(compact, 4), mask), top, 2, 2, checker), query + i + 32, &sum);
        contribute_ip_neon(add_top_bits_neon(vandq_u8(vshrq_n_u8(compact, 6), mask), top, 3, 2, checker), query + i + 48, &sum);
        compact_code += 24;
    }
    return vaddvq_f32(sum);
}

float ip16_fxu4_neon(const float *query, const uint8_t *compact_code, int64_t dim) {
    const uint8x8_t mask = vdup_n_u8(15);
    float32x4_t sum = vdupq_n_f32(0);
    for (int64_t i = 0; i < dim; i += 16) {
        const uint8x8_t compact = vld1_u8(compact_code);
        const uint8x16_t values = vcombine_u8(
            vand_u8(compact, mask), vshr_n_u8(compact, 4)
        );
        contribute_ip_neon(values, query + i, &sum);
        compact_code += 8;
    }
    return vaddvq_f32(sum);
}

float ip64_fxu5_neon(const float *query, const uint8_t *compact_code, int64_t dim) {
    const uint8x16_t mask = vdupq_n_u8(15);
    const uint8x8_t checker = bit_checker_neon();
    float32x4_t sum = vdupq_n_f32(0);
    for (int64_t i = 0; i < dim; i += 64) {
        const uint8x16_t first = vld1q_u8(compact_code);
        const uint8x16_t second = vld1q_u8(compact_code + 16);
        const uint8_t *top = compact_code + 32;
        contribute_ip_neon(add_top_bits_neon(vandq_u8(first, mask), top, 0, 4, checker), query + i, &sum);
        contribute_ip_neon(add_top_bits_neon(vshrq_n_u8(first, 4), top, 1, 4, checker), query + i + 16, &sum);
        contribute_ip_neon(add_top_bits_neon(vandq_u8(second, mask), top, 2, 4, checker), query + i + 32, &sum);
        contribute_ip_neon(add_top_bits_neon(vshrq_n_u8(second, 4), top, 3, 4, checker), query + i + 48, &sum);
        compact_code += 40;
    }
    return vaddvq_f32(sum);
}

float ip64_fxu6_neon(const float *query, const uint8_t *compact_code, int64_t dim) {
    const uint8x16_t mask6 = vdupq_n_u8(0x3f);
    const uint8x16_t mask2 = vdupq_n_u8(0xc0);
    float32x4_t sum = vdupq_n_f32(0);
    for (int64_t i = 0; i < dim; i += 64) {
        const uint8x16_t first = vld1q_u8(compact_code);
        const uint8x16_t second = vld1q_u8(compact_code + 16);
        const uint8x16_t third = vld1q_u8(compact_code + 32);
        const uint8x16_t fourth = vorrq_u8(
            vorrq_u8(
                vshrq_n_u8(vandq_u8(first, mask2), 6),
                vshrq_n_u8(vandq_u8(second, mask2), 4)
            ),
            vshrq_n_u8(vandq_u8(third, mask2), 2)
        );
        contribute_ip_neon(vandq_u8(first, mask6), query + i, &sum);
        contribute_ip_neon(vandq_u8(second, mask6), query + i + 16, &sum);
        contribute_ip_neon(vandq_u8(third, mask6), query + i + 32, &sum);
        contribute_ip_neon(fourth, query + i + 48, &sum);
        compact_code += 48;
    }
    return vaddvq_f32(sum);
}

float ip64_fxu7_neon(const float *query, const uint8_t *compact_code, int64_t dim) {
    const uint8x16_t mask6 = vdupq_n_u8(0x3f);
    const uint8x16_t mask2 = vdupq_n_u8(0xc0);
    const uint8x8_t checker = bit_checker_neon();
    float32x4_t sum = vdupq_n_f32(0);
    for (int64_t i = 0; i < dim; i += 64) {
        const uint8x16_t first = vld1q_u8(compact_code);
        const uint8x16_t second = vld1q_u8(compact_code + 16);
        const uint8x16_t third = vld1q_u8(compact_code + 32);
        const uint8_t *top = compact_code + 48;
        const uint8x16_t fourth = vorrq_u8(
            vorrq_u8(
                vshrq_n_u8(vandq_u8(first, mask2), 6),
                vshrq_n_u8(vandq_u8(second, mask2), 4)
            ),
            vshrq_n_u8(vandq_u8(third, mask2), 2)
        );
        contribute_ip_neon(add_top_bits_neon(vandq_u8(first, mask6), top, 0, 6, checker), query + i, &sum);
        contribute_ip_neon(add_top_bits_neon(vandq_u8(second, mask6), top, 1, 6, checker), query + i + 16, &sum);
        contribute_ip_neon(add_top_bits_neon(vandq_u8(third, mask6), top, 2, 6, checker), query + i + 32, &sum);
        contribute_ip_neon(add_top_bits_neon(fourth, top, 3, 6, checker), query + i + 48, &sum);
        compact_code += 56;
    }
    return vaddvq_f32(sum);
}

float ip16_fxu8_neon(const float *query, const uint8_t *code, int64_t dim) {
    float32x4_t sum = vdupq_n_f32(0);
    for (int64_t i = 0; i < dim; i += 16) {
        contribute_ip_neon(vld1q_u8(code), query + i, &sum);
        code += 16;
    }
    return vaddvq_f32(sum);
}
