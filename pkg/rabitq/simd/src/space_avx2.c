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

#include <immintrin.h>
#include <stdint.h>

// Ported from VectorDB-NTU/RaBitQ-Library src/simd/space_avx2.cpp at
// revision 540242ea0a68926f1b827bf1f9add844f07a427b.

static inline uint32_t reverse_bits32(uint32_t n) {
    n = ((n >> 1) & 0x55555555U) | ((n << 1) & 0xaaaaaaaaU);
    n = ((n >> 2) & 0x33333333U) | ((n << 2) & 0xccccccccU);
    n = ((n >> 4) & 0x0f0f0f0fU) | ((n << 4) & 0xf0f0f0f0U);
    n = ((n >> 8) & 0x00ff00ffU) | ((n << 8) & 0xff00ff00U);
    return (n >> 16) | (n << 16);
}

static inline uint64_t reverse_bits64(uint64_t n) {
    n = ((n >> 1) & 0x5555555555555555ULL) | ((n << 1) & 0xaaaaaaaaaaaaaaaaULL);
    n = ((n >> 2) & 0x3333333333333333ULL) | ((n << 2) & 0xccccccccccccccccULL);
    n = ((n >> 4) & 0x0f0f0f0f0f0f0f0fULL) | ((n << 4) & 0xf0f0f0f0f0f0f0f0ULL);
    n = ((n >> 8) & 0x00ff00ff00ff00ffULL) | ((n << 8) & 0xff00ff00ff00ff00ULL);
    n = ((n >> 16) & 0x0000ffff0000ffffULL) | ((n << 16) & 0xffff0000ffff0000ULL);
    return (n >> 32) | (n << 32);
}

void scalar_quantize_uint8_avx2(
    uint8_t *result, const float *data, int64_t dimension, float lo, float delta
) {
    const int64_t vectorized = dimension & ~7LL;
    volatile float one = 1.0F;
    volatile float half = 0.5F;
    const float one_over_delta = one / delta;
    const __m256 lo256 = _mm256_set1_ps(lo);
    const __m256 reciprocal256 = _mm256_set1_ps(one_over_delta);
    const __m256 half256 = _mm256_set1_ps(half);
    const __m128i zero = _mm_setzero_si128();
    int64_t i = 0;
    for (; i < vectorized; i += 8) {
        __m256 current = _mm256_loadu_ps(data + i);
        current = _mm256_mul_ps(_mm256_sub_ps(current, lo256), reciprocal256);
        const __m256i i32 = _mm256_cvttps_epi32(_mm256_add_ps(current, half256));
        const __m128i i16 = _mm_packus_epi32(
            _mm256_castsi256_si128(i32), _mm256_extracti128_si256(i32, 1)
        );
        const __m128i i8 = _mm_packus_epi16(i16, zero);
        _mm_storel_epi64((__m128i *)(result + i), i8);
    }
    for (; i < dimension; ++i) {
        result[i] = (uint8_t)_mm_cvttss_si32(
            _mm_set_ss((data[i] - lo) * one_over_delta + half)
        );
    }
}

void scalar_quantize_uint16_avx2(
    uint16_t *result, const float *data, int64_t dimension, float lo, float delta
) {
    const int64_t vectorized = dimension & ~7LL;
    volatile float one = 1.0F;
    volatile float half = 0.5F;
    const float one_over_delta = one / delta;
    const __m256 lo256 = _mm256_set1_ps(lo);
    const __m256 reciprocal256 = _mm256_set1_ps(one_over_delta);
    const __m256 half256 = _mm256_set1_ps(half);
    int64_t i = 0;
    for (; i < vectorized; i += 8) {
        __m256 current = _mm256_loadu_ps(data + i);
        current = _mm256_mul_ps(_mm256_sub_ps(current, lo256), reciprocal256);
        const __m256i i32 = _mm256_cvttps_epi32(_mm256_add_ps(current, half256));
        const __m128i i16 = _mm_packus_epi32(
            _mm256_castsi256_si128(i32), _mm256_extracti128_si256(i32, 1)
        );
        _mm_storeu_si128((__m128i *)(result + i), i16);
    }
    for (; i < dimension; ++i) {
        result[i] = (uint16_t)_mm_cvttss_si32(
            _mm_set_ss((data[i] - lo) * one_over_delta + half)
        );
    }
}

void new_transpose_bin_avx2(
    const uint16_t *data, uint64_t *result, int64_t padded_dimension, int64_t bits
) {
    for (int64_t i = 0; i < padded_dimension; i += 64) {
        __m256i v0 = _mm256_loadu_si256((const __m256i *)(data));
        __m256i v1 = _mm256_loadu_si256((const __m256i *)(data + 16));
        __m256i v2 = _mm256_loadu_si256((const __m256i *)(data + 32));
        __m256i v3 = _mm256_loadu_si256((const __m256i *)(data + 48));
        v0 = _mm256_slli_epi32(v0, 16 - bits);
        v1 = _mm256_slli_epi32(v1, 16 - bits);
        v2 = _mm256_slli_epi32(v2, 16 - bits);
        v3 = _mm256_slli_epi32(v3, 16 - bits);

        for (int64_t j = 0; j < bits; ++j) {
            uint32_t m0 = (uint32_t)_mm256_movemask_epi8(_mm256_packs_epi16(v0, v1));
            uint32_t m1 = (uint32_t)_mm256_movemask_epi8(_mm256_packs_epi16(v2, v3));
            m0 = (m0 & 0xff0000ffU) | ((m0 & 0x00ff0000U) >> 8) | ((m0 & 0x0000ff00U) << 8);
            m1 = (m1 & 0xff0000ffU) | ((m1 & 0x00ff0000U) >> 8) | ((m1 & 0x0000ff00U) << 8);
            result[bits - j - 1] = ((uint64_t)reverse_bits32(m0) << 32) | reverse_bits32(m1);
            v0 = _mm256_slli_epi16(v0, 1);
            v1 = _mm256_slli_epi16(v1, 1);
            v2 = _mm256_slli_epi16(v2, 1);
            v3 = _mm256_slli_epi16(v3, 1);
        }
        result += bits;
        data += 64;
    }
}

void new_transpose_bin_512_avx2(
    const uint8_t *data, uint64_t *result, int64_t padded_dimension, int64_t bits
) {
    for (int64_t i = 0; i < padded_dimension;) {
        const int64_t block_size = padded_dimension - i < 512 ? padded_dimension - i : 512;
        const int64_t chunks = block_size / 64;
        for (int64_t chunk = 0; chunk < chunks; ++chunk) {
            const __m256i lo = _mm256_loadu_si256((const __m256i *)(data + i + chunk * 64));
            const __m256i hi = _mm256_loadu_si256((const __m256i *)(data + i + chunk * 64 + 32));
            for (int64_t j = 0; j < bits; ++j) {
                const __m256i mask = _mm256_set1_epi8((char)(1 << (bits - 1 - j)));
                const uint32_t mlo = ~(uint32_t)_mm256_movemask_epi8(
                    _mm256_cmpeq_epi8(_mm256_and_si256(lo, mask), _mm256_setzero_si256())
                );
                const uint32_t mhi = ~(uint32_t)_mm256_movemask_epi8(
                    _mm256_cmpeq_epi8(_mm256_and_si256(hi, mask), _mm256_setzero_si256())
                );
                result[(bits - j - 1) * chunks + chunk] =
                    reverse_bits64(((uint64_t)mhi << 32) | mlo);
            }
        }
        i += block_size;
        result += chunks * bits;
    }
}

float mask_ip_x0_q_avx2(
    const float *query, const uint64_t *data, int64_t padded_dimension
) {
    volatile int32_t bit_lanes[8];
    bit_lanes[0] = 1;
    bit_lanes[1] = 2;
    bit_lanes[2] = 4;
    bit_lanes[3] = 8;
    bit_lanes[4] = 16;
    bit_lanes[5] = 32;
    bit_lanes[6] = 64;
    bit_lanes[7] = 128;
    const __m256i bit_checker = _mm256_loadu_si256((const __m256i *)bit_lanes);
    __m256 sum = _mm256_setzero_ps();
    for (int64_t block = 0; block < padded_dimension / 64; ++block) {
        const uint64_t mask_bits = reverse_bits64(data[block]);
        for (int j = 0; j < 8; ++j) {
            const __m256i byte = _mm256_set1_epi32((uint8_t)(mask_bits >> (j * 8)));
            const __m256i mask = _mm256_cmpgt_epi32(
                _mm256_and_si256(byte, bit_checker), _mm256_setzero_si256()
            );
            sum = _mm256_add_ps(
                sum, _mm256_and_ps(_mm256_loadu_ps(query), _mm256_castsi256_ps(mask))
            );
            query += 8;
        }
    }
    float lanes[8];
    _mm256_storeu_ps(lanes, sum);
    float result = 0;
    for (int i = 0; i < 8; ++i) {
        result += lanes[i];
    }
    return result;
}
