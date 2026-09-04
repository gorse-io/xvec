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

// Ported from VectorDB-NTU/RaBitQ-Library src/simd/space_avx512.cpp at
// revision 540242ea0a68926f1b827bf1f9add844f07a427b.

static inline uint32_t reverse_bits32(uint32_t n) {
    volatile uint32_t value = n;
    value = ((value >> 1) & 0x55555555U) | ((value << 1) & 0xaaaaaaaaU);
    value = ((value >> 2) & 0x33333333U) | ((value << 2) & 0xccccccccU);
    value = ((value >> 4) & 0x0f0f0f0fU) | ((value << 4) & 0xf0f0f0f0U);
    value = ((value >> 8) & 0x00ff00ffU) | ((value << 8) & 0xff00ff00U);
    return (value >> 16) | (value << 16);
}

static inline uint64_t reverse_bits64(uint64_t n) {
    volatile uint64_t value = n;
    value = ((value >> 1) & 0x5555555555555555ULL) |
        ((value << 1) & 0xaaaaaaaaaaaaaaaaULL);
    value = ((value >> 2) & 0x3333333333333333ULL) |
        ((value << 2) & 0xccccccccccccccccULL);
    value = ((value >> 4) & 0x0f0f0f0f0f0f0f0fULL) |
        ((value << 4) & 0xf0f0f0f0f0f0f0f0ULL);
    value = ((value >> 8) & 0x00ff00ff00ff00ffULL) |
        ((value << 8) & 0xff00ff00ff00ff00ULL);
    value = ((value >> 16) & 0x0000ffff0000ffffULL) |
        ((value << 16) & 0xffff0000ffff0000ULL);
    return (value >> 32) | (value << 32);
}

void scalar_quantize_uint8_avx512(
    uint8_t *result, const float *data, int64_t dimension, float lo, float delta
) {
    const int64_t vectorized = dimension & ~15LL;
    volatile float one = 1.0F;
    volatile float half = 0.5F;
    const float one_over_delta = one / delta;
    const __m512 lo512 = _mm512_set1_ps(lo);
    const __m512 reciprocal512 = _mm512_set1_ps(one_over_delta);
    const __m512 half512 = _mm512_set1_ps(half);
    int64_t i = 0;
    for (; i < vectorized; i += 16) {
        __m512 current = _mm512_loadu_ps(data + i);
        current = _mm512_mul_ps(_mm512_sub_ps(current, lo512), reciprocal512);
        const __m128i packed = _mm512_cvtusepi32_epi8(
            _mm512_cvttps_epi32(_mm512_add_ps(current, half512))
        );
        _mm_storeu_si128((__m128i *)(result + i), packed);
    }
    for (; i < dimension; ++i) {
        result[i] = (uint8_t)_mm_cvttss_si32(
            _mm_set_ss((data[i] - lo) * one_over_delta + half)
        );
    }
}

void scalar_quantize_uint16_avx512(
    uint16_t *result, const float *data, int64_t dimension, float lo, float delta
) {
    const int64_t vectorized = dimension & ~15LL;
    volatile float one = 1.0F;
    volatile float half = 0.5F;
    const float one_over_delta = one / delta;
    const __m512 lo512 = _mm512_set1_ps(lo);
    const __m512 reciprocal512 = _mm512_set1_ps(one_over_delta);
    const __m512 half512 = _mm512_set1_ps(half);
    int64_t i = 0;
    for (; i < vectorized; i += 16) {
        __m512 current = _mm512_loadu_ps(data + i);
        current = _mm512_mul_ps(_mm512_sub_ps(current, lo512), reciprocal512);
        const __m256i packed = _mm512_cvtusepi32_epi16(
            _mm512_cvttps_epi32(_mm512_add_ps(current, half512))
        );
        _mm256_storeu_si256((__m256i *)(result + i), packed);
    }
    for (; i < dimension; ++i) {
        result[i] = (uint16_t)_mm_cvttss_si32(
            _mm_set_ss((data[i] - lo) * one_over_delta + half)
        );
    }
}

void new_transpose_bin_avx512(
    const uint16_t *data, uint64_t *result, int64_t padded_dimension, int64_t bits
) {
    for (int64_t i = 0; i < padded_dimension; i += 64) {
        __m512i lo = _mm512_loadu_si512(data);
        __m512i hi = _mm512_loadu_si512(data + 32);
        lo = _mm512_slli_epi32(lo, 16 - bits);
        hi = _mm512_slli_epi32(hi, 16 - bits);
        for (int64_t j = 0; j < bits; ++j) {
            const uint32_t mlo = reverse_bits32((uint32_t)_mm512_movepi16_mask(lo));
            const uint32_t mhi = reverse_bits32((uint32_t)_mm512_movepi16_mask(hi));
            result[bits - j - 1] = ((uint64_t)mlo << 32) | mhi;
            lo = _mm512_slli_epi16(lo, 1);
            hi = _mm512_slli_epi16(hi, 1);
        }
        result += bits;
        data += 64;
    }
}

void new_transpose_bin_512_avx512(
    const uint8_t *data, uint64_t *result, int64_t padded_dimension, int64_t bits
) {
    for (int64_t i = 0; i < padded_dimension;) {
        const int64_t block_size = padded_dimension - i < 512 ? padded_dimension - i : 512;
        const int64_t chunks = block_size / 64;
        for (int64_t chunk = 0; chunk < chunks; ++chunk) {
            const __m512i values = _mm512_loadu_si512(data + i + chunk * 64);
            for (int64_t j = 0; j < bits; ++j) {
                const __mmask64 mask = _mm512_test_epi8_mask(
                    values, _mm512_set1_epi8((char)(1 << (bits - 1 - j)))
                );
                result[(bits - j - 1) * chunks + chunk] = reverse_bits64((uint64_t)mask);
            }
        }
        i += block_size;
        result += chunks * bits;
    }
}

float mask_ip_x0_q_avx512(
    const float *query, const uint64_t *data, int64_t padded_dimension
) {
    __m512 sum = _mm512_setzero_ps();
    for (int64_t block = 0; block < padded_dimension / 64; ++block) {
        const uint64_t bits = reverse_bits64(data[block]);
        sum = _mm512_add_ps(sum, _mm512_maskz_loadu_ps((__mmask16)bits, query));
        sum = _mm512_add_ps(sum, _mm512_maskz_loadu_ps((__mmask16)(bits >> 16), query + 16));
        sum = _mm512_add_ps(sum, _mm512_maskz_loadu_ps((__mmask16)(bits >> 32), query + 32));
        sum = _mm512_add_ps(sum, _mm512_maskz_loadu_ps((__mmask16)(bits >> 48), query + 48));
        query += 64;
    }
    return _mm512_reduce_add_ps(sum);
}
