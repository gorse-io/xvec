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

// Ported from VectorDB-NTU/RaBitQ-Library src/simd/space_excode_avx2.cpp at
// revision 540242ea0a68926f1b827bf1f9add844f07a427b.

static inline __m128i make_epi64x(int64_t high, int64_t low) {
    volatile int64_t lanes[2];
    lanes[0] = low;
    lanes[1] = high;
    return _mm_loadu_si128((const __m128i *)lanes);
}

// helper function for AVX2 inner product
static inline void contribute_ip(__m128i vec, const float* __restrict__ query, __m256 *sum) {
    __m256 q = _mm256_loadu_ps(query);
    __m256 cf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(vec));
    *sum = _mm256_add_ps(*sum, _mm256_mul_ps(q, cf));

    q = _mm256_loadu_ps(query + 8);
    cf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(vec, 8)));
    *sum = _mm256_add_ps(*sum, _mm256_mul_ps(q, cf));
};

static inline void contribute_ip_signed(
    __m128i vec, const float* __restrict__ query, __m256 *sum
) {
    __m256 q = _mm256_loadu_ps(query);
    __m256 cf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(vec));
    *sum = _mm256_add_ps(*sum, _mm256_mul_ps(cf, q));

    q = _mm256_loadu_ps(query + 8);
    cf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(vec, 8)));
    *sum = _mm256_add_ps(*sum, _mm256_mul_ps(cf, q));
};


static inline void contribute_top_bits(
    uint64_t bits, const float *query, float scale, __m256 *sum
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
    volatile float scale_value = scale;
    const __m256i checker = _mm256_loadu_si256((const __m256i *)bit_lanes);
    const __m256 scale_vector = _mm256_set1_ps(scale_value);
    for (int group = 0; group < 8; ++group) {
        const __m256i byte = _mm256_set1_epi32((uint8_t)(bits >> (group * 8)));
        const __m256i mask = _mm256_cmpeq_epi32(_mm256_and_si256(byte, checker), checker);
        const __m256 selected = _mm256_and_ps(
            _mm256_loadu_ps(query + group * 8), _mm256_castsi256_ps(mask)
        );
        *sum = _mm256_add_ps(*sum, _mm256_mul_ps(selected, scale_vector));
    }
}

static inline float mm256_reduce_add_ps(__m256 v) {
    float accumulator[8];
    _mm256_storeu_ps(accumulator, v);
    float result = 0.0F;
    for (int i = 0; i < 8; ++i) {
        result += accumulator[i];
    }
    return result;
}

// ip16: this function is used to compute inner product of
// vectors padded to multiple of 16
// fxu1: the inner product is computed between float and 1-bit unsigned int (lay out can be
// found rabitq_impl.hpp)
// avx512: only applicable for avx512
float ip16_fxu1_avx2(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, int64_t dim
) {
    float result = 0;
    __m256 sum = _mm256_setzero_ps();

    volatile int32_t bit_lanes[8];
    bit_lanes[0] = 1;
    bit_lanes[1] = 2;
    bit_lanes[2] = 4;
    bit_lanes[3] = 8;
    bit_lanes[4] = 16;
    bit_lanes[5] = 32;
    bit_lanes[6] = 64;
    bit_lanes[7] = 128;
    const __m256i bitmask = _mm256_loadu_si256((const __m256i *)bit_lanes);

    for (int64_t i = 0; i < dim; i += 8) {
        __m256 q = _mm256_loadu_ps(query);

        __m256i byte_v = _mm256_set1_epi32(*compact_code);
        __m256i isolated = _mm256_and_si256(byte_v, bitmask);
        __m256i mask = _mm256_cmpeq_epi32(isolated, bitmask);
        __m256 masked = _mm256_and_ps(q, _mm256_castsi256_ps(mask));

        sum = _mm256_add_ps(sum, masked);
        query += 8;
        ++compact_code;
    }
    result = mm256_reduce_add_ps(sum);

    return result;
}

float ip64_fxu2_avx2(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, int64_t dim
) {
    __m256 sum = _mm256_setzero_ps();

    float result = 0;
    volatile uint8_t mask_value = 0b00000011;
    const __m128i mask = _mm_set1_epi8((char)mask_value);

    for (int64_t i = 0; i < dim; i += 64) {
        __m128i compact = _mm_loadu_si128((const __m128i *)(compact_code));

        __m128i vec_00_to_15 = _mm_and_si128(compact, mask);
        __m128i vec_16_to_31 = _mm_and_si128(_mm_srli_epi16(compact, 2), mask);
        __m128i vec_32_to_47 = _mm_and_si128(_mm_srli_epi16(compact, 4), mask);
        __m128i vec_48_to_63 = _mm_and_si128(_mm_srli_epi16(compact, 6), mask);
        contribute_ip(vec_00_to_15, &query[i], &sum);
        contribute_ip(vec_16_to_31, &query[i + 16], &sum);
        contribute_ip(vec_32_to_47, &query[i + 32], &sum);
        contribute_ip(vec_48_to_63, &query[i + 48], &sum);

        compact_code += 16;
    }

    result = mm256_reduce_add_ps(sum);

    return result;
}

float ip64_fxu3_avx2(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, int64_t dim
) {
    __m256 sum = _mm256_setzero_ps();

    float result = 0;
    volatile uint8_t mask_value = 0b11;
    const __m128i mask = _mm_set1_epi8((char)mask_value);

    for (int64_t i = 0; i < dim; i += 64) {
        __m128i compact2 = _mm_loadu_si128((const __m128i *)(compact_code));
        compact_code += 16;

        uint64_t top_bit = *(const uint64_t *)(compact_code);
        compact_code += 8;

        __m128i vec_00_to_15 = _mm_and_si128(compact2, mask);
        __m128i vec_16_to_31 = _mm_and_si128(_mm_srli_epi16(compact2, 2), mask);
        __m128i vec_32_to_47 = _mm_and_si128(_mm_srli_epi16(compact2, 4), mask);
        __m128i vec_48_to_63 = _mm_and_si128(_mm_srli_epi16(compact2, 6), mask);

        contribute_ip(vec_00_to_15, &query[i], &sum);
        contribute_ip(vec_16_to_31, &query[i + 16], &sum);
        contribute_ip(vec_32_to_47, &query[i + 32], &sum);
        contribute_ip(vec_48_to_63, &query[i + 48], &sum);
        contribute_top_bits(top_bit, &query[i], 4.0F, &sum);

    }

    result = mm256_reduce_add_ps(sum);

    return result;
}

float ip16_fxu4_avx2(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, int64_t dim
) {
    __m256 sum = _mm256_setzero_ps();

    float result = 0.0F;
    volatile int64_t kMask = 0x0f0f0f0f0f0f0f0f;
    for (int64_t i = 0; i < dim; i += 16) {
        int64_t compact = *(const int64_t *)(compact_code);
        int64_t code0 = compact & kMask;
        int64_t code1 = (compact >> 4) & kMask;

        __m128i c8 = make_epi64x(code1, code0);
        contribute_ip_signed(c8, &query[i], &sum);

        compact_code += 8;
    }
    result = mm256_reduce_add_ps(sum);

    return result;
}

float ip64_fxu5_avx2(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, int64_t dim
) {
    __m256 sum = _mm256_setzero_ps();


    float result = 0.0F;
    volatile uint8_t mask_value = 0b1111;
    const __m128i mask = _mm_set1_epi8((char)mask_value);

    for (int64_t i = 0; i < dim; i += 64) {
        __m128i compact4_1 =
            _mm_loadu_si128((const __m128i *)(compact_code));
        __m128i compact4_2 =
            _mm_loadu_si128((const __m128i *)(compact_code + 16));
        compact_code += 32;

        uint64_t top_bit = *(const uint64_t *)(compact_code);
        compact_code += 8;

        __m128i vec_00_to_15 = _mm_and_si128(compact4_1, mask);
        __m128i vec_16_to_31 = _mm_and_si128(_mm_srli_epi16(compact4_1, 4), mask);
        __m128i vec_32_to_47 = _mm_and_si128(compact4_2, mask);
        __m128i vec_48_to_63 = _mm_and_si128(_mm_srli_epi16(compact4_2, 4), mask);


        contribute_ip(vec_00_to_15, &query[i], &sum);
        contribute_ip(vec_16_to_31, &query[i + 16], &sum);
        contribute_ip(vec_32_to_47, &query[i + 32], &sum);
        contribute_ip(vec_48_to_63, &query[i + 48], &sum);
        contribute_top_bits(top_bit, &query[i], 16.0F, &sum);

    }
    result = mm256_reduce_add_ps(sum);

    return result;
}

float ip64_fxu6_avx2(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, int64_t dim
) {
    __m256 sum = _mm256_setzero_ps();

    float result = 0.0F;
    volatile uint8_t mask6_value = 0b00111111;
    volatile uint8_t mask2_value = 0b11000000;
    const __m128i mask6 = _mm_set1_epi8((char)mask6_value);
    const __m128i mask2 = _mm_set1_epi8((char)mask2_value);

    for (int64_t i = 0; i < dim; i += 64) {
        __m128i cpt1 = _mm_loadu_si128((const __m128i *)(compact_code));
        __m128i cpt2 = _mm_loadu_si128((const __m128i *)(compact_code + 16));
        __m128i cpt3 = _mm_loadu_si128((const __m128i *)(compact_code + 32));

        compact_code += 48;

        __m128i vec_00_to_15 = _mm_and_si128(cpt1, mask6);
        __m128i vec_16_to_31 = _mm_and_si128(cpt2, mask6);
        __m128i vec_32_to_47 = _mm_and_si128(cpt3, mask6);
        __m128i vec_48_to_63 = _mm_or_si128(
            _mm_or_si128(
                _mm_srli_epi16(_mm_and_si128(cpt1, mask2), 6),
                _mm_srli_epi16(_mm_and_si128(cpt2, mask2), 4)
            ),
            _mm_srli_epi16(_mm_and_si128(cpt3, mask2), 2)
        );

        contribute_ip(vec_00_to_15, &query[i], &sum);
        contribute_ip(vec_16_to_31, &query[i + 16], &sum);
        contribute_ip(vec_32_to_47, &query[i + 32], &sum);
        contribute_ip(vec_48_to_63, &query[i + 48], &sum);

    }
    result = mm256_reduce_add_ps(sum);

    return result;
}

float ip64_fxu7_avx2(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, int64_t dim
) {
    __m256 sum = _mm256_setzero_ps();


    float result = 0.0F;
    volatile uint8_t mask6_value = 0b00111111;
    volatile uint8_t mask2_value = 0b11000000;
    const __m128i mask6 = _mm_set1_epi8((char)mask6_value);
    const __m128i mask2 = _mm_set1_epi8((char)mask2_value);

    for (int64_t i = 0; i < dim; i += 64) {
        __m128i cpt1 = _mm_loadu_si128((const __m128i *)(compact_code));
        __m128i cpt2 = _mm_loadu_si128((const __m128i *)(compact_code + 16));
        __m128i cpt3 = _mm_loadu_si128((const __m128i *)(compact_code + 32));
        compact_code += 48;

        __m128i vec_00_to_15 = _mm_and_si128(cpt1, mask6);
        __m128i vec_16_to_31 = _mm_and_si128(cpt2, mask6);
        __m128i vec_32_to_47 = _mm_and_si128(cpt3, mask6);
        __m128i vec_48_to_63 = _mm_or_si128(
            _mm_or_si128(
                _mm_srli_epi16(_mm_and_si128(cpt1, mask2), 6),
                _mm_srli_epi16(_mm_and_si128(cpt2, mask2), 4)
            ),
            _mm_srli_epi16(_mm_and_si128(cpt3, mask2), 2)
        );

        uint64_t top_bit = *(const uint64_t *)(compact_code);
        compact_code += 8;


        contribute_ip(vec_00_to_15, &query[i], &sum);
        contribute_ip(vec_16_to_31, &query[i + 16], &sum);
        contribute_ip(vec_32_to_47, &query[i + 32], &sum);
        contribute_ip(vec_48_to_63, &query[i + 48], &sum);
        contribute_top_bits(top_bit, &query[i], 64.0F, &sum);

    }

    result = mm256_reduce_add_ps(sum);

    return result;
}

float ip16_fxu8_avx2(
    const float* __restrict__ query, const uint8_t* __restrict__ code, int64_t dim
) {
    __m256 sum = _mm256_setzero_ps();
    for (int64_t i = 0; i < dim; i += 16) {
        __m128i c8 = _mm_loadu_si128((const __m128i *)(code));
        contribute_ip(c8, &query[i], &sum);
        code += 16;
    }
    return mm256_reduce_add_ps(sum);
}
