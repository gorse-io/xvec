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

// Ported from VectorDB-NTU/RaBitQ-Library src/simd/space_excode_avx512.cpp at
// revision 540242ea0a68926f1b827bf1f9add844f07a427b.

static inline __m128i make_epi64x(int64_t high, int64_t low) {
    volatile int64_t lanes[2];
    lanes[0] = low;
    lanes[1] = high;
    return _mm_loadu_si128((const __m128i *)lanes);
}

static inline void contribute_top_bits(
    uint64_t bits, const float *query, float scale, __m512 *sum
) {
    volatile float scale_value = scale;
    const __m512 scale_vector = _mm512_set1_ps(scale_value);
    for (int group = 0; group < 4; ++group) {
        const __mmask16 mask = (__mmask16)(bits >> (group * 16));
        const __m512 selected = _mm512_maskz_loadu_ps(mask, query + group * 16);
        *sum = _mm512_add_ps(*sum, _mm512_mul_ps(selected, scale_vector));
    }
}

// ip16: this function is used to compute inner product of
// vectors padded to multiple of 16
// fxu1: the inner product is computed between float and 1-bit unsigned int (lay out can be
// found rabitq_impl.hpp)
// avx512: only applicable for avx512
float ip16_fxu1_avx512(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, int64_t dim
) {
    float result = 0;
    __m512 sum = _mm512_setzero_ps();

    for (int64_t i = 0; i < dim; i += 16) {
        __mmask16 mask = *(const __mmask16 *)(compact_code);
        __m512 q = _mm512_loadu_ps(query);

        sum = _mm512_add_ps(_mm512_maskz_mov_ps(mask, q), sum);

        compact_code += 2;
        query += 16;
    }
    result = _mm512_reduce_add_ps(sum);

    return result;
}

float ip64_fxu2_avx512(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, int64_t dim
) {
    __m512 sum = _mm512_setzero_ps();

    float result = 0;
    volatile uint8_t mask_value = 0b00000011;
    const __m128i mask = _mm_set1_epi8((char)mask_value);

    for (int64_t i = 0; i < dim; i += 64) {
        __m128i compact = _mm_loadu_si128((const __m128i *)(compact_code));

        __m128i vec_00_to_15 = _mm_and_si128(compact, mask);
        __m128i vec_16_to_31 = _mm_and_si128(_mm_srli_epi16(compact, 2), mask);
        __m128i vec_32_to_47 = _mm_and_si128(_mm_srli_epi16(compact, 4), mask);
        __m128i vec_48_to_63 = _mm_and_si128(_mm_srli_epi16(compact, 6), mask);
        __m512 q;
        __m512 cf;

        q = _mm512_loadu_ps(&query[i]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum = _mm512_add_ps(sum, _mm512_mul_ps(q, cf));

        q = _mm512_loadu_ps(&query[i + 16]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_16_to_31));
        sum = _mm512_add_ps(sum, _mm512_mul_ps(q, cf));

        q = _mm512_loadu_ps(&query[i + 32]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_32_to_47));
        sum = _mm512_add_ps(sum, _mm512_mul_ps(q, cf));

        q = _mm512_loadu_ps(&query[i + 48]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_48_to_63));
        sum = _mm512_add_ps(sum, _mm512_mul_ps(q, cf));

        compact_code += 16;
    }

    result = _mm512_reduce_add_ps(sum);

    return result;
}

float ip64_fxu3_avx512(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, int64_t dim
) {
    __m512 sum = _mm512_setzero_ps();

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

        __m512 q;
        __m512 cf;

        q = _mm512_loadu_ps(&query[i]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum = _mm512_add_ps(sum, _mm512_mul_ps(q, cf));

        q = _mm512_loadu_ps(&query[i + 16]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_16_to_31));
        sum = _mm512_add_ps(sum, _mm512_mul_ps(q, cf));

        q = _mm512_loadu_ps(&query[i + 32]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_32_to_47));
        sum = _mm512_add_ps(sum, _mm512_mul_ps(q, cf));

        q = _mm512_loadu_ps(&query[i + 48]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_48_to_63));
        sum = _mm512_add_ps(sum, _mm512_mul_ps(q, cf));
        contribute_top_bits(top_bit, &query[i], 4.0F, &sum);

    }

    result = _mm512_reduce_add_ps(sum);

    return result;
}

float ip16_fxu4_avx512(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, int64_t dim
) {
    __m512 sum = _mm512_setzero_ps();

    float result = 0.0F;
    volatile int64_t kMask = 0x0f0f0f0f0f0f0f0f;
    for (int64_t i = 0; i < dim; i += 16) {
        int64_t compact = *(const int64_t *)(compact_code);
        int64_t code0 = compact & kMask;
        int64_t code1 = (compact >> 4) & kMask;

        __m128i c8 = make_epi64x(code1, code0);
        __m512 q = _mm512_loadu_ps(&query[i]);
        __m512 cf = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(c8));
        sum = _mm512_add_ps(sum, _mm512_mul_ps(cf, q));

        compact_code += 8;
    }
    result = _mm512_reduce_add_ps(sum);

    return result;
}

float ip64_fxu5_avx512(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, int64_t dim
) {
    __m512 sum = _mm512_setzero_ps();


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


        __m512 q;
        __m512 cf;

        q = _mm512_loadu_ps(&query[i]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum = _mm512_add_ps(sum, _mm512_mul_ps(q, cf));

        q = _mm512_loadu_ps(&query[i + 16]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_16_to_31));
        sum = _mm512_add_ps(sum, _mm512_mul_ps(q, cf));

        q = _mm512_loadu_ps(&query[i + 32]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_32_to_47));
        sum = _mm512_add_ps(sum, _mm512_mul_ps(q, cf));

        q = _mm512_loadu_ps(&query[i + 48]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_48_to_63));
        sum = _mm512_add_ps(sum, _mm512_mul_ps(q, cf));
        contribute_top_bits(top_bit, &query[i], 16.0F, &sum);

    }
    result = _mm512_reduce_add_ps(sum);

    return result;
}

float ip64_fxu6_avx512(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, int64_t dim
) {
    __m512 sum = _mm512_setzero_ps();

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

        __m512 q;
        __m512 cf;

        q = _mm512_loadu_ps(&query[i]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum = _mm512_add_ps(sum, _mm512_mul_ps(q, cf));

        q = _mm512_loadu_ps(&query[i + 16]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_16_to_31));
        sum = _mm512_add_ps(sum, _mm512_mul_ps(q, cf));

        q = _mm512_loadu_ps(&query[i + 32]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_32_to_47));
        sum = _mm512_add_ps(sum, _mm512_mul_ps(q, cf));

        q = _mm512_loadu_ps(&query[i + 48]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_48_to_63));
        sum = _mm512_add_ps(sum, _mm512_mul_ps(q, cf));

    }
    result = _mm512_reduce_add_ps(sum);

    return result;
}

float ip64_fxu7_avx512(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, int64_t dim
) {
    __m512 sum = _mm512_setzero_ps();


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


        __m512 q;
        __m512 cf;

        q = _mm512_loadu_ps(&query[i]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum = _mm512_add_ps(sum, _mm512_mul_ps(q, cf));

        q = _mm512_loadu_ps(&query[i + 16]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_16_to_31));
        sum = _mm512_add_ps(sum, _mm512_mul_ps(q, cf));

        q = _mm512_loadu_ps(&query[i + 32]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_32_to_47));
        sum = _mm512_add_ps(sum, _mm512_mul_ps(q, cf));

        q = _mm512_loadu_ps(&query[i + 48]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_48_to_63));
        sum = _mm512_add_ps(sum, _mm512_mul_ps(q, cf));
        contribute_top_bits(top_bit, &query[i], 64.0F, &sum);

    }

    result = _mm512_reduce_add_ps(sum);

    return result;
}

float ip16_fxu8_avx512(
    const float* __restrict__ query, const uint8_t* __restrict__ code, int64_t dim
) {
    __m512 sum = _mm512_setzero_ps();
    for (int64_t i = 0; i < dim; i += 16) {
        __m128i c8 = _mm_loadu_si128((const __m128i *)(code));
        __m512 q = _mm512_loadu_ps(&query[i]);
        __m512 cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(c8));
        sum = _mm512_add_ps(sum, _mm512_mul_ps(cf, q));
        code += 16;
    }
    return _mm512_reduce_add_ps(sum);
}
