// Copyright 2026-present the xvec project
// Licensed under the Apache License, Version 2.0.
//
// C intrinsic translation of VectorDB-NTU/RaBitQ-Library src/simd/space_*.cpp.

#ifndef XVEC_RABITQ_SPACE_KERNELS_H
#define XVEC_RABITQ_SPACE_KERNELS_H

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

typedef uint64_t __attribute__((aligned(1), may_alias)) rabitq_space_u64;

static __attribute__((always_inline)) inline uint32_t rabitq_space_reverse32(uint32_t value) {
    value = ((value & 0x55555555U) << 1) | ((value >> 1) & 0x55555555U);
    value = ((value & 0x33333333U) << 2) | ((value >> 2) & 0x33333333U);
    value = ((value & 0x0f0f0f0fU) << 4) | ((value >> 4) & 0x0f0f0f0fU);
    return __builtin_bswap32(value);
}

static __attribute__((always_inline)) inline uint64_t rabitq_space_reverse64(uint64_t value) {
    value = ((value & 0x5555555555555555ULL) << 1) |
            ((value >> 1) & 0x5555555555555555ULL);
    value = ((value & 0x3333333333333333ULL) << 2) |
            ((value >> 2) & 0x3333333333333333ULL);
    value = ((value & 0x0f0f0f0f0f0f0f0fULL) << 4) |
            ((value >> 4) & 0x0f0f0f0f0f0f0f0fULL);
    return __builtin_bswap64(value);
}

static __attribute__((always_inline)) inline int32_t rabitq_quantize_scalar(
    float value, float maximum) {
    if (value <= 0.0F) return 0;
    if (value >= maximum) return (int32_t)maximum;
    return _mm_cvtss_si32(_mm_set_ss(value));
}

static __attribute__((always_inline)) inline void rabitq_quantize_u8(
    uint8_t *result, const float *vec, int64_t dim, float lo, float delta
) {
    int64_t i = 0;
    const float one_over_delta = 1.0F / delta;
#if defined(RABITQ_AVX512) && RABITQ_AVX512
    const int64_t vector_end = dim - (dim & 15);
    const __m512 vlo = _mm512_set1_ps(lo);
    const __m512 vod = _mm512_set1_ps(one_over_delta);
    const __m512 vzero = _mm512_setzero_ps();
    const __m512 vmax = _mm512_set1_ps(255.0F);
    for (; i < vector_end; i += 16) {
        __m512 cur = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(vec + i), vlo), vod);
        cur = _mm512_min_ps(_mm512_max_ps(cur, vzero), vmax);
        const __m128i i8 = _mm512_cvtusepi32_epi8(_mm512_cvtps_epi32(cur));
        _mm_storeu_si128((__m128i *)(result + i), i8);
    }
#else
    const int64_t vector_end = dim - (dim & 7);
    const __m256 vlo = _mm256_set1_ps(lo);
    const __m256 vod = _mm256_set1_ps(one_over_delta);
    const __m256 vzero = _mm256_setzero_ps();
    const __m256 vmax = _mm256_set1_ps(255.0F);
    const __m128i zero = _mm_setzero_si128();
    for (; i < vector_end; i += 8) {
        __m256 cur = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(vec + i), vlo), vod);
        cur = _mm256_min_ps(_mm256_max_ps(cur, vzero), vmax);
        const __m256i i32 = _mm256_cvtps_epi32(cur);
        const __m128i i16 = _mm_packus_epi32(_mm256_castsi256_si128(i32), _mm256_extracti128_si256(i32, 1));
        _mm_storel_epi64((__m128i *)(result + i), _mm_packus_epi16(i16, zero));
    }
#endif
    for (; i < dim; ++i)
        result[i] = (uint8_t)rabitq_quantize_scalar((vec[i] - lo) * one_over_delta, 255.0F);
}

static __attribute__((always_inline)) inline void rabitq_quantize_u16(
    uint16_t *result, const float *vec, int64_t dim, float lo, float delta
) {
    int64_t i = 0;
    const float one_over_delta = 1.0F / delta;
#if defined(RABITQ_AVX512) && RABITQ_AVX512
    const int64_t vector_end = dim - (dim & 15);
    const __m512 vlo = _mm512_set1_ps(lo);
    const __m512 vod = _mm512_set1_ps(one_over_delta);
    const __m512 vzero = _mm512_setzero_ps();
    const __m512 vmax = _mm512_set1_ps(65535.0F);
    for (; i < vector_end; i += 16) {
        __m512 cur = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(vec + i), vlo), vod);
        cur = _mm512_min_ps(_mm512_max_ps(cur, vzero), vmax);
        _mm256_storeu_si256((__m256i *)(result + i), _mm512_cvtusepi32_epi16(_mm512_cvtps_epi32(cur)));
    }
#else
    const int64_t vector_end = dim - (dim & 7);
    const __m256 vlo = _mm256_set1_ps(lo);
    const __m256 vod = _mm256_set1_ps(one_over_delta);
    const __m256 vzero = _mm256_setzero_ps();
    const __m256 vmax = _mm256_set1_ps(65535.0F);
    for (; i < vector_end; i += 8) {
        __m256 cur = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(vec + i), vlo), vod);
        cur = _mm256_min_ps(_mm256_max_ps(cur, vzero), vmax);
        const __m256i i32 = _mm256_cvtps_epi32(cur);
        _mm_storeu_si128((__m128i *)(result + i), _mm_packus_epi32(_mm256_castsi256_si128(i32), _mm256_extracti128_si256(i32, 1)));
    }
#endif
    for (; i < dim; ++i)
        result[i] = (uint16_t)rabitq_quantize_scalar((vec[i] - lo) * one_over_delta, 65535.0F);
}

static __attribute__((always_inline)) inline void rabitq_transpose_bin(
    const uint16_t *q, uint64_t *tq, int64_t padded_dim, int64_t b_query
) {
    for (int64_t i = 0; i < padded_dim; i += 64) {
#if defined(RABITQ_AVX512) && RABITQ_AVX512
        const __m512i shift_count = _mm512_set1_epi32((int)(16 - b_query));
        __m512i vec0 = _mm512_loadu_si512((const void *)q);
        __m512i vec1 = _mm512_loadu_si512((const void *)(q + 32));
        vec0 = _mm512_sllv_epi32(vec0, shift_count);
        vec1 = _mm512_sllv_epi32(vec1, shift_count);

        for (int64_t j = 0; j < b_query; ++j) {
            uint32_t v0 = (uint32_t)_mm512_movepi16_mask(vec0);
            uint32_t v1 = (uint32_t)_mm512_movepi16_mask(vec1);
            v0 = rabitq_space_reverse32(v0);
            v1 = rabitq_space_reverse32(v1);
            tq[b_query - j - 1] = ((uint64_t)v0 << 32) | (uint64_t)v1;
            vec0 = _mm512_slli_epi16(vec0, 1);
            vec1 = _mm512_slli_epi16(vec1, 1);
        }
#else
        const __m256i shift_count = _mm256_set1_epi32((int)(16 - b_query));
        __m256i vec0 = _mm256_loadu_si256((const __m256i *)q);
        __m256i vec1 = _mm256_loadu_si256((const __m256i *)(q + 16));
        __m256i vec2 = _mm256_loadu_si256((const __m256i *)(q + 32));
        __m256i vec3 = _mm256_loadu_si256((const __m256i *)(q + 48));
        vec0 = _mm256_sllv_epi32(vec0, shift_count);
        vec1 = _mm256_sllv_epi32(vec1, shift_count);
        vec2 = _mm256_sllv_epi32(vec2, shift_count);
        vec3 = _mm256_sllv_epi32(vec3, shift_count);

        for (int64_t j = 0; j < b_query; ++j) {
            const __m256i packed0 = _mm256_packs_epi16(vec0, vec1);
            const __m256i packed1 = _mm256_packs_epi16(vec2, vec3);
            uint32_t m0 = (uint32_t)_mm256_movemask_epi8(packed0);
            uint32_t m1 = (uint32_t)_mm256_movemask_epi8(packed1);
            m0 = (m0 & 0xff0000ffU) | ((m0 & 0x00ff0000U) >> 8) |
                 ((m0 & 0x0000ff00U) << 8);
            m1 = (m1 & 0xff0000ffU) | ((m1 & 0x00ff0000U) >> 8) |
                 ((m1 & 0x0000ff00U) << 8);
            m0 = rabitq_space_reverse32(m0);
            m1 = rabitq_space_reverse32(m1);
            tq[b_query - j - 1] = ((uint64_t)m0 << 32) | (uint64_t)m1;
            vec0 = _mm256_slli_epi16(vec0, 1);
            vec1 = _mm256_slli_epi16(vec1, 1);
            vec2 = _mm256_slli_epi16(vec2, 1);
            vec3 = _mm256_slli_epi16(vec3, 1);
        }
#endif
        q += 64;
        tq += b_query;
    }
}

static __attribute__((always_inline)) inline void rabitq_transpose_bin_512(
    const uint8_t *q, uint64_t *tq, int64_t padded_dim, int64_t b_query
) {
    for (int64_t i = 0; i < padded_dim;) {
        int64_t block_size = 512;
        if (i + block_size > padded_dim) {
            block_size = padded_dim - i;
        }
        const int64_t num_chunks = block_size / 64;

        for (int64_t k = 0; k < num_chunks; ++k) {
#if defined(RABITQ_AVX512) && RABITQ_AVX512
            const __m512i vec = _mm512_loadu_si512((const void *)(q + i + k * 64));
            for (int64_t j = 0; j < b_query; ++j) {
                const int bit_idx = (int)(b_query - 1 - j);
                const __m512i bit_mask = _mm512_set1_epi8((char)(1U << bit_idx));
                const __mmask64 mask = _mm512_test_epi8_mask(vec, bit_mask);
                tq[(b_query - j - 1) * num_chunks + k] =
                    rabitq_space_reverse64((uint64_t)mask);
            }
#else
            const uint8_t *current = q + i + k * 64;
            const __m256i vec_lo = _mm256_loadu_si256((const __m256i *)current);
            const __m256i vec_hi = _mm256_loadu_si256((const __m256i *)(current + 32));
            for (int64_t j = 0; j < b_query; ++j) {
                const int bit_idx = (int)(b_query - 1 - j);
                const __m256i bit_mask = _mm256_set1_epi8((char)(1U << bit_idx));
                const __m256i lo_bits = _mm256_and_si256(vec_lo, bit_mask);
                const __m256i hi_bits = _mm256_and_si256(vec_hi, bit_mask);
                const __m256i zero = _mm256_setzero_si256();
                const uint32_t m_lo = ~(uint32_t)_mm256_movemask_epi8(
                    _mm256_cmpeq_epi8(lo_bits, zero)
                );
                const uint32_t m_hi = ~(uint32_t)_mm256_movemask_epi8(
                    _mm256_cmpeq_epi8(hi_bits, zero)
                );
                const uint64_t mask = ((uint64_t)m_hi << 32) | (uint64_t)m_lo;
                tq[(b_query - j - 1) * num_chunks + k] = rabitq_space_reverse64(mask);
            }
#endif
        }

        i += block_size;
        tq += num_chunks * b_query;
    }
}

static __attribute__((always_inline)) inline float rabitq_mask_ip(
    const float *query, const uint8_t *data, int64_t padded_dim
) {
    const int64_t num_blocks = padded_dim / 64;
#if defined(RABITQ_AVX512) && RABITQ_AVX512
    __m512 sum = _mm512_setzero_ps();
    for (int64_t i = 0; i < num_blocks; ++i) {
        uint64_t stored_bits = *(const rabitq_space_u64 *)(data + i * 8);
        const uint64_t bits = rabitq_space_reverse64(stored_bits);
        const float *block = query + i * 64;
        sum = _mm512_add_ps(sum, _mm512_maskz_loadu_ps((__mmask16)bits, block));
        sum = _mm512_add_ps(sum, _mm512_maskz_loadu_ps((__mmask16)(bits >> 16), block + 16));
        sum = _mm512_add_ps(sum, _mm512_maskz_loadu_ps((__mmask16)(bits >> 32), block + 32));
        sum = _mm512_add_ps(sum, _mm512_maskz_loadu_ps((__mmask16)(bits >> 48), block + 48));
    }
    return _mm512_reduce_add_ps(sum);
#else
    __m256 sum = _mm256_setzero_ps();
    const __m256i bit_checker =
        _mm256_set_epi32(0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01);
    const __m256i zero = _mm256_setzero_si256();
    const float *current_query = query;

    for (int64_t i = 0; i < num_blocks; ++i) {
        uint64_t stored_bits = *(const rabitq_space_u64 *)(data + i * 8);
        const uint64_t bits = rabitq_space_reverse64(stored_bits);
        for (int j = 0; j < 8; ++j) {
            const __m256i byte = _mm256_set1_epi32((int)((bits >> (j * 8)) & 0xffU));
            const __m256i selected = _mm256_and_si256(byte, bit_checker);
            const __m256i mask = _mm256_cmpgt_epi32(selected, zero);
            const __m256 values = _mm256_loadu_ps(current_query);
            sum = _mm256_add_ps(sum, _mm256_and_ps(values, _mm256_castsi256_ps(mask)));
            current_query += 8;
        }
    }

    float lanes[8];
    _mm256_storeu_ps(lanes, sum);
    float result = 0.0F;
    for (int i = 0; i < 8; ++i) {
        result += lanes[i];
    }
    return result;
#endif
}

#endif  // XVEC_RABITQ_SPACE_KERNELS_H
