// Copyright 2026-present the xvec project
// Licensed under the Apache License, Version 2.0.
//
// C intrinsic translation of VectorDB-NTU/RaBitQ-Library's
// src/simd/space_excode_avx2.cpp and space_excode_avx512.cpp.

#ifndef XVEC_RABITQ_EXCODE_KERNELS_H
#define XVEC_RABITQ_EXCODE_KERNELS_H

#include <immintrin.h>
#include <stdint.h>
#include <string.h>

#ifndef RABITQ_FN
#error RABITQ_FN must be defined before including excode_kernels.h
#endif
#ifndef RABITQ_AVX512
#error RABITQ_AVX512 must be defined to 0 or 1
#endif
#if RABITQ_AVX512 != 0 && RABITQ_AVX512 != 1
#error RABITQ_AVX512 must be 0 or 1
#endif

#if defined(__clang__) || defined(__GNUC__)
#define RABITQ_EXCODE_INLINE static inline __attribute__((always_inline))
#else
#define RABITQ_EXCODE_INLINE static inline
#endif

typedef uint16_t __attribute__((aligned(1), may_alias)) rabitq_excode_u16;
typedef uint64_t __attribute__((aligned(1), may_alias)) rabitq_excode_u64;

RABITQ_EXCODE_INLINE uint64_t rabitq_excode_load_u64(const uint8_t *data) {
    return *(const rabitq_excode_u64 *)data;
}

RABITQ_EXCODE_INLINE __m128i rabitq_excode_set_u64x(uint64_t high, uint64_t low) {
    return _mm_set_epi64x((int64_t)high, (int64_t)low);
}

RABITQ_EXCODE_INLINE __m128i rabitq_excode_expand_top(
    uint64_t top, int segment, uint8_t weight) {
    uint8_t values[16];
    top >>= segment * 16;
    for (int i = 0; i < 16; ++i)
        values[i] = (uint8_t)(((top >> i) & 1) * weight);
    return _mm_loadu_si128((const __m128i *)values);
}

#if RABITQ_AVX512

typedef __m512 rabitq_excode_sum_t;

RABITQ_EXCODE_INLINE rabitq_excode_sum_t rabitq_excode_zero(void) {
    return _mm512_setzero_ps();
}

RABITQ_EXCODE_INLINE rabitq_excode_sum_t rabitq_excode_contribute(
    __m128i code, const float *query, rabitq_excode_sum_t sum) {
    __m512 q = _mm512_loadu_ps(query);
    __m512 cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(code));
    return _mm512_fmadd_ps(q, cf, sum);
}

RABITQ_EXCODE_INLINE float rabitq_excode_reduce(rabitq_excode_sum_t sum) {
    return _mm512_reduce_add_ps(sum);
}

RABITQ_EXCODE_INLINE float rabitq_excode_ip_1_intrinsics(
    const float *query, const uint8_t *compact, int64_t dim) {
    __m512 sum = _mm512_setzero_ps();
    for (int64_t i = 0; i < dim; i += 16) {
        uint16_t bits = *(const rabitq_excode_u16 *)compact;
        sum = _mm512_add_ps(sum, _mm512_maskz_mov_ps((__mmask16)bits, _mm512_loadu_ps(query)));
        compact += 2;
        query += 16;
    }
    return _mm512_reduce_add_ps(sum);
}

#else

typedef __m256 rabitq_excode_sum_t;

RABITQ_EXCODE_INLINE rabitq_excode_sum_t rabitq_excode_zero(void) {
    return _mm256_setzero_ps();
}

RABITQ_EXCODE_INLINE rabitq_excode_sum_t rabitq_excode_contribute(
    __m128i code, const float *query, rabitq_excode_sum_t sum) {
    __m256 q0 = _mm256_loadu_ps(query);
    __m256 c0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(code));
    sum = _mm256_fmadd_ps(q0, c0, sum);
    __m256 q1 = _mm256_loadu_ps(query + 8);
    __m256 c1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(code, 8)));
    return _mm256_fmadd_ps(q1, c1, sum);
}

RABITQ_EXCODE_INLINE float rabitq_excode_reduce(rabitq_excode_sum_t sum) {
    float lanes[8];
    _mm256_storeu_ps(lanes, sum);
    float result = 0.0f;
    for (int i = 0; i < 8; ++i) result += lanes[i];
    return result;
}

RABITQ_EXCODE_INLINE float rabitq_excode_ip_1_intrinsics(
    const float *query, const uint8_t *compact, int64_t dim) {
    __m256 sum = _mm256_setzero_ps();
    const __m256i bitmask = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
    for (int64_t i = 0; i < dim; i += 8) {
        __m256 q = _mm256_loadu_ps(query);
        __m256i bytes = _mm256_set1_epi32((int)*compact);
        __m256i isolated = _mm256_and_si256(bytes, bitmask);
        __m256i mask = _mm256_cmpeq_epi32(isolated, bitmask);
        sum = _mm256_add_ps(sum, _mm256_and_ps(q, _mm256_castsi256_ps(mask)));
        query += 8;
        ++compact;
    }
    return rabitq_excode_reduce(sum);
}

#endif

RABITQ_EXCODE_INLINE float rabitq_excode_ip_2_intrinsics(
    const float *query, const uint8_t *compact, int64_t dim) {
    rabitq_excode_sum_t sum = rabitq_excode_zero();
    const __m128i mask = _mm_set1_epi8(0x03);
    for (int64_t i = 0; i < dim; i += 64) {
        __m128i c = _mm_loadu_si128((const __m128i *)compact);
        sum = rabitq_excode_contribute(_mm_and_si128(c, mask), query + i, sum);
        sum = rabitq_excode_contribute(_mm_and_si128(_mm_srli_epi16(c, 2), mask), query + i + 16, sum);
        sum = rabitq_excode_contribute(_mm_and_si128(_mm_srli_epi16(c, 4), mask), query + i + 32, sum);
        sum = rabitq_excode_contribute(_mm_and_si128(_mm_srli_epi16(c, 6), mask), query + i + 48, sum);
        compact += 16;
    }
    return rabitq_excode_reduce(sum);
}

RABITQ_EXCODE_INLINE float rabitq_excode_ip_3_intrinsics(
    const float *query, const uint8_t *compact, int64_t dim) {
    rabitq_excode_sum_t sum = rabitq_excode_zero();
    const __m128i low_mask = _mm_set1_epi8(0x03);

    for (int64_t i = 0; i < dim; i += 64) {
        __m128i c = _mm_loadu_si128((const __m128i *)compact);
        uint64_t top = rabitq_excode_load_u64(compact + 16);
        __m128i v0 = _mm_or_si128(_mm_and_si128(c, low_mask), rabitq_excode_expand_top(top, 0, 4));
        __m128i v1 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(c, 2), low_mask), rabitq_excode_expand_top(top, 1, 4));
        __m128i v2 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(c, 4), low_mask), rabitq_excode_expand_top(top, 2, 4));
        __m128i v3 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(c, 6), low_mask), rabitq_excode_expand_top(top, 3, 4));
        sum = rabitq_excode_contribute(v0, query + i, sum);
        sum = rabitq_excode_contribute(v1, query + i + 16, sum);
        sum = rabitq_excode_contribute(v2, query + i + 32, sum);
        sum = rabitq_excode_contribute(v3, query + i + 48, sum);
        compact += 24;
    }
    return rabitq_excode_reduce(sum);
}

RABITQ_EXCODE_INLINE float rabitq_excode_ip_4_intrinsics(
    const float *query, const uint8_t *compact, int64_t dim) {
    rabitq_excode_sum_t sum = rabitq_excode_zero();
    const uint64_t mask = UINT64_C(0x0f0f0f0f0f0f0f0f);
    for (int64_t i = 0; i < dim; i += 16) {
        uint64_t c = rabitq_excode_load_u64(compact);
        __m128i values = rabitq_excode_set_u64x((c >> 4) & mask, c & mask);
        sum = rabitq_excode_contribute(values, query + i, sum);
        compact += 8;
    }
    return rabitq_excode_reduce(sum);
}

RABITQ_EXCODE_INLINE float rabitq_excode_ip_5_intrinsics(
    const float *query, const uint8_t *compact, int64_t dim) {
    rabitq_excode_sum_t sum = rabitq_excode_zero();
    const uint64_t low_mask = UINT64_C(0x0f0f0f0f0f0f0f0f);
    for (int64_t i = 0; i < dim; i += 64) {
        uint64_t top = rabitq_excode_load_u64(compact + 32);
        for (int segment = 0; segment < 4; ++segment) {
            uint64_t codes = rabitq_excode_load_u64(compact + segment * 8);
            __m128i values = rabitq_excode_set_u64x((codes >> 4) & low_mask,
                                                    codes & low_mask);
            values = _mm_or_si128(values, rabitq_excode_expand_top(top, segment, 16));
            sum = rabitq_excode_contribute(values, query + i + segment * 16, sum);
        }
        compact += 40;
    }
    return rabitq_excode_reduce(sum);
}

RABITQ_EXCODE_INLINE __m128i rabitq_excode_unpack_6bit_tail(
    __m128i c0, __m128i c1, __m128i c2, __m128i mask2) {
    return _mm_or_si128(
        _mm_or_si128(_mm_srli_epi16(_mm_and_si128(c0, mask2), 6),
                     _mm_srli_epi16(_mm_and_si128(c1, mask2), 4)),
        _mm_srli_epi16(_mm_and_si128(c2, mask2), 2));
}

RABITQ_EXCODE_INLINE float rabitq_excode_ip_6_intrinsics(
    const float *query, const uint8_t *compact, int64_t dim) {
    rabitq_excode_sum_t sum = rabitq_excode_zero();
    const __m128i mask6 = _mm_set1_epi8(0x3f);
    const __m128i mask2 = _mm_set1_epi8((char)0xc0);
    for (int64_t i = 0; i < dim; i += 64) {
        __m128i c0 = _mm_loadu_si128((const __m128i *)(compact + 0));
        __m128i c1 = _mm_loadu_si128((const __m128i *)(compact + 16));
        __m128i c2 = _mm_loadu_si128((const __m128i *)(compact + 32));
        sum = rabitq_excode_contribute(_mm_and_si128(c0, mask6), query + i, sum);
        sum = rabitq_excode_contribute(_mm_and_si128(c1, mask6), query + i + 16, sum);
        sum = rabitq_excode_contribute(_mm_and_si128(c2, mask6), query + i + 32, sum);
        sum = rabitq_excode_contribute(rabitq_excode_unpack_6bit_tail(c0, c1, c2, mask2), query + i + 48, sum);
        compact += 48;
    }
    return rabitq_excode_reduce(sum);
}

RABITQ_EXCODE_INLINE float rabitq_excode_ip_7_intrinsics(
    const float *query, const uint8_t *compact, int64_t dim) {
    rabitq_excode_sum_t sum = rabitq_excode_zero();
    const __m128i mask6 = _mm_set1_epi8(0x3f);
    const __m128i mask2 = _mm_set1_epi8((char)0xc0);

    for (int64_t i = 0; i < dim; i += 64) {
        __m128i c0 = _mm_loadu_si128((const __m128i *)(compact + 0));
        __m128i c1 = _mm_loadu_si128((const __m128i *)(compact + 16));
        __m128i c2 = _mm_loadu_si128((const __m128i *)(compact + 32));
        uint64_t top = rabitq_excode_load_u64(compact + 48);
        __m128i v0 = _mm_or_si128(_mm_and_si128(c0, mask6), rabitq_excode_expand_top(top, 0, 64));
        __m128i v1 = _mm_or_si128(_mm_and_si128(c1, mask6), rabitq_excode_expand_top(top, 1, 64));
        __m128i v2 = _mm_or_si128(_mm_and_si128(c2, mask6), rabitq_excode_expand_top(top, 2, 64));
        __m128i v3 = _mm_or_si128(rabitq_excode_unpack_6bit_tail(c0, c1, c2, mask2), rabitq_excode_expand_top(top, 3, 64));
        sum = rabitq_excode_contribute(v0, query + i, sum);
        sum = rabitq_excode_contribute(v1, query + i + 16, sum);
        sum = rabitq_excode_contribute(v2, query + i + 32, sum);
        sum = rabitq_excode_contribute(v3, query + i + 48, sum);
        compact += 56;
    }
    return rabitq_excode_reduce(sum);
}

RABITQ_EXCODE_INLINE float rabitq_excode_ip_8_intrinsics(
    const float *query, const uint8_t *code, int64_t dim) {
    rabitq_excode_sum_t sum = rabitq_excode_zero();
    for (int64_t i = 0; i < dim; i += 16) {
        sum = rabitq_excode_contribute(_mm_loadu_si128((const __m128i *)code), query + i, sum);
        code += 16;
    }
    return rabitq_excode_reduce(sum);
}

float RABITQ_FN(rabitq_excode_ip)(
    const float *query, const uint8_t *packed, int64_t dim, int64_t bits) {
    switch (bits) {
    case 1: return rabitq_excode_ip_1_intrinsics(query, packed, dim);
    case 2: return rabitq_excode_ip_2_intrinsics(query, packed, dim);
    case 3: return rabitq_excode_ip_3_intrinsics(query, packed, dim);
    case 4: return rabitq_excode_ip_4_intrinsics(query, packed, dim);
    case 5: return rabitq_excode_ip_5_intrinsics(query, packed, dim);
    case 6: return rabitq_excode_ip_6_intrinsics(query, packed, dim);
    case 7: return rabitq_excode_ip_7_intrinsics(query, packed, dim);
    case 8: return rabitq_excode_ip_8_intrinsics(query, packed, dim);
    default: return 0.0f;
    }
}

#undef RABITQ_EXCODE_INLINE

#endif
