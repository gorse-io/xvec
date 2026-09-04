// Copyright 2026-present the xvec project
// Licensed under the Apache License, Version 2.0.
//
// C intrinsic translation of VectorDB-NTU/RaBitQ-Library src/simd/rotator_*.cpp.

#ifndef XVEC_RABITQ_ROTATOR_KERNELS_H
#define XVEC_RABITQ_ROTATOR_KERNELS_H

#include <immintrin.h>
#include <stdint.h>
#include <string.h>

typedef uint32_t __attribute__((aligned(1), may_alias)) rabitq_rotator_u32;
typedef uint64_t __attribute__((aligned(1), may_alias)) rabitq_rotator_u64;

static __attribute__((always_inline)) inline void rabitq_flip_sign(
    const uint8_t *flip, float *data, int64_t dim
) {
#if defined(RABITQ_AVX512) && RABITQ_AVX512
    const __m512i sign_flip = _mm512_set1_epi32((int32_t)0x80000000U);

    for (int64_t i = 0; i < dim; i += 64) {
        uint64_t mask_bits = *(const rabitq_rotator_u64 *)(flip + i / 8);

        for (int b = 0; b < 4; ++b) {
            const __mmask16 mask = (__mmask16)(mask_bits >> (b * 16));
            __m512i vec = _mm512_loadu_si512((const void *)(data + i + b * 16));
            vec = _mm512_mask_xor_epi32(vec, mask, vec, sign_flip);
            _mm512_storeu_si512((void *)(data + i + b * 16), vec);
        }
    }
#else
    const __m256i bit_select =
        _mm256_setr_epi32(0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80);
    const __m256i sign_flip = _mm256_set1_epi32((int32_t)0x80000000U);

    for (int64_t i = 0; i < dim; i += 32) {
        uint32_t mask_bits = *(const rabitq_rotator_u32 *)(flip + i / 8);

        for (int b = 0; b < 4; ++b) {
            const __m256i byte_mask = _mm256_set1_epi32((int)((mask_bits >> (b * 8)) & 0xffU));
            const __m256i selected = _mm256_and_si256(byte_mask, bit_select);
            const __m256i lane_mask = _mm256_cmpeq_epi32(selected, bit_select);
            const __m256i xor_mask = _mm256_and_si256(lane_mask, sign_flip);
            __m256i vec = _mm256_loadu_si256((const __m256i *)(data + i + b * 8));
            vec = _mm256_xor_si256(vec, xor_mask);
            _mm256_storeu_si256((__m256i *)(data + i + b * 8), vec);
        }
    }
#endif
}

static __attribute__((always_inline)) inline void rabitq_kacs_walk(
    float *data, int64_t len
) {
    const int64_t half = len / 2;
#if defined(RABITQ_AVX512) && RABITQ_AVX512
    for (int64_t i = 0; i < half; i += 16) {
        const __m512 x = _mm512_loadu_ps(data + i);
        const __m512 y = _mm512_loadu_ps(data + half + i);
        _mm512_storeu_ps(data + i, _mm512_add_ps(x, y));
        _mm512_storeu_ps(data + half + i, _mm512_sub_ps(x, y));
    }
#else
    for (int64_t i = 0; i < half; i += 8) {
        const __m256 x = _mm256_loadu_ps(data + i);
        const __m256 y = _mm256_loadu_ps(data + half + i);
        _mm256_storeu_ps(data + i, _mm256_add_ps(x, y));
        _mm256_storeu_ps(data + half + i, _mm256_sub_ps(x, y));
    }
#endif
}

#endif  // XVEC_RABITQ_ROTATOR_KERNELS_H
