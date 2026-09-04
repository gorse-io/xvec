// Copyright 2026-present the xvec project
// Licensed under the Apache License, Version 2.0.
//
// This file is derived from VectorDB-NTU/RaBitQ-Library src/simd.
// It was changed to provide C intrinsic templates suitable for GoAT.

#ifndef XVEC_RABITQ_FASTSCAN_KERNELS_H_
#define XVEC_RABITQ_FASTSCAN_KERNELS_H_

#include <immintrin.h>
#include <stdint.h>

#ifndef RABITQ_FN
#error RABITQ_FN must be defined before including fastscan_kernels.h
#endif

#define RABITQ_FASTSCAN_INLINE __attribute__((always_inline)) static inline

#if RABITQ_AVX512

RABITQ_FASTSCAN_INLINE __m512i rabitq_fastscan_combine_avx512(__m512i a, __m512i b) {
    return _mm512_add_epi16(
        _mm512_mask_blend_epi64((__mmask8)0xf0, a, b),
        _mm512_shuffle_i64x2(a, b, 0x4e));
}

void RABITQ_FN(rabitq_fastscan_accumulate)(const uint8_t *codes, const uint8_t *lut,
                                           uint16_t *out, int64_t dim) {
    const __m512i low_mask = _mm512_set1_epi8(0x0f);
    __m512i accu0 = _mm512_setzero_si512();
    __m512i accu1 = _mm512_setzero_si512();
    __m512i accu2 = _mm512_setzero_si512();
    __m512i accu3 = _mm512_setzero_si512();
    const int64_t code_length = dim << 2;

    for (int64_t i = 0; i < code_length; i += 64) {
        const __m512i c = _mm512_loadu_si512((const void *)(codes + i));
        const __m512i table = _mm512_loadu_si512((const void *)(lut + i));
        const __m512i lo = _mm512_and_si512(c, low_mask);
        const __m512i hi = _mm512_and_si512(_mm512_srli_epi16(c, 4), low_mask);
        const __m512i res_lo = _mm512_shuffle_epi8(table, lo);
        const __m512i res_hi = _mm512_shuffle_epi8(table, hi);

        /* Deliberately wrapping 16-bit component accumulators, as upstream. */
        accu0 = _mm512_add_epi16(accu0, res_lo);
        accu1 = _mm512_add_epi16(accu1, _mm512_srli_epi16(res_lo, 8));
        accu2 = _mm512_add_epi16(accu2, res_hi);
        accu3 = _mm512_add_epi16(accu3, _mm512_srli_epi16(res_hi, 8));
    }

    accu0 = _mm512_sub_epi16(accu0, _mm512_slli_epi16(accu1, 8));
    accu2 = _mm512_sub_epi16(accu2, _mm512_slli_epi16(accu3, 8));

    const __m512i ret1 = rabitq_fastscan_combine_avx512(accu0, accu1);
    const __m512i ret2 = rabitq_fastscan_combine_avx512(accu2, accu3);
    const __m512i ret = _mm512_add_epi16(
        _mm512_shuffle_i64x2(ret1, ret2, 0x88),
        _mm512_shuffle_i64x2(ret1, ret2, 0xdd));
    _mm512_storeu_si512((void *)out, ret);
}

void RABITQ_FN(rabitq_fastscan_transfer_hacc)(const uint16_t *lut, uint8_t *out,
                                              int64_t dim) {
    const int64_t num_codebook = dim >> 2;
    for (int64_t i = 0; i < num_codebook; ++i) {
        uint8_t *fill_lo = out + (i / 4) * 128 + (i % 4) * 16;
        uint8_t *fill_hi = fill_lo + 64;
        const __m256i packed = _mm256_loadu_si256((const __m256i *)lut);
        const __m512i widened = _mm512_cvtepi16_epi32(packed);
        const __m128i lo = _mm512_cvtepi32_epi8(widened);
        const __m128i hi = _mm512_cvtepi32_epi8(_mm512_srli_epi32(widened, 8));
        _mm_storeu_si128((__m128i *)fill_lo, lo);
        _mm_storeu_si128((__m128i *)fill_hi, hi);
        lut += 16;
    }
}

void RABITQ_FN(rabitq_fastscan_accumulate_hacc)(const uint8_t *codes,
                                                const uint8_t *lut,
                                                int32_t *out, int64_t dim) {
    const __m512i low_mask = _mm512_set1_epi8(0x0f);
    __m512i accu[2][4];
    for (int q = 0; q < 2; ++q)
        for (int k = 0; k < 4; ++k)
            accu[q][k] = _mm512_setzero_si512();

    const int64_t num_codebook = dim >> 2;
    for (int64_t m = 0; m < num_codebook; m += 4) {
        const __m512i c = _mm512_loadu_si512((const void *)codes);
        const __m512i lo = _mm512_and_si512(c, low_mask);
        const __m512i hi = _mm512_and_si512(_mm512_srli_epi16(c, 4), low_mask);

        for (int q = 0; q < 2; ++q) {
            const __m512i table = _mm512_loadu_si512((const void *)lut);
            const __m512i res_lo = _mm512_shuffle_epi8(table, lo);
            const __m512i res_hi = _mm512_shuffle_epi8(table, hi);
            accu[q][0] = _mm512_add_epi16(accu[q][0], res_lo);
            accu[q][1] = _mm512_add_epi16(accu[q][1], _mm512_srli_epi16(res_lo, 8));
            accu[q][2] = _mm512_add_epi16(accu[q][2], res_hi);
            accu[q][3] = _mm512_add_epi16(accu[q][3], _mm512_srli_epi16(res_hi, 8));
            lut += 64;
        }
        codes += 64;
    }

    __m512i dis0[2];
    __m512i dis1[2];
    for (int q = 0; q < 2; ++q) {
        __m256i a0 = _mm256_add_epi16(_mm512_castsi512_si256(accu[q][0]),
                                      _mm512_extracti64x4_epi64(accu[q][0], 1));
        const __m256i a1 = _mm256_add_epi16(_mm512_castsi512_si256(accu[q][1]),
                                            _mm512_extracti64x4_epi64(accu[q][1], 1));
        a0 = _mm256_sub_epi16(a0, _mm256_slli_epi16(a1, 8));
        dis0[q] = _mm512_add_epi32(
            _mm512_cvtepu16_epi32(_mm256_permute2f128_si256(a0, a1, 0x21)),
            _mm512_cvtepu16_epi32(_mm256_blend_epi32(a0, a1, 0xf0)));

        __m256i a2 = _mm256_add_epi16(_mm512_castsi512_si256(accu[q][2]),
                                      _mm512_extracti64x4_epi64(accu[q][2], 1));
        const __m256i a3 = _mm256_add_epi16(_mm512_castsi512_si256(accu[q][3]),
                                            _mm512_extracti64x4_epi64(accu[q][3], 1));
        a2 = _mm256_sub_epi16(a2, _mm256_slli_epi16(a3, 8));
        dis1[q] = _mm512_add_epi32(
            _mm512_cvtepu16_epi32(_mm256_permute2f128_si256(a2, a3, 0x21)),
            _mm512_cvtepu16_epi32(_mm256_blend_epi32(a2, a3, 0xf0)));
    }

    _mm512_storeu_si512((void *)out,
                        _mm512_add_epi32(dis0[0], _mm512_slli_epi32(dis0[1], 8)));
    _mm512_storeu_si512((void *)(out + 16),
                        _mm512_add_epi32(dis1[0], _mm512_slli_epi32(dis1[1], 8)));
}

#else

RABITQ_FASTSCAN_INLINE __m256i rabitq_fastscan_combine_avx2(__m256i a, __m256i b) {
    return _mm256_add_epi16(_mm256_permute2f128_si256(a, b, 0x21),
                            _mm256_blend_epi32(a, b, 0xf0));
}

void RABITQ_FN(rabitq_fastscan_accumulate)(const uint8_t *codes, const uint8_t *lut,
                                           uint16_t *out, int64_t dim) {
    const __m256i low_mask = _mm256_set1_epi8(0x0f);
    __m256i accu0 = _mm256_setzero_si256();
    __m256i accu1 = _mm256_setzero_si256();
    __m256i accu2 = _mm256_setzero_si256();
    __m256i accu3 = _mm256_setzero_si256();
    const int64_t code_length = dim << 2;

    for (int64_t i = 0; i < code_length; i += 64) {
        for (int half = 0; half < 2; ++half) {
            const int64_t off = i + (int64_t)half * 32;
            const __m256i c = _mm256_loadu_si256((const __m256i *)(codes + off));
            const __m256i table = _mm256_loadu_si256((const __m256i *)(lut + off));
            const __m256i lo = _mm256_and_si256(c, low_mask);
            const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);
            const __m256i res_lo = _mm256_shuffle_epi8(table, lo);
            const __m256i res_hi = _mm256_shuffle_epi8(table, hi);
            accu0 = _mm256_add_epi16(accu0, res_lo);
            accu1 = _mm256_add_epi16(accu1, _mm256_srli_epi16(res_lo, 8));
            accu2 = _mm256_add_epi16(accu2, res_hi);
            accu3 = _mm256_add_epi16(accu3, _mm256_srli_epi16(res_hi, 8));
        }
    }

    accu0 = _mm256_sub_epi16(accu0, _mm256_slli_epi16(accu1, 8));
    accu2 = _mm256_sub_epi16(accu2, _mm256_slli_epi16(accu3, 8));
    _mm256_storeu_si256((__m256i *)out, rabitq_fastscan_combine_avx2(accu0, accu1));
    _mm256_storeu_si256((__m256i *)(out + 16),
                        rabitq_fastscan_combine_avx2(accu2, accu3));
}

void RABITQ_FN(rabitq_fastscan_transfer_hacc)(const uint16_t *lut, uint8_t *out,
                                              int64_t dim) {
    const int64_t num_codebook = dim >> 2;
    for (int64_t i = 0; i < num_codebook; ++i) {
        uint8_t *fill_lo = out + (i / 2) * 64 + (i % 2) * 16;
        uint8_t *fill_hi = fill_lo + 32;
        for (int entry = 0; entry < 16; ++entry) {
            const uint16_t value = lut[entry];
            fill_lo[entry] = (uint8_t)value;
            fill_hi[entry] = (uint8_t)(value >> 8);
        }
        lut += 16;
    }
}

void RABITQ_FN(rabitq_fastscan_accumulate_hacc)(const uint8_t *codes,
                                                const uint8_t *lut,
                                                int32_t *out, int64_t dim) {
    const __m256i low_mask = _mm256_set1_epi8(0x0f);
    __m256i accu[2][4];
    for (int q = 0; q < 2; ++q)
        for (int k = 0; k < 4; ++k)
            accu[q][k] = _mm256_setzero_si256();

    const int64_t num_codebook = dim >> 2;
    for (int64_t m = 0; m < num_codebook; m += 2) {
        const __m256i c = _mm256_loadu_si256((const __m256i *)codes);
        const __m256i lo = _mm256_and_si256(c, low_mask);
        const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);
        codes += 32;

        for (int q = 0; q < 2; ++q) {
            const __m256i table = _mm256_loadu_si256((const __m256i *)lut);
            const __m256i res_lo = _mm256_shuffle_epi8(table, lo);
            const __m256i res_hi = _mm256_shuffle_epi8(table, hi);
            accu[q][0] = _mm256_add_epi16(accu[q][0], res_lo);
            accu[q][1] = _mm256_add_epi16(accu[q][1], _mm256_srli_epi16(res_lo, 8));
            accu[q][2] = _mm256_add_epi16(accu[q][2], res_hi);
            accu[q][3] = _mm256_add_epi16(accu[q][3], _mm256_srli_epi16(res_hi, 8));
            lut += 32;
        }
    }

    __m256i dis0[2];
    __m256i dis1[2];
    for (int q = 0; q < 2; ++q) {
        accu[q][0] = _mm256_sub_epi16(accu[q][0], _mm256_slli_epi16(accu[q][1], 8));
        dis0[q] = rabitq_fastscan_combine_avx2(accu[q][0], accu[q][1]);
        accu[q][2] = _mm256_sub_epi16(accu[q][2], _mm256_slli_epi16(accu[q][3], 8));
        dis1[q] = rabitq_fastscan_combine_avx2(accu[q][2], accu[q][3]);
    }

    const __m256i d00 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(dis0[0]));
    const __m256i d01 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(dis0[0], 1));
    const __m256i d10 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(dis0[1]));
    const __m256i d11 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(dis0[1], 1));
    _mm256_storeu_si256((__m256i *)out, _mm256_add_epi32(d00, _mm256_slli_epi32(d10, 8)));
    _mm256_storeu_si256((__m256i *)(out + 8),
                        _mm256_add_epi32(d01, _mm256_slli_epi32(d11, 8)));

    const __m256i e00 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(dis1[0]));
    const __m256i e01 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(dis1[0], 1));
    const __m256i e10 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(dis1[1]));
    const __m256i e11 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(dis1[1], 1));
    _mm256_storeu_si256((__m256i *)(out + 16),
                        _mm256_add_epi32(e00, _mm256_slli_epi32(e10, 8)));
    _mm256_storeu_si256((__m256i *)(out + 24),
                        _mm256_add_epi32(e01, _mm256_slli_epi32(e11, 8)));
}

#endif

#undef RABITQ_FASTSCAN_INLINE

#endif  // XVEC_RABITQ_FASTSCAN_KERNELS_H_
