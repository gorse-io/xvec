// Copyright 2026-present the xvec project
// Licensed under the Apache License, Version 2.0.
//
// This file is derived from VectorDB-NTU/RaBitQ-Library src/simd.
// It was changed to provide C intrinsic templates suitable for GoAT.

#ifndef XVEC_RABITQ_WARMUP_KERNELS_H_
#define XVEC_RABITQ_WARMUP_KERNELS_H_

#include <immintrin.h>
#include <stdint.h>

#ifndef RABITQ_FN
#error RABITQ_FN must be defined before including warmup_kernels.h
#endif

#define RABITQ_WARMUP_INLINE __attribute__((always_inline)) static inline

#if RABITQ_AVX512

float RABITQ_FN(rabitq_warmup_ip)(const uint64_t *data, const uint64_t *query,
                                 int64_t words, int64_t width, float delta, float vl) {
    __m512i acc_ip = _mm512_setzero_si512();
    __m512i acc_pop = _mm512_setzero_si512();
    __m512i acc_bits[16];
    for (int64_t bit = 0; bit < width; ++bit)
        acc_bits[bit] = _mm512_setzero_si512();

    int64_t data_base = 0;
    for (; data_base + 8 <= words; data_base += 8) {
        const __m512i data_vec = _mm512_loadu_si512((const void *)data);
        data += 8;
        acc_pop = _mm512_add_epi64(acc_pop, _mm512_popcnt_epi64(data_vec));

        for (int64_t bit = 0; bit < width; ++bit) {
            const __m512i query_vec = _mm512_loadu_si512((const void *)query);
            query += 8;
            acc_bits[bit] = _mm512_add_epi64(
                acc_bits[bit],
                _mm512_popcnt_epi64(_mm512_and_si512(data_vec, query_vec)));
        }
    }

    const int64_t remaining = words - data_base;
    if (remaining != 0) {
        const __mmask8 valid = (__mmask8)((1u << (unsigned)remaining) - 1u);
        const __m512i data_vec = _mm512_maskz_loadu_epi64(valid, (const void *)data);
        acc_pop = _mm512_add_epi64(acc_pop, _mm512_popcnt_epi64(data_vec));

        for (int64_t bit = 0; bit < width; ++bit) {
            const __m512i query_vec = _mm512_maskz_loadu_epi64(valid, (const void *)query);
            query += remaining;
            acc_bits[bit] = _mm512_add_epi64(
                acc_bits[bit],
                _mm512_popcnt_epi64(_mm512_and_si512(data_vec, query_vec)));
        }
    }

    for (int64_t bit = 0; bit < width; ++bit) {
        const __m128i shift = _mm_cvtsi64_si128(bit);
        /* These are intentionally wrapping 64-bit lane operations. */
        acc_ip = _mm512_add_epi64(acc_ip, _mm512_sll_epi64(acc_bits[bit], shift));
    }

    const uint64_t ip = (uint64_t)_mm512_reduce_add_epi64(acc_ip);
    const uint64_t pop = (uint64_t)_mm512_reduce_add_epi64(acc_pop);
    return delta * (float)ip + vl * (float)pop;
}

#else

RABITQ_WARMUP_INLINE __m256i rabitq_warmup_popcount_avx2(__m256i value) {
    const __m256i lookup = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    const __m256i low_mask = _mm256_set1_epi8(0x0f);
    const __m256i lo = _mm256_and_si256(value, low_mask);
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(value, 4), low_mask);
    const __m256i byte_counts = _mm256_add_epi8(
        _mm256_shuffle_epi8(lookup, lo), _mm256_shuffle_epi8(lookup, hi));
    return _mm256_sad_epu8(byte_counts, _mm256_setzero_si256());
}

RABITQ_WARMUP_INLINE uint64_t rabitq_warmup_reduce_avx2(__m256i value) {
    const __m128i sum = _mm_add_epi64(_mm256_castsi256_si128(value),
                                      _mm256_extracti128_si256(value, 1));
    return (uint64_t)_mm_cvtsi128_si64(sum) +
           (uint64_t)_mm_cvtsi128_si64(_mm_unpackhi_epi64(sum, sum));
}

float RABITQ_FN(rabitq_warmup_ip)(const uint64_t *data, const uint64_t *query,
                                 int64_t words, int64_t width, float delta, float vl) {
    __m256i acc_ip = _mm256_setzero_si256();
    __m256i acc_pop = _mm256_setzero_si256();
    __m256i acc_bits[16];
    for (int64_t bit = 0; bit < width; ++bit)
        acc_bits[bit] = _mm256_setzero_si256();

    int64_t data_base = 0;
    for (; data_base + 8 <= words; data_base += 8) {
        const __m256i data_lo = _mm256_loadu_si256((const __m256i *)data);
        const __m256i data_hi = _mm256_loadu_si256((const __m256i *)(data + 4));
        data += 8;
        acc_pop = _mm256_add_epi64(acc_pop, rabitq_warmup_popcount_avx2(data_lo));
        acc_pop = _mm256_add_epi64(acc_pop, rabitq_warmup_popcount_avx2(data_hi));

        for (int64_t bit = 0; bit < width; ++bit) {
            const __m256i query_lo = _mm256_loadu_si256((const __m256i *)query);
            const __m256i query_hi = _mm256_loadu_si256((const __m256i *)(query + 4));
            query += 8;
            acc_bits[bit] = _mm256_add_epi64(
                acc_bits[bit],
                rabitq_warmup_popcount_avx2(_mm256_and_si256(data_lo, query_lo)));
            acc_bits[bit] = _mm256_add_epi64(
                acc_bits[bit],
                rabitq_warmup_popcount_avx2(_mm256_and_si256(data_hi, query_hi)));
        }
    }

    const int64_t remaining = words - data_base;
    if (remaining != 0) {
        const int32_t lo_words = (int32_t)(remaining < 4 ? remaining : 4);
        const int32_t hi_words = (int32_t)(remaining > 4 ? remaining - 4 : 0);
        const __m256i sequence = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        const __m256i lo_limit = _mm256_set1_epi32(lo_words * 2);
        const __m256i hi_limit = _mm256_set1_epi32(hi_words * 2);
        const __m256i lo_mask = _mm256_cmpgt_epi32(lo_limit, sequence);
        const __m256i hi_mask = _mm256_cmpgt_epi32(hi_limit, sequence);
        const __m256i data_lo = _mm256_maskload_epi32((const int *)data, lo_mask);
        const __m256i data_hi = _mm256_maskload_epi32((const int *)(data + 4), hi_mask);
        acc_pop = _mm256_add_epi64(acc_pop, rabitq_warmup_popcount_avx2(data_lo));
        acc_pop = _mm256_add_epi64(acc_pop, rabitq_warmup_popcount_avx2(data_hi));

        for (int64_t bit = 0; bit < width; ++bit) {
            const __m256i query_lo = _mm256_maskload_epi32((const int *)query, lo_mask);
            const __m256i query_hi = _mm256_maskload_epi32((const int *)(query + 4), hi_mask);
            query += remaining;
            acc_bits[bit] = _mm256_add_epi64(
                acc_bits[bit],
                rabitq_warmup_popcount_avx2(_mm256_and_si256(data_lo, query_lo)));
            acc_bits[bit] = _mm256_add_epi64(
                acc_bits[bit],
                rabitq_warmup_popcount_avx2(_mm256_and_si256(data_hi, query_hi)));
        }
    }

    for (int64_t bit = 0; bit < width; ++bit) {
        const __m128i shift = _mm_cvtsi64_si128(bit);
        /* These are intentionally wrapping 64-bit lane operations. */
        acc_ip = _mm256_add_epi64(acc_ip, _mm256_sll_epi64(acc_bits[bit], shift));
    }

    const uint64_t ip = rabitq_warmup_reduce_avx2(acc_ip);
    const uint64_t pop = rabitq_warmup_reduce_avx2(acc_pop);
    return delta * (float)ip + vl * (float)pop;
}

#endif

#undef RABITQ_WARMUP_INLINE

#endif  // XVEC_RABITQ_WARMUP_KERNELS_H_
