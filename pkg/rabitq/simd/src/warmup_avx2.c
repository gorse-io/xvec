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

// Ported from VectorDB-NTU/RaBitQ-Library src/simd/warmup_avx2.cpp at
// revision 540242ea0a68926f1b827bf1f9add844f07a427b.

static inline __m256i popcount_avx2(__m256i value) {
    volatile uint64_t lookup0 = 0x0302020102010100ULL;
    volatile uint64_t lookup1 = 0x0403030203020201ULL;
    volatile uint64_t lookup2 = 0x0302020102010100ULL;
    volatile uint64_t lookup3 = 0x0403030203020201ULL;
    volatile uint8_t mask_value = 0x0f;
    const __m256i lookup = _mm256_set_epi64x(
        (int64_t)lookup3, (int64_t)lookup2, (int64_t)lookup1, (int64_t)lookup0
    );
    const __m256i low_mask = _mm256_set1_epi8((char)mask_value);
    const __m256i low = _mm256_and_si256(value, low_mask);
    const __m256i high = _mm256_and_si256(_mm256_srli_epi16(value, 4), low_mask);
    const __m256i counts = _mm256_add_epi8(
        _mm256_shuffle_epi8(lookup, low),
        _mm256_shuffle_epi8(lookup, high)
    );
    return _mm256_sad_epu8(counts, _mm256_setzero_si256());
}

static inline uint64_t reduce_add_epi64_avx2(__m256i value) {
    const __m128i sum = _mm_add_epi64(
        _mm256_castsi256_si128(value),
        _mm256_extracti128_si256(value, 1)
    );
    return (uint64_t)_mm_extract_epi64(sum, 0) + (uint64_t)_mm_extract_epi64(sum, 1);
}

float warmup_ip_x0_q_512_avx2(
    const uint64_t *data,
    const uint64_t *query,
    float delta,
    float vl,
    int64_t padded_dimension,
    int64_t query_bits
) {
    __m256i accumulated_ip = _mm256_setzero_si256();
    __m256i accumulated_popcount = _mm256_setzero_si256();
    __m256i accumulated_bits[8];
    for (int64_t bit = 0; bit < query_bits; ++bit) {
        accumulated_bits[bit] = _mm256_setzero_si256();
    }

    int64_t dimension = 0;
    const int64_t dimension_end_512 = padded_dimension & ~511LL;
    for (; dimension < dimension_end_512; dimension += 512) {
        const __m256i data_low = _mm256_loadu_si256((const __m256i *)data);
        const __m256i data_high = _mm256_loadu_si256((const __m256i *)(data + 4));
        data += 8;
        accumulated_popcount = _mm256_add_epi64(accumulated_popcount, popcount_avx2(data_low));
        accumulated_popcount = _mm256_add_epi64(accumulated_popcount, popcount_avx2(data_high));
        for (int64_t bit = 0; bit < query_bits; ++bit) {
            const __m256i query_low = _mm256_loadu_si256((const __m256i *)query);
            const __m256i query_high = _mm256_loadu_si256((const __m256i *)(query + 4));
            query += 8;
            accumulated_bits[bit] = _mm256_add_epi64(
                accumulated_bits[bit],
                popcount_avx2(_mm256_and_si256(data_low, query_low))
            );
            accumulated_bits[bit] = _mm256_add_epi64(
                accumulated_bits[bit],
                popcount_avx2(_mm256_and_si256(data_high, query_high))
            );
        }
    }

    const int64_t remaining_dimension = padded_dimension - dimension;
    if (remaining_dimension > 0) {
        const int64_t chunks64 = remaining_dimension / 64;
        const int64_t chunks32 = remaining_dimension / 32;
        const int64_t low_chunks = chunks32 < 8 ? chunks32 : 8;
        const int64_t high_chunks = chunks32 > 8 ? chunks32 - 8 : 0;
        volatile int32_t sequence0 = 0;
        volatile int32_t sequence1 = 1;
        volatile int32_t sequence2 = 2;
        volatile int32_t sequence3 = 3;
        volatile int32_t sequence4 = 4;
        volatile int32_t sequence5 = 5;
        volatile int32_t sequence6 = 6;
        volatile int32_t sequence7 = 7;
        const __m256i sequence = _mm256_setr_epi32(
            sequence0, sequence1, sequence2, sequence3,
            sequence4, sequence5, sequence6, sequence7
        );
        const __m256i low_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32((int)low_chunks), sequence);
        const __m256i high_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32((int)high_chunks), sequence);
        const __m256i data_low = _mm256_maskload_epi32((const int *)data, low_mask);
        const __m256i data_high = _mm256_maskload_epi32((const int *)(data + 4), high_mask);
        accumulated_popcount = _mm256_add_epi64(accumulated_popcount, popcount_avx2(data_low));
        accumulated_popcount = _mm256_add_epi64(accumulated_popcount, popcount_avx2(data_high));
        for (int64_t bit = 0; bit < query_bits; ++bit) {
            const __m256i query_low = _mm256_maskload_epi32((const int *)query, low_mask);
            const __m256i query_high = _mm256_maskload_epi32((const int *)(query + 4), high_mask);
            query += chunks64;
            accumulated_bits[bit] = _mm256_add_epi64(
                accumulated_bits[bit],
                popcount_avx2(_mm256_and_si256(data_low, query_low))
            );
            accumulated_bits[bit] = _mm256_add_epi64(
                accumulated_bits[bit],
                popcount_avx2(_mm256_and_si256(data_high, query_high))
            );
        }
    }

    for (int64_t bit = 0; bit < query_bits; ++bit) {
        const __m128i shift = _mm_cvtsi64_si128(bit);
        accumulated_ip = _mm256_add_epi64(
            accumulated_ip,
            _mm256_sll_epi64(accumulated_bits[bit], shift)
        );
    }
    const uint64_t ip = reduce_add_epi64_avx2(accumulated_ip);
    const uint64_t popcount = reduce_add_epi64_avx2(accumulated_popcount);
    return delta * (float)ip + vl * (float)popcount;
}
