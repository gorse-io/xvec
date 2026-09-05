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

// Ported from VectorDB-NTU/RaBitQ-Library src/simd/warmup_avx512.cpp at
// revision 540242ea0a68926f1b827bf1f9add844f07a427b.

float warmup_ip_x0_q_512_avx512(
    const uint64_t *data,
    const uint64_t *query,
    float delta,
    float vl,
    int64_t padded_dimension,
    int64_t query_bits
) {
    __m512i accumulated_ip = _mm512_setzero_si512();
    __m512i accumulated_popcount = _mm512_setzero_si512();
    __m512i accumulated_bits[8];
    for (int64_t bit = 0; bit < query_bits; ++bit) {
        accumulated_bits[bit] = _mm512_setzero_si512();
    }

    int64_t dimension = 0;
    const int64_t dimension_end_512 = padded_dimension & ~511LL;
    for (; dimension < dimension_end_512; dimension += 512) {
        const __m512i data_values = _mm512_loadu_si512(data);
        data += 8;
        accumulated_popcount = _mm512_add_epi64(
            accumulated_popcount, _mm512_popcnt_epi64(data_values)
        );
        for (int64_t bit = 0; bit < query_bits; ++bit) {
            const __m512i query_values = _mm512_loadu_si512(query);
            query += 8;
            accumulated_bits[bit] = _mm512_add_epi64(
                accumulated_bits[bit],
                _mm512_popcnt_epi64(_mm512_and_si512(data_values, query_values))
            );
        }
    }

    const int64_t remaining_dimension = padded_dimension - dimension;
    if (remaining_dimension > 0) {
        const int64_t chunks = remaining_dimension / 64;
        const __mmask8 valid = (__mmask8)((1U << chunks) - 1U);
        const __m512i data_values = _mm512_maskz_loadu_epi64(valid, data);
        accumulated_popcount = _mm512_add_epi64(
            accumulated_popcount, _mm512_popcnt_epi64(data_values)
        );
        for (int64_t bit = 0; bit < query_bits; ++bit) {
            const __m512i query_values = _mm512_maskz_loadu_epi64(valid, query);
            query += chunks;
            accumulated_bits[bit] = _mm512_add_epi64(
                accumulated_bits[bit],
                _mm512_popcnt_epi64(_mm512_and_si512(data_values, query_values))
            );
        }
    }

    for (int64_t bit = 0; bit < query_bits; ++bit) {
        const __m128i shift = _mm_cvtsi64_si128(bit);
        accumulated_ip = _mm512_add_epi64(
            accumulated_ip,
            _mm512_sll_epi64(accumulated_bits[bit], shift)
        );
    }
    const uint64_t ip = (uint64_t)_mm512_reduce_add_epi64(accumulated_ip);
    const uint64_t popcount = (uint64_t)_mm512_reduce_add_epi64(accumulated_popcount);
    return delta * (float)ip + vl * (float)popcount;
}
