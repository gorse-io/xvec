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

// Ported from VectorDB-NTU/RaBitQ-Library src/simd/fastscan_avx512.cpp at
// revision 540242ea0a68926f1b827bf1f9add844f07a427b.

void accumulate_avx512(
    const uint8_t *codes, const uint8_t *lp_table, uint16_t *result, int64_t dim
) {
    const int64_t code_length = dim << 2;
    volatile uint8_t mask_value = 0x0f;
    const __m512i low_mask = _mm512_set1_epi8((char)mask_value);
    __m512i accu0 = _mm512_setzero_si512();
    __m512i accu1 = _mm512_setzero_si512();
    __m512i accu2 = _mm512_setzero_si512();
    __m512i accu3 = _mm512_setzero_si512();

    for (int64_t i = 0; i < code_length; i += 64) {
        const __m512i c = _mm512_loadu_si512(codes + i);
        const __m512i lut = _mm512_loadu_si512(lp_table + i);
        const __m512i lo = _mm512_and_si512(c, low_mask);
        const __m512i hi = _mm512_and_si512(_mm512_srli_epi16(c, 4), low_mask);
        const __m512i res_lo = _mm512_shuffle_epi8(lut, lo);
        const __m512i res_hi = _mm512_shuffle_epi8(lut, hi);
        accu0 = _mm512_add_epi16(accu0, res_lo);
        accu1 = _mm512_add_epi16(accu1, _mm512_srli_epi16(res_lo, 8));
        accu2 = _mm512_add_epi16(accu2, res_hi);
        accu3 = _mm512_add_epi16(accu3, _mm512_srli_epi16(res_hi, 8));
    }

    accu0 = _mm512_sub_epi16(accu0, _mm512_slli_epi16(accu1, 8));
    accu2 = _mm512_sub_epi16(accu2, _mm512_slli_epi16(accu3, 8));
    const __m512i ret1 = _mm512_add_epi16(
        _mm512_mask_blend_epi64(0xf0, accu0, accu1),
        _mm512_shuffle_i64x2(accu0, accu1, 0x4e)
    );
    const __m512i ret2 = _mm512_add_epi16(
        _mm512_mask_blend_epi64(0xf0, accu2, accu3),
        _mm512_shuffle_i64x2(accu2, accu3, 0x4e)
    );
    __m512i ret = _mm512_shuffle_i64x2(ret1, ret2, 0x88);
    ret = _mm512_add_epi16(ret, _mm512_shuffle_i64x2(ret1, ret2, 0xdd));
    _mm512_storeu_si512(result, ret);
}

void transfer_lut_hacc_avx512(const uint16_t *lut, int64_t dim, uint8_t *hc_lut) {
    const int64_t num_codebook = dim >> 2;
    for (int64_t i = 0; i < num_codebook; ++i) {
        uint8_t *fill_lo = hc_lut + (i / 4 * 128) + ((i % 4) * 16);
        uint8_t *fill_hi = fill_lo + 64;
        const __m512i tmp = _mm512_cvtepi16_epi32(
            _mm256_loadu_si256((const __m256i *)lut)
        );
        const __m128i lo = _mm512_cvtepi32_epi8(tmp);
        const __m128i hi = _mm512_cvtepi32_epi8(_mm512_srli_epi32(tmp, 8));
        _mm_storeu_si128((__m128i *)fill_lo, lo);
        _mm_storeu_si128((__m128i *)fill_hi, hi);
        lut += 16;
    }
}

void accumulate_hacc_avx512(
    const uint8_t *codes, const uint8_t *hc_lut, int32_t *result, int64_t dim
) {
    volatile uint8_t mask_value = 0xf;
    const __m512i low_mask = _mm512_set1_epi8((char)mask_value);
    __m512i accu[2][4];
    for (int q = 0; q < 2; ++q) {
        for (int i = 0; i < 4; ++i) {
            accu[q][i] = _mm512_setzero_si512();
        }
    }

    const int64_t num_codebook = dim >> 2;
    for (int64_t m = 0; m < num_codebook; m += 4) {
        const __m512i c = _mm512_loadu_si512(codes);
        const __m512i lo = _mm512_and_si512(c, low_mask);
        const __m512i hi = _mm512_and_si512(_mm512_srli_epi16(c, 4), low_mask);
        for (int q = 0; q < 2; ++q) {
            const __m512i lut = _mm512_loadu_si512(hc_lut);
            const __m512i res_lo = _mm512_shuffle_epi8(lut, lo);
            const __m512i res_hi = _mm512_shuffle_epi8(lut, hi);
            accu[q][0] = _mm512_add_epi16(accu[q][0], res_lo);
            accu[q][1] = _mm512_add_epi16(accu[q][1], _mm512_srli_epi16(res_lo, 8));
            accu[q][2] = _mm512_add_epi16(accu[q][2], res_hi);
            accu[q][3] = _mm512_add_epi16(accu[q][3], _mm512_srli_epi16(res_hi, 8));
            hc_lut += 64;
        }
        codes += 64;
    }

    __m512i dis0[2];
    __m512i dis1[2];
    for (int i = 0; i < 2; ++i) {
        const __m256i tmp1 = _mm256_add_epi16(
            _mm512_castsi512_si256(accu[i][1]), _mm512_extracti64x4_epi64(accu[i][1], 1)
        );
        __m256i tmp0 = _mm256_add_epi16(
            _mm512_castsi512_si256(accu[i][0]), _mm512_extracti64x4_epi64(accu[i][0], 1)
        );
        tmp0 = _mm256_sub_epi16(tmp0, _mm256_slli_epi16(tmp1, 8));
        dis0[i] = _mm512_add_epi32(
            _mm512_cvtepu16_epi32(_mm256_permute2f128_si256(tmp0, tmp1, 0x21)),
            _mm512_cvtepu16_epi32(_mm256_blend_epi32(tmp0, tmp1, 0xf0))
        );

        const __m256i tmp3 = _mm256_add_epi16(
            _mm512_castsi512_si256(accu[i][3]), _mm512_extracti64x4_epi64(accu[i][3], 1)
        );
        __m256i tmp2 = _mm256_add_epi16(
            _mm512_castsi512_si256(accu[i][2]), _mm512_extracti64x4_epi64(accu[i][2], 1)
        );
        tmp2 = _mm256_sub_epi16(tmp2, _mm256_slli_epi16(tmp3, 8));
        dis1[i] = _mm512_add_epi32(
            _mm512_cvtepu16_epi32(_mm256_permute2f128_si256(tmp2, tmp3, 0x21)),
            _mm512_cvtepu16_epi32(_mm256_blend_epi32(tmp2, tmp3, 0xf0))
        );
    }

    _mm512_storeu_si512(result, _mm512_add_epi32(dis0[0], _mm512_slli_epi32(dis0[1], 8)));
    _mm512_storeu_si512(result + 16, _mm512_add_epi32(dis1[0], _mm512_slli_epi32(dis1[1], 8)));
}
