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

// Ported from VectorDB-NTU/RaBitQ-Library src/simd/fastscan_avx2.cpp at
// revision 540242ea0a68926f1b827bf1f9add844f07a427b.

void accumulate_avx2(
    const uint8_t *codes, const uint8_t *lp_table, uint16_t *result, int64_t dim
) {
    const int64_t code_length = dim << 2;
    volatile uint8_t mask_value = 0xf;
    const __m256i low_mask = _mm256_set1_epi8((char)mask_value);
    __m256i accu0 = _mm256_setzero_si256();
    __m256i accu1 = _mm256_setzero_si256();
    __m256i accu2 = _mm256_setzero_si256();
    __m256i accu3 = _mm256_setzero_si256();

    for (int64_t i = 0; i < code_length; i += 64) {
        __m256i c = _mm256_loadu_si256((const __m256i *)(codes + i));
        __m256i lut = _mm256_loadu_si256((const __m256i *)(lp_table + i));
        __m256i lo = _mm256_and_si256(c, low_mask);
        __m256i hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);
        __m256i res_lo = _mm256_shuffle_epi8(lut, lo);
        __m256i res_hi = _mm256_shuffle_epi8(lut, hi);
        accu0 = _mm256_add_epi16(accu0, res_lo);
        accu1 = _mm256_add_epi16(accu1, _mm256_srli_epi16(res_lo, 8));
        accu2 = _mm256_add_epi16(accu2, res_hi);
        accu3 = _mm256_add_epi16(accu3, _mm256_srli_epi16(res_hi, 8));

        c = _mm256_loadu_si256((const __m256i *)(codes + i + 32));
        lut = _mm256_loadu_si256((const __m256i *)(lp_table + i + 32));
        lo = _mm256_and_si256(c, low_mask);
        hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);
        res_lo = _mm256_shuffle_epi8(lut, lo);
        res_hi = _mm256_shuffle_epi8(lut, hi);
        accu0 = _mm256_add_epi16(accu0, res_lo);
        accu1 = _mm256_add_epi16(accu1, _mm256_srli_epi16(res_lo, 8));
        accu2 = _mm256_add_epi16(accu2, res_hi);
        accu3 = _mm256_add_epi16(accu3, _mm256_srli_epi16(res_hi, 8));
    }

    accu0 = _mm256_sub_epi16(accu0, _mm256_slli_epi16(accu1, 8));
    const __m256i dis0 = _mm256_add_epi16(
        _mm256_permute2f128_si256(accu0, accu1, 0x21),
        _mm256_blend_epi32(accu0, accu1, 0xf0)
    );
    _mm256_storeu_si256((__m256i *)result, dis0);

    accu2 = _mm256_sub_epi16(accu2, _mm256_slli_epi16(accu3, 8));
    const __m256i dis1 = _mm256_add_epi16(
        _mm256_permute2f128_si256(accu2, accu3, 0x21),
        _mm256_blend_epi32(accu2, accu3, 0xf0)
    );
    _mm256_storeu_si256((__m256i *)(result + 16), dis1);
}

void transfer_lut_hacc_avx2(const uint16_t *lut, int64_t dim, uint8_t *hc_lut) {
    const int64_t num_codebook = dim >> 2;
    for (int64_t i = 0; i < num_codebook; ++i) {
        uint8_t *fill_lo = hc_lut + (i / 2 * 64) + ((i % 2) * 16);
        uint8_t *fill_hi = fill_lo + 32;
        for (int64_t j = 0; j < 16; ++j) {
            const uint16_t value = lut[j];
            fill_lo[j] = (uint8_t)value;
            fill_hi[j] = (uint8_t)(value >> 8);
        }
        lut += 16;
    }
}

static inline __m256i combine2x2_avx2(__m256i a, __m256i b) {
    return _mm256_add_epi16(
        _mm256_permute2f128_si256(a, b, 0x21), _mm256_blend_epi32(a, b, 0xf0)
    );
}

void accumulate_hacc_avx2(
    const uint8_t *codes, const uint8_t *hc_lut, int32_t *result, int64_t dim
) {
    volatile uint8_t mask_value = 0xf;
    const __m256i low_mask = _mm256_set1_epi8((char)mask_value);
    __m256i accu[2][4];
    for (int q = 0; q < 2; ++q) {
        for (int i = 0; i < 4; ++i) {
            accu[q][i] = _mm256_setzero_si256();
        }
    }

    const int64_t num_codebook = dim >> 2;
    for (int64_t m = 0; m < num_codebook; m += 2) {
        const __m256i c = _mm256_loadu_si256((const __m256i *)codes);
        codes += 32;
        const __m256i lo = _mm256_and_si256(c, low_mask);
        const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);
        for (int q = 0; q < 2; ++q) {
            const __m256i lut = _mm256_loadu_si256((const __m256i *)hc_lut);
            hc_lut += 32;
            const __m256i res_lo = _mm256_shuffle_epi8(lut, lo);
            const __m256i res_hi = _mm256_shuffle_epi8(lut, hi);
            accu[q][0] = _mm256_add_epi16(accu[q][0], res_lo);
            accu[q][1] = _mm256_add_epi16(accu[q][1], _mm256_srli_epi16(res_lo, 8));
            accu[q][2] = _mm256_add_epi16(accu[q][2], res_hi);
            accu[q][3] = _mm256_add_epi16(accu[q][3], _mm256_srli_epi16(res_hi, 8));
        }
    }

    __m256i dis0[2];
    __m256i dis1[2];
    for (int i = 0; i < 2; ++i) {
        accu[i][0] = _mm256_sub_epi16(accu[i][0], _mm256_slli_epi16(accu[i][1], 8));
        dis0[i] = combine2x2_avx2(accu[i][0], accu[i][1]);
        accu[i][2] = _mm256_sub_epi16(accu[i][2], _mm256_slli_epi16(accu[i][3], 8));
        dis1[i] = combine2x2_avx2(accu[i][2], accu[i][3]);
    }

    const __m256i a00 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(dis0[0]));
    const __m256i a01 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(dis0[0], 1));
    const __m256i a10 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(dis0[1]));
    const __m256i a11 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(dis0[1], 1));
    const __m256i b00 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(dis1[0]));
    const __m256i b01 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(dis1[0], 1));
    const __m256i b10 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(dis1[1]));
    const __m256i b11 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(dis1[1], 1));

    _mm256_storeu_si256((__m256i *)result, _mm256_add_epi32(a00, _mm256_slli_epi32(a10, 8)));
    _mm256_storeu_si256((__m256i *)(result + 8), _mm256_add_epi32(a01, _mm256_slli_epi32(a11, 8)));
    _mm256_storeu_si256((__m256i *)(result + 16), _mm256_add_epi32(b00, _mm256_slli_epi32(b10, 8)));
    _mm256_storeu_si256((__m256i *)(result + 24), _mm256_add_epi32(b01, _mm256_slli_epi32(b11, 8)));
}
