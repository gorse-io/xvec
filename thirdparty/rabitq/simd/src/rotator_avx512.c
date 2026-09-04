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
#include <string.h>

// Ported from VectorDB-NTU/RaBitQ-Library src/simd/rotator_avx512.cpp at
// revision 540242ea0a68926f1b827bf1f9add844f07a427b.

void flip_sign_avx512(const uint8_t *flip, float *data, int64_t dim) {
    volatile int32_t sign = (int32_t)0x80000000U;
    const __m512 sign_flip = _mm512_castsi512_ps(_mm512_set1_epi32(sign));

    for (int64_t i = 0; i < dim; i += 64) {
        uint64_t mask_bits;
        __builtin_memcpy(&mask_bits, flip + i / 8, sizeof(mask_bits));

        for (int64_t block = 0; block < 4; block++) {
            const __mmask16 mask = (__mmask16)(mask_bits >> (block * 16));
            __m512 values = _mm512_loadu_ps(data + i + block * 16);
            values = _mm512_mask_xor_ps(values, mask, values, sign_flip);
            _mm512_storeu_ps(data + i + block * 16, values);
        }
    }
}

void kacs_walk_avx512(float *data, int64_t len) {
    for (int64_t i = 0; i < len / 2; i += 16) {
        const __m512 x = _mm512_loadu_ps(data + i);
        const __m512 y = _mm512_loadu_ps(data + i + len / 2);

        _mm512_storeu_ps(data + i, _mm512_add_ps(x, y));
        _mm512_storeu_ps(data + i + len / 2, _mm512_sub_ps(x, y));
    }
}
