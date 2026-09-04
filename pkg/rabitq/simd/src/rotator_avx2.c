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

// Ported from VectorDB-NTU/RaBitQ-Library src/simd/rotator_avx2.cpp at
// revision 540242ea0a68926f1b827bf1f9add844f07a427b.

void flip_sign_avx2(const uint8_t *flip, float *data, int64_t dim) {
    for (int64_t i = 0; i < dim; i += 32) {
        uint32_t mask_bits;
        __builtin_memcpy(&mask_bits, flip + i / 8, sizeof(mask_bits));

        for (int64_t block = 0; block < 4; block++) {
            uint64_t byte = (mask_bits >> (block * 8)) & 0xffU;
            volatile uint64_t mask0 = ((byte & 0x01U) << 31) | ((byte & 0x02U) << 62);
            volatile uint64_t mask1 = ((byte & 0x04U) << 29) | ((byte & 0x08U) << 60);
            volatile uint64_t mask2 = ((byte & 0x10U) << 27) | ((byte & 0x20U) << 58);
            volatile uint64_t mask3 = ((byte & 0x40U) << 25) | ((byte & 0x80U) << 56);
            const __m256i mask = _mm256_set_epi64x(mask3, mask2, mask1, mask0);
            __m256 values = _mm256_loadu_ps(data + i + block * 8);
            values = _mm256_xor_ps(values, _mm256_castsi256_ps(mask));
            _mm256_storeu_ps(data + i + block * 8, values);
        }
    }
}

void kacs_walk_avx2(float *data, int64_t len) {
    for (int64_t i = 0; i < len / 2; i += 8) {
        const __m256 x = _mm256_loadu_ps(data + i);
        const __m256 y = _mm256_loadu_ps(data + i + len / 2);

        _mm256_storeu_ps(data + i, _mm256_add_ps(x, y));
        _mm256_storeu_ps(data + i + len / 2, _mm256_sub_ps(x, y));
    }
}
