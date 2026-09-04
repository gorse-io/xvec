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

// Ported from VectorDB-NTU/RaBitQ-Library src/simd/pack_excode_kernels.hpp at
// revision 540242ea0a68926f1b827bf1f9add844f07a427b.

static inline void packing_2bit_excode_intrinsics(
    const uint8_t *raw, uint8_t *compact, int64_t dimension
) {
    for (int64_t offset = 0; offset < dimension; offset += 64) {
        const __m128i values0 = _mm_loadu_si128((const __m128i *)(raw));
        const __m128i values1 = _mm_loadu_si128((const __m128i *)(raw + 16));
        const __m128i values2 = _mm_loadu_si128((const __m128i *)(raw + 32));
        const __m128i values3 = _mm_loadu_si128((const __m128i *)(raw + 48));
        const __m128i packed = _mm_or_si128(
            _mm_or_si128(values0, _mm_slli_epi16(values1, 2)),
            _mm_or_si128(_mm_slli_epi16(values2, 4), _mm_slli_epi16(values3, 6))
        );
        _mm_storeu_si128((__m128i *)(compact), packed);
        raw += 64;
        compact += 16;
    }
}

static inline void packing_3bit_excode_intrinsics(
    const uint8_t *raw, uint8_t *compact, int64_t dimension
) {
    volatile char mask_value = 3;
    const __m128i mask = _mm_set1_epi8(mask_value);
    for (int64_t offset = 0; offset < dimension; offset += 64) {
        const __m128i raw0 = _mm_loadu_si128((const __m128i *)(raw));
        const __m128i raw1 = _mm_loadu_si128((const __m128i *)(raw + 16));
        const __m128i raw2 = _mm_loadu_si128((const __m128i *)(raw + 32));
        const __m128i raw3 = _mm_loadu_si128((const __m128i *)(raw + 48));
        const __m128i values0 = _mm_and_si128(raw0, mask);
        const __m128i values1 = _mm_slli_epi16(_mm_and_si128(raw1, mask), 2);
        const __m128i values2 = _mm_slli_epi16(_mm_and_si128(raw2, mask), 4);
        const __m128i values3 = _mm_slli_epi16(_mm_and_si128(raw3, mask), 6);
        const __m128i packed = _mm_or_si128(
            _mm_or_si128(values0, values1), _mm_or_si128(values2, values3)
        );
        _mm_storeu_si128((__m128i *)(compact), packed);
        const uint64_t top_bit =
            (uint16_t)_mm_movemask_epi8(_mm_slli_epi16(raw0, 5)) |
            ((uint64_t)(uint16_t)_mm_movemask_epi8(_mm_slli_epi16(raw1, 5)) << 16) |
            ((uint64_t)(uint16_t)_mm_movemask_epi8(_mm_slli_epi16(raw2, 5)) << 32) |
            ((uint64_t)(uint16_t)_mm_movemask_epi8(_mm_slli_epi16(raw3, 5)) << 48);
        _mm_storel_epi64((__m128i *)(compact + 16), _mm_cvtsi64_si128((int64_t)top_bit));
        raw += 64;
        compact += 24;
    }
}

static inline void packing_4bit_excode_intrinsics(
    const uint8_t *raw, uint8_t *compact, int64_t dimension
) {
#pragma clang loop vectorize(disable) interleave(disable)
    for (int64_t offset = 0; offset < dimension; offset += 16) {
        const uint64_t values0 = (uint64_t)_mm_cvtsi128_si64(
            _mm_loadl_epi64((const __m128i *)(raw))
        );
        const uint64_t values1 = (uint64_t)_mm_cvtsi128_si64(
            _mm_loadl_epi64((const __m128i *)(raw + 8))
        );
        const uint64_t packed = values0 | (values1 << 4);
        _mm_storel_epi64((__m128i *)(compact), _mm_cvtsi64_si128((int64_t)packed));
        raw += 16;
        compact += 8;
    }
}

static inline void packing_5bit_excode_intrinsics(
    const uint8_t *raw, uint8_t *compact, int64_t dimension
) {
    volatile char mask_value = 15;
    const __m128i mask = _mm_set1_epi8(mask_value);
    for (int64_t offset = 0; offset < dimension; offset += 64) {
        const __m128i raw0 = _mm_loadu_si128((const __m128i *)(raw));
        const __m128i raw1 = _mm_loadu_si128((const __m128i *)(raw + 16));
        const __m128i raw2 = _mm_loadu_si128((const __m128i *)(raw + 32));
        const __m128i raw3 = _mm_loadu_si128((const __m128i *)(raw + 48));
        const __m128i values0 = _mm_and_si128(raw0, mask);
        const __m128i values1 = _mm_slli_epi16(_mm_and_si128(raw1, mask), 4);
        const __m128i values2 = _mm_and_si128(raw2, mask);
        const __m128i values3 = _mm_slli_epi16(_mm_and_si128(raw3, mask), 4);
        _mm_storeu_si128((__m128i *)(compact), _mm_or_si128(values0, values1));
        _mm_storeu_si128((__m128i *)(compact + 16), _mm_or_si128(values2, values3));
        const uint64_t top_bit =
            (uint16_t)_mm_movemask_epi8(_mm_slli_epi16(raw0, 3)) |
            ((uint64_t)(uint16_t)_mm_movemask_epi8(_mm_slli_epi16(raw1, 3)) << 16) |
            ((uint64_t)(uint16_t)_mm_movemask_epi8(_mm_slli_epi16(raw2, 3)) << 32) |
            ((uint64_t)(uint16_t)_mm_movemask_epi8(_mm_slli_epi16(raw3, 3)) << 48);
        _mm_storel_epi64((__m128i *)(compact + 32), _mm_cvtsi64_si128((int64_t)top_bit));
        raw += 64;
        compact += 40;
    }
}

static inline void packing_6bit_body(
    const uint8_t *raw, uint8_t *compact, int64_t dimension, int top_bit
) {
    volatile char mask2_value = (char)0xc0;
    volatile char mask6_value = 0x3f;
    const __m128i mask2 = _mm_set1_epi8(mask2_value);
    const __m128i mask6 = _mm_set1_epi8(mask6_value);
    const int64_t compact_stride = top_bit ? 56 : 48;
    for (int64_t offset = 0; offset < dimension; offset += 64) {
        const __m128i values0 = _mm_loadu_si128((const __m128i *)(raw));
        const __m128i values1 = _mm_loadu_si128((const __m128i *)(raw + 16));
        const __m128i values2 = _mm_loadu_si128((const __m128i *)(raw + 32));
        const __m128i values3 = _mm_loadu_si128((const __m128i *)(raw + 48));
        _mm_storeu_si128(
            (__m128i *)(compact),
            _mm_or_si128(
                _mm_and_si128(values0, mask6),
                _mm_and_si128(_mm_slli_epi16(values3, 6), mask2)
            )
        );
        _mm_storeu_si128(
            (__m128i *)(compact + 16),
            _mm_or_si128(
                _mm_and_si128(values1, mask6),
                _mm_and_si128(_mm_slli_epi16(values3, 4), mask2)
            )
        );
        _mm_storeu_si128(
            (__m128i *)(compact + 32),
            _mm_or_si128(
                _mm_and_si128(values2, mask6),
                _mm_and_si128(_mm_slli_epi16(values3, 2), mask2)
            )
        );
        if (top_bit) {
            const uint64_t packed_top_bit =
                (uint16_t)_mm_movemask_epi8(_mm_slli_epi16(values0, 1)) |
                ((uint64_t)(uint16_t)_mm_movemask_epi8(_mm_slli_epi16(values1, 1)) << 16) |
                ((uint64_t)(uint16_t)_mm_movemask_epi8(_mm_slli_epi16(values2, 1)) << 32) |
                ((uint64_t)(uint16_t)_mm_movemask_epi8(_mm_slli_epi16(values3, 1)) << 48);
            _mm_storel_epi64(
                (__m128i *)(compact + 48), _mm_cvtsi64_si128((int64_t)packed_top_bit)
            );
        }
        raw += 64;
        compact += compact_stride;
    }
}

static inline void packing_6bit_excode_intrinsics(
    const uint8_t *raw, uint8_t *compact, int64_t dimension
) {
    packing_6bit_body(raw, compact, dimension, 0);
}

static inline void packing_7bit_excode_intrinsics(
    const uint8_t *raw, uint8_t *compact, int64_t dimension
) {
    packing_6bit_body(raw, compact, dimension, 1);
}
