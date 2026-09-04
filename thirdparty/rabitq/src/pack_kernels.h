// Copyright 2026-present the xvec project
// Licensed under the Apache License, Version 2.0.
//
// C intrinsic translation of VectorDB-NTU/RaBitQ-Library's
// src/simd/pack_excode_kernels.hpp.

#ifndef XVEC_RABITQ_PACK_KERNELS_H
#define XVEC_RABITQ_PACK_KERNELS_H

#include <immintrin.h>
#include <stdint.h>
#include <string.h>

#ifndef RABITQ_FN
#error RABITQ_FN must be defined before including pack_kernels.h
#endif
#ifndef RABITQ_AVX512
#error RABITQ_AVX512 must be defined to 0 or 1
#endif
#if RABITQ_AVX512 != 0 && RABITQ_AVX512 != 1
#error RABITQ_AVX512 must be 0 or 1
#endif

#if defined(__clang__) || defined(__GNUC__)
#define RABITQ_PACK_INLINE static inline __attribute__((always_inline))
#else
#define RABITQ_PACK_INLINE static inline
#endif

typedef uint16_t __attribute__((aligned(1), may_alias)) rabitq_pack_u16;
typedef uint64_t __attribute__((aligned(1), may_alias)) rabitq_pack_u64;

RABITQ_PACK_INLINE void rabitq_pack_1bit_intrinsics(
    const uint8_t *raw, uint8_t *compact, int64_t dim) {
    for (int64_t base = 0; base < dim; base += 64) {
        for (int part = 0; part < 4; ++part) {
            __m128i v = _mm_loadu_si128((const __m128i *)(raw + part * 16));
            uint16_t mask = (uint16_t)_mm_movemask_epi8(_mm_slli_epi16(v, 7));
            *(rabitq_pack_u16 *)(compact + part * 2) = mask;
        }
        raw += 64;
        compact += 8;
    }
}

RABITQ_PACK_INLINE void rabitq_pack_2bit_intrinsics(
    const uint8_t *raw, uint8_t *compact, int64_t dim) {
    for (int64_t base = 0; base < dim; base += 64) {
        __m128i v0 = _mm_loadu_si128((const __m128i *)(raw + 0));
        __m128i v1 = _mm_loadu_si128((const __m128i *)(raw + 16));
        __m128i v2 = _mm_loadu_si128((const __m128i *)(raw + 32));
        __m128i v3 = _mm_loadu_si128((const __m128i *)(raw + 48));
        __m128i packed = _mm_or_si128(
            _mm_or_si128(v0, _mm_slli_epi16(v1, 2)),
            _mm_or_si128(_mm_slli_epi16(v2, 4), _mm_slli_epi16(v3, 6)));
        _mm_storeu_si128((__m128i *)compact, packed);
        raw += 64;
        compact += 16;
    }
}

RABITQ_PACK_INLINE uint64_t rabitq_pack_top_bit_intrinsics(
    const uint8_t *raw, int bit) {
    uint64_t result = 0;
    for (int part = 0; part < 4; ++part) {
        const __m128i codes = _mm_loadu_si128((const __m128i *)(raw + part * 16));
        const uint16_t mask = (uint16_t)_mm_movemask_epi8(_mm_slli_epi16(codes, 7 - bit));
        result |= (uint64_t)mask << (part * 16);
    }
    return result;
}

RABITQ_PACK_INLINE void rabitq_pack_3bit_intrinsics(
    const uint8_t *raw, uint8_t *compact, int64_t dim) {
    const __m128i mask2 = _mm_set1_epi8(0x03);
    for (int64_t base = 0; base < dim; base += 64) {
        __m128i v0 = _mm_and_si128(_mm_loadu_si128((const __m128i *)(raw + 0)), mask2);
        __m128i v1 = _mm_slli_epi16(_mm_and_si128(_mm_loadu_si128((const __m128i *)(raw + 16)), mask2), 2);
        __m128i v2 = _mm_slli_epi16(_mm_and_si128(_mm_loadu_si128((const __m128i *)(raw + 32)), mask2), 4);
        __m128i v3 = _mm_slli_epi16(_mm_and_si128(_mm_loadu_si128((const __m128i *)(raw + 48)), mask2), 6);
        __m128i packed = _mm_or_si128(_mm_or_si128(v0, v1), _mm_or_si128(v2, v3));
        uint64_t top = rabitq_pack_top_bit_intrinsics(raw, 2);
        _mm_storeu_si128((__m128i *)compact, packed);
        *(rabitq_pack_u64 *)(compact + 16) = top;
        raw += 64;
        compact += 24;
    }
}

RABITQ_PACK_INLINE void rabitq_pack_4bit_intrinsics(
    const uint8_t *raw, uint8_t *compact, int64_t dim) {
    for (int64_t base = 0; base < dim; base += 16) {
        uint64_t lo = *(const rabitq_pack_u64 *)raw;
        uint64_t hi = *(const rabitq_pack_u64 *)(raw + 8);
        lo |= hi << 4;
        *(rabitq_pack_u64 *)compact = lo;
        raw += 16;
        compact += 8;
    }
}

RABITQ_PACK_INLINE void rabitq_pack_5bit_intrinsics(
    const uint8_t *raw, uint8_t *compact, int64_t dim) {
    for (int64_t base = 0; base < dim; base += 64) {
        for (int group = 0; group < 4; ++group) {
            uint64_t lo = *(const rabitq_pack_u64 *)(raw + group * 16);
            uint64_t hi = *(const rabitq_pack_u64 *)(raw + group * 16 + 8);
            *(rabitq_pack_u64 *)(compact + group * 8) =
                (lo & UINT64_C(0x0f0f0f0f0f0f0f0f)) |
                ((hi & UINT64_C(0x0f0f0f0f0f0f0f0f)) << 4);
        }
        uint64_t top = rabitq_pack_top_bit_intrinsics(raw, 4);
        *(rabitq_pack_u64 *)(compact + 32) = top;
        raw += 64;
        compact += 40;
    }
}

RABITQ_PACK_INLINE void rabitq_pack_6bit_intrinsics(
    const uint8_t *raw, uint8_t *compact, int64_t dim) {
    const __m128i mask6 = _mm_set1_epi8(0x3f);
    const __m128i mask2 = _mm_set1_epi8((char)0xc0);
    for (int64_t base = 0; base < dim; base += 64) {
        __m128i v0 = _mm_loadu_si128((const __m128i *)(raw + 0));
        __m128i v1 = _mm_loadu_si128((const __m128i *)(raw + 16));
        __m128i v2 = _mm_loadu_si128((const __m128i *)(raw + 32));
        __m128i v3 = _mm_loadu_si128((const __m128i *)(raw + 48));
        _mm_storeu_si128((__m128i *)(compact + 0), _mm_or_si128(_mm_and_si128(v0, mask6), _mm_and_si128(_mm_slli_epi16(v3, 6), mask2)));
        _mm_storeu_si128((__m128i *)(compact + 16), _mm_or_si128(_mm_and_si128(v1, mask6), _mm_and_si128(_mm_slli_epi16(v3, 4), mask2)));
        _mm_storeu_si128((__m128i *)(compact + 32), _mm_or_si128(_mm_and_si128(v2, mask6), _mm_and_si128(_mm_slli_epi16(v3, 2), mask2)));
        raw += 64;
        compact += 48;
    }
}

RABITQ_PACK_INLINE void rabitq_pack_7bit_intrinsics(
    const uint8_t *raw, uint8_t *compact, int64_t dim) {
    const __m128i mask6 = _mm_set1_epi8(0x3f);
    const __m128i mask2 = _mm_set1_epi8((char)0xc0);
    for (int64_t base = 0; base < dim; base += 64) {
        __m128i v0 = _mm_loadu_si128((const __m128i *)(raw + 0));
        __m128i v1 = _mm_loadu_si128((const __m128i *)(raw + 16));
        __m128i v2 = _mm_loadu_si128((const __m128i *)(raw + 32));
        __m128i v3 = _mm_loadu_si128((const __m128i *)(raw + 48));
        uint64_t top = rabitq_pack_top_bit_intrinsics(raw, 6);
        _mm_storeu_si128((__m128i *)(compact + 0), _mm_or_si128(_mm_and_si128(v0, mask6), _mm_and_si128(_mm_slli_epi16(v3, 6), mask2)));
        _mm_storeu_si128((__m128i *)(compact + 16), _mm_or_si128(_mm_and_si128(v1, mask6), _mm_and_si128(_mm_slli_epi16(v3, 4), mask2)));
        _mm_storeu_si128((__m128i *)(compact + 32), _mm_or_si128(_mm_and_si128(v2, mask6), _mm_and_si128(_mm_slli_epi16(v3, 2), mask2)));
        *(rabitq_pack_u64 *)(compact + 48) = top;
        raw += 64;
        compact += 56;
    }
}

RABITQ_PACK_INLINE void rabitq_pack_8bit_intrinsics(
    const uint8_t *raw, uint8_t *compact, int64_t dim) {
    for (int64_t base = 0; base < dim; base += 64) {
        _mm_storeu_si128((__m128i *)(compact + 0), _mm_loadu_si128((const __m128i *)(raw + 0)));
        _mm_storeu_si128((__m128i *)(compact + 16), _mm_loadu_si128((const __m128i *)(raw + 16)));
        _mm_storeu_si128((__m128i *)(compact + 32), _mm_loadu_si128((const __m128i *)(raw + 32)));
        _mm_storeu_si128((__m128i *)(compact + 48), _mm_loadu_si128((const __m128i *)(raw + 48)));
        raw += 64;
        compact += 64;
    }
}

void RABITQ_FN(rabitq_pack_excode)(
    const uint8_t *values, uint8_t *out, int64_t dim, int64_t bits) {
    switch (bits) {
    case 1: rabitq_pack_1bit_intrinsics(values, out, dim); break;
    case 2: rabitq_pack_2bit_intrinsics(values, out, dim); break;
    case 3: rabitq_pack_3bit_intrinsics(values, out, dim); break;
    case 4: rabitq_pack_4bit_intrinsics(values, out, dim); break;
    case 5: rabitq_pack_5bit_intrinsics(values, out, dim); break;
    case 6: rabitq_pack_6bit_intrinsics(values, out, dim); break;
    case 7: rabitq_pack_7bit_intrinsics(values, out, dim); break;
    case 8: rabitq_pack_8bit_intrinsics(values, out, dim); break;
    default: break;
    }
}

#undef RABITQ_PACK_INLINE

#endif
