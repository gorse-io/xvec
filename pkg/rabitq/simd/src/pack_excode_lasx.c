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

#include <stdint.h>

typedef uint8_t u8x8 __attribute__((vector_size(8)));
typedef uint8_t u8x16 __attribute__((vector_size(16)));
typedef uint8_t u8x32 __attribute__((vector_size(32)));
typedef uint16_t u16x16 __attribute__((vector_size(32)));
typedef uint32_t u32x8 __attribute__((vector_size(32)));
typedef int32_t i32x8 __attribute__((vector_size(32)));
typedef uint64_t u64x4 __attribute__((vector_size(32)));
typedef float f32x8 __attribute__((vector_size(32)));

static inline u8x32 load_u8x32(const void *source) {
    u8x32 value;
    __builtin_memcpy(&value, source, sizeof(value));
    return value;
}

static inline u8x16 load_u8x16(const void *source) {
    u8x16 value;
    __builtin_memcpy(&value, source, sizeof(value));
    return value;
}

static inline void store_u8x32(void *destination, u8x32 value) {
    __builtin_memcpy(destination, &value, sizeof(value));
}

static inline void store_u8x16(void *destination, u8x16 value) {
    __builtin_memcpy(destination, &value, sizeof(value));
}

// LASX port of VectorDB-NTU/RaBitQ-Library pack-excode kernels at revision
// 540242ea0a68926f1b827bf1f9add844f07a427b.

static inline u8x16 splat_u8x16(uint8_t value) {
    return (u8x16){
        value, value, value, value, value, value, value, value,
        value, value, value, value, value, value, value, value,
    };
}

static inline uint64_t pack_top_bit_lasx(
    u8x32 first, u8x32 second, unsigned int bit
) {
    uint8_t lanes[64];
    store_u8x32(lanes, first >> bit);
    store_u8x32(lanes + 32, second >> bit);
    uint64_t packed = 0;
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
    for (int lane = 0; lane < 64; ++lane) {
        packed |= (uint64_t)(lanes[lane] & 1) << lane;
    }
    return packed;
}

void packing_2bit_excode_lasx(const uint8_t *raw, uint8_t *compact, int64_t dimension) {
    for (int64_t offset = 0; offset < dimension; offset += 64) {
        const u8x16 raw0 = load_u8x16(raw);
        const u8x16 raw1 = load_u8x16(raw + 16);
        const u8x16 raw2 = load_u8x16(raw + 32);
        const u8x16 raw3 = load_u8x16(raw + 48);
        store_u8x16(compact, raw0 | raw1 << 2 | raw2 << 4 | raw3 << 6);
        raw += 64;
        compact += 16;
    }
}

void packing_3bit_excode_lasx(const uint8_t *raw, uint8_t *compact, int64_t dimension) {
    volatile uint8_t mask_value = 3;
    const u8x16 mask = splat_u8x16(mask_value);
    for (int64_t offset = 0; offset < dimension; offset += 64) {
        const u8x32 first = load_u8x32(raw);
        const u8x32 second = load_u8x32(raw + 32);
        const u8x16 raw0 = load_u8x16(raw);
        const u8x16 raw1 = load_u8x16(raw + 16);
        const u8x16 raw2 = load_u8x16(raw + 32);
        const u8x16 raw3 = load_u8x16(raw + 48);
        const u8x16 values = (raw0 & mask) | (raw1 & mask) << 2 |
                             (raw2 & mask) << 4 | (raw3 & mask) << 6;
        store_u8x16(compact, values);
        const uint64_t top = pack_top_bit_lasx(first, second, 2);
        __builtin_memcpy(compact + 16, &top, sizeof(top));
        raw += 64;
        compact += 24;
    }
}

void packing_4bit_excode_lasx(const uint8_t *raw, uint8_t *compact, int64_t dimension) {
    for (int64_t offset = 0; offset < dimension; offset += 16) {
        uint64_t low;
        uint64_t high;
        __builtin_memcpy(&low, raw, sizeof(low));
        __builtin_memcpy(&high, raw + 8, sizeof(high));
        const uint64_t packed = low | high << 4;
        __builtin_memcpy(compact, &packed, sizeof(packed));
        raw += 16;
        compact += 8;
    }
}

void packing_5bit_excode_lasx(const uint8_t *raw, uint8_t *compact, int64_t dimension) {
    volatile uint8_t mask_value = 15;
    const u8x16 mask = splat_u8x16(mask_value);
    for (int64_t offset = 0; offset < dimension; offset += 64) {
        const u8x32 first = load_u8x32(raw);
        const u8x32 second = load_u8x32(raw + 32);
        const u8x16 raw0 = load_u8x16(raw);
        const u8x16 raw1 = load_u8x16(raw + 16);
        const u8x16 raw2 = load_u8x16(raw + 32);
        const u8x16 raw3 = load_u8x16(raw + 48);
        store_u8x16(compact, (raw0 & mask) | (raw1 & mask) << 4);
        store_u8x16(compact + 16, (raw2 & mask) | (raw3 & mask) << 4);
        const uint64_t top = pack_top_bit_lasx(first, second, 4);
        __builtin_memcpy(compact + 32, &top, sizeof(top));
        raw += 64;
        compact += 40;
    }
}

static inline void packing_6bit_body_lasx(
    const uint8_t *raw, uint8_t *compact, int64_t dimension, int top_bit
) {
    volatile uint8_t mask_value = 63;
    const u8x16 mask6 = splat_u8x16(mask_value);
    const u8x16 mask2 = ~mask6;
    const int64_t stride = top_bit ? 56 : 48;
    for (int64_t offset = 0; offset < dimension; offset += 64) {
        const u8x32 first = load_u8x32(raw);
        const u8x32 second = load_u8x32(raw + 32);
        const u8x16 raw0 = load_u8x16(raw);
        const u8x16 raw1 = load_u8x16(raw + 16);
        const u8x16 raw2 = load_u8x16(raw + 32);
        const u8x16 raw3 = load_u8x16(raw + 48);
        store_u8x16(compact, (raw0 & mask6) | (raw3 << 6 & mask2));
        store_u8x16(compact + 16, (raw1 & mask6) | (raw3 << 4 & mask2));
        store_u8x16(compact + 32, (raw2 & mask6) | (raw3 << 2 & mask2));
        if (top_bit) {
            const uint64_t top = pack_top_bit_lasx(first, second, 6);
            __builtin_memcpy(compact + 48, &top, sizeof(top));
        }
        raw += 64;
        compact += stride;
    }
}

void packing_6bit_excode_lasx(const uint8_t *raw, uint8_t *compact, int64_t dimension) {
    packing_6bit_body_lasx(raw, compact, dimension, 0);
}

void packing_7bit_excode_lasx(const uint8_t *raw, uint8_t *compact, int64_t dimension) {
    packing_6bit_body_lasx(raw, compact, dimension, 1);
}
