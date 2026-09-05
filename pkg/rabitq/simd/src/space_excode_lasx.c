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

static inline u8x32 splat_u8x32(uint8_t value) {
    return (u8x32){
        value, value, value, value, value, value, value, value,
        value, value, value, value, value, value, value, value,
        value, value, value, value, value, value, value, value,
        value, value, value, value, value, value, value, value,
    };
}

static inline u8x32 load_u8x32(const void *source) {
    u8x32 value;
    __builtin_memcpy(&value, source, sizeof(value));
    return value;
}

static inline f32x8 load_f32x8(const void *source) {
    f32x8 value;
    __builtin_memcpy(&value, source, sizeof(value));
    return value;
}

static inline void store_f32x8(void *destination, f32x8 value) {
    __builtin_memcpy(destination, &value, sizeof(value));
}

static inline float reduce_f32x8(f32x8 value) {
    float lanes[8];
    store_f32x8(lanes, value);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3] +
           lanes[4] + lanes[5] + lanes[6] + lanes[7];
}

static inline f32x8 convert_u8x8_f32x8(u8x8 value) {
    return __builtin_convertvector(value, f32x8);
}

// LASX port of VectorDB-NTU/RaBitQ-Library src/simd/space_excode_avx2.cpp and
// space_excode_avx512.cpp at revision 540242ea0a68926f1b827bf1f9add844f07a427b.

static inline u8x32 bit_checker_lasx(void) {
    volatile uint8_t lanes[32];
    for (int i = 0; i < 32; i += 8) {
        lanes[i] = 1;
        lanes[i + 1] = 2;
        lanes[i + 2] = 4;
        lanes[i + 3] = 8;
        lanes[i + 4] = 16;
        lanes[i + 5] = 32;
        lanes[i + 6] = 64;
        lanes[i + 7] = 128;
    }
    return load_u8x32((const uint8_t *)lanes);
}

static inline u8x32 expand_bits_lasx(uint8_t low, uint8_t high, u8x32 checker) {
    const u8x32 one = splat_u8x32(1);
    const u8x32 low_bits = ((splat_u8x32(low) & checker) == checker) & one;
    const u8x32 high_bits = ((splat_u8x32(high) & checker) == checker) & one;
    return __builtin_shufflevector(
        low_bits, high_bits,
        0, 1, 2, 3, 4, 5, 6, 7,
        32, 33, 34, 35, 36, 37, 38, 39,
        -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1
    );
}

static inline u8x32 add_top_bits_lasx(
    u8x32 values, const uint8_t *top_bits, int group, int shift, u8x32 checker
) {
    return values | expand_bits_lasx(
        top_bits[group * 2], top_bits[group * 2 + 1], checker
    ) << shift;
}

static inline void contribute_ip_lasx(u8x32 values, const float *query, f32x8 *sum) {
    const u8x8 low = __builtin_shufflevector(values, values, 0, 1, 2, 3, 4, 5, 6, 7);
    const u8x8 high = __builtin_shufflevector(values, values, 8, 9, 10, 11, 12, 13, 14, 15);
    *sum += load_f32x8(query) * convert_u8x8_f32x8(low);
    *sum += load_f32x8(query + 8) * convert_u8x8_f32x8(high);
}

float ip16_fxu1_lasx(const float *query, const uint8_t *compact_code, int64_t dim) {
    const u8x32 checker = bit_checker_lasx();
    f32x8 sum = (f32x8){0};
    for (int64_t i = 0; i < dim; i += 16) {
        contribute_ip_lasx(
            expand_bits_lasx(compact_code[0], compact_code[1], checker), query + i, &sum
        );
        compact_code += 2;
    }
    return reduce_f32x8(sum);
}

float ip64_fxu2_lasx(const float *query, const uint8_t *compact_code, int64_t dim) {
    const u8x32 mask = splat_u8x32(3);
    f32x8 sum = (f32x8){0};
    for (int64_t i = 0; i < dim; i += 64) {
        const u8x32 compact = load_u8x32(compact_code);
        contribute_ip_lasx(compact & mask, query + i, &sum);
        contribute_ip_lasx((compact >> 2) & mask, query + i + 16, &sum);
        contribute_ip_lasx((compact >> 4) & mask, query + i + 32, &sum);
        contribute_ip_lasx((compact >> 6) & mask, query + i + 48, &sum);
        compact_code += 16;
    }
    return reduce_f32x8(sum);
}

float ip64_fxu3_lasx(const float *query, const uint8_t *compact_code, int64_t dim) {
    const u8x32 mask = splat_u8x32(3);
    const u8x32 checker = bit_checker_lasx();
    f32x8 sum = (f32x8){0};
    for (int64_t i = 0; i < dim; i += 64) {
        const u8x32 compact = load_u8x32(compact_code);
        const uint8_t *top = compact_code + 16;
        contribute_ip_lasx(add_top_bits_lasx(compact & mask, top, 0, 2, checker), query + i, &sum);
        contribute_ip_lasx(add_top_bits_lasx((compact >> 2) & mask, top, 1, 2, checker), query + i + 16, &sum);
        contribute_ip_lasx(add_top_bits_lasx((compact >> 4) & mask, top, 2, 2, checker), query + i + 32, &sum);
        contribute_ip_lasx(add_top_bits_lasx((compact >> 6) & mask, top, 3, 2, checker), query + i + 48, &sum);
        compact_code += 24;
    }
    return reduce_f32x8(sum);
}

float ip16_fxu4_lasx(const float *query, const uint8_t *compact_code, int64_t dim) {
    const u8x32 mask = splat_u8x32(15);
    f32x8 sum = (f32x8){0};
    for (int64_t i = 0; i < dim; i += 16) {
        const u8x32 compact = load_u8x32(compact_code);
        const u8x32 values = __builtin_shufflevector(
            compact & mask, compact >> 4,
            0, 1, 2, 3, 4, 5, 6, 7,
            32, 33, 34, 35, 36, 37, 38, 39,
            -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1
        );
        contribute_ip_lasx(values, query + i, &sum);
        compact_code += 8;
    }
    return reduce_f32x8(sum);
}

float ip64_fxu5_lasx(const float *query, const uint8_t *compact_code, int64_t dim) {
    const u8x32 mask = splat_u8x32(15);
    const u8x32 checker = bit_checker_lasx();
    f32x8 sum = (f32x8){0};
    for (int64_t i = 0; i < dim; i += 64) {
        const u8x32 first = load_u8x32(compact_code);
        const u8x32 second = load_u8x32(compact_code + 16);
        const uint8_t *top = compact_code + 32;
        contribute_ip_lasx(add_top_bits_lasx(first & mask, top, 0, 4, checker), query + i, &sum);
        contribute_ip_lasx(add_top_bits_lasx(first >> 4, top, 1, 4, checker), query + i + 16, &sum);
        contribute_ip_lasx(add_top_bits_lasx(second & mask, top, 2, 4, checker), query + i + 32, &sum);
        contribute_ip_lasx(add_top_bits_lasx(second >> 4, top, 3, 4, checker), query + i + 48, &sum);
        compact_code += 40;
    }
    return reduce_f32x8(sum);
}

float ip64_fxu6_lasx(const float *query, const uint8_t *compact_code, int64_t dim) {
    const u8x32 mask6 = splat_u8x32(0x3f);
    const u8x32 mask2 = splat_u8x32(0xc0);
    f32x8 sum = (f32x8){0};
    for (int64_t i = 0; i < dim; i += 64) {
        const u8x32 first = load_u8x32(compact_code);
        const u8x32 second = load_u8x32(compact_code + 16);
        const u8x32 third = load_u8x32(compact_code + 32);
        const u8x32 fourth = ((first & mask2) >> 6) |
                             ((second & mask2) >> 4) |
                             ((third & mask2) >> 2);
        contribute_ip_lasx(first & mask6, query + i, &sum);
        contribute_ip_lasx(second & mask6, query + i + 16, &sum);
        contribute_ip_lasx(third & mask6, query + i + 32, &sum);
        contribute_ip_lasx(fourth, query + i + 48, &sum);
        compact_code += 48;
    }
    return reduce_f32x8(sum);
}

float ip64_fxu7_lasx(const float *query, const uint8_t *compact_code, int64_t dim) {
    const u8x32 mask6 = splat_u8x32(0x3f);
    const u8x32 mask2 = splat_u8x32(0xc0);
    const u8x32 checker = bit_checker_lasx();
    f32x8 sum = (f32x8){0};
    for (int64_t i = 0; i < dim; i += 64) {
        const u8x32 first = load_u8x32(compact_code);
        const u8x32 second = load_u8x32(compact_code + 16);
        const u8x32 third = load_u8x32(compact_code + 32);
        const uint8_t *top = compact_code + 48;
        const u8x32 fourth = ((first & mask2) >> 6) |
                             ((second & mask2) >> 4) |
                             ((third & mask2) >> 2);
        contribute_ip_lasx(add_top_bits_lasx(first & mask6, top, 0, 6, checker), query + i, &sum);
        contribute_ip_lasx(add_top_bits_lasx(second & mask6, top, 1, 6, checker), query + i + 16, &sum);
        contribute_ip_lasx(add_top_bits_lasx(third & mask6, top, 2, 6, checker), query + i + 32, &sum);
        contribute_ip_lasx(add_top_bits_lasx(fourth, top, 3, 6, checker), query + i + 48, &sum);
        compact_code += 56;
    }
    return reduce_f32x8(sum);
}

float ip16_fxu8_lasx(const float *query, const uint8_t *code, int64_t dim) {
    f32x8 sum = (f32x8){0};
    for (int64_t i = 0; i < dim; i += 16) {
        contribute_ip_lasx(load_u8x32(code), query + i, &sum);
        code += 16;
    }
    return reduce_f32x8(sum);
}
