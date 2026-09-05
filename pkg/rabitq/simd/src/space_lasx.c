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

#include "lasx_common.h"

// LASX port of VectorDB-NTU/RaBitQ-Library space kernels at revision
// 540242ea0a68926f1b827bf1f9add844f07a427b.

void scalar_quantize_uint8_lasx(
    uint8_t *result, const float *data, int64_t dimension, float lo, float delta
) {
    const f32x8 lo8 = (f32x8){lo, lo, lo, lo, lo, lo, lo, lo};
    const float reciprocal = 1.0F / delta;
    const f32x8 reciprocal8 = (f32x8){
        reciprocal, reciprocal, reciprocal, reciprocal,
        reciprocal, reciprocal, reciprocal, reciprocal,
    };
    const f32x8 half8 = (f32x8){0.5F, 0.5F, 0.5F, 0.5F, 0.5F, 0.5F, 0.5F, 0.5F};
    int64_t i = 0;
    for (; i + 8 <= dimension; i += 8) {
        const f32x8 scaled = (load_f32x8(data + i) - lo8) * reciprocal8 + half8;
        const i32x8 quantized = __builtin_convertvector(scaled, i32x8);
        int32_t lanes[8];
        __builtin_memcpy(lanes, &quantized, sizeof(lanes));
        for (int lane = 0; lane < 8; ++lane) result[i + lane] = (uint8_t)lanes[lane];
    }
    for (; i < dimension; ++i) {
        result[i] = (uint8_t)((data[i] - lo) * reciprocal + 0.5F);
    }
}

void scalar_quantize_uint16_lasx(
    uint16_t *result, const float *data, int64_t dimension, float lo, float delta
) {
    const f32x8 lo8 = (f32x8){lo, lo, lo, lo, lo, lo, lo, lo};
    const float reciprocal = 1.0F / delta;
    const f32x8 reciprocal8 = (f32x8){
        reciprocal, reciprocal, reciprocal, reciprocal,
        reciprocal, reciprocal, reciprocal, reciprocal,
    };
    const f32x8 half8 = (f32x8){0.5F, 0.5F, 0.5F, 0.5F, 0.5F, 0.5F, 0.5F, 0.5F};
    int64_t i = 0;
    for (; i + 8 <= dimension; i += 8) {
        const f32x8 scaled = (load_f32x8(data + i) - lo8) * reciprocal8 + half8;
        const i32x8 quantized = __builtin_convertvector(scaled, i32x8);
        int32_t lanes[8];
        __builtin_memcpy(lanes, &quantized, sizeof(lanes));
        for (int lane = 0; lane < 8; ++lane) result[i + lane] = (uint16_t)lanes[lane];
    }
    for (; i < dimension; ++i) {
        result[i] = (uint16_t)((data[i] - lo) * reciprocal + 0.5F);
    }
}

void new_transpose_bin_lasx(
    const uint16_t *data, uint64_t *result, int64_t padded_dimension, int64_t bits
) {
    const u16x16 one = (u16x16){1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    for (int64_t block = 0; block < padded_dimension / 64; ++block) {
        for (int64_t bit = 0; bit < bits; ++bit) {
            volatile uint64_t plane = 0;
            for (int64_t offset = 0; offset < 64; offset += 16) {
                const u16x16 selected = (load_u16x16(data + block * 64 + offset) >> bit) & one;
                uint16_t lanes[16];
                store_u16x16(lanes, selected);
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
                for (int lane = 0; lane < 16; ++lane) {
                    plane |= (uint64_t)lanes[lane] << (63 - offset - lane);
                }
            }
            result[block * bits + bit] = plane;
        }
    }
}

void new_transpose_bin_512_lasx(
    const uint8_t *data, uint64_t *result, int64_t padded_dimension, int64_t bits
) {
    const u8x32 one = (u8x32){
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    };
    for (int64_t block_start = 0; block_start < padded_dimension; block_start += 512) {
        const int64_t block_size = padded_dimension - block_start < 512
            ? padded_dimension - block_start
            : 512;
        const int64_t chunks = block_size / 64;
        const int64_t output_base = block_start / 64 * bits;
        for (int64_t bit = 0; bit < bits; ++bit) {
            for (int64_t chunk = 0; chunk < chunks; ++chunk) {
                volatile uint64_t plane = 0;
                const int64_t input_base = block_start + chunk * 64;
                for (int64_t offset = 0; offset < 64; offset += 32) {
                    const u8x32 selected = (load_u8x32(data + input_base + offset) >> bit) & one;
                    uint8_t lanes[32];
                    store_u8x32(lanes, selected);
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
                    for (int lane = 0; lane < 32; ++lane) {
                        plane |= (uint64_t)lanes[lane] << (63 - offset - lane);
                    }
                }
                result[output_base + bit * chunks + chunk] = plane;
            }
        }
    }
}

float mask_ip_x0_q_lasx(
    const float *query, const uint64_t *data, int64_t padded_dimension
) {
    f32x8 sum = (f32x8){0};
    for (int64_t i = 0; i < padded_dimension; i += 8) {
        const uint64_t bits = data[i / 64];
        volatile u32x8 mask;
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
        for (int lane = 0; lane < 8; ++lane) {
            volatile int shift = 63 - i % 64 - lane;
            mask[lane] = bits & (1ULL << shift) ? UINT32_MAX : 0;
        }
        u32x8 values;
        __builtin_memcpy(&values, query + i, sizeof(values));
        values &= (u32x8)mask;
        f32x8 selected;
        __builtin_memcpy(&selected, &values, sizeof(selected));
        sum += selected;
    }
    return reduce_f32x8(sum);
}
