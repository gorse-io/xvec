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

static inline u16x16 load_u16x16(const void *source) {
    u16x16 value;
    __builtin_memcpy(&value, source, sizeof(value));
    return value;
}

static inline void store_u16x16(void *destination, u16x16 value) {
    __builtin_memcpy(destination, &value, sizeof(value));
}

static inline void store_u32x8(void *destination, u32x8 value) {
    __builtin_memcpy(destination, &value, sizeof(value));
}

// LASX port of VectorDB-NTU/RaBitQ-Library fast-scan kernels at revision
// 540242ea0a68926f1b827bf1f9add844f07a427b.

void accumulate_lasx(
    const uint8_t *codes, const uint8_t *lp_table, uint16_t *result, int64_t dim
) {
    u16x16 low_accumulator = (u16x16){0};
    u16x16 high_accumulator = (u16x16){0};
    for (int64_t codebook = 0; codebook < dim / 4; ++codebook) {
        uint16_t low[16] = {0};
        uint16_t high[16] = {0};
        for (int packed = 0; packed < 16; ++packed) {
            const uint8_t code = codes[packed];
            const int vector = (packed >> 1) + (packed & 1) * 8;
            low[vector] = lp_table[code & 15];
            high[vector] = lp_table[code >> 4];
        }
        low_accumulator += load_u16x16(low);
        high_accumulator += load_u16x16(high);
        codes += 16;
        lp_table += 16;
    }
    store_u16x16(result, low_accumulator);
    store_u16x16(result + 16, high_accumulator);
}

void transfer_lut_hacc_lasx(const uint16_t *lut, int64_t dim, uint8_t *hc_lut) {
    for (int64_t codebook = 0; codebook < dim / 4; ++codebook) {
        const u16x16 values = load_u16x16(lut);
        const u16x16 high = values >> 8;
        uint16_t low_lanes[16];
        uint16_t high_lanes[16];
        store_u16x16(low_lanes, values);
        store_u16x16(high_lanes, high);
        uint8_t *low_output = hc_lut + codebook / 4 * 128 + codebook % 4 * 16;
        for (int lane = 0; lane < 16; ++lane) {
            low_output[lane] = (uint8_t)low_lanes[lane];
            low_output[64 + lane] = (uint8_t)high_lanes[lane];
        }
        lut += 16;
    }
}

void accumulate_hacc_lasx(
    const uint8_t *codes, const uint8_t *hc_lut, int32_t *result, int64_t dim
) {
    u32x8 accumulators[4] = {
        (u32x8){0}, (u32x8){0}, (u32x8){0}, (u32x8){0},
    };
    for (int64_t codebook = 0; codebook < dim / 4; ++codebook) {
        volatile uint32_t values[32];
        for (int i = 0; i < 32; ++i) {
            values[i] = 0;
        }
        const uint8_t *low_lut = hc_lut + codebook / 4 * 128 + codebook % 4 * 16;
        for (int packed = 0; packed < 16; ++packed) {
            const uint8_t code = codes[packed];
            const int vector = (packed >> 1) + (packed & 1) * 8;
            const uint8_t low_code = code & 15;
            const uint8_t high_code = code >> 4;
            values[vector] = (uint32_t)low_lut[low_code] |
                             (uint32_t)low_lut[64 + low_code] << 8;
            values[vector + 16] = (uint32_t)low_lut[high_code] |
                                  (uint32_t)low_lut[64 + high_code] << 8;
        }
        for (int lane = 0; lane < 4; ++lane) {
            u32x8 current;
            __builtin_memcpy(&current, values + lane * 8, sizeof(current));
            accumulators[lane] += current;
        }
        codes += 16;
    }
    for (int lane = 0; lane < 4; ++lane) {
        store_u32x8(result + lane * 8, accumulators[lane]);
    }
}
