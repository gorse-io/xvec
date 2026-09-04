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

#include <arm_neon.h>
#include <stdint.h>

// NEON port of VectorDB-NTU/RaBitQ-Library src/simd/pack_excode_kernels.hpp at
// revision 540242ea0a68926f1b827bf1f9add844f07a427b.

static inline uint8_t movemask8_u8_neon(uint8x8_t values) {
    const uint64_t bytes = vget_lane_u64(vreinterpret_u64_u8(values), 0);
    return (uint8_t)(((bytes & 0x0101010101010101ULL) * 0x0102040810204080ULL) >> 56);
}

static inline uint16_t movemask_u8_neon(uint8x16_t values) {
    return (uint16_t)movemask8_u8_neon(vget_low_u8(values)) |
           ((uint16_t)movemask8_u8_neon(vget_high_u8(values)) << 8);
}

void packing_2bit_excode_neon(const uint8_t *raw, uint8_t *compact, int64_t dimension) {
    for (int64_t offset = 0; offset < dimension; offset += 64) {
        const uint8x16_t values0 = vld1q_u8(raw);
        const uint8x16_t values1 = vld1q_u8(raw + 16);
        const uint8x16_t values2 = vld1q_u8(raw + 32);
        const uint8x16_t values3 = vld1q_u8(raw + 48);
        vst1q_u8(
            compact,
            vorrq_u8(
                vorrq_u8(values0, vshlq_n_u8(values1, 2)),
                vorrq_u8(vshlq_n_u8(values2, 4), vshlq_n_u8(values3, 6))
            )
        );
        raw += 64;
        compact += 16;
    }
}

void packing_3bit_excode_neon(const uint8_t *raw, uint8_t *compact, int64_t dimension) {
    const uint8x16_t mask = vdupq_n_u8(3);
    for (int64_t offset = 0; offset < dimension; offset += 64) {
        const uint8x16_t raw0 = vld1q_u8(raw);
        const uint8x16_t raw1 = vld1q_u8(raw + 16);
        const uint8x16_t raw2 = vld1q_u8(raw + 32);
        const uint8x16_t raw3 = vld1q_u8(raw + 48);
        vst1q_u8(
            compact,
            vorrq_u8(
                vorrq_u8(vandq_u8(raw0, mask), vshlq_n_u8(vandq_u8(raw1, mask), 2)),
                vorrq_u8(
                    vshlq_n_u8(vandq_u8(raw2, mask), 4),
                    vshlq_n_u8(vandq_u8(raw3, mask), 6)
                )
            )
        );
        const uint64_t top_bit =
            (uint64_t)movemask_u8_neon(vshrq_n_u8(raw0, 2)) |
            ((uint64_t)movemask_u8_neon(vshrq_n_u8(raw1, 2)) << 16) |
            ((uint64_t)movemask_u8_neon(vshrq_n_u8(raw2, 2)) << 32) |
            ((uint64_t)movemask_u8_neon(vshrq_n_u8(raw3, 2)) << 48);
        vst1_u8(compact + 16, vreinterpret_u8_u64(vcreate_u64(top_bit)));
        raw += 64;
        compact += 24;
    }
}

void packing_4bit_excode_neon(const uint8_t *raw, uint8_t *compact, int64_t dimension) {
    for (int64_t offset = 0; offset < dimension; offset += 16) {
        const uint8x8_t values0 = vld1_u8(raw);
        const uint8x8_t values1 = vld1_u8(raw + 8);
        vst1_u8(compact, vorr_u8(values0, vshl_n_u8(values1, 4)));
        raw += 16;
        compact += 8;
    }
}

void packing_5bit_excode_neon(const uint8_t *raw, uint8_t *compact, int64_t dimension) {
    const uint8x16_t mask = vdupq_n_u8(15);
    for (int64_t offset = 0; offset < dimension; offset += 64) {
        const uint8x16_t raw0 = vld1q_u8(raw);
        const uint8x16_t raw1 = vld1q_u8(raw + 16);
        const uint8x16_t raw2 = vld1q_u8(raw + 32);
        const uint8x16_t raw3 = vld1q_u8(raw + 48);
        vst1q_u8(
            compact,
            vorrq_u8(vandq_u8(raw0, mask), vshlq_n_u8(vandq_u8(raw1, mask), 4))
        );
        vst1q_u8(
            compact + 16,
            vorrq_u8(vandq_u8(raw2, mask), vshlq_n_u8(vandq_u8(raw3, mask), 4))
        );
        const uint64_t top_bit =
            (uint64_t)movemask_u8_neon(vshrq_n_u8(raw0, 4)) |
            ((uint64_t)movemask_u8_neon(vshrq_n_u8(raw1, 4)) << 16) |
            ((uint64_t)movemask_u8_neon(vshrq_n_u8(raw2, 4)) << 32) |
            ((uint64_t)movemask_u8_neon(vshrq_n_u8(raw3, 4)) << 48);
        vst1_u8(compact + 32, vreinterpret_u8_u64(vcreate_u64(top_bit)));
        raw += 64;
        compact += 40;
    }
}

static inline void packing_6bit_body_neon(
    const uint8_t *raw, uint8_t *compact, int64_t dimension, int top_bit
) {
    const uint8x16_t mask2 = vdupq_n_u8(0xc0);
    const uint8x16_t mask6 = vdupq_n_u8(0x3f);
    const int64_t compact_stride = top_bit ? 56 : 48;
    for (int64_t offset = 0; offset < dimension; offset += 64) {
        const uint8x16_t values0 = vld1q_u8(raw);
        const uint8x16_t values1 = vld1q_u8(raw + 16);
        const uint8x16_t values2 = vld1q_u8(raw + 32);
        const uint8x16_t values3 = vld1q_u8(raw + 48);
        vst1q_u8(
            compact,
            vorrq_u8(
                vandq_u8(values0, mask6), vandq_u8(vshlq_n_u8(values3, 6), mask2)
            )
        );
        vst1q_u8(
            compact + 16,
            vorrq_u8(
                vandq_u8(values1, mask6), vandq_u8(vshlq_n_u8(values3, 4), mask2)
            )
        );
        vst1q_u8(
            compact + 32,
            vorrq_u8(
                vandq_u8(values2, mask6), vandq_u8(vshlq_n_u8(values3, 2), mask2)
            )
        );
        if (top_bit) {
            const uint64_t packed_top_bit =
                (uint64_t)movemask_u8_neon(vshrq_n_u8(values0, 6)) |
                ((uint64_t)movemask_u8_neon(vshrq_n_u8(values1, 6)) << 16) |
                ((uint64_t)movemask_u8_neon(vshrq_n_u8(values2, 6)) << 32) |
                ((uint64_t)movemask_u8_neon(vshrq_n_u8(values3, 6)) << 48);
            vst1_u8(compact + 48, vreinterpret_u8_u64(vcreate_u64(packed_top_bit)));
        }
        raw += 64;
        compact += compact_stride;
    }
}

void packing_6bit_excode_neon(const uint8_t *raw, uint8_t *compact, int64_t dimension) {
    packing_6bit_body_neon(raw, compact, dimension, 0);
}

void packing_7bit_excode_neon(const uint8_t *raw, uint8_t *compact, int64_t dimension) {
    packing_6bit_body_neon(raw, compact, dimension, 1);
}
