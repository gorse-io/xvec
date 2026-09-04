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

// NEON port of VectorDB-NTU/RaBitQ-Library src/simd/fastscan_avx2.cpp and
// fastscan_avx512.cpp at revision 540242ea0a68926f1b827bf1f9add844f07a427b.

static inline void accumulate_lookup_neon(
    uint16x8_t *low, uint16x8_t *high, uint8x16_t values
) {
    const uint8x8_t first = vget_low_u8(values);
    const uint8x8_t second = vget_high_u8(values);
    *low = vaddq_u16(*low, vmovl_u8(vuzp1_u8(first, second)));
    *high = vaddq_u16(*high, vmovl_u8(vuzp2_u8(first, second)));
}

void accumulate_neon(
    const uint8_t *codes, const uint8_t *lp_table, uint16_t *result, int64_t dim
) {
    volatile uint8_t mask_value = 0x0f;
    const uint8x16_t low_mask = vdupq_n_u8(mask_value);
    uint16x8_t accu[4] = {
        vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0)
    };

    const int64_t num_codebook = dim >> 2;
    for (int64_t i = 0; i < num_codebook; ++i) {
        const uint8x16_t c = vld1q_u8(codes);
        const uint8x16_t lut = vld1q_u8(lp_table);
        const uint8x16_t lo = vandq_u8(c, low_mask);
        const uint8x16_t hi = vshrq_n_u8(c, 4);
        accumulate_lookup_neon(&accu[0], &accu[1], vqtbl1q_u8(lut, lo));
        accumulate_lookup_neon(&accu[2], &accu[3], vqtbl1q_u8(lut, hi));
        codes += 16;
        lp_table += 16;
    }

    vst1q_u16(result, accu[0]);
    vst1q_u16(result + 8, accu[1]);
    vst1q_u16(result + 16, accu[2]);
    vst1q_u16(result + 24, accu[3]);
}

void transfer_lut_hacc_neon(const uint16_t *lut, int64_t dim, uint8_t *hc_lut) {
    const int64_t num_codebook = dim >> 2;
    for (int64_t i = 0; i < num_codebook; ++i) {
        const uint16x8_t first = vld1q_u16(lut);
        const uint16x8_t second = vld1q_u16(lut + 8);
        const uint8x16_t lo = vcombine_u8(vmovn_u16(first), vmovn_u16(second));
        const uint8x16_t hi = vcombine_u8(
            vmovn_u16(vshrq_n_u16(first, 8)), vmovn_u16(vshrq_n_u16(second, 8))
        );
        uint8_t *fill_lo = hc_lut + (i / 4 * 128) + ((i % 4) * 16);
        vst1q_u8(fill_lo, lo);
        vst1q_u8(fill_lo + 64, hi);
        lut += 16;
    }
}

static inline void accumulate_lookup_hacc_neon(
    uint32x4_t *accu,
    uint8x16_t lo_values,
    uint8x16_t hi_values
) {
    const uint8x8_t lo_first = vget_low_u8(lo_values);
    const uint8x8_t lo_second = vget_high_u8(lo_values);
    const uint8x8_t hi_first = vget_low_u8(hi_values);
    const uint8x8_t hi_second = vget_high_u8(hi_values);
    const uint16x8_t first = vaddq_u16(
        vmovl_u8(vuzp1_u8(lo_first, lo_second)),
        vshlq_n_u16(vmovl_u8(vuzp1_u8(hi_first, hi_second)), 8)
    );
    const uint16x8_t second = vaddq_u16(
        vmovl_u8(vuzp2_u8(lo_first, lo_second)),
        vshlq_n_u16(vmovl_u8(vuzp2_u8(hi_first, hi_second)), 8)
    );
    accu[0] = vaddq_u32(accu[0], vmovl_u16(vget_low_u16(first)));
    accu[1] = vaddq_u32(accu[1], vmovl_high_u16(first));
    accu[2] = vaddq_u32(accu[2], vmovl_u16(vget_low_u16(second)));
    accu[3] = vaddq_u32(accu[3], vmovl_high_u16(second));
}

void accumulate_hacc_neon(
    const uint8_t *codes, const uint8_t *hc_lut, int32_t *result, int64_t dim
) {
    volatile uint8_t mask_value = 0x0f;
    const uint8x16_t low_mask = vdupq_n_u8(mask_value);
    uint32x4_t accu[8] = {
        vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0),
        vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0)
    };

    const int64_t num_codebook = dim >> 2;
    for (int64_t i = 0; i < num_codebook; ++i) {
        const uint8x16_t c = vld1q_u8(codes);
        const uint8x16_t lo = vandq_u8(c, low_mask);
        const uint8x16_t hi = vshrq_n_u8(c, 4);
        const uint8_t *low_lut = hc_lut + (i / 4 * 128) + ((i % 4) * 16);
        const uint8x16_t lut_lo = vld1q_u8(low_lut);
        const uint8x16_t lut_hi = vld1q_u8(low_lut + 64);
        accumulate_lookup_hacc_neon(
            &accu[0], vqtbl1q_u8(lut_lo, lo), vqtbl1q_u8(lut_hi, lo)
        );
        accumulate_lookup_hacc_neon(
            &accu[4], vqtbl1q_u8(lut_lo, hi), vqtbl1q_u8(lut_hi, hi)
        );
        codes += 16;
    }

    for (int i = 0; i < 8; ++i) {
        vst1q_s32(result + i * 4, vreinterpretq_s32_u32(accu[i]));
    }
}
