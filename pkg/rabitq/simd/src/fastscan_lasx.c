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

#include <lasxintrin.h>
#include <stdint.h>

// LASX port of VectorDB-NTU/RaBitQ-Library fast-scan kernels at revision
// 540242ea0a68926f1b827bf1f9add844f07a427b.

void accumulate_lasx(
    const uint8_t *codes, const uint8_t *lp_table, uint16_t *result, int64_t dim
) {
    __m256i low_accumulator = __lasx_xvldi(0);
    __m256i high_accumulator = __lasx_xvldi(0);
    for (int64_t codebook = 0; codebook < dim / 4; ++codebook) {
        uint16_t low[16] = {0};
        uint16_t high[16] = {0};
        for (int packed = 0; packed < 16; ++packed) {
            const uint8_t code = codes[packed];
            const int vector = (packed >> 1) + (packed & 1) * 8;
            low[vector] = lp_table[code & 15];
            high[vector] = lp_table[code >> 4];
        }
        low_accumulator = __lasx_xvadd_h(low_accumulator, __lasx_xvld(low, 0));
        high_accumulator = __lasx_xvadd_h(high_accumulator, __lasx_xvld(high, 0));
        codes += 16;
        lp_table += 16;
    }
    __lasx_xvst(low_accumulator, result, 0);
    __lasx_xvst(high_accumulator, result + 16, 0);
}

void transfer_lut_hacc_lasx(const uint16_t *lut, int64_t dim, uint8_t *hc_lut) {
    for (int64_t codebook = 0; codebook < dim / 4; ++codebook) {
        const __m256i values = __lasx_xvld(lut, 0);
        const __m256i high = __lasx_xvsrli_h(values, 8);
        uint16_t low_lanes[16];
        uint16_t high_lanes[16];
        __lasx_xvst(values, low_lanes, 0);
        __lasx_xvst(high, high_lanes, 0);
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
    __m256i accumulators[4] = {
        __lasx_xvldi(0), __lasx_xvldi(0), __lasx_xvldi(0), __lasx_xvldi(0),
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
            const __m256i current = __lasx_xvld((const void *)(values + lane * 8), 0);
            accumulators[lane] = __lasx_xvadd_w(accumulators[lane], current);
        }
        codes += 16;
    }
    for (int lane = 0; lane < 4; ++lane) {
        __lasx_xvst(accumulators[lane], result + lane * 8, 0);
    }
}
