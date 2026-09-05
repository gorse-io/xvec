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

// NEON port of VectorDB-NTU/RaBitQ-Library src/simd/warmup_avx2.cpp and
// warmup_avx512.cpp at revision 540242ea0a68926f1b827bf1f9add844f07a427b.

static inline uint64x2_t popcount_neon(uint64x2_t value) {
    const uint8x16_t byte_counts = vcntq_u8(vreinterpretq_u8_u64(value));
    const uint16x8_t counts16 = vpaddlq_u8(byte_counts);
    const uint32x4_t counts32 = vpaddlq_u16(counts16);
    return vpaddlq_u32(counts32);
}

static inline uint64x2_t load_words_neon(const uint64_t *values, int64_t count) {
    if (count >= 2) {
        return vld1q_u64(values);
    }
    return vsetq_lane_u64(values[0], vdupq_n_u64(0), 0);
}

float warmup_ip_x0_q_512_neon(
    const uint64_t *data,
    const uint64_t *query,
    float delta,
    float vl,
    int64_t padded_dimension,
    int64_t query_bits
) {
    uint64x2_t accumulated_popcount = vdupq_n_u64(0);
    uint64x2_t accumulated_bits[8] = {
        vdupq_n_u64(0), vdupq_n_u64(0), vdupq_n_u64(0), vdupq_n_u64(0),
        vdupq_n_u64(0), vdupq_n_u64(0), vdupq_n_u64(0), vdupq_n_u64(0),
    };

    const int64_t words = padded_dimension / 64;
    for (int64_t block = 0; block < words; block += 8) {
        const int64_t block_words = words - block < 8 ? words - block : 8;
        for (int64_t pair = 0; pair < block_words; pair += 2) {
            const int64_t pair_words = block_words - pair < 2 ? block_words - pair : 2;
            const uint64x2_t data_values = load_words_neon(data + block + pair, pair_words);
            accumulated_popcount = vaddq_u64(
                accumulated_popcount, popcount_neon(data_values)
            );
            for (int64_t bit = 0; bit < query_bits; ++bit) {
                const uint64x2_t query_values = load_words_neon(
                    query + bit * block_words + pair, pair_words
                );
                accumulated_bits[bit] = vaddq_u64(
                    accumulated_bits[bit],
                    popcount_neon(vandq_u64(data_values, query_values))
                );
            }
        }
        query += block_words * query_bits;
    }

    uint64_t ip = 0;
#pragma clang loop vectorize(disable)
    for (int64_t bit = 0; bit < query_bits; ++bit) {
        const uint64_t count = vgetq_lane_u64(accumulated_bits[bit], 0) +
            vgetq_lane_u64(accumulated_bits[bit], 1);
        ip += count << bit;
    }
    const uint64_t popcount = vgetq_lane_u64(accumulated_popcount, 0) +
        vgetq_lane_u64(accumulated_popcount, 1);
    return vl * (float)popcount + delta * (float)ip;
}
