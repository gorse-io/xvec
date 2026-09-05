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

// LASX port of VectorDB-NTU/RaBitQ-Library warmup kernels at revision
// 540242ea0a68926f1b827bf1f9add844f07a427b.

static inline __m256i load_partial_u64x4(const uint64_t *source, int64_t count) {
    if (count >= 4) {
        return __lasx_xvld(source, 0);
    }
    __m256i value = __lasx_xvldi(0);
    value = __lasx_xvinsgr2vr_d(value, source[0], 0);
    if (count >= 2) {
        value = __lasx_xvinsgr2vr_d(value, source[1], 1);
    }
    if (count >= 3) {
        value = __lasx_xvinsgr2vr_d(value, source[2], 2);
    }
    return value;
}

float warmup_ip_x0_q_512_lasx(
    const uint64_t *data,
    const uint64_t *query,
    float delta,
    float vl,
    int64_t padded_dimension,
    int64_t query_bits
) {
    __m256i accumulated_popcount = __lasx_xvldi(0);
    __m256i accumulated_bits[8];
    accumulated_bits[0] = __lasx_xvldi(0);
    accumulated_bits[1] = __lasx_xvldi(0);
    accumulated_bits[2] = __lasx_xvldi(0);
    accumulated_bits[3] = __lasx_xvldi(0);
    accumulated_bits[4] = __lasx_xvldi(0);
    accumulated_bits[5] = __lasx_xvldi(0);
    accumulated_bits[6] = __lasx_xvldi(0);
    accumulated_bits[7] = __lasx_xvldi(0);
    const int64_t words = padded_dimension / 64;
    for (int64_t block = 0; block < words; block += 8) {
        const int64_t block_words = words - block < 8 ? words - block : 8;
        for (int64_t group = 0; group < block_words; group += 4) {
            const int64_t group_words = block_words - group < 4 ? block_words - group : 4;
            const __m256i data_values = load_partial_u64x4(data + block + group, group_words);
            accumulated_popcount = __lasx_xvadd_d(
                accumulated_popcount, __lasx_xvpcnt_d(data_values)
            );
            for (int64_t bit = 0; bit < query_bits; ++bit) {
                const __m256i query_values = load_partial_u64x4(
                    query + bit * block_words + group, group_words
                );
                accumulated_bits[bit] = __lasx_xvadd_d(
                    accumulated_bits[bit],
                    __lasx_xvpcnt_d(__lasx_xvand_v(data_values, query_values))
                );
            }
        }
        query += block_words * query_bits;
    }

    const v4u64 popcount_lanes = (v4u64)accumulated_popcount;
    const uint64_t popcount = popcount_lanes[0] + popcount_lanes[1] +
                              popcount_lanes[2] + popcount_lanes[3];
    uint64_t ip = 0;
    for (int64_t bit = 0; bit < query_bits; ++bit) {
        const v4u64 lanes = (v4u64)accumulated_bits[bit];
        ip += (lanes[0] + lanes[1] + lanes[2] + lanes[3]) << bit;
    }
    return __builtin_fmaf(delta, (float)ip, vl * (float)popcount);
}
