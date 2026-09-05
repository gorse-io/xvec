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

// LASX port of VectorDB-NTU/RaBitQ-Library warmup kernels at revision
// 540242ea0a68926f1b827bf1f9add844f07a427b.

static inline u64x4 load_partial_u64x4(const uint64_t *source, int64_t count) {
    if (count >= 4) {
        return load_u64x4(source);
    }
    if (count == 3) {
        return (u64x4){source[0], source[1], source[2], 0};
    }
    if (count == 2) {
        return (u64x4){source[0], source[1], 0, 0};
    }
    volatile uint64_t lanes[4];
    lanes[0] = source[0];
    lanes[1] = 0;
    lanes[2] = 0;
    lanes[3] = 0;
    return load_u64x4((const uint64_t *)lanes);
}

float warmup_ip_x0_q_512_lasx(
    const uint64_t *data,
    const uint64_t *query,
    float delta,
    float vl,
    int64_t padded_dimension,
    int64_t query_bits
) {
    u64x4 accumulated_popcount = (u64x4){0};
    volatile u64x4 accumulated_bits[8];
    for (int bit = 0; bit < 8; ++bit) {
        accumulated_bits[bit] = (u64x4){0};
    }
    const int64_t words = padded_dimension / 64;
    for (int64_t block = 0; block < words; block += 8) {
        const int64_t block_words = words - block < 8 ? words - block : 8;
        for (int64_t group = 0; group < block_words; group += 4) {
            const int64_t group_words = block_words - group < 4 ? block_words - group : 4;
            const u64x4 data_values = load_partial_u64x4(data + block + group, group_words);
            accumulated_popcount += __builtin_elementwise_popcount(data_values);
            for (int64_t bit = 0; bit < query_bits; ++bit) {
                const u64x4 query_values = load_partial_u64x4(
                    query + bit * block_words + group, group_words
                );
                accumulated_bits[bit] +=
                    __builtin_elementwise_popcount(data_values & query_values);
            }
        }
        query += block_words * query_bits;
    }

    const uint64_t popcount = accumulated_popcount[0] + accumulated_popcount[1] +
                              accumulated_popcount[2] + accumulated_popcount[3];
    uint64_t ip = 0;
    for (int64_t bit = 0; bit < query_bits; ++bit) {
        const u64x4 lanes = accumulated_bits[bit];
        ip += (lanes[0] + lanes[1] + lanes[2] + lanes[3]) << bit;
    }
    return __builtin_fmaf(delta, (float)ip, vl * (float)popcount);
}
