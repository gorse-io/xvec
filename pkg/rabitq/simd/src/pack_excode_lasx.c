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
#include <lsxintrin.h>
#include <stdint.h>

// LASX port of VectorDB-NTU/RaBitQ-Library pack-excode kernels at revision
// 540242ea0a68926f1b827bf1f9add844f07a427b.

static inline uint64_t pack_top_bit_lasx(
    __m256i first, __m256i second, unsigned int bit
) {
    uint8_t lanes[64];
    __lasx_xvst(__lasx_xvsrl_b(first, __lasx_xvreplgr2vr_b(bit)), lanes, 0);
    __lasx_xvst(__lasx_xvsrl_b(second, __lasx_xvreplgr2vr_b(bit)), lanes + 32, 0);
    uint64_t packed = 0;
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
    for (int lane = 0; lane < 64; ++lane) {
        packed |= (uint64_t)(lanes[lane] & 1) << lane;
    }
    return packed;
}

void packing_2bit_excode_lasx(const uint8_t *raw, uint8_t *compact, int64_t dimension) {
    for (int64_t offset = 0; offset < dimension; offset += 64) {
        const __m128i raw0 = __lsx_vld(raw, 0);
        const __m128i raw1 = __lsx_vld(raw, 16);
        const __m128i raw2 = __lsx_vld(raw, 32);
        const __m128i raw3 = __lsx_vld(raw, 48);
        const __m128i values = __lsx_vor_v(
            __lsx_vor_v(raw0, __lsx_vslli_b(raw1, 2)),
            __lsx_vor_v(__lsx_vslli_b(raw2, 4), __lsx_vslli_b(raw3, 6))
        );
        __lsx_vst(values, compact, 0);
        raw += 64;
        compact += 16;
    }
}

void packing_3bit_excode_lasx(const uint8_t *raw, uint8_t *compact, int64_t dimension) {
    const __m128i mask = __lsx_vrepli_b(3);
    for (int64_t offset = 0; offset < dimension; offset += 64) {
        const __m256i first = __lasx_xvld(raw, 0);
        const __m256i second = __lasx_xvld(raw, 32);
        const __m128i raw0 = __lsx_vld(raw, 0);
        const __m128i raw1 = __lsx_vld(raw, 16);
        const __m128i raw2 = __lsx_vld(raw, 32);
        const __m128i raw3 = __lsx_vld(raw, 48);
        const __m128i values = __lsx_vor_v(
            __lsx_vor_v(__lsx_vand_v(raw0, mask), __lsx_vslli_b(__lsx_vand_v(raw1, mask), 2)),
            __lsx_vor_v(__lsx_vslli_b(__lsx_vand_v(raw2, mask), 4), __lsx_vslli_b(__lsx_vand_v(raw3, mask), 6))
        );
        __lsx_vst(values, compact, 0);
        const uint64_t top = pack_top_bit_lasx(first, second, 2);
        __lsx_vstelm_d(__lsx_vreplgr2vr_d(top), compact + 16, 0, 0);
        raw += 64;
        compact += 24;
    }
}

void packing_4bit_excode_lasx(const uint8_t *raw, uint8_t *compact, int64_t dimension) {
    for (int64_t offset = 0; offset < dimension; offset += 16) {
        const __m128i values = __lsx_vor_v(
            __lsx_vldrepl_d(raw, 0),
            __lsx_vslli_b(__lsx_vldrepl_d(raw + 8, 0), 4)
        );
        __lsx_vstelm_d(values, compact, 0, 0);
        raw += 16;
        compact += 8;
    }
}

void packing_5bit_excode_lasx(const uint8_t *raw, uint8_t *compact, int64_t dimension) {
    const __m128i mask = __lsx_vrepli_b(15);
    for (int64_t offset = 0; offset < dimension; offset += 64) {
        const __m256i first = __lasx_xvld(raw, 0);
        const __m256i second = __lasx_xvld(raw, 32);
        const __m128i raw0 = __lsx_vld(raw, 0);
        const __m128i raw1 = __lsx_vld(raw, 16);
        const __m128i raw2 = __lsx_vld(raw, 32);
        const __m128i raw3 = __lsx_vld(raw, 48);
        __lsx_vst(__lsx_vor_v(__lsx_vand_v(raw0, mask), __lsx_vslli_b(__lsx_vand_v(raw1, mask), 4)), compact, 0);
        __lsx_vst(__lsx_vor_v(__lsx_vand_v(raw2, mask), __lsx_vslli_b(__lsx_vand_v(raw3, mask), 4)), compact, 16);
        const uint64_t top = pack_top_bit_lasx(first, second, 4);
        __lsx_vstelm_d(__lsx_vreplgr2vr_d(top), compact + 32, 0, 0);
        raw += 64;
        compact += 40;
    }
}

static inline void packing_6bit_body_lasx(
    const uint8_t *raw, uint8_t *compact, int64_t dimension, int top_bit
) {
    const __m128i mask6 = __lsx_vrepli_b(63);
    const __m128i mask2 = __lsx_vnor_v(mask6, mask6);
    const int64_t stride = top_bit ? 56 : 48;
    for (int64_t offset = 0; offset < dimension; offset += 64) {
        const __m256i first = __lasx_xvld(raw, 0);
        const __m256i second = __lasx_xvld(raw, 32);
        const __m128i raw0 = __lsx_vld(raw, 0);
        const __m128i raw1 = __lsx_vld(raw, 16);
        const __m128i raw2 = __lsx_vld(raw, 32);
        const __m128i raw3 = __lsx_vld(raw, 48);
        __lsx_vst(__lsx_vor_v(__lsx_vand_v(raw0, mask6), __lsx_vand_v(__lsx_vslli_b(raw3, 6), mask2)), compact, 0);
        __lsx_vst(__lsx_vor_v(__lsx_vand_v(raw1, mask6), __lsx_vand_v(__lsx_vslli_b(raw3, 4), mask2)), compact, 16);
        __lsx_vst(__lsx_vor_v(__lsx_vand_v(raw2, mask6), __lsx_vand_v(__lsx_vslli_b(raw3, 2), mask2)), compact, 32);
        if (top_bit) {
            const uint64_t top = pack_top_bit_lasx(first, second, 6);
            __lsx_vstelm_d(__lsx_vreplgr2vr_d(top), compact + 48, 0, 0);
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
