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

static inline float reduce_f32x8(__m256 value) {
    float lanes[8];
    __lasx_xvst((__m256i)value, lanes, 0);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3] +
           lanes[4] + lanes[5] + lanes[6] + lanes[7];
}

// LASX port of VectorDB-NTU/RaBitQ-Library src/simd/space_excode_avx2.cpp and
// space_excode_avx512.cpp at revision 540242ea0a68926f1b827bf1f9add844f07a427b.

static inline __m256i bit_checker_lasx(void) {
    return __lasx_xvreplgr2vr_d((long)UINT64_C(0x8040201008040201));
}

static inline __m256i expand_bits_lasx(
    uint8_t low, uint8_t high, __m256i checker
) {
    const __m256i one = __lasx_xvrepli_b(1);
    const __m256i low_bits = __lasx_xvand_v(
        __lasx_xvseq_b(
            __lasx_xvand_v(__lasx_xvreplgr2vr_b(low), checker), checker
        ),
        one
    );
    const __m256i high_bits = __lasx_xvand_v(
        __lasx_xvseq_b(
            __lasx_xvand_v(__lasx_xvreplgr2vr_b(high), checker), checker
        ),
        one
    );
    return __lasx_xvinsve0_d(low_bits, high_bits, 1);
}

static inline __m256i add_top_bits_lasx(
    __m256i values, const uint8_t *top_bits, int group, int shift, __m256i checker
) {
    return __lasx_xvor_v(
        values,
        __lasx_xvsll_b(
            expand_bits_lasx(top_bits[group * 2], top_bits[group * 2 + 1], checker),
            __lasx_xvreplgr2vr_b(shift)
        )
    );
}

static inline void contribute_ip_lasx(
    __m256i values, const float *query, __m256 *sum
) {
    const v32u8 bytes = (v32u8)values;
    v8u32 low;
    v8u32 high;
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
    for (int lane = 0; lane < 8; ++lane) {
        low[lane] = bytes[lane];
        high[lane] = bytes[lane + 8];
    }
    *sum = __lasx_xvfadd_s(
        *sum,
        __lasx_xvfmul_s(
            (__m256)__lasx_xvld(query, 0),
            __lasx_xvffint_s_wu((__m256i)low)
        )
    );
    *sum = __lasx_xvfadd_s(
        *sum,
        __lasx_xvfmul_s(
            (__m256)__lasx_xvld(query + 8, 0),
            __lasx_xvffint_s_wu((__m256i)high)
        )
    );
}

float ip16_fxu1_lasx(const float *query, const uint8_t *compact_code, int64_t dim) {
    const __m256i checker = bit_checker_lasx();
    __m256 sum = (__m256)__lasx_xvldi(0);
    for (int64_t i = 0; i < dim; i += 16) {
        contribute_ip_lasx(
            expand_bits_lasx(compact_code[0], compact_code[1], checker), query + i, &sum
        );
        compact_code += 2;
    }
    return reduce_f32x8(sum);
}

float ip64_fxu2_lasx(const float *query, const uint8_t *compact_code, int64_t dim) {
    const __m256i mask = __lasx_xvrepli_b(3);
    __m256 sum = (__m256)__lasx_xvldi(0);
    for (int64_t i = 0; i < dim; i += 64) {
        const __m256i compact = __lasx_xvld(compact_code, 0);
        contribute_ip_lasx(__lasx_xvand_v(compact, mask), query + i, &sum);
        contribute_ip_lasx(__lasx_xvand_v(__lasx_xvsrli_b(compact, 2), mask), query + i + 16, &sum);
        contribute_ip_lasx(__lasx_xvand_v(__lasx_xvsrli_b(compact, 4), mask), query + i + 32, &sum);
        contribute_ip_lasx(__lasx_xvand_v(__lasx_xvsrli_b(compact, 6), mask), query + i + 48, &sum);
        compact_code += 16;
    }
    return reduce_f32x8(sum);
}

float ip64_fxu3_lasx(const float *query, const uint8_t *compact_code, int64_t dim) {
    const __m256i mask = __lasx_xvrepli_b(3);
    const __m256i checker = bit_checker_lasx();
    __m256 sum = (__m256)__lasx_xvldi(0);
    for (int64_t i = 0; i < dim; i += 64) {
        const __m256i compact = __lasx_xvld(compact_code, 0);
        const uint8_t *top = compact_code + 16;
        contribute_ip_lasx(add_top_bits_lasx(__lasx_xvand_v(compact, mask), top, 0, 2, checker), query + i, &sum);
        contribute_ip_lasx(add_top_bits_lasx(__lasx_xvand_v(__lasx_xvsrli_b(compact, 2), mask), top, 1, 2, checker), query + i + 16, &sum);
        contribute_ip_lasx(add_top_bits_lasx(__lasx_xvand_v(__lasx_xvsrli_b(compact, 4), mask), top, 2, 2, checker), query + i + 32, &sum);
        contribute_ip_lasx(add_top_bits_lasx(__lasx_xvsrli_b(compact, 6), top, 3, 2, checker), query + i + 48, &sum);
        compact_code += 24;
    }
    return reduce_f32x8(sum);
}

float ip16_fxu4_lasx(const float *query, const uint8_t *compact_code, int64_t dim) {
    const __m256i mask = __lasx_xvrepli_b(15);
    __m256 sum = (__m256)__lasx_xvldi(0);
    for (int64_t i = 0; i < dim; i += 16) {
        const __m256i compact = __lasx_xvld(compact_code, 0);
        const __m256i values = __lasx_xvinsve0_d(
            __lasx_xvand_v(compact, mask), __lasx_xvsrli_b(compact, 4), 1
        );
        contribute_ip_lasx(values, query + i, &sum);
        compact_code += 8;
    }
    return reduce_f32x8(sum);
}

float ip64_fxu5_lasx(const float *query, const uint8_t *compact_code, int64_t dim) {
    const __m256i mask = __lasx_xvrepli_b(15);
    const __m256i checker = bit_checker_lasx();
    __m256 sum = (__m256)__lasx_xvldi(0);
    for (int64_t i = 0; i < dim; i += 64) {
        const __m256i first = __lasx_xvld(compact_code, 0);
        const __m256i second = __lasx_xvld(compact_code, 16);
        const uint8_t *top = compact_code + 32;
        contribute_ip_lasx(add_top_bits_lasx(__lasx_xvand_v(first, mask), top, 0, 4, checker), query + i, &sum);
        contribute_ip_lasx(add_top_bits_lasx(__lasx_xvsrli_b(first, 4), top, 1, 4, checker), query + i + 16, &sum);
        contribute_ip_lasx(add_top_bits_lasx(__lasx_xvand_v(second, mask), top, 2, 4, checker), query + i + 32, &sum);
        contribute_ip_lasx(add_top_bits_lasx(__lasx_xvsrli_b(second, 4), top, 3, 4, checker), query + i + 48, &sum);
        compact_code += 40;
    }
    return reduce_f32x8(sum);
}

float ip64_fxu6_lasx(const float *query, const uint8_t *compact_code, int64_t dim) {
    const __m256i mask6 = __lasx_xvrepli_b(0x3f);
    const __m256i mask2 = __lasx_xvrepli_b(0xc0);
    __m256 sum = (__m256)__lasx_xvldi(0);
    for (int64_t i = 0; i < dim; i += 64) {
        const __m256i first = __lasx_xvld(compact_code, 0);
        const __m256i second = __lasx_xvld(compact_code, 16);
        const __m256i third = __lasx_xvld(compact_code, 32);
        const __m256i fourth = __lasx_xvor_v(
            __lasx_xvor_v(
                __lasx_xvsrli_b(__lasx_xvand_v(first, mask2), 6),
                __lasx_xvsrli_b(__lasx_xvand_v(second, mask2), 4)
            ),
            __lasx_xvsrli_b(__lasx_xvand_v(third, mask2), 2)
        );
        contribute_ip_lasx(__lasx_xvand_v(first, mask6), query + i, &sum);
        contribute_ip_lasx(__lasx_xvand_v(second, mask6), query + i + 16, &sum);
        contribute_ip_lasx(__lasx_xvand_v(third, mask6), query + i + 32, &sum);
        contribute_ip_lasx(fourth, query + i + 48, &sum);
        compact_code += 48;
    }
    return reduce_f32x8(sum);
}

float ip64_fxu7_lasx(const float *query, const uint8_t *compact_code, int64_t dim) {
    const __m256i mask6 = __lasx_xvrepli_b(0x3f);
    const __m256i mask2 = __lasx_xvrepli_b(0xc0);
    const __m256i checker = bit_checker_lasx();
    __m256 sum = (__m256)__lasx_xvldi(0);
    for (int64_t i = 0; i < dim; i += 64) {
        const __m256i first = __lasx_xvld(compact_code, 0);
        const __m256i second = __lasx_xvld(compact_code, 16);
        const __m256i third = __lasx_xvld(compact_code, 32);
        const uint8_t *top = compact_code + 48;
        const __m256i fourth = __lasx_xvor_v(
            __lasx_xvor_v(
                __lasx_xvsrli_b(__lasx_xvand_v(first, mask2), 6),
                __lasx_xvsrli_b(__lasx_xvand_v(second, mask2), 4)
            ),
            __lasx_xvsrli_b(__lasx_xvand_v(third, mask2), 2)
        );
        contribute_ip_lasx(add_top_bits_lasx(__lasx_xvand_v(first, mask6), top, 0, 6, checker), query + i, &sum);
        contribute_ip_lasx(add_top_bits_lasx(__lasx_xvand_v(second, mask6), top, 1, 6, checker), query + i + 16, &sum);
        contribute_ip_lasx(add_top_bits_lasx(__lasx_xvand_v(third, mask6), top, 2, 6, checker), query + i + 32, &sum);
        contribute_ip_lasx(add_top_bits_lasx(fourth, top, 3, 6, checker), query + i + 48, &sum);
        compact_code += 56;
    }
    return reduce_f32x8(sum);
}

float ip16_fxu8_lasx(const float *query, const uint8_t *code, int64_t dim) {
    __m256 sum = (__m256)__lasx_xvldi(0);
    for (int64_t i = 0; i < dim; i += 16) {
        contribute_ip_lasx(__lasx_xvld(code, 0), query + i, &sum);
        code += 16;
    }
    return reduce_f32x8(sum);
}
