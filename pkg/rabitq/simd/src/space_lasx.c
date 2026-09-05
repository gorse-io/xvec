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

// LASX port of VectorDB-NTU/RaBitQ-Library space kernels at revision
// 540242ea0a68926f1b827bf1f9add844f07a427b.

void scalar_quantize_uint8_lasx(
    uint8_t *result, const float *data, int64_t dimension, float lo, float delta
) {
    const __m256 lo8 = (__m256)__lasx_xvldrepl_w(&lo, 0);
    const float reciprocal = 1.0F / delta;
    const __m256 reciprocal8 = (__m256)__lasx_xvldrepl_w(&reciprocal, 0);
    const __m256 half8 = (__m256)__lasx_xvreplgr2vr_w(0x3f000000);
    int64_t i = 0;
    for (; i + 8 <= dimension; i += 8) {
        const __m256 scaled = __lasx_xvfadd_s(
            __lasx_xvfmul_s(
                __lasx_xvfsub_s((__m256)__lasx_xvld(data + i, 0), lo8),
                reciprocal8
            ),
            half8
        );
        const __m256i quantized = __lasx_xvftintrz_w_s(scaled);
        int32_t lanes[8];
        __lasx_xvst(quantized, lanes, 0);
        for (int lane = 0; lane < 8; ++lane) {
            result[i + lane] = (uint8_t)lanes[lane];
        }
    }
    for (; i < dimension; ++i) {
        result[i] = (uint8_t)((data[i] - lo) * reciprocal + 0.5F);
    }
}

void scalar_quantize_uint16_lasx(
    uint16_t *result, const float *data, int64_t dimension, float lo, float delta
) {
    const __m256 lo8 = (__m256)__lasx_xvldrepl_w(&lo, 0);
    const float reciprocal = 1.0F / delta;
    const __m256 reciprocal8 = (__m256)__lasx_xvldrepl_w(&reciprocal, 0);
    const __m256 half8 = (__m256)__lasx_xvreplgr2vr_w(0x3f000000);
    int64_t i = 0;
    for (; i + 8 <= dimension; i += 8) {
        const __m256 scaled = __lasx_xvfadd_s(
            __lasx_xvfmul_s(
                __lasx_xvfsub_s((__m256)__lasx_xvld(data + i, 0), lo8),
                reciprocal8
            ),
            half8
        );
        const __m256i quantized = __lasx_xvftintrz_w_s(scaled);
        int32_t lanes[8];
        __lasx_xvst(quantized, lanes, 0);
        for (int lane = 0; lane < 8; ++lane) {
            result[i + lane] = (uint16_t)lanes[lane];
        }
    }
    for (; i < dimension; ++i) {
        result[i] = (uint16_t)((data[i] - lo) * reciprocal + 0.5F);
    }
}

void new_transpose_bin_lasx(
    const uint16_t *data, uint64_t *result, int64_t padded_dimension, int64_t bits
) {
    const __m256i one = __lasx_xvrepli_h(1);
    for (int64_t block = 0; block < padded_dimension / 64; ++block) {
        for (int64_t bit = 0; bit < bits; ++bit) {
            volatile uint64_t plane = 0;
            const __m256i shift = __lasx_xvreplgr2vr_h(bit);
            for (int64_t offset = 0; offset < 64; offset += 16) {
                const __m256i selected = __lasx_xvand_v(
                    __lasx_xvsrl_h(__lasx_xvld(data + block * 64 + offset, 0), shift),
                    one
                );
                uint16_t lanes[16];
                __lasx_xvst(selected, lanes, 0);
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
    const __m256i one = __lasx_xvrepli_b(1);
    for (int64_t block_start = 0; block_start < padded_dimension; block_start += 512) {
        const int64_t block_size = padded_dimension - block_start < 512
            ? padded_dimension - block_start
            : 512;
        const int64_t chunks = block_size / 64;
        const int64_t output_base = block_start / 64 * bits;
        for (int64_t bit = 0; bit < bits; ++bit) {
            const __m256i shift = __lasx_xvreplgr2vr_b(bit);
            for (int64_t chunk = 0; chunk < chunks; ++chunk) {
                volatile uint64_t plane = 0;
                const int64_t input_base = block_start + chunk * 64;
                for (int64_t offset = 0; offset < 64; offset += 32) {
                    const __m256i selected = __lasx_xvand_v(
                        __lasx_xvsrl_b(__lasx_xvld(data + input_base + offset, 0), shift),
                        one
                    );
                    uint8_t lanes[32];
                    __lasx_xvst(selected, lanes, 0);
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
    __m256 sum = (__m256)__lasx_xvldi(0);
    for (int64_t i = 0; i < padded_dimension; i += 8) {
        const uint64_t bits = data[i / 64];
        v8u32 mask;
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
        for (int lane = 0; lane < 8; ++lane) {
            const int shift = 63 - i % 64 - lane;
            mask[lane] = bits & (1ULL << shift) ? UINT32_MAX : 0;
        }
        const __m256 selected = (__m256)__lasx_xvand_v(
            __lasx_xvld(query + i, 0), (__m256i)mask
        );
        sum = __lasx_xvfadd_s(sum, selected);
    }
    return reduce_f32x8(sum);
}
