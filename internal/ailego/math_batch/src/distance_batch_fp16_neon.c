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

static inline float reduce_fp16_batch_neon(float32x4_t value) {
    return vaddvq_f32(value);
}

static inline float fp16_batch_to_fp32(uint16_t bits) {
    float16_t value;
    __builtin_memcpy(&value, &bits, sizeof(value));
    return (float)value;
}

static inline void load_fp16_batch_neon(const uint16_t *input,
                                        float32x4_t *low,
                                        float32x4_t *high) {
    float16x8_t value = vreinterpretq_f16_u16(vld1q_u16(input));
    *low = vcvt_f32_f16(vget_low_f16(value));
    *high = vcvt_high_f32_f16(value);
}

void fp16_l2_neon4(const uint16_t *query, const uint16_t *first,
                   const uint16_t *second, const uint16_t *third,
                   const uint16_t *fourth, int64_t size, float *output) {
    float32x4_t sums[4] = {vdupq_n_f32(0), vdupq_n_f32(0),
                           vdupq_n_f32(0), vdupq_n_f32(0)};
    const uint16_t *candidates[4] = {first, second, third, fourth};
    int64_t i = 0;
    for (; i + 8 <= size; i += 8) {
        float32x4_t query_low, query_high;
        load_fp16_batch_neon(query + i, &query_low, &query_high);
        for (int candidate = 0; candidate < 4; candidate++) {
            float32x4_t value_low, value_high;
            load_fp16_batch_neon(candidates[candidate] + i, &value_low, &value_high);
            float32x4_t delta = vsubq_f32(query_low, value_low);
            sums[candidate] = vmlaq_f32(sums[candidate], delta, delta);
            delta = vsubq_f32(query_high, value_high);
            sums[candidate] = vmlaq_f32(sums[candidate], delta, delta);
        }
    }
    for (int candidate = 0; candidate < 4; candidate++) {
        output[candidate] = reduce_fp16_batch_neon(sums[candidate]);
    }
    for (; i < size; i++) {
        float query_value = fp16_batch_to_fp32(query[i]);
        for (int candidate = 0; candidate < 4; candidate++) {
            float delta = query_value - fp16_batch_to_fp32(candidates[candidate][i]);
            output[candidate] += delta * delta;
        }
    }
}

void fp16_dot_neon4(const uint16_t *query, const uint16_t *first,
                    const uint16_t *second, const uint16_t *third,
                    const uint16_t *fourth, int64_t size, float *output) {
    float32x4_t sums[4] = {vdupq_n_f32(0), vdupq_n_f32(0),
                           vdupq_n_f32(0), vdupq_n_f32(0)};
    const uint16_t *candidates[4] = {first, second, third, fourth};
    int64_t i = 0;
    for (; i + 8 <= size; i += 8) {
        float32x4_t query_low, query_high;
        load_fp16_batch_neon(query + i, &query_low, &query_high);
        for (int candidate = 0; candidate < 4; candidate++) {
            float32x4_t value_low, value_high;
            load_fp16_batch_neon(candidates[candidate] + i, &value_low, &value_high);
            sums[candidate] = vmlaq_f32(sums[candidate], query_low, value_low);
            sums[candidate] = vmlaq_f32(sums[candidate], query_high, value_high);
        }
    }
    for (int candidate = 0; candidate < 4; candidate++) {
        output[candidate] = reduce_fp16_batch_neon(sums[candidate]);
    }
    for (; i < size; i++) {
        float query_value = fp16_batch_to_fp32(query[i]);
        for (int candidate = 0; candidate < 4; candidate++) {
            output[candidate] += query_value * fp16_batch_to_fp32(candidates[candidate][i]);
        }
    }
}

void fp16_products_neon4(const uint16_t *query, const uint16_t *first,
                         const uint16_t *second, const uint16_t *third,
                         const uint16_t *fourth, int64_t size, float *output) {
    float32x4_t dots[4] = {vdupq_n_f32(0), vdupq_n_f32(0),
                           vdupq_n_f32(0), vdupq_n_f32(0)};
    float32x4_t norms[4] = {vdupq_n_f32(0), vdupq_n_f32(0),
                            vdupq_n_f32(0), vdupq_n_f32(0)};
    float32x4_t query_norm = vdupq_n_f32(0);
    const uint16_t *candidates[4] = {first, second, third, fourth};
    int64_t i = 0;
    for (; i + 8 <= size; i += 8) {
        float32x4_t query_low, query_high;
        load_fp16_batch_neon(query + i, &query_low, &query_high);
        query_norm = vmlaq_f32(query_norm, query_low, query_low);
        query_norm = vmlaq_f32(query_norm, query_high, query_high);
        for (int candidate = 0; candidate < 4; candidate++) {
            float32x4_t value_low, value_high;
            load_fp16_batch_neon(candidates[candidate] + i, &value_low, &value_high);
            dots[candidate] = vmlaq_f32(dots[candidate], query_low, value_low);
            dots[candidate] = vmlaq_f32(dots[candidate], query_high, value_high);
            norms[candidate] = vmlaq_f32(norms[candidate], value_low, value_low);
            norms[candidate] = vmlaq_f32(norms[candidate], value_high, value_high);
        }
    }
    for (int candidate = 0; candidate < 4; candidate++) {
        output[candidate] = reduce_fp16_batch_neon(dots[candidate]);
        output[5 + candidate] = reduce_fp16_batch_neon(norms[candidate]);
    }
    output[4] = reduce_fp16_batch_neon(query_norm);
    for (; i < size; i++) {
        float query_value = fp16_batch_to_fp32(query[i]);
        output[4] += query_value * query_value;
        for (int candidate = 0; candidate < 4; candidate++) {
            float value = fp16_batch_to_fp32(candidates[candidate][i]);
            output[candidate] += query_value * value;
            output[5 + candidate] += value * value;
        }
    }
}


// Only these entry points require native FP16 arithmetic. Do not raise the
// translation unit ISA floor: the original NEON kernels remain available.
#define ASIMDHP __attribute__((target("fullfp16")))
// GoAT assembles Clang's textual output separately. Enable its assembler to
// encode FP16 instructions without enabling them for unrelated C functions.
__asm__(".arch_extension fp16");

ASIMDHP static inline float32x4_t add_half_products(float32x4_t sum, float16x8_t products) {
    sum = vaddq_f32(sum, vcvt_f32_f16(vget_low_f16(products)));
    return vaddq_f32(sum, vcvt_high_f32_f16(products));
}

static inline float16x8_t broadcast_half_bits(uint16_t bits) {
    return vreinterpretq_f16_u16(vdupq_n_u16(bits));
}

ASIMDHP static inline float half_product_lane(float16x8_t products) {
    return vgetq_lane_f32(vcvt_f32_f16(vget_low_f16(products)), 0);
}

ASIMDHP void fp16_l2_asimdhp4(const uint16_t *query, const uint16_t *first,
                   const uint16_t *second, const uint16_t *third,
                   const uint16_t *fourth, int64_t size, float *output) {
    const uint16_t *candidates[4] = {first, second, third, fourth};
    float32x4_t sums[4] = {vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0)};
    int64_t i = 0;
    for (; i + 8 <= size; i += 8) {
        float16x8_t q = vreinterpretq_f16_u16(vld1q_u16(query + i));
        for (int j = 0; j < 4; j++) {
            float16x8_t v = vreinterpretq_f16_u16(vld1q_u16(candidates[j] + i));
            float16x8_t delta = vsubq_f16(q, v);
            sums[j] = add_half_products(sums[j], vmulq_f16(delta, delta));
        }
    }
    for (int j = 0; j < 4; j++) {
        output[j] = vaddvq_f32(sums[j]);
    }
    // Broadcast single elements so the tail uses exactly the same rounded
    // subtraction/multiplication as vector blocks, including sizes below 8.
    for (; i < size; i++) {
        float16x8_t q = broadcast_half_bits(query[i]);
        for (int j = 0; j < 4; j++) {
            float16x8_t v = broadcast_half_bits(candidates[j][i]);
            float16x8_t delta = vsubq_f16(q, v);
            output[j] += half_product_lane(vmulq_f16(delta, delta));
        }
    }
}

ASIMDHP void fp16_dot_asimdhp4(const uint16_t *query, const uint16_t *first,
                   const uint16_t *second, const uint16_t *third,
                   const uint16_t *fourth, int64_t size, float *output) {
    const uint16_t *candidates[4] = {first, second, third, fourth};
    float32x4_t sums[4] = {vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0)};
    int64_t i = 0;
    for (; i + 8 <= size; i += 8) {
        float16x8_t q = vreinterpretq_f16_u16(vld1q_u16(query + i));
        for (int j = 0; j < 4; j++) {
            float16x8_t v = vreinterpretq_f16_u16(vld1q_u16(candidates[j] + i));
            sums[j] = add_half_products(sums[j], vmulq_f16(q, v));
        }
    }
    for (int j = 0; j < 4; j++) {
        output[j] = vaddvq_f32(sums[j]);
    }
    // Broadcast single elements so the tail uses exactly the same rounded
    // subtraction/multiplication as vector blocks, including sizes below 8.
    for (; i < size; i++) {
        float16x8_t q = broadcast_half_bits(query[i]);
        for (int j = 0; j < 4; j++) {
            float16x8_t v = broadcast_half_bits(candidates[j][i]);
            output[j] += half_product_lane(vmulq_f16(q, v));
        }
    }
}

ASIMDHP void fp16_products_asimdhp4(const uint16_t *query, const uint16_t *first,
                   const uint16_t *second, const uint16_t *third,
                   const uint16_t *fourth, int64_t size, float *output) {
    const uint16_t *candidates[4] = {first, second, third, fourth};
    float32x4_t sums[4] = {vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0)};
    float32x4_t norms[4] = {vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0)};
    float32x4_t query_norm = vdupq_n_f32(0);
    int64_t i = 0;
    for (; i + 8 <= size; i += 8) {
        float16x8_t q = vreinterpretq_f16_u16(vld1q_u16(query + i));
        query_norm = add_half_products(query_norm, vmulq_f16(q, q));
        for (int j = 0; j < 4; j++) {
            float16x8_t v = vreinterpretq_f16_u16(vld1q_u16(candidates[j] + i));
            sums[j] = add_half_products(sums[j], vmulq_f16(q, v));
            norms[j] = add_half_products(norms[j], vmulq_f16(v, v));
        }
    }
    for (int j = 0; j < 4; j++) {
        output[j] = vaddvq_f32(sums[j]);
        output[5+j] = vaddvq_f32(norms[j]);
    }
    output[4] = vaddvq_f32(query_norm);
    // Broadcast single elements so the tail uses exactly the same rounded
    // subtraction/multiplication as vector blocks, including sizes below 8.
    for (; i < size; i++) {
        float16x8_t q = broadcast_half_bits(query[i]);
        output[4] += half_product_lane(vmulq_f16(q, q));
        for (int j = 0; j < 4; j++) {
            float16x8_t v = broadcast_half_bits(candidates[j][i]);
            output[j] += half_product_lane(vmulq_f16(q, v));
            output[5+j] += half_product_lane(vmulq_f16(v, v));
        }
    }
}
