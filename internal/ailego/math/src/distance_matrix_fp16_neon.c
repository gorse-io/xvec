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

static inline float horizontal_add_fp32x4(float32x4_t value) {
  return vaddvq_f32(value);
}

static inline float fp16_to_fp32(uint16_t bits) {
  float16_t value;
  __builtin_memcpy(&value, &bits, sizeof(value));
  return (float)value;
}

float squared_euclidean_distance_fp16_neon(const uint16_t *lhs,
                                            const uint16_t *rhs, int64_t size) {
  float32x4_t sum = vdupq_n_f32(0.0f);
  int64_t index = 0;
  for (; index + 8 <= size; index += 8) {
    float16x8_t left_half = vreinterpretq_f16_u16(vld1q_u16(lhs + index));
    float16x8_t right_half = vreinterpretq_f16_u16(vld1q_u16(rhs + index));
    float32x4_t difference = vsubq_f32(vcvt_f32_f16(vget_low_f16(left_half)),
                                      vcvt_f32_f16(vget_low_f16(right_half)));
    sum = vmlaq_f32(sum, difference, difference);
    difference = vsubq_f32(vcvt_high_f32_f16(left_half),
                           vcvt_high_f32_f16(right_half));
    sum = vmlaq_f32(sum, difference, difference);
  }
  float result = horizontal_add_fp32x4(sum);
  for (; index < size; ++index) {
    float difference = fp16_to_fp32(lhs[index]) - fp16_to_fp32(rhs[index]);
    result += difference * difference;
  }
  return result;
}

float inner_product_fp16_neon(const uint16_t *lhs, const uint16_t *rhs,
                              int64_t size) {
  float32x4_t sum = vdupq_n_f32(0.0f);
  int64_t index = 0;
  for (; index + 8 <= size; index += 8) {
    float16x8_t left_half = vreinterpretq_f16_u16(vld1q_u16(lhs + index));
    float16x8_t right_half = vreinterpretq_f16_u16(vld1q_u16(rhs + index));
    sum = vmlaq_f32(sum, vcvt_f32_f16(vget_low_f16(left_half)),
                    vcvt_f32_f16(vget_low_f16(right_half)));
    sum = vmlaq_f32(sum, vcvt_high_f32_f16(left_half),
                    vcvt_high_f32_f16(right_half));
  }
  float result = horizontal_add_fp32x4(sum);
  for (; index < size; ++index) {
    result += fp16_to_fp32(lhs[index]) * fp16_to_fp32(rhs[index]);
  }
  return result;
}

float inner_product_and_squared_norm_fp16_neon(
    const uint16_t *lhs, const uint16_t *rhs, int64_t size, float *lhs_norm,
    float *rhs_norm) {
  float32x4_t dot = vdupq_n_f32(0.0f);
  float32x4_t left_sum = vdupq_n_f32(0.0f);
  float32x4_t right_sum = vdupq_n_f32(0.0f);
  int64_t index = 0;
  for (; index + 8 <= size; index += 8) {
    float16x8_t left_half = vreinterpretq_f16_u16(vld1q_u16(lhs + index));
    float16x8_t right_half = vreinterpretq_f16_u16(vld1q_u16(rhs + index));
    float32x4_t left = vcvt_f32_f16(vget_low_f16(left_half));
    float32x4_t right = vcvt_f32_f16(vget_low_f16(right_half));
    dot = vmlaq_f32(dot, left, right);
    left_sum = vmlaq_f32(left_sum, left, left);
    right_sum = vmlaq_f32(right_sum, right, right);
    left = vcvt_high_f32_f16(left_half);
    right = vcvt_high_f32_f16(right_half);
    dot = vmlaq_f32(dot, left, right);
    left_sum = vmlaq_f32(left_sum, left, left);
    right_sum = vmlaq_f32(right_sum, right, right);
  }
  float dot_result = horizontal_add_fp32x4(dot);
  float left_result = horizontal_add_fp32x4(left_sum);
  float right_result = horizontal_add_fp32x4(right_sum);
  for (; index < size; ++index) {
    float left = fp16_to_fp32(lhs[index]);
    float right = fp16_to_fp32(rhs[index]);
    dot_result += left * right;
    left_result += left * left;
    right_result += right * right;
  }
  *lhs_norm = left_result;
  *rhs_norm = right_result;
  return dot_result;
}
