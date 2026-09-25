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

#include <riscv_vector.h>
#include <stdint.h>

static inline vint8m1_t decode_low_rvv(vuint8m1_t packed, size_t vl) {
  vuint8m1_t nibble = __riscv_vand_vx_u8m1(packed, 15, vl);
  return __riscv_vreinterpret_v_u8m1_i8m1(
      __riscv_vsub_vx_u8m1(__riscv_vxor_vx_u8m1(nibble, 8, vl), 8, vl));
}

static inline vint8m1_t decode_high_rvv(vuint8m1_t packed, size_t vl) {
  vuint8m1_t nibble = __riscv_vsrl_vx_u8m1(packed, 4, vl);
  return __riscv_vreinterpret_v_u8m1_i8m1(
      __riscv_vsub_vx_u8m1(__riscv_vxor_vx_u8m1(nibble, 8, vl), 8, vl));
}

static inline int64_t dot_int8_rvv(vint8m1_t left, vint8m1_t right,
                                   size_t vl) {
  vint16m2_t left16 = __riscv_vsext_vf2_i16m2(left, vl);
  vint16m2_t right16 = __riscv_vsext_vf2_i16m2(right, vl);
  vint32m4_t product = __riscv_vwmul_vv_i32m4(left16, right16, vl);
  vint64m1_t zero = __riscv_vmv_v_x_i64m1(0, 1);
  vint64m1_t sum = __riscv_vwredsum_vs_i32m4_i64m1(product, zero, vl);
  return __riscv_vmv_x_s_i64m1_i64(sum);
}

int64_t inner_product_int4_rvv(const uint8_t *left, const uint8_t *right,
                               int64_t size) {
  int64_t result = 0;
  while (size > 0) {
    size_t vl = __riscv_vsetvl_e8m1(size);
    vuint8m1_t packed_left = __riscv_vle8_v_u8m1(left, vl);
    vuint8m1_t packed_right = __riscv_vle8_v_u8m1(right, vl);
    result += dot_int8_rvv(decode_low_rvv(packed_left, vl),
                           decode_low_rvv(packed_right, vl), vl);
    result += dot_int8_rvv(decode_high_rvv(packed_left, vl),
                           decode_high_rvv(packed_right, vl), vl);
    left += vl;
    right += vl;
    size -= vl;
  }
  return result;
}

int64_t squared_euclidean_int4_rvv(const uint8_t *left,
                                   const uint8_t *right, int64_t size) {
  int64_t result = 0;
  while (size > 0) {
    size_t vl = __riscv_vsetvl_e8m1(size);
    vuint8m1_t packed_left = __riscv_vle8_v_u8m1(left, vl);
    vuint8m1_t packed_right = __riscv_vle8_v_u8m1(right, vl);
    vint8m1_t low_difference = __riscv_vsub_vv_i8m1(
        decode_low_rvv(packed_left, vl), decode_low_rvv(packed_right, vl), vl);
    vint8m1_t high_difference = __riscv_vsub_vv_i8m1(
        decode_high_rvv(packed_left, vl), decode_high_rvv(packed_right, vl),
        vl);
    result += dot_int8_rvv(low_difference, low_difference, vl);
    result += dot_int8_rvv(high_difference, high_difference, vl);
    left += vl;
    right += vl;
    size -= vl;
  }
  return result;
}

void dot_norms_int4_rvv(const uint8_t *left, const uint8_t *right,
                        int64_t size, int64_t *out) {
  int64_t dot = 0;
  int64_t left_norm = 0;
  int64_t right_norm = 0;
  while (size > 0) {
    size_t vl = __riscv_vsetvl_e8m1(size);
    vuint8m1_t packed_left = __riscv_vle8_v_u8m1(left, vl);
    vuint8m1_t packed_right = __riscv_vle8_v_u8m1(right, vl);
    vint8m1_t left_low = decode_low_rvv(packed_left, vl);
    vint8m1_t left_high = decode_high_rvv(packed_left, vl);
    vint8m1_t right_low = decode_low_rvv(packed_right, vl);
    vint8m1_t right_high = decode_high_rvv(packed_right, vl);
    dot += dot_int8_rvv(left_low, right_low, vl);
    dot += dot_int8_rvv(left_high, right_high, vl);
    left_norm += dot_int8_rvv(left_low, left_low, vl);
    left_norm += dot_int8_rvv(left_high, left_high, vl);
    right_norm += dot_int8_rvv(right_low, right_low, vl);
    right_norm += dot_int8_rvv(right_high, right_high, vl);
    left += vl;
    right += vl;
    size -= vl;
  }
  out[0] = dot;
  out[1] = left_norm;
  out[2] = right_norm;
}
