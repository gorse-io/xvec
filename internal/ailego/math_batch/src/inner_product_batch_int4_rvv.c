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

static inline vint8m1_t batch_decode_low_rvv(vuint8m1_t packed, size_t vl) {
  vuint8m1_t nibble = __riscv_vand_vx_u8m1(packed, 15, vl);
  return __riscv_vreinterpret_v_u8m1_i8m1(
      __riscv_vsub_vx_u8m1(__riscv_vxor_vx_u8m1(nibble, 8, vl), 8, vl));
}

static inline vint8m1_t batch_decode_high_rvv(vuint8m1_t packed, size_t vl) {
  vuint8m1_t nibble = __riscv_vsrl_vx_u8m1(packed, 4, vl);
  return __riscv_vreinterpret_v_u8m1_i8m1(
      __riscv_vsub_vx_u8m1(__riscv_vxor_vx_u8m1(nibble, 8, vl), 8, vl));
}

static inline int64_t batch_dot_int8_rvv(vint8m1_t left,
                                         vint8m1_t right, size_t vl) {
  vint16m2_t left16 = __riscv_vsext_vf2_i16m2(left, vl);
  vint16m2_t right16 = __riscv_vsext_vf2_i16m2(right, vl);
  vint32m4_t product = __riscv_vwmul_vv_i32m4(left16, right16, vl);
  vint64m1_t zero = __riscv_vmv_v_x_i64m1(0, 1);
  vint64m1_t sum = __riscv_vwredsum_vs_i32m4_i64m1(product, zero, vl);
  return __riscv_vmv_x_s_i64m1_i64(sum);
}

void xvec_rvv_batch_inner_products_int4_4(
    const uint8_t *query, const uint8_t *first, const uint8_t *second,
    const uint8_t *third, const uint8_t *fourth, int64_t size,
    int64_t *output) {
  const uint8_t *candidates[4] = {first, second, third, fourth};
  int64_t totals[4] = {0, 0, 0, 0};
  while (size > 0) {
    size_t vl = __riscv_vsetvl_e8m1(size);
    vuint8m1_t packed_query = __riscv_vle8_v_u8m1(query, vl);
    vint8m1_t query_low = batch_decode_low_rvv(packed_query, vl);
    vint8m1_t query_high = batch_decode_high_rvv(packed_query, vl);
    for (int j = 0; j < 4; ++j) {
      vuint8m1_t packed = __riscv_vle8_v_u8m1(candidates[j], vl);
      totals[j] += batch_dot_int8_rvv(
          query_low, batch_decode_low_rvv(packed, vl), vl);
      totals[j] += batch_dot_int8_rvv(
          query_high, batch_decode_high_rvv(packed, vl), vl);
      candidates[j] += vl;
    }
    query += vl;
    size -= vl;
  }
  for (int j = 0; j < 4; ++j) {
    output[j] = totals[j];
  }
}
