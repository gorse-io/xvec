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

static inline int64_t batch_dot_int8_rvv(vint8m1_t left, vint8m1_t right,
                                          size_t vl) {
  vint16m2_t left16 = __riscv_vsext_vf2_i16m2(left, vl);
  vint16m2_t right16 = __riscv_vsext_vf2_i16m2(right, vl);
  vint32m4_t product = __riscv_vwmul_vv_i32m4(left16, right16, vl);
  vint64m1_t zero = __riscv_vmv_v_x_i64m1(0, 1);
  vint64m1_t sum = __riscv_vwredsum_vs_i32m4_i64m1(product, zero, vl);
  return __riscv_vmv_x_s_i64m1_i64(sum);
}

void xvec_rvv_batch_inner_products_int8_4(const int8_t *const *vectors,
                                           int64_t size, int64_t *output) {
  const int8_t *query = vectors[0];
  const int8_t *candidates[4] = {vectors[1], vectors[2], vectors[3], vectors[4]};
  int64_t totals[4] = {0, 0, 0, 0};
  while (size > 0) {
    size_t vl = __riscv_vsetvl_e8m1(size);
    vint8m1_t q = __riscv_vle8_v_i8m1(query, vl);
    for (int j = 0; j < 4; ++j) {
      totals[j] += batch_dot_int8_rvv(
          q, __riscv_vle8_v_i8m1(candidates[j], vl), vl);
      candidates[j] += vl;
    }
    query += vl;
    size -= vl;
  }
  for (int j = 0; j < 4; ++j) output[j] = totals[j];
}
