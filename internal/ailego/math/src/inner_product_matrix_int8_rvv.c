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

int64_t inner_product_int8_rvv(const int8_t *left, const int8_t *right,
                               int64_t size) {
  int64_t result = 0;
  while (size > 0) {
    size_t vl = __riscv_vsetvl_e8m1(size);
    vint8m1_t left8 = __riscv_vle8_v_i8m1(left, vl);
    vint8m1_t right8 = __riscv_vle8_v_i8m1(right, vl);
    vint16m2_t left16 = __riscv_vsext_vf2_i16m2(left8, vl);
    vint16m2_t right16 = __riscv_vsext_vf2_i16m2(right8, vl);
    vint32m4_t product = __riscv_vwmul_vv_i32m4(left16, right16, vl);
    vint64m1_t zero = __riscv_vmv_v_x_i64m1(0, 1);
    vint64m1_t sum = __riscv_vwredsum_vs_i32m4_i64m1(product, zero, vl);
    result += __riscv_vmv_x_s_i64m1_i64(sum);
    left += vl;
    right += vl;
    size -= vl;
  }
  return result;
}
