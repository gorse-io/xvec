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

static inline __m256i multiply_add_int8_lasx(__m256i left,
                                             __m256i right) {
  const __m256i zero = __lasx_xvrepli_w(0);
  __m256i even = __lasx_xvmulwev_h_b(left, right);
  __m256i odd = __lasx_xvmulwod_h_b(left, right);
  __m256i even32 = __lasx_xvadd_w(__lasx_xvaddwev_w_h(even, zero),
                                  __lasx_xvaddwod_w_h(even, zero));
  __m256i odd32 = __lasx_xvadd_w(__lasx_xvaddwev_w_h(odd, zero),
                                 __lasx_xvaddwod_w_h(odd, zero));
  return __lasx_xvadd_w(even32, odd32);
}

static inline long horizontal_sum_int32_lasx(__m256i values) {
  int lanes[8];
  __lasx_xvst(values, lanes, 0);
  long sum = 0;
  for (int lane = 0; lane < 8; ++lane) {
    sum += lanes[lane];
  }
  return sum;
}

long inner_product_int8_lasx(const signed char *left,
                             const signed char *right, long size) {
  long result = 0;
  while (size != 0) {
    // Each lane receives four products per 32-byte vector. Reducing after at
    // most 524288 bytes keeps worst-case signed products within int32.
    long count = size < 524288 ? size : 524288;
    __m256i sum = __lasx_xvrepli_w(0);
    for (long i = 0; i < count; i += 32) {
      __m256i left8 = __lasx_xvld(left + i, 0);
      __m256i right8 = __lasx_xvld(right + i, 0);
      sum = __lasx_xvadd_w(sum, multiply_add_int8_lasx(left8, right8));
    }
    result += horizontal_sum_int32_lasx(sum);
    left += count;
    right += count;
    size -= count;
  }
  return result;
}
