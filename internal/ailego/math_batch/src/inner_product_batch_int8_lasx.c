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

static inline __m256i batch_dot_int8_lasx(__m256i left, __m256i right) {
  const __m256i zero = __lasx_xvrepli_w(0);
  __m256i even = __lasx_xvmulwev_h_b(left, right);
  __m256i odd = __lasx_xvmulwod_h_b(left, right);
  __m256i even32 = __lasx_xvadd_w(__lasx_xvaddwev_w_h(even, zero),
                                  __lasx_xvaddwod_w_h(even, zero));
  __m256i odd32 = __lasx_xvadd_w(__lasx_xvaddwev_w_h(odd, zero),
                                 __lasx_xvaddwod_w_h(odd, zero));
  return __lasx_xvadd_w(even32, odd32);
}

static inline long batch_reduce_int8_lasx(__m256i values) {
  int lanes[8];
  __lasx_xvst(values, lanes, 0);
  long sum = 0;
  for (int lane = 0; lane < 8; ++lane) sum += lanes[lane];
  return sum;
}

void xvec_lasx_batch_inner_products_int8_4(const signed char *const *vectors,
                                            long size, long *output) {
  const signed char *query = vectors[0];
  const signed char *candidates[4] = {vectors[1], vectors[2], vectors[3], vectors[4]};
  long totals[4] = {0, 0, 0, 0};
  while (size != 0) {
    long count = size < 65536 ? size : 65536;
    __m256i sums[4];
    for (int j = 0; j < 4; ++j) sums[j] = __lasx_xvrepli_w(0);
    for (long i = 0; i < count; i += 32) {
      __m256i q = __lasx_xvld(query + i, 0);
      for (int j = 0; j < 4; ++j) {
        sums[j] = __lasx_xvadd_w(
            sums[j], batch_dot_int8_lasx(q, __lasx_xvld(candidates[j] + i, 0)));
      }
    }
    for (int j = 0; j < 4; ++j) {
      totals[j] += batch_reduce_int8_lasx(sums[j]);
      candidates[j] += count;
    }
    query += count;
    size -= count;
  }
  for (int j = 0; j < 4; ++j) output[j] = totals[j];
}
