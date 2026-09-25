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

static inline __m256i batch_decode_low_lasx(__m256i packed) {
  const __m256i sign = __lasx_xvrepli_b(8);
  __m256i nibble = __lasx_xvand_v(packed, __lasx_xvrepli_b(15));
  return __lasx_xvsub_b(__lasx_xvxor_v(nibble, sign), sign);
}

static inline __m256i batch_decode_high_lasx(__m256i packed) {
  const __m256i sign = __lasx_xvrepli_b(8);
  __m256i nibble = __lasx_xvsrli_b(packed, 4);
  return __lasx_xvsub_b(__lasx_xvxor_v(nibble, sign), sign);
}

static inline __m256i batch_dot_bytes_lasx(__m256i left, __m256i right) {
  const __m256i zero = __lasx_xvrepli_w(0);
  __m256i even = __lasx_xvmulwev_h_b(left, right);
  __m256i odd = __lasx_xvmulwod_h_b(left, right);
  __m256i even32 = __lasx_xvadd_w(__lasx_xvaddwev_w_h(even, zero),
                                  __lasx_xvaddwod_w_h(even, zero));
  __m256i odd32 = __lasx_xvadd_w(__lasx_xvaddwev_w_h(odd, zero),
                                 __lasx_xvaddwod_w_h(odd, zero));
  return __lasx_xvadd_w(even32, odd32);
}

static inline long batch_reduce_lasx(__m256i values) {
  int lanes[8];
  __lasx_xvst(values, lanes, 0);
  long sum = 0;
  for (int lane = 0; lane < 8; ++lane) {
    sum += lanes[lane];
  }
  return sum;
}

void xvec_lasx_batch_inner_products_int4_4(
    const unsigned char *query, const unsigned char *first, const unsigned char *second,
    const unsigned char *third, const unsigned char *fourth, long size,
    long *output) {
  const unsigned char *candidates[4] = {first, second, third, fourth};
  long totals[4] = {0, 0, 0, 0};
  while (size != 0) {
    long count = size < 1048576 ? size : 1048576;
    __m256i sums[4];
    for (int j = 0; j < 4; ++j) {
      sums[j] = __lasx_xvrepli_w(0);
    }
    for (long i = 0; i < count; i += 32) {
      __m256i packed_query = __lasx_xvld(query + i, 0);
      __m256i query_low = batch_decode_low_lasx(packed_query);
      __m256i query_high = batch_decode_high_lasx(packed_query);
      for (int j = 0; j < 4; ++j) {
        __m256i packed = __lasx_xvld(candidates[j] + i, 0);
        sums[j] = __lasx_xvadd_w(
            sums[j], batch_dot_bytes_lasx(
                         query_low, batch_decode_low_lasx(packed)));
        sums[j] = __lasx_xvadd_w(
            sums[j], batch_dot_bytes_lasx(
                         query_high, batch_decode_high_lasx(packed)));
      }
    }
    for (int j = 0; j < 4; ++j) {
      totals[j] += batch_reduce_lasx(sums[j]);
      candidates[j] += count;
    }
    query += count;
    size -= count;
  }
  for (int j = 0; j < 4; ++j) {
    output[j] = totals[j];
  }
}
