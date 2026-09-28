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

#include <immintrin.h>
#include <stdint.h>

static inline float reduce_half_batch_avx512(__m512 value) {
 float partial[16];
 _mm512_storeu_ps(partial, value);
 float sum = 0;
 for (int index = 0; index < 16; index++) {
  sum += partial[index];
 }
 return sum;
}

void fp16_l2_avx512_4(const uint16_t *query, const uint16_t *first,
 const uint16_t *second, const uint16_t *third, const uint16_t *fourth,
 int64_t size, float *output) {
 __m512 sum0 = _mm512_setzero_ps();
 __m512 sum1 = _mm512_setzero_ps();
 __m512 sum2 = _mm512_setzero_ps();
 __m512 sum3 = _mm512_setzero_ps();
 int64_t i = 0;
 for (; i + 16 <= size; i += 16) {
  __m512 q = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(query+i)));
  __m512 v0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(first+i)));
  __m512 delta0 = _mm512_sub_ps(q, v0);
  sum0 = _mm512_add_ps(sum0, _mm512_mul_ps(delta0, delta0));
  __m512 v1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(second+i)));
  __m512 delta1 = _mm512_sub_ps(q, v1);
  sum1 = _mm512_add_ps(sum1, _mm512_mul_ps(delta1, delta1));
  __m512 v2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(third+i)));
  __m512 delta2 = _mm512_sub_ps(q, v2);
  sum2 = _mm512_add_ps(sum2, _mm512_mul_ps(delta2, delta2));
  __m512 v3 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(fourth+i)));
  __m512 delta3 = _mm512_sub_ps(q, v3);
  sum3 = _mm512_add_ps(sum3, _mm512_mul_ps(delta3, delta3));
 }
 output[0] = reduce_half_batch_avx512(sum0);
 output[1] = reduce_half_batch_avx512(sum1);
 output[2] = reduce_half_batch_avx512(sum2);
 output[3] = reduce_half_batch_avx512(sum3);
 for (; i < size; ++i) {
  float q = _cvtsh_ss(query[i]);
  float v0 = _cvtsh_ss(first[i]);
  float delta0 = q-v0;
  output[0] += delta0*delta0;
  float v1 = _cvtsh_ss(second[i]);
  float delta1 = q-v1;
  output[1] += delta1*delta1;
  float v2 = _cvtsh_ss(third[i]);
  float delta2 = q-v2;
  output[2] += delta2*delta2;
  float v3 = _cvtsh_ss(fourth[i]);
  float delta3 = q-v3;
  output[3] += delta3*delta3;
 }
}

void fp16_dot_avx512_4(const uint16_t *query, const uint16_t *first,
 const uint16_t *second, const uint16_t *third, const uint16_t *fourth,
 int64_t size, float *output) {
 __m512 sum0 = _mm512_setzero_ps();
 __m512 sum1 = _mm512_setzero_ps();
 __m512 sum2 = _mm512_setzero_ps();
 __m512 sum3 = _mm512_setzero_ps();
 int64_t i = 0;
 for (; i + 16 <= size; i += 16) {
  __m512 q = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(query+i)));
  __m512 v0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(first+i)));
  sum0 = _mm512_add_ps(sum0, _mm512_mul_ps(q, v0));
  __m512 v1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(second+i)));
  sum1 = _mm512_add_ps(sum1, _mm512_mul_ps(q, v1));
  __m512 v2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(third+i)));
  sum2 = _mm512_add_ps(sum2, _mm512_mul_ps(q, v2));
  __m512 v3 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(fourth+i)));
  sum3 = _mm512_add_ps(sum3, _mm512_mul_ps(q, v3));
 }
 output[0] = reduce_half_batch_avx512(sum0);
 output[1] = reduce_half_batch_avx512(sum1);
 output[2] = reduce_half_batch_avx512(sum2);
 output[3] = reduce_half_batch_avx512(sum3);
 for (; i < size; ++i) {
  float q = _cvtsh_ss(query[i]);
  float v0 = _cvtsh_ss(first[i]);
  output[0] += q*v0;
  float v1 = _cvtsh_ss(second[i]);
  output[1] += q*v1;
  float v2 = _cvtsh_ss(third[i]);
  output[2] += q*v2;
  float v3 = _cvtsh_ss(fourth[i]);
  output[3] += q*v3;
 }
}

void fp16_products_avx512_4(const uint16_t *query, const uint16_t *first,
 const uint16_t *second, const uint16_t *third, const uint16_t *fourth,
 int64_t size, float *output) {
 __m512 sum0 = _mm512_setzero_ps();
 __m512 sum1 = _mm512_setzero_ps();
 __m512 sum2 = _mm512_setzero_ps();
 __m512 sum3 = _mm512_setzero_ps();
 __m512 query_norm = _mm512_setzero_ps();
 __m512 norm0 = _mm512_setzero_ps();
 __m512 norm1 = _mm512_setzero_ps();
 __m512 norm2 = _mm512_setzero_ps();
 __m512 norm3 = _mm512_setzero_ps();
 int64_t i = 0;
 for (; i + 16 <= size; i += 16) {
  __m512 q = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(query+i)));
  query_norm = _mm512_add_ps(query_norm, _mm512_mul_ps(q, q));
  __m512 v0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(first+i)));
  sum0 = _mm512_add_ps(sum0, _mm512_mul_ps(q, v0));
  norm0 = _mm512_add_ps(norm0, _mm512_mul_ps(v0, v0));
  __m512 v1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(second+i)));
  sum1 = _mm512_add_ps(sum1, _mm512_mul_ps(q, v1));
  norm1 = _mm512_add_ps(norm1, _mm512_mul_ps(v1, v1));
  __m512 v2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(third+i)));
  sum2 = _mm512_add_ps(sum2, _mm512_mul_ps(q, v2));
  norm2 = _mm512_add_ps(norm2, _mm512_mul_ps(v2, v2));
  __m512 v3 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(fourth+i)));
  sum3 = _mm512_add_ps(sum3, _mm512_mul_ps(q, v3));
  norm3 = _mm512_add_ps(norm3, _mm512_mul_ps(v3, v3));
 }
 output[0] = reduce_half_batch_avx512(sum0);
 output[1] = reduce_half_batch_avx512(sum1);
 output[2] = reduce_half_batch_avx512(sum2);
 output[3] = reduce_half_batch_avx512(sum3);
 output[4] = reduce_half_batch_avx512(query_norm);
 output[5] = reduce_half_batch_avx512(norm0);
 output[6] = reduce_half_batch_avx512(norm1);
 output[7] = reduce_half_batch_avx512(norm2);
 output[8] = reduce_half_batch_avx512(norm3);
 for (; i < size; ++i) {
  float q = _cvtsh_ss(query[i]);
  output[4] += q*q;
  float v0 = _cvtsh_ss(first[i]);
  output[0] += q*v0;
  output[5] += v0*v0;
  float v1 = _cvtsh_ss(second[i]);
  output[1] += q*v1;
  output[6] += v1*v1;
  float v2 = _cvtsh_ss(third[i]);
  output[2] += q*v2;
  output[7] += v2*v2;
  float v3 = _cvtsh_ss(fourth[i]);
  output[3] += q*v3;
  output[8] += v3*v3;
 }
}
