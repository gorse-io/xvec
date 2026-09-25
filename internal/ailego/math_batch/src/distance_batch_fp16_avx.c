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

// Match the existing single-pair AVX kernel's reduction order.
static inline float reduce_half_batch(__m256 value) {
 __m256 x1 = _mm256_hadd_ps(value, value);
 __m256 x2 = _mm256_hadd_ps(x1, x1);
 return _mm_cvtss_f32(_mm_add_ss(_mm256_castps256_ps128(x2), _mm256_extractf128_ps(x2, 1)));
}

void fp16_l2_avx4(const uint16_t *query, const uint16_t *first,
 const uint16_t *second, const uint16_t *third, const uint16_t *fourth,
 int64_t size, float *output) {
 __m256 sum0 = _mm256_setzero_ps();
 __m256 sum1 = _mm256_setzero_ps();
 __m256 sum2 = _mm256_setzero_ps();
 __m256 sum3 = _mm256_setzero_ps();
 int64_t i = 0;
 for (; i + 8 <= size; i += 8) {
  __m256 q = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(query+i)));
  __m256 v0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(first+i)));
  __m256 delta0 = _mm256_sub_ps(q, v0);
  sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(delta0, delta0));
  __m256 v1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(second+i)));
  __m256 delta1 = _mm256_sub_ps(q, v1);
  sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(delta1, delta1));
  __m256 v2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(third+i)));
  __m256 delta2 = _mm256_sub_ps(q, v2);
  sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(delta2, delta2));
  __m256 v3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(fourth+i)));
  __m256 delta3 = _mm256_sub_ps(q, v3);
  sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(delta3, delta3));
 }
 output[0] = reduce_half_batch(sum0);
 output[1] = reduce_half_batch(sum1);
 output[2] = reduce_half_batch(sum2);
 output[3] = reduce_half_batch(sum3);
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

void fp16_dot_avx4(const uint16_t *query, const uint16_t *first,
 const uint16_t *second, const uint16_t *third, const uint16_t *fourth,
 int64_t size, float *output) {
 __m256 sum0 = _mm256_setzero_ps();
 __m256 sum1 = _mm256_setzero_ps();
 __m256 sum2 = _mm256_setzero_ps();
 __m256 sum3 = _mm256_setzero_ps();
 int64_t i = 0;
 for (; i + 8 <= size; i += 8) {
  __m256 q = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(query+i)));
  __m256 v0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(first+i)));
  sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(q, v0));
  __m256 v1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(second+i)));
  sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(q, v1));
  __m256 v2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(third+i)));
  sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(q, v2));
  __m256 v3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(fourth+i)));
  sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(q, v3));
 }
 output[0] = reduce_half_batch(sum0);
 output[1] = reduce_half_batch(sum1);
 output[2] = reduce_half_batch(sum2);
 output[3] = reduce_half_batch(sum3);
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

void fp16_products_avx4(const uint16_t *query, const uint16_t *first,
 const uint16_t *second, const uint16_t *third, const uint16_t *fourth,
 int64_t size, float *output) {
 __m256 sum0 = _mm256_setzero_ps();
 __m256 sum1 = _mm256_setzero_ps();
 __m256 sum2 = _mm256_setzero_ps();
 __m256 sum3 = _mm256_setzero_ps();
 __m256 query_norm = _mm256_setzero_ps();
 __m256 norm0 = _mm256_setzero_ps();
 __m256 norm1 = _mm256_setzero_ps();
 __m256 norm2 = _mm256_setzero_ps();
 __m256 norm3 = _mm256_setzero_ps();
 int64_t i = 0;
 for (; i + 8 <= size; i += 8) {
  __m256 q = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(query+i)));
  query_norm = _mm256_add_ps(query_norm, _mm256_mul_ps(q, q));
  __m256 v0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(first+i)));
  sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(q, v0));
  norm0 = _mm256_add_ps(norm0, _mm256_mul_ps(v0, v0));
  __m256 v1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(second+i)));
  sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(q, v1));
  norm1 = _mm256_add_ps(norm1, _mm256_mul_ps(v1, v1));
  __m256 v2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(third+i)));
  sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(q, v2));
  norm2 = _mm256_add_ps(norm2, _mm256_mul_ps(v2, v2));
  __m256 v3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(fourth+i)));
  sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(q, v3));
  norm3 = _mm256_add_ps(norm3, _mm256_mul_ps(v3, v3));
 }
 output[0] = reduce_half_batch(sum0);
 output[1] = reduce_half_batch(sum1);
 output[2] = reduce_half_batch(sum2);
 output[3] = reduce_half_batch(sum3);
 output[4] = reduce_half_batch(query_norm);
 output[5] = reduce_half_batch(norm0);
 output[6] = reduce_half_batch(norm1);
 output[7] = reduce_half_batch(norm2);
 output[8] = reduce_half_batch(norm3);
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
