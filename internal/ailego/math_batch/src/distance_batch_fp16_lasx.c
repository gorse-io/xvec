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

static inline float reduce_fp16_batch_lasx(__m256 value) {
    volatile float partial[8];
    __lasx_xvst((__m256i)value, (void *)partial, 0);
    float sum = 0;
    for (int index = 0; index < 8; index++) {
        sum += partial[index];
    }
    return sum;
}

static inline void load_fp16_batch_lasx(const unsigned short *input,
                                        __m256 *low, __m256 *high) {
    __m256i packed = __lasx_xvld((const void *)input, 0);
    *low = __lasx_xvfcvtl_s_h(packed);
    *high = __lasx_xvfcvth_s_h(packed);
}

void fp16_l2_lasx4(const unsigned short *query, const unsigned short *first,
                   const unsigned short *second, const unsigned short *third,
                   const unsigned short *fourth, long size, float *output) {
    __m256 sum0 = (__m256)__lasx_xvreplgr2vr_w(0);
    __m256 sum1 = (__m256)__lasx_xvreplgr2vr_w(0);
    __m256 sum2 = (__m256)__lasx_xvreplgr2vr_w(0);
    __m256 sum3 = (__m256)__lasx_xvreplgr2vr_w(0);
    for (long index = 0; index < size; index += 16) {
        __m256 ql, qh, vl, vh, delta;
        load_fp16_batch_lasx(query + index, &ql, &qh);
        load_fp16_batch_lasx(first + index, &vl, &vh);
        delta = __lasx_xvfsub_s(ql, vl);
        sum0 = __lasx_xvfadd_s(sum0, __lasx_xvfmul_s(delta, delta));
        delta = __lasx_xvfsub_s(qh, vh);
        sum0 = __lasx_xvfadd_s(sum0, __lasx_xvfmul_s(delta, delta));
        load_fp16_batch_lasx(second + index, &vl, &vh);
        delta = __lasx_xvfsub_s(ql, vl);
        sum1 = __lasx_xvfadd_s(sum1, __lasx_xvfmul_s(delta, delta));
        delta = __lasx_xvfsub_s(qh, vh);
        sum1 = __lasx_xvfadd_s(sum1, __lasx_xvfmul_s(delta, delta));
        load_fp16_batch_lasx(third + index, &vl, &vh);
        delta = __lasx_xvfsub_s(ql, vl);
        sum2 = __lasx_xvfadd_s(sum2, __lasx_xvfmul_s(delta, delta));
        delta = __lasx_xvfsub_s(qh, vh);
        sum2 = __lasx_xvfadd_s(sum2, __lasx_xvfmul_s(delta, delta));
        load_fp16_batch_lasx(fourth + index, &vl, &vh);
        delta = __lasx_xvfsub_s(ql, vl);
        sum3 = __lasx_xvfadd_s(sum3, __lasx_xvfmul_s(delta, delta));
        delta = __lasx_xvfsub_s(qh, vh);
        sum3 = __lasx_xvfadd_s(sum3, __lasx_xvfmul_s(delta, delta));
    }
    output[0] = reduce_fp16_batch_lasx(sum0);
    output[1] = reduce_fp16_batch_lasx(sum1);
    output[2] = reduce_fp16_batch_lasx(sum2);
    output[3] = reduce_fp16_batch_lasx(sum3);
}

void fp16_dot_lasx4(const unsigned short *query, const unsigned short *first,
                    const unsigned short *second, const unsigned short *third,
                    const unsigned short *fourth, long size, float *output) {
    __m256 sum0 = (__m256)__lasx_xvreplgr2vr_w(0);
    __m256 sum1 = (__m256)__lasx_xvreplgr2vr_w(0);
    __m256 sum2 = (__m256)__lasx_xvreplgr2vr_w(0);
    __m256 sum3 = (__m256)__lasx_xvreplgr2vr_w(0);
    for (long index = 0; index < size; index += 16) {
        __m256 ql, qh, vl, vh;
        load_fp16_batch_lasx(query + index, &ql, &qh);
        load_fp16_batch_lasx(first + index, &vl, &vh);
        sum0 = __lasx_xvfadd_s(sum0, __lasx_xvfmul_s(ql, vl));
        sum0 = __lasx_xvfadd_s(sum0, __lasx_xvfmul_s(qh, vh));
        load_fp16_batch_lasx(second + index, &vl, &vh);
        sum1 = __lasx_xvfadd_s(sum1, __lasx_xvfmul_s(ql, vl));
        sum1 = __lasx_xvfadd_s(sum1, __lasx_xvfmul_s(qh, vh));
        load_fp16_batch_lasx(third + index, &vl, &vh);
        sum2 = __lasx_xvfadd_s(sum2, __lasx_xvfmul_s(ql, vl));
        sum2 = __lasx_xvfadd_s(sum2, __lasx_xvfmul_s(qh, vh));
        load_fp16_batch_lasx(fourth + index, &vl, &vh);
        sum3 = __lasx_xvfadd_s(sum3, __lasx_xvfmul_s(ql, vl));
        sum3 = __lasx_xvfadd_s(sum3, __lasx_xvfmul_s(qh, vh));
    }
    output[0] = reduce_fp16_batch_lasx(sum0);
    output[1] = reduce_fp16_batch_lasx(sum1);
    output[2] = reduce_fp16_batch_lasx(sum2);
    output[3] = reduce_fp16_batch_lasx(sum3);
}

void fp16_products_lasx4(const unsigned short *query,
                         const unsigned short *first,
                         const unsigned short *second,
                         const unsigned short *third,
                         const unsigned short *fourth, long size,
                         float *output) {
    __m256 dot0 = (__m256)__lasx_xvreplgr2vr_w(0);
    __m256 dot1 = (__m256)__lasx_xvreplgr2vr_w(0);
    __m256 dot2 = (__m256)__lasx_xvreplgr2vr_w(0);
    __m256 dot3 = (__m256)__lasx_xvreplgr2vr_w(0);
    __m256 query_norm = (__m256)__lasx_xvreplgr2vr_w(0);
    __m256 norm0 = (__m256)__lasx_xvreplgr2vr_w(0);
    __m256 norm1 = (__m256)__lasx_xvreplgr2vr_w(0);
    __m256 norm2 = (__m256)__lasx_xvreplgr2vr_w(0);
    __m256 norm3 = (__m256)__lasx_xvreplgr2vr_w(0);
    for (long index = 0; index < size; index += 16) {
        __m256 ql, qh, vl, vh;
        load_fp16_batch_lasx(query + index, &ql, &qh);
        query_norm = __lasx_xvfadd_s(query_norm, __lasx_xvfmul_s(ql, ql));
        query_norm = __lasx_xvfadd_s(query_norm, __lasx_xvfmul_s(qh, qh));
        load_fp16_batch_lasx(first + index, &vl, &vh);
        dot0 = __lasx_xvfadd_s(dot0, __lasx_xvfmul_s(ql, vl));
        dot0 = __lasx_xvfadd_s(dot0, __lasx_xvfmul_s(qh, vh));
        norm0 = __lasx_xvfadd_s(norm0, __lasx_xvfmul_s(vl, vl));
        norm0 = __lasx_xvfadd_s(norm0, __lasx_xvfmul_s(vh, vh));
        load_fp16_batch_lasx(second + index, &vl, &vh);
        dot1 = __lasx_xvfadd_s(dot1, __lasx_xvfmul_s(ql, vl));
        dot1 = __lasx_xvfadd_s(dot1, __lasx_xvfmul_s(qh, vh));
        norm1 = __lasx_xvfadd_s(norm1, __lasx_xvfmul_s(vl, vl));
        norm1 = __lasx_xvfadd_s(norm1, __lasx_xvfmul_s(vh, vh));
        load_fp16_batch_lasx(third + index, &vl, &vh);
        dot2 = __lasx_xvfadd_s(dot2, __lasx_xvfmul_s(ql, vl));
        dot2 = __lasx_xvfadd_s(dot2, __lasx_xvfmul_s(qh, vh));
        norm2 = __lasx_xvfadd_s(norm2, __lasx_xvfmul_s(vl, vl));
        norm2 = __lasx_xvfadd_s(norm2, __lasx_xvfmul_s(vh, vh));
        load_fp16_batch_lasx(fourth + index, &vl, &vh);
        dot3 = __lasx_xvfadd_s(dot3, __lasx_xvfmul_s(ql, vl));
        dot3 = __lasx_xvfadd_s(dot3, __lasx_xvfmul_s(qh, vh));
        norm3 = __lasx_xvfadd_s(norm3, __lasx_xvfmul_s(vl, vl));
        norm3 = __lasx_xvfadd_s(norm3, __lasx_xvfmul_s(vh, vh));
    }
    output[0] = reduce_fp16_batch_lasx(dot0);
    output[1] = reduce_fp16_batch_lasx(dot1);
    output[2] = reduce_fp16_batch_lasx(dot2);
    output[3] = reduce_fp16_batch_lasx(dot3);
    output[4] = reduce_fp16_batch_lasx(query_norm);
    output[5] = reduce_fp16_batch_lasx(norm0);
    output[6] = reduce_fp16_batch_lasx(norm1);
    output[7] = reduce_fp16_batch_lasx(norm2);
    output[8] = reduce_fp16_batch_lasx(norm3);
}
