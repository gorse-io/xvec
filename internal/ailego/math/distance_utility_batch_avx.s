//go:build !noasm && amd64

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

#include "textflag.h"

// xvec_avx_batch_inner_products2 computes two query/candidate inner products
// in one pass, loading every query cache line once for both candidates.
TEXT ·xvec_avx_batch_inner_products2(SB), NOSPLIT, $64-48
	MOVQ query+0(FP), AX
	MOVQ first+8(FP), BX
	MOVQ second+16(FP), CX
	MOVQ size+24(FP), DX
	MOVQ firstOutput+32(FP), SI
	MOVQ secondOutput+40(FP), DI

	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	MOVQ DX, R8
	SHRQ $3, R8
	JE reduce

vector_loop:
	VMOVUPS (AX), Y2
	VMULPS (BX), Y2, Y3
	VMULPS (CX), Y2, Y4
	VADDPS Y3, Y0, Y0
	VADDPS Y4, Y1, Y1
	ADDQ $32, AX
	ADDQ $32, BX
	ADDQ $32, CX
	DECQ R8
	JNE vector_loop

reduce:
	VMOVUPS Y0, 0(SP)
	VMOVUPS Y1, 32(SP)
	VXORPS X2, X2, X2
	VXORPS X3, X3, X3
	VADDSS 0(SP), X2, X2
	VADDSS 4(SP), X2, X2
	VADDSS 8(SP), X2, X2
	VADDSS 12(SP), X2, X2
	VADDSS 16(SP), X2, X2
	VADDSS 20(SP), X2, X2
	VADDSS 24(SP), X2, X2
	VADDSS 28(SP), X2, X2
	VADDSS 32(SP), X3, X3
	VADDSS 36(SP), X3, X3
	VADDSS 40(SP), X3, X3
	VADDSS 44(SP), X3, X3
	VADDSS 48(SP), X3, X3
	VADDSS 52(SP), X3, X3
	VADDSS 56(SP), X3, X3
	VADDSS 60(SP), X3, X3

	ANDQ $7, DX
	JE done

tail_loop:
	VMOVSS (AX), X4
	VMULSS (BX), X4, X5
	VMULSS (CX), X4, X6
	VADDSS X5, X2, X2
	VADDSS X6, X3, X3
	ADDQ $4, AX
	ADDQ $4, BX
	ADDQ $4, CX
	DECQ DX
	JNE tail_loop

done:
	VMOVSS X2, (SI)
	VMOVSS X3, (DI)
	VZEROUPPER
	RET

// xvec_avx_batch_inner_products4 computes four query/candidate inner products
// in one pass, loading every query cache line once for all four candidates.
TEXT ·xvec_avx_batch_inner_products4(SB), NOSPLIT, $128-80
	MOVQ query+0(FP), AX
	MOVQ first+8(FP), BX
	MOVQ second+16(FP), CX
	MOVQ third+24(FP), DX
	MOVQ fourth+32(FP), SI
	MOVQ size+40(FP), R8
	MOVQ firstOutput+48(FP), R9
	MOVQ secondOutput+56(FP), R10
	MOVQ thirdOutput+64(FP), R11
	MOVQ fourthOutput+72(FP), R12

	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	MOVQ R8, R13
	SHRQ $3, R13
	JE reduce4

vector_loop4:
	VMOVUPS (AX), Y4
	VMULPS (BX), Y4, Y5
	VMULPS (CX), Y4, Y6
	VMULPS (DX), Y4, Y7
	VMULPS (SI), Y4, Y8
	VADDPS Y5, Y0, Y0
	VADDPS Y6, Y1, Y1
	VADDPS Y7, Y2, Y2
	VADDPS Y8, Y3, Y3
	ADDQ $32, AX
	ADDQ $32, BX
	ADDQ $32, CX
	ADDQ $32, DX
	ADDQ $32, SI
	DECQ R13
	JNE vector_loop4

reduce4:
	VMOVUPS Y0, 0(SP)
	VMOVUPS Y1, 32(SP)
	VMOVUPS Y2, 64(SP)
	VMOVUPS Y3, 96(SP)
	VXORPS X4, X4, X4
	VXORPS X5, X5, X5
	VXORPS X6, X6, X6
	VXORPS X7, X7, X7
	VADDSS 0(SP), X4, X4
	VADDSS 4(SP), X4, X4
	VADDSS 8(SP), X4, X4
	VADDSS 12(SP), X4, X4
	VADDSS 16(SP), X4, X4
	VADDSS 20(SP), X4, X4
	VADDSS 24(SP), X4, X4
	VADDSS 28(SP), X4, X4
	VADDSS 32(SP), X5, X5
	VADDSS 36(SP), X5, X5
	VADDSS 40(SP), X5, X5
	VADDSS 44(SP), X5, X5
	VADDSS 48(SP), X5, X5
	VADDSS 52(SP), X5, X5
	VADDSS 56(SP), X5, X5
	VADDSS 60(SP), X5, X5
	VADDSS 64(SP), X6, X6
	VADDSS 68(SP), X6, X6
	VADDSS 72(SP), X6, X6
	VADDSS 76(SP), X6, X6
	VADDSS 80(SP), X6, X6
	VADDSS 84(SP), X6, X6
	VADDSS 88(SP), X6, X6
	VADDSS 92(SP), X6, X6
	VADDSS 96(SP), X7, X7
	VADDSS 100(SP), X7, X7
	VADDSS 104(SP), X7, X7
	VADDSS 108(SP), X7, X7
	VADDSS 112(SP), X7, X7
	VADDSS 116(SP), X7, X7
	VADDSS 120(SP), X7, X7
	VADDSS 124(SP), X7, X7

	ANDQ $7, R8
	JE done4

tail_loop4:
	VMOVSS (AX), X8
	VMULSS (BX), X8, X9
	VMULSS (CX), X8, X10
	VMULSS (DX), X8, X11
	VMULSS (SI), X8, X12
	VADDSS X9, X4, X4
	VADDSS X10, X5, X5
	VADDSS X11, X6, X6
	VADDSS X12, X7, X7
	ADDQ $4, AX
	ADDQ $4, BX
	ADDQ $4, CX
	ADDQ $4, DX
	ADDQ $4, SI
	DECQ R8
	JNE tail_loop4

done4:
	VMOVSS X4, (R9)
	VMOVSS X5, (R10)
	VMOVSS X6, (R11)
	VMOVSS X7, (R12)
	VZEROUPPER
	RET
