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
