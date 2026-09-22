//go:build amd64 && !noasm

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

// One native call for the bounded neighbor batch; no vector values are read.
TEXT ·prefetchDenseVectors(SB), NOSPLIT, $0-48
	MOVQ vectors+0(FP), R8
	MOVQ stride+8(FP), R9
	MOVQ neighbors_base+16(FP), SI
	MOVQ neighbors_len+24(FP), CX
	MOVQ lines+40(FP), DX
	TESTQ CX, CX
	JE done
	TESTQ DX, DX
	JE done
neighbor:
	MOVQ (SI), AX
	IMULQ R9, AX
	LEAQ (R8)(AX*1), DI
	MOVQ DX, BX
line:
	PREFETCHT0 (DI)
	ADDQ $64, DI
	DECQ BX
	JNZ line
	ADDQ $8, SI
	DECQ CX
	JNZ neighbor
done:
	RET
