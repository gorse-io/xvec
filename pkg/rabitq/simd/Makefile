GO ?= go
GOAT ?= $(GO) tool goat
GOAT_OUTPUT_DIR ?= .goat/simd
MV ?= mv
CLANG_RESOURCE_DIR ?= $(shell clang -print-resource-dir)
GOAT_RVV_INCLUDE ?= $(CLANG_RESOURCE_DIR)/include
GOAT_LASX_FLAGS ?= -e=-ffreestanding -e=-isystem -e=$(CLANG_RESOURCE_DIR)/include -e=-fno-vectorize -e=-fno-slp-vectorize
RVV_OBJDUMP ?= riscv64-linux-gnu-objdump
LASX_OBJDUMP ?= loongarch64-linux-gnu-objdump
RISCV64_QEMU ?= qemu-riscv64
RISCV64_QEMU_FLAGS ?= -cpu max,v=true
LOONG64_QEMU ?= qemu-loongarch64
LOONG64_QEMU_FLAGS ?= -cpu max
TEST_RVV_BINARY ?= .test-rvv.riscv64.test
TEST_LASX_BINARY ?= .test-lasx.loong64.test

.PHONY: generate avx2 avx512 neon rvv lasx clean test-rvv test-lasx

generate: avx2 avx512 neon rvv lasx clean

avx2:
	$(RM) -r $(GOAT_OUTPUT_DIR)
	mkdir -p $(GOAT_OUTPUT_DIR)
	$(GOAT) src/fastscan_avx2.c --target amd64 -O3 -mavx2 -o $(GOAT_OUTPUT_DIR)
	$(GOAT) src/pack_excode_avx2.c --target amd64 -O3 -mavx2 -o $(GOAT_OUTPUT_DIR)
	$(GOAT) src/rotator_avx2.c --target amd64 -O3 -mavx2 -o $(GOAT_OUTPUT_DIR)
	$(GOAT) src/space_avx2.c --target amd64 -O3 -mavx2 -o $(GOAT_OUTPUT_DIR)
	$(GOAT) src/space_excode_avx2.c --target amd64 -O3 -mavx2 -o $(GOAT_OUTPUT_DIR)
	$(GOAT) src/warmup_avx2.c --target amd64 -O3 -mavx2 -o $(GOAT_OUTPUT_DIR)
	$(MV) $(GOAT_OUTPUT_DIR)/fastscan_avx2.go $(GOAT_OUTPUT_DIR)/fastscan_avx2.s .
	$(MV) $(GOAT_OUTPUT_DIR)/pack_excode_avx2.go $(GOAT_OUTPUT_DIR)/pack_excode_avx2.s .
	$(MV) $(GOAT_OUTPUT_DIR)/rotator_avx2.go $(GOAT_OUTPUT_DIR)/rotator_avx2.s .
	$(MV) $(GOAT_OUTPUT_DIR)/space_avx2.go $(GOAT_OUTPUT_DIR)/space_avx2.s .
	$(MV) $(GOAT_OUTPUT_DIR)/space_excode_avx2.go $(GOAT_OUTPUT_DIR)/space_excode_avx2.s .
	$(MV) $(GOAT_OUTPUT_DIR)/warmup_avx2.go $(GOAT_OUTPUT_DIR)/warmup_avx2.s .

avx512:
	$(RM) -r $(GOAT_OUTPUT_DIR)
	mkdir -p $(GOAT_OUTPUT_DIR)
	$(GOAT) src/fastscan_avx512.c --target amd64 -O3 -mavx512f -mavx512dq -mavx512bw -o $(GOAT_OUTPUT_DIR)
	$(GOAT) src/pack_excode_avx512.c --target amd64 -O3 -mavx512f -mavx512dq -o $(GOAT_OUTPUT_DIR)
	$(GOAT) src/rotator_avx512.c --target amd64 -O3 -mavx512f -mavx512dq -o $(GOAT_OUTPUT_DIR)
	$(GOAT) src/space_avx512.c --target amd64 -O3 -mavx512f -mavx512dq -mavx512bw -o $(GOAT_OUTPUT_DIR)
	$(GOAT) src/space_excode_avx512.c --target amd64 -O3 -mavx512f -mavx512dq -mavx512bw -o $(GOAT_OUTPUT_DIR)
	$(GOAT) src/warmup_avx512.c --target amd64 -O3 -mavx512f -mavx512vpopcntdq -o $(GOAT_OUTPUT_DIR)
	$(MV) $(GOAT_OUTPUT_DIR)/fastscan_avx512.go $(GOAT_OUTPUT_DIR)/fastscan_avx512.s .
	$(MV) $(GOAT_OUTPUT_DIR)/pack_excode_avx512.go $(GOAT_OUTPUT_DIR)/pack_excode_avx512.s .
	$(MV) $(GOAT_OUTPUT_DIR)/rotator_avx512.go $(GOAT_OUTPUT_DIR)/rotator_avx512.s .
	$(MV) $(GOAT_OUTPUT_DIR)/space_avx512.go $(GOAT_OUTPUT_DIR)/space_avx512.s .
	$(MV) $(GOAT_OUTPUT_DIR)/space_excode_avx512.go $(GOAT_OUTPUT_DIR)/space_excode_avx512.s .
	$(MV) $(GOAT_OUTPUT_DIR)/warmup_avx512.go $(GOAT_OUTPUT_DIR)/warmup_avx512.s .

neon:
	$(RM) -r $(GOAT_OUTPUT_DIR)
	mkdir -p $(GOAT_OUTPUT_DIR)
	$(GOAT) src/fastscan_neon.c --target arm64 -O3 -o $(GOAT_OUTPUT_DIR)
	$(GOAT) src/pack_excode_neon.c --target arm64 -O3 -o $(GOAT_OUTPUT_DIR)
	$(GOAT) src/rotator_neon.c --target arm64 -O3 -o $(GOAT_OUTPUT_DIR)
	$(GOAT) src/space_neon.c --target arm64 -O3 -o $(GOAT_OUTPUT_DIR)
	$(GOAT) src/space_excode_neon.c --target arm64 -O3 -o $(GOAT_OUTPUT_DIR)
	$(GOAT) src/warmup_neon.c --target arm64 -O3 -o $(GOAT_OUTPUT_DIR)
	$(MV) $(GOAT_OUTPUT_DIR)/fastscan_neon.go $(GOAT_OUTPUT_DIR)/fastscan_neon.s .
	$(MV) $(GOAT_OUTPUT_DIR)/pack_excode_neon.go $(GOAT_OUTPUT_DIR)/pack_excode_neon.s .
	$(MV) $(GOAT_OUTPUT_DIR)/rotator_neon.go $(GOAT_OUTPUT_DIR)/rotator_neon.s .
	$(MV) $(GOAT_OUTPUT_DIR)/space_neon.go $(GOAT_OUTPUT_DIR)/space_neon.s .
	$(MV) $(GOAT_OUTPUT_DIR)/space_excode_neon.go $(GOAT_OUTPUT_DIR)/space_excode_neon.s .
	$(MV) $(GOAT_OUTPUT_DIR)/warmup_neon.go $(GOAT_OUTPUT_DIR)/warmup_neon.s .

rvv:
	$(RM) -r $(GOAT_OUTPUT_DIR)
	mkdir -p $(GOAT_OUTPUT_DIR)
	OBJDUMP=$(RVV_OBJDUMP) $(GOAT) src/fastscan_rvv.c --target riscv64 -O3 -march=rv64imafdv -e=-isystem -e=$(GOAT_RVV_INCLUDE) -o $(GOAT_OUTPUT_DIR)
	OBJDUMP=$(RVV_OBJDUMP) $(GOAT) src/pack_excode_rvv.c --target riscv64 -O3 -march=rv64imafdv -e=-isystem -e=$(GOAT_RVV_INCLUDE) -o $(GOAT_OUTPUT_DIR)
	OBJDUMP=$(RVV_OBJDUMP) $(GOAT) src/rotator_rvv.c --target riscv64 -O3 -march=rv64imafdv -e=-isystem -e=$(GOAT_RVV_INCLUDE) -o $(GOAT_OUTPUT_DIR)
	OBJDUMP=$(RVV_OBJDUMP) $(GOAT) src/space_rvv.c --target riscv64 -O3 -march=rv64imafdv -e=-isystem -e=$(GOAT_RVV_INCLUDE) -o $(GOAT_OUTPUT_DIR)
	OBJDUMP=$(RVV_OBJDUMP) $(GOAT) src/space_excode_rvv.c --target riscv64 -O3 -march=rv64imafdv -e=-isystem -e=$(GOAT_RVV_INCLUDE) -o $(GOAT_OUTPUT_DIR)
	OBJDUMP=$(RVV_OBJDUMP) $(GOAT) src/warmup_rvv.c --target riscv64 -O3 -march=rv64imafdv -e=-isystem -e=$(GOAT_RVV_INCLUDE) -o $(GOAT_OUTPUT_DIR)
	$(MV) $(GOAT_OUTPUT_DIR)/*.go $(GOAT_OUTPUT_DIR)/*.s .

lasx:
	$(RM) -r $(GOAT_OUTPUT_DIR)
	mkdir -p $(GOAT_OUTPUT_DIR)
	OBJDUMP=$(LASX_OBJDUMP) $(GOAT) src/fastscan_lasx.c --target loong64 -O3 -mlasx $(GOAT_LASX_FLAGS) -o $(GOAT_OUTPUT_DIR)
	OBJDUMP=$(LASX_OBJDUMP) $(GOAT) src/pack_excode_lasx.c --target loong64 -O3 -mlasx $(GOAT_LASX_FLAGS) -o $(GOAT_OUTPUT_DIR)
	OBJDUMP=$(LASX_OBJDUMP) $(GOAT) src/rotator_lasx.c --target loong64 -O3 -mlasx $(GOAT_LASX_FLAGS) -o $(GOAT_OUTPUT_DIR)
	OBJDUMP=$(LASX_OBJDUMP) $(GOAT) src/space_lasx.c --target loong64 -O3 -mlasx $(GOAT_LASX_FLAGS) -o $(GOAT_OUTPUT_DIR)
	OBJDUMP=$(LASX_OBJDUMP) $(GOAT) src/space_excode_lasx.c --target loong64 -O3 -mlasx $(GOAT_LASX_FLAGS) -o $(GOAT_OUTPUT_DIR)
	OBJDUMP=$(LASX_OBJDUMP) $(GOAT) src/warmup_lasx.c --target loong64 -O3 -mlasx $(GOAT_LASX_FLAGS) -o $(GOAT_OUTPUT_DIR)
	$(MV) $(GOAT_OUTPUT_DIR)/*.go $(GOAT_OUTPUT_DIR)/*.s .

clean:
	$(RM) src/*.o src/*.s
	$(RM) -r .goat

test-rvv:
	@set -e; \
	trap '$(RM) ./$(TEST_RVV_BINARY)' EXIT; \
	GOOS=linux GOARCH=riscv64 CGO_ENABLED=0 $(GO) test -c -o $(TEST_RVV_BINARY) .; \
	$(RISCV64_QEMU) $(RISCV64_QEMU_FLAGS) ./$(TEST_RVV_BINARY)

test-lasx:
	@set -e; \
	trap '$(RM) ./$(TEST_LASX_BINARY)' EXIT; \
	GOOS=linux GOARCH=loong64 CGO_ENABLED=0 $(GO) test -c -o $(TEST_LASX_BINARY) .; \
	$(LOONG64_QEMU) $(LOONG64_QEMU_FLAGS) ./$(TEST_LASX_BINARY)
