GO ?= go
GOAT ?= $(GO) tool goat
GOAT_TARGET_AMD64 ?= amd64
GOAT_TARGET_ARM64 ?= arm64
GOAT_TARGET_RVV ?= riscv64
GOAT_TARGET_LASX ?= loong64
GOAT_OUTPUT_DIR ?= .goat/mathutil
MV ?= mv
CLANG_RESOURCE_DIR ?= $(shell clang -print-resource-dir)
GOAT_RVV_INCLUDE ?= $(CLANG_RESOURCE_DIR)/include
LASX_OBJDUMP ?= loongarch64-linux-gnu-objdump
RISCV64_QEMU ?= qemu-riscv64
RISCV64_QEMU_FLAGS ?= -cpu max,v=true
LOONG64_QEMU ?= qemu-loongarch64
LOONG64_QEMU_FLAGS ?= -cpu max
TEST_RVV_BINARY ?= .test-rvv.riscv64.test
TEST_LASX_BINARY ?= .test-lasx.loong64.test

.PHONY: generate avx avx2 avx512 neon rvv lasx fht-avx2 fht-avx512 fht-neon distance-avx distance-avx512 distance-neon distance-rvv distance-lasx clean test-rvv test-lasx

generate: fht-avx2 fht-avx512 fht-neon distance-avx distance-avx512 distance-neon distance-rvv distance-lasx clean

avx: distance-avx

avx2: fht-avx2

avx512: fht-avx512 distance-avx512

neon: fht-neon distance-neon

rvv: distance-rvv

lasx: distance-lasx

fht-avx2:
	mkdir -p $(GOAT_OUTPUT_DIR)
	$(GOAT) src/fht_avx2.c --target $(GOAT_TARGET_AMD64) -O3 -mavx2 -o $(GOAT_OUTPUT_DIR)
	$(MV) $(GOAT_OUTPUT_DIR)/fht_avx2.go $(GOAT_OUTPUT_DIR)/fht_avx2.s .

fht-avx512:
	mkdir -p $(GOAT_OUTPUT_DIR)
	$(GOAT) src/fht_avx512.c --target $(GOAT_TARGET_AMD64) -O3 -mavx512f -mavx512dq -o $(GOAT_OUTPUT_DIR)
	$(MV) $(GOAT_OUTPUT_DIR)/fht_avx512.go $(GOAT_OUTPUT_DIR)/fht_avx512.s .

fht-neon:
	mkdir -p $(GOAT_OUTPUT_DIR)
	$(GOAT) src/fht_neon.c --target $(GOAT_TARGET_ARM64) -O3 -o $(GOAT_OUTPUT_DIR)
	$(MV) $(GOAT_OUTPUT_DIR)/fht_neon.go $(GOAT_OUTPUT_DIR)/fht_neon.s .

distance-avx:
	$(GOAT) src/distance_utility_avx.c --target $(GOAT_TARGET_AMD64) -O3 -mavx
	$(GOAT) src/distance_utility_batch_avx.c --target $(GOAT_TARGET_AMD64) -O3 -mavx

distance-avx512:
	$(GOAT) src/distance_utility_avx512.c --target $(GOAT_TARGET_AMD64) -O3 -mavx -mfma -mavx512f

distance-neon:
	$(RM) -r $(GOAT_OUTPUT_DIR)
	mkdir -p $(GOAT_OUTPUT_DIR)
	$(GOAT) src/euclidean_distance_matrix_fp32_neon.c --target $(GOAT_TARGET_ARM64) -O3 -o $(GOAT_OUTPUT_DIR)
	$(GOAT) src/inner_product_matrix_fp32_neon.c --target $(GOAT_TARGET_ARM64) -O3 -o $(GOAT_OUTPUT_DIR)
	$(GOAT) src/mips_euclidean_distance_matrix_fp32_neon.c --target $(GOAT_TARGET_ARM64) -O3 -o $(GOAT_OUTPUT_DIR)
	$(MV) $(GOAT_OUTPUT_DIR)/*.go $(GOAT_OUTPUT_DIR)/*.s .

distance-rvv:
	$(GOAT) src/distance_utility_rvv.c --target $(GOAT_TARGET_RVV) -O3 -march=rv64imafdv -e=-isystem -e=$(GOAT_RVV_INCLUDE)

distance-lasx:
	OBJDUMP=$(LASX_OBJDUMP) $(GOAT) src/distance_utility_lasx.c --target $(GOAT_TARGET_LASX) -O3 -mlasx

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
