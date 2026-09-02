GO ?= go
GOAT ?= $(GO) tool goat
GOAT_TARGET_AMD64 ?= amd64
GOAT_TARGET_ARM64 ?= arm64
GOAT_TARGET_RVV ?= riscv64
GOAT_TARGET_LASX ?= loong64
GOAT_OUTPUT_DIR ?= .goat/mathbatch
MV ?= mv
CLANG_RESOURCE_DIR ?= $(shell clang -print-resource-dir)
GOAT_RVV_INCLUDE ?= $(CLANG_RESOURCE_DIR)/include
LASX_OBJDUMP ?= loongarch64-linux-gnu-objdump
ARM64_QEMU ?= qemu-aarch64
ARM64_QEMU_FLAGS ?= -cpu max
RISCV64_QEMU ?= qemu-riscv64
RISCV64_QEMU_FLAGS ?= -cpu max,v=true
LOONG64_QEMU ?= qemu-loongarch64
LOONG64_QEMU_FLAGS ?= -cpu max
TEST_NEON_BINARY ?= .test-neon.arm64.test
TEST_RVV_BINARY ?= .test-rvv.riscv64.test
TEST_LASX_BINARY ?= .test-lasx.loong64.test

.PHONY: generate avx neon rvv lasx clean test-neon test-rvv test-lasx

generate: avx neon rvv lasx clean

avx:
	$(RM) -r $(GOAT_OUTPUT_DIR)
	mkdir -p $(GOAT_OUTPUT_DIR)
	$(GOAT) src/inner_product_batch_fp32_avx.c --target $(GOAT_TARGET_AMD64) -O3 -mavx -o $(GOAT_OUTPUT_DIR)
	$(MV) $(GOAT_OUTPUT_DIR)/*.go $(GOAT_OUTPUT_DIR)/*.s .

neon:
	$(RM) -r $(GOAT_OUTPUT_DIR)
	mkdir -p $(GOAT_OUTPUT_DIR)
	$(GOAT) src/distance_batch_fp32_neon.c --target $(GOAT_TARGET_ARM64) -O3 -o $(GOAT_OUTPUT_DIR)
	$(MV) $(GOAT_OUTPUT_DIR)/*.go $(GOAT_OUTPUT_DIR)/*.s .

rvv:
	$(RM) -r $(GOAT_OUTPUT_DIR)
	mkdir -p $(GOAT_OUTPUT_DIR)
	$(GOAT) src/distance_batch_fp32_rvv.c --target $(GOAT_TARGET_RVV) -O3 -march=rv64imafdv -e=-isystem -e=$(GOAT_RVV_INCLUDE) -o $(GOAT_OUTPUT_DIR)
	$(MV) $(GOAT_OUTPUT_DIR)/*.go $(GOAT_OUTPUT_DIR)/*.s .

lasx:
	$(RM) -r $(GOAT_OUTPUT_DIR)
	mkdir -p $(GOAT_OUTPUT_DIR)
	OBJDUMP=$(LASX_OBJDUMP) $(GOAT) src/distance_batch_fp32_lasx.c --target $(GOAT_TARGET_LASX) -O3 -mlasx -o $(GOAT_OUTPUT_DIR)
	$(MV) $(GOAT_OUTPUT_DIR)/*.go $(GOAT_OUTPUT_DIR)/*.s .

clean:
	$(RM) src/*.o src/*.s
	$(RM) -r .goat

test-neon:
	@set -e; \
	trap '$(RM) ./$(TEST_NEON_BINARY)' EXIT; \
	GOOS=linux GOARCH=arm64 CGO_ENABLED=0 $(GO) test -c -o $(TEST_NEON_BINARY) .; \
	$(ARM64_QEMU) $(ARM64_QEMU_FLAGS) ./$(TEST_NEON_BINARY)

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
