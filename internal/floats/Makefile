GO ?= go
GOAT ?= $(GO) tool goat
GOAT_TARGET_AVX ?= amd64
GOAT_TARGET_NEON ?= arm64
GOAT_TARGET_RVV ?= riscv64
GOAT_TARGET_LASX ?= loong64
CLANG_RESOURCE_DIR ?= $(shell clang -print-resource-dir)
GOAT_RVV_INCLUDE ?= $(CLANG_RESOURCE_DIR)/include
LASX_OBJDUMP ?= loongarch64-linux-gnu-objdump
RISCV64_QEMU ?= qemu-riscv64
RISCV64_QEMU_FLAGS ?= -cpu max,v=true
LOONG64_QEMU ?= qemu-loongarch64
LOONG64_QEMU_FLAGS ?= -cpu max
TEST_RVV_BINARY ?= .test-rvv.riscv64.test
TEST_LASX_BINARY ?= .test-lasx.loong64.test

.PHONY: generate avx avx512 neon rvv lasx clean test-rvv test-lasx

generate: avx avx512 neon rvv lasx clean

avx:
	$(GOAT) src/floats_avx.c --target $(GOAT_TARGET_AVX) -O3 -mavx

avx512:
	$(GOAT) src/floats_avx512.c --target $(GOAT_TARGET_AVX) -O3 -mavx -mfma -mavx512f

neon:
	$(GOAT) src/floats_neon.c --target $(GOAT_TARGET_NEON) -O3

rvv:
	$(GOAT) src/floats_rvv.c --target $(GOAT_TARGET_RVV) -O3 -march=rv64imafdv -e=-isystem -e=$(GOAT_RVV_INCLUDE)

lasx:
	OBJDUMP=$(LASX_OBJDUMP) $(GOAT) src/floats_lasx.c --target $(GOAT_TARGET_LASX) -O3 -mlasx

clean:
	$(RM) src/*.o src/*.s

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
