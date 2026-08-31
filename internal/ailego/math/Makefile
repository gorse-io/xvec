GO ?= go
GOAT ?= $(GO) tool goat
GOAT_TARGET_AMD64 ?= amd64
GOAT_TARGET_ARM64 ?= arm64
GOAT_OUTPUT_DIR ?= .goat/mathutil
MV ?= mv

.PHONY: generate avx2 avx512 neon clean

generate: avx2 avx512 neon clean

avx2:
	mkdir -p $(GOAT_OUTPUT_DIR)
	$(GOAT) src/fht_avx2.c --target $(GOAT_TARGET_AMD64) -O3 -mavx2 -o $(GOAT_OUTPUT_DIR)
	$(MV) $(GOAT_OUTPUT_DIR)/fht_avx2.go $(GOAT_OUTPUT_DIR)/fht_avx2.s .

avx512:
	mkdir -p $(GOAT_OUTPUT_DIR)
	$(GOAT) src/fht_avx512.c --target $(GOAT_TARGET_AMD64) -O3 -mavx512f -mavx512dq -o $(GOAT_OUTPUT_DIR)
	$(MV) $(GOAT_OUTPUT_DIR)/fht_avx512.go $(GOAT_OUTPUT_DIR)/fht_avx512.s .

neon:
	mkdir -p $(GOAT_OUTPUT_DIR)
	$(GOAT) src/fht_neon.c --target $(GOAT_TARGET_ARM64) -O3 -o $(GOAT_OUTPUT_DIR)
	$(MV) $(GOAT_OUTPUT_DIR)/fht_neon.go $(GOAT_OUTPUT_DIR)/fht_neon.s .

clean:
	$(RM) src/*.o src/*.s
	$(RM) -r .goat
