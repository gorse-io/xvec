GO ?= go
GOAT ?= $(GO) tool goat
GOAT_OUTPUT_DIR ?= .goat/simd
MV ?= mv

.PHONY: generate avx2 avx512 neon clean

generate: avx2 avx512 neon clean

avx2:
	$(RM) -r $(GOAT_OUTPUT_DIR)
	mkdir -p $(GOAT_OUTPUT_DIR)
	$(GOAT) src/rotator_avx2.c --target amd64 -O3 -mavx2 -o $(GOAT_OUTPUT_DIR)
	$(MV) $(GOAT_OUTPUT_DIR)/rotator_avx2.go $(GOAT_OUTPUT_DIR)/rotator_avx2.s .

avx512:
	$(RM) -r $(GOAT_OUTPUT_DIR)
	mkdir -p $(GOAT_OUTPUT_DIR)
	$(GOAT) src/rotator_avx512.c --target amd64 -O3 -mavx512f -mavx512dq -o $(GOAT_OUTPUT_DIR)
	$(MV) $(GOAT_OUTPUT_DIR)/rotator_avx512.go $(GOAT_OUTPUT_DIR)/rotator_avx512.s .

neon:
	$(RM) -r $(GOAT_OUTPUT_DIR)
	mkdir -p $(GOAT_OUTPUT_DIR)
	$(GOAT) src/rotator_neon.c --target arm64 -O3 -o $(GOAT_OUTPUT_DIR)
	$(MV) $(GOAT_OUTPUT_DIR)/rotator_neon.go $(GOAT_OUTPUT_DIR)/rotator_neon.s .

clean:
	$(RM) src/*.o src/*.s
	$(RM) -r .goat
