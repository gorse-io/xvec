GO ?= go
GOAT ?= $(GO) tool goat
GOAT_OUTPUT_DIR ?= .goat/simd
MV ?= mv

.PHONY: generate avx512 clean

generate: avx512 clean

avx512:
	$(RM) -r $(GOAT_OUTPUT_DIR)
	mkdir -p $(GOAT_OUTPUT_DIR)
	$(GOAT) src/rotator_avx512.c --target amd64 -O3 -mavx512f -mavx512dq -o $(GOAT_OUTPUT_DIR)
	$(MV) $(GOAT_OUTPUT_DIR)/rotator_avx512.go $(GOAT_OUTPUT_DIR)/rotator_avx512.s .

clean:
	$(RM) src/*.o src/*.s
	$(RM) -r .goat
