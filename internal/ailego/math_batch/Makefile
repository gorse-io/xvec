GO ?= go
GOAT ?= $(GO) tool goat
GOAT_TARGET_AMD64 ?= amd64
GOAT_OUTPUT_DIR ?= .goat/mathbatch
MV ?= mv

.PHONY: generate avx clean

generate: avx clean

avx:
	$(RM) -r $(GOAT_OUTPUT_DIR)
	mkdir -p $(GOAT_OUTPUT_DIR)
	$(GOAT) src/inner_product_batch_fp32_avx.c --target $(GOAT_TARGET_AMD64) -O3 -mavx -o $(GOAT_OUTPUT_DIR)
	$(MV) $(GOAT_OUTPUT_DIR)/*.go $(GOAT_OUTPUT_DIR)/*.s .

clean:
	$(RM) src/*.o src/*.s
	$(RM) -r .goat
