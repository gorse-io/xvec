GO ?= go
GOAT ?= $(GO) tool goat
GOAT_OUTPUT_DIR ?= .goat/rabitq
CLANG ?= $(CURDIR)/src/clang_wrapper.py
MV ?= mv

.PHONY: generate avx2 avx512 clean

generate: avx2 avx512 clean

avx2:
	$(RM) -r $(GOAT_OUTPUT_DIR)
	mkdir -p $(GOAT_OUTPUT_DIR)
	python3 src/expand_kernels.py avx2 $(GOAT_OUTPUT_DIR)/kernels_avx2.c
	CLANG=$(CLANG) $(GOAT) $(GOAT_OUTPUT_DIR)/kernels_avx2.c --target amd64 -O3 -mavx2 -mfma -o $(GOAT_OUTPUT_DIR)
	$(MV) $(GOAT_OUTPUT_DIR)/*.go $(GOAT_OUTPUT_DIR)/*.s .

avx512:
	$(RM) -r $(GOAT_OUTPUT_DIR)
	mkdir -p $(GOAT_OUTPUT_DIR)
	python3 src/expand_kernels.py avx512 $(GOAT_OUTPUT_DIR)/kernels_avx512.c
	CLANG=$(CLANG) $(GOAT) $(GOAT_OUTPUT_DIR)/kernels_avx512.c --target amd64 -O3 -mavx2 -mfma -mavx512f -mavx512bw -mavx512vl -mavx512vpopcntdq -mprefer-vector-width=512 -o $(GOAT_OUTPUT_DIR)
	$(MV) $(GOAT_OUTPUT_DIR)/*.go $(GOAT_OUTPUT_DIR)/*.s .

clean:
	$(RM) src/*.o src/*.s
	$(RM) -r .goat
