GO ?= go
GOAT ?= $(GO) tool goat

.PHONY: generate avx avx512 neon clean

generate: avx avx512 neon clean

avx:
	$(GOAT) src/floats_avx.c --target amd64 -O3 -mavx

avx512:
	$(GOAT) src/floats_avx512.c --target amd64 -O3 -mavx -mfma -mavx512f

neon:
	$(GOAT) src/floats_neon.c --target arm64 -O3

clean:
	$(RM) src/*.o src/*.s
