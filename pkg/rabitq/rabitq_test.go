// Copyright 2026-present the xvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package rabitq

import (
	"encoding/binary"
	"encoding/hex"
	"math"
	"testing"
)

func close32(a, b float32) bool {
	return float32(math.Abs(float64(a-b))) <= 2e-4*max(1, float32(math.Abs(float64(b))))
}

func TestFasterConfigMatchesUpstream(t *testing.T) {
	cases := []struct {
		dim, bits int
		want      float64
	}{
		{64, 2, 7.7370669667450729},
		{64, 4, 28.033980929074072},
		{128, 8, 511.38024451470136},
	}
	for _, tc := range cases {
		if got := FasterConfig(tc.dim, tc.bits).TConst; math.Abs(got-tc.want) > 1e-6*math.Max(1, math.Abs(tc.want)) {
			t.Fatalf("FasterConfig(%d, %d) = %.17g, want %.17g", tc.dim, tc.bits, got, tc.want)
		}
	}
}

func TestDataMapsPreservePackedLayout(t *testing.T) {
	const dim, exBits = 64, 3
	batchData := make([]byte, BatchDataBytes(dim))
	batch := NewBatchDataMap(batchData, dim)
	if got, want := len(batch.BinCode()), dim*BatchSize/8; got != want {
		t.Fatalf("batch binary length = %d, want %d", got, want)
	}
	batch.SetFAdd(0, 1.25)
	batch.SetFRescale(0, -2.5)
	batch.SetFError(0, 3.75)
	metadata := dim * BatchSize / 8
	for i, want := range []float32{1.25, -2.5, 3.75} {
		offset := metadata + i*BatchSize*4
		if got := math.Float32frombits(binary.LittleEndian.Uint32(batchData[offset:])); got != want {
			t.Fatalf("batch metadata %d = %v, want %v", i, got, want)
		}
	}

	binData := make([]byte, BinDataBytes(dim))
	bin := NewBinDataMap(binData, dim)
	binary.LittleEndian.PutUint64(bin.BinCode(), 0x0123456789abcdef)
	bin.SetFAdd(4.5)
	bin.SetFRescale(-6.25)
	bin.SetFError(7.75)
	if got := binary.LittleEndian.Uint64(binData); got != 0x0123456789abcdef {
		t.Fatalf("binary code = %#x", got)
	}
	if bin.FAdd() != 4.5 || bin.FRescale() != -6.25 || bin.FError() != 7.75 {
		t.Fatalf("bin factors = %v, %v, %v", bin.FAdd(), bin.FRescale(), bin.FError())
	}

	exData := make([]byte, ExDataBytes(dim, exBits))
	ex := NewExDataMap(exData, dim, exBits)
	if got, want := len(ex.ExCode()), dim*exBits/8; got != want {
		t.Fatalf("extra code length = %d, want %d", got, want)
	}
	ex.SetFAddEx(8.5)
	ex.SetFRescaleEx(-9.5)
	if ex.FAddEx() != 8.5 || ex.FRescaleEx() != -9.5 {
		t.Fatalf("extra factors = %v, %v", ex.FAddEx(), ex.FRescaleEx())
	}
	if ExDataBytes(dim, 0) != 0 {
		t.Fatal("zero extra bits must occupy no bytes")
	}
}

func TestDataMapsSupportUnalignedStorage(t *testing.T) {
	const dim = 64
	batchStorage := make([]byte, BatchDataBytes(dim)+1)[1:]
	batch := NewBatchDataMap(batchStorage, dim)
	batch.SetFAdd(0, 1.25)
	batch.SetFRescale(1, -2.5)
	batch.SetFError(2, 3.75)
	if batch.FAdd(0) != 1.25 || batch.FRescale(1) != -2.5 || batch.FError(2) != 3.75 {
		t.Fatal("unaligned batch metadata did not round trip")
	}
	binStorage := make([]byte, BinDataBytes(dim)+1)[1:]
	bin := NewBinDataMap(binStorage, dim)
	binary.LittleEndian.PutUint64(bin.BinCode(), 0x0123456789abcdef)
	bin.SetFAdd(4.5)
	if binary.LittleEndian.Uint64(bin.BinCode()) != 0x0123456789abcdef || bin.FAdd() != 4.5 {
		t.Fatal("unaligned binary data did not round trip")
	}
}

func TestSpaceAndExcodeSelection(t *testing.T) {
	a := []float32{1, -2, 3, -4}
	b := []float32{4, 3, 2, 1}
	if got := DotProduct(a, b); got != 0 {
		t.Fatalf("dot product = %v", got)
	}
	if got := EuclideanSqr(a, b); got != 60 {
		t.Fatalf("euclidean square = %v", got)
	}

	query := make([]float32, 64)
	raw := make([]byte, 64)
	for i := range query {
		query[i] = float32(i%7 - 3)
		raw[i] = byte(i % 8)
	}
	for bits := 0; bits <= 8; bits++ {
		fn, err := SelectExcodeIPFunc(bits)
		if err != nil {
			t.Fatalf("bits %d: %v", bits, err)
		}
		if bits == 0 {
			if fn(query, nil) != 0 {
				t.Fatal("zero-bit inner product must be zero")
			}
			continue
		}
		code := make([]byte, len(query)*bits/8)
		packExcodeForTest(raw, code, bits)
		var want float32
		for i := range query {
			want += query[i] * float32(raw[i]&byte((1<<bits)-1))
		}
		if got := fn(query, code); got != want {
			t.Fatalf("bits %d inner product = %v, want %v", bits, got, want)
		}
	}
	if _, err := SelectExcodeIPFunc(9); err == nil {
		t.Fatal("unsupported bit width accepted")
	}
}

func packExcodeForTest(raw, code []byte, bits int) {
	switch bits {
	case 1:
		for i, v := range raw {
			code[i/8] |= (v & 1) << (i % 8)
		}
	case 2:
		copyRaw := append([]byte(nil), raw...)
		pack2ForTest(copyRaw, code)
	case 3:
		for block := 0; block < len(raw); block += 64 {
			pack2ForTest(raw[block:block+64], code[block*3/8:block*3/8+16])
			for i := 0; i < 64; i++ {
				code[block*3/8+16+i/8] |= (raw[block+i] >> 2 & 1) << (i % 8)
			}
		}
	case 4:
		for i := 0; i < len(raw); i += 16 {
			for j := 0; j < 8; j++ {
				code[i/2+j] = raw[i+j]&15 | raw[i+8+j]<<4
			}
		}
	case 5, 6, 7:
		// Production quantization packing is covered end-to-end below. Generate
		// these layouts through the same public quantization-compatible rules.
		for block := 0; block < len(raw); block += 64 {
			base := block * bits / 8
			if bits == 5 {
				for i := 0; i < 16; i++ {
					code[base+i] = raw[block+i]&15 | raw[block+16+i]<<4
					code[base+16+i] = raw[block+32+i]&15 | raw[block+48+i]<<4
				}
				for i := 0; i < 64; i++ {
					code[base+32+i/8] |= (raw[block+i] >> 4 & 1) << (i % 8)
				}
			} else {
				for i := 0; i < 16; i++ {
					last := raw[block+48+i]
					code[base+i] = raw[block+i]&63 | (last&3)<<6
					code[base+16+i] = raw[block+16+i]&63 | (last>>2&3)<<6
					code[base+32+i] = raw[block+32+i]&63 | (last>>4&3)<<6
				}
				if bits == 7 {
					for i := 0; i < 64; i++ {
						code[base+48+i/8] |= (raw[block+i] >> 6 & 1) << (i % 8)
					}
				}
			}
		}
	case 8:
		copy(code, raw)
	}
}

func pack2ForTest(raw, code []byte) {
	for i := 0; i < 16; i++ {
		code[i] = raw[i]&3 | (raw[16+i]&3)<<2 | (raw[32+i]&3)<<4 | (raw[48+i]&3)<<6
	}
}

func TestZeroExtraBitsFullDistanceUsesOneBitEstimate(t *testing.T) {
	const dim = 64
	data, centroid, query := make([]float32, dim), make([]float32, dim), make([]float32, dim)
	for i := range data {
		data[i] = float32(i%7 - 3)
		query[i] = float32(i%5 - 2)
	}
	binData := make([]byte, BinDataBytes(dim))
	if err := QuantizeSplitSingle(data, centroid, dim, 0, binData, nil, MetricL2, RaBitQConfig{}); err != nil {
		t.Fatal(err)
	}
	q, err := NewSplitSingleQuery(query, dim, 0, FasterConfig(dim, SplitSingleQueryNumBits), MetricL2)
	if err != nil {
		t.Fatal(err)
	}
	ip, estimate, low := SplitSingleEstDist(binData, q, dim, 4, 2)
	full, fullLow, fullIP := SplitSingleFullDist(binData, nil, mustIP(t, 0), q, dim, 0, 4, 2)
	if full != estimate || fullLow != low || fullIP != ip {
		t.Fatalf("full=(%v,%v,%v), one-bit=(%v,%v,%v)", full, fullLow, fullIP, estimate, low, ip)
	}
}

func TestQuantizeSplitSingleLayoutAndEstimators(t *testing.T) {
	const dim, exBits = 64, 2
	data, centroid, query := make([]float32, dim), make([]float32, dim), make([]float32, dim)
	for i := range data {
		data[i] = 1
		if i%2 != 0 {
			data[i] = -1
		}
		query[i] = float32((i%9)-4) / 3
	}
	binData := make([]byte, BinDataBytes(dim))
	exData := make([]byte, ExDataBytes(dim, exBits))
	if err := QuantizeSplitSingle(data, centroid, dim, exBits, binData, exData, MetricL2, RaBitQConfig{}); err != nil {
		t.Fatal(err)
	}
	if got, want := hex.EncodeToString(binData), "aaaaaaaaaaaaaaaa00008042000080c000000000"; got != want {
		t.Fatalf("binary data = %s, want %s", got, want)
	}
	if got, want := hex.EncodeToString(exData), "aa55aa55aa55aa55aa55aa55aa55aa5500008042cdcc4cbf"; got != want {
		t.Fatalf("extra data = %s, want %s", got, want)
	}
	for i, b := range binData[:dim/8] {
		if b != 0xaa {
			t.Fatalf("binary byte %d = %#x, want 0xaa", i, b)
		}
	}
	bin := NewBinDataMap(binData, dim)
	if !close32(bin.FAdd(), 64) || !close32(bin.FRescale(), -4) || !close32(bin.FError(), 0) {
		t.Fatalf("one-bit factors = %v, %v, %v", bin.FAdd(), bin.FRescale(), bin.FError())
	}

	q, err := NewSplitSingleQuery(query, dim, exBits, FasterConfig(dim, SplitSingleQueryNumBits), MetricL2)
	if err != nil {
		t.Fatal(err)
	}
	q.SetGAdd(2, 0)
	ip, est, low := SplitSingleEstDist(binData, q, dim, q.GAdd(), q.GError())
	if !close32(est, low) {
		t.Fatalf("zero quantization error: estimate %v, lower %v", est, low)
	}
	full, fullLow, fullIP := SplitSingleFullDist(binData, exData, mustIP(t, exBits), q, dim, exBits, q.GAdd(), q.GError())
	boost := SplitDistanceBoosting(exData, mustIP(t, exBits), q, dim, exBits, fullIP)
	if !close32(full, boost) || fullLow > full {
		t.Fatalf("full=%v low=%v ip=%v boost=%v warmup=%v", full, fullLow, fullIP, boost, ip)
	}
	for name, pair := range map[string][2]float32{
		"warmup": {ip, -0.47397095}, "estimate": {est, 67.2292175},
		"low": {low, 67.2292175}, "full": {full, 65.3333359},
		"full low": {fullLow, 65.3333359}, "full IP": {fullIP, -1.1920929e-07},
	} {
		if !close32(pair[0], pair[1]) {
			t.Fatalf("%s = %v, want %v", name, pair[0], pair[1])
		}
	}
}

func mustIP(t *testing.T, bits int) ExcodeIPFunc {
	t.Helper()
	fn, err := SelectExcodeIPFunc(bits)
	if err != nil {
		t.Fatal(err)
	}
	return fn
}

func requirePanic(t *testing.T, fn func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic")
		}
	}()
	fn()
}

func TestEstimatorsRejectMismatchedQueryConfiguration(t *testing.T) {
	const dim, exBits = 64, 2
	query := make([]float32, dim)
	batchQuery, err := NewSplitBatchQuery(query, dim, exBits, MetricL2, false)
	if err != nil {
		t.Fatal(err)
	}
	batchData := make([]byte, BatchDataBytes(dim))
	results := make([]float32, BatchSize)
	t.Run("batch accuracy", func(t *testing.T) {
		requirePanic(t, func() {
			SplitBatchEstDist(batchData, batchQuery, dim, results, results, results, true)
		})
	})

	singleQuery, err := NewSplitSingleQuery(query, dim, exBits, FasterConfig(dim, SplitSingleQueryNumBits), MetricL2)
	if err != nil {
		t.Fatal(err)
	}
	binData := make([]byte, BinDataBytes(dim))
	exData := make([]byte, ExDataBytes(dim, exBits))
	t.Run("single dimension", func(t *testing.T) {
		requirePanic(t, func() {
			SplitSingleEstDist(binData, singleQuery, dim*2, 0, 0)
		})
	})
	t.Run("full extra bits", func(t *testing.T) {
		requirePanic(t, func() {
			SplitSingleFullDist(binData, exData, mustIP(t, exBits), singleQuery, dim, exBits+1, 0, 0)
		})
	})
	t.Run("boost dimension", func(t *testing.T) {
		requirePanic(t, func() {
			SplitDistanceBoosting(exData, mustIP(t, exBits), singleQuery, dim*2, exBits, 0)
		})
	})
}

func TestBatchQuantizationReusesScratch(t *testing.T) {
	const dim, exBits, num = 64, 3, BatchSize
	data := make([]float32, dim*num)
	centroid := make([]float32, dim)
	for i := range data {
		data[i] = float32(i%17 - 8)
	}
	batchData := make([]byte, BatchDataBytes(dim))
	exData := make([]byte, num*ExDataBytes(dim, exBits))
	allocations := testing.AllocsPerRun(10, func() {
		if err := QuantizeSplitBatch(data, centroid, num, dim, exBits, batchData, exData, MetricL2, RaBitQConfig{TConst: 10}); err != nil {
			t.Fatal(err)
		}
	})
	if allocations > 10 {
		t.Fatalf("batch quantization allocated %.0f times, want at most 10", allocations)
	}
}

func TestQuantizeSplitBatchAndReset(t *testing.T) {
	const dim, exBits, num = 64, 2, 3
	data, centroid := make([]float32, num*dim), make([]float32, dim)
	for row := 0; row < num; row++ {
		for i := 0; i < dim; i++ {
			data[row*dim+i] = float32((row+i)%7 - 3)
		}
	}
	batchData := make([]byte, BatchDataBytes(dim))
	exData := make([]byte, num*ExDataBytes(dim, exBits))
	if err := QuantizeSplitBatch(data, centroid, num, dim, exBits, batchData, exData, MetricL2, RaBitQConfig{}); err != nil {
		t.Fatal(err)
	}

	queryA, queryB := make([]float32, dim), make([]float32, dim)
	for i := range queryA {
		queryA[i], queryB[i] = float32(i%5-2), float32(i%11-5)/2
	}
	q, err := NewSplitBatchQuery(queryA, dim, exBits, MetricL2, true)
	if err != nil {
		t.Fatal(err)
	}
	q.SetGAdd(3, 0)
	oldLUTCap := q.LUTCapacity()
	if err := q.Reset(queryB, dim, exBits, MetricIP, true); err != nil {
		t.Fatal(err)
	}
	if q.GAdd() != 0 || q.GError() != 0 {
		t.Fatalf("Reset retained g state: %v, %v", q.GAdd(), q.GError())
	}
	if q.LUTCapacity() != oldLUTCap {
		t.Fatalf("Reset reallocated same-sized LUT: cap %d -> %d", oldLUTCap, q.LUTCapacity())
	}
	queryB[0] = 99
	if q.RotatedQuery()[0] != -2.5 {
		t.Fatal("Reset retained caller-owned query storage")
	}
	rotated := q.RotatedQuery()
	rotated[0] = 88
	if q.RotatedQuery()[0] != -2.5 {
		t.Fatal("RotatedQuery exposed mutable query state")
	}
	q.SetGAdd(4, 6)
	if q.GAdd() != -6 || q.GError() != 4 {
		t.Fatalf("IP g factors = %v, %v", q.GAdd(), q.GError())
	}

	est, low, ips := make([]float32, BatchSize), make([]float32, BatchSize), make([]float32, BatchSize)
	SplitBatchEstDist(batchData, q, dim, est, low, ips, true)
	want := [][4]float32{
		{247.940704, 229.488159, -1.49944305, 252.934708},
		{215.086151, 196.854523, 1.50051117, 233.12204},
		{235.390381, 217.042068, -0.999557495, 235.889771},
	}
	for i := 0; i < num; i++ {
		boost := SplitDistanceBoosting(exData[i*ExDataBytes(dim, exBits):], mustIP(t, exBits), q, dim, exBits, ips[i])
		if math.IsNaN(float64(est[i])) || math.IsNaN(float64(boost)) || low[i] > est[i] {
			t.Fatalf("lane %d invalid distances est=%v low=%v boost=%v", i, est[i], low[i], boost)
		}
		for j, got := range [...]float32{est[i], low[i], ips[i], boost} {
			if !close32(got, want[i][j]) {
				t.Fatalf("lane %d value %d = %v, want %v", i, j, got, want[i][j])
			}
		}
	}
}
