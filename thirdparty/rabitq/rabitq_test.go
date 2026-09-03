// Copyright 2026-present the xvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// This package is an independent scalar Go implementation of the RaBitQ
// algorithms described by the Apache-2.0 RaBitQ-Library project.

package rabitq

import (
	"bytes"
	"math"
	"testing"
)

func close32(a, b float32) bool {
	return float32(math.Abs(float64(a-b))) <= 2e-4*max(1, float32(math.Abs(float64(b))))
}

func fixture(dim int) (data, centroid, query []float32) {
	data = make([]float32, dim)
	centroid = make([]float32, dim)
	query = make([]float32, dim)
	for i := range dim {
		data[i] = float32(math.Sin(float64(i)*0.17) + 0.25*math.Cos(float64(i)*0.07))
		centroid[i] = float32(0.1 * math.Cos(float64(i)*0.11))
		query[i] = float32(math.Cos(float64(i)*0.13) - 0.2*math.Sin(float64(i)*0.03))
	}
	return
}

func TestDistancePrimitives(t *testing.T) {
	a := []float32{1, -2, 3}
	b := []float32{4, 5, -6}
	if got, err := DotProduct(a, b); err != nil || got != -24 {
		t.Fatalf("DotProduct = %v, %v", got, err)
	}
	if got, err := EuclideanSquared(a, b); err != nil || got != 139 {
		t.Fatalf("EuclideanSquared = %v, %v", got, err)
	}
	if _, err := DotProduct(a, b[:2]); err == nil {
		t.Fatal("DotProduct accepted mismatched dimensions")
	}
	if _, err := EuclideanSquared([]float32{float32(math.NaN())}, []float32{0}); err == nil {
		t.Fatal("EuclideanSquared accepted NaN")
	}
}

func TestDataLayouts(t *testing.T) {
	const dim = 64
	if got := BinDataBytes(dim); got != 20 {
		t.Fatalf("BinDataBytes = %d", got)
	}
	if got := BatchDataBytes(dim); got != 640 {
		t.Fatalf("BatchDataBytes = %d", got)
	}
	for bits := 0; bits <= 8; bits++ {
		want := 0
		if bits > 0 {
			want = dim*bits/8 + 8
		}
		if got := ExDataBytes(dim, bits); got != want {
			t.Fatalf("ExDataBytes(%d) = %d, want %d", bits, got, want)
		}
	}
	if _, err := NewBinData(make([]byte, 19), dim); err == nil {
		t.Fatal("NewBinData accepted short data")
	}
	if got := BinDataBytes(8192); got != 1036 {
		t.Fatalf("BinDataBytes(8192) = %d, want 1036", got)
	}
}

func TestSplitSingleQuantizationAllBits(t *testing.T) {
	data, centroid, _ := fixture(64)
	for metric := MetricL2; metric <= MetricIP; metric++ {
		for exBits := 0; exBits <= 8; exBits++ {
			config, err := NewFasterConfig(64, exBits+1)
			if err != nil {
				t.Fatal(err)
			}
			bin, ex, err := QuantizeSplitSingle(data, centroid, exBits, metric, config)
			if err != nil {
				t.Fatalf("metric=%d bits=%d: %v", metric, exBits, err)
			}
			if len(bin) != BinDataBytes(64) || len(ex) != ExDataBytes(64, exBits) {
				t.Fatalf("bad layouts: %d %d", len(bin), len(ex))
			}
			bm, _ := NewBinData(bin, 64)
			if len(bm.Code()) != 8 || !isFinite(bm.FAdd()) || !isFinite(bm.FRescale()) || bm.FError() < 0 {
				t.Fatalf("invalid bin factors: %+v", bm)
			}
			if exBits > 0 {
				em, _ := NewExData(ex, 64, exBits)
				values, err := em.Values()
				if err != nil || len(values) != 64 {
					t.Fatalf("Values: %v, %v", values, err)
				}
				for _, value := range values {
					if int(value) >= 1<<exBits {
						t.Fatalf("value %d exceeds %d bits", value, exBits)
					}
				}
			}
		}
	}
}

func TestFasterConfig(t *testing.T) {
	for _, tt := range []struct {
		totalBits int
		want      float64
	}{
		{totalBits: 2, want: 10.164274220048007},
		{totalBits: 9, want: 788.09039464199338},
	} {
		got, err := NewFasterConfig(64, tt.totalBits)
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(got.TConst-tt.want) > 1e-12 {
			t.Fatalf("totalBits=%d TConst=%.17g, want %.17g", tt.totalBits, got.TConst, tt.want)
		}
	}
}

func TestPinnedLibraryQuantizationFixture(t *testing.T) {
	centroid := make([]float32, 64)
	vector := make([]float32, 64)
	for index := range vector {
		vector[index] = float32(int(index*17%29)-14) * .125
		centroid[index] = float32(int(index*5%11)-5) * .05
	}
	want := []uint8{
		61, 64, 61, 65, 62, 67, 63, 61, 65, 62, 66, 64, 60, 64, 62, 65,
		63, 66, 64, 61, 66, 62, 67, 0, 60, 65, 61, 66, 63, 61, 64, 62,
		65, 63, 67, 63, 61, 65, 62, 66, 64, 60, 65, 62, 66, 63, 66, 64,
		61, 65, 62, 67, 63, 61, 65, 62, 66, 63, 60, 64, 62, 65, 63, 66,
	}
	tests := []struct {
		metric                                Metric
		coarseAdd, coarseRescale, coarseError float32
		fullAdd, fullRescale                  float32
	}{
		{MetricL2, 70.4307632, -4.88890362, 2.37279153, 69.6409988, -1.05726922},
		{MetricIP, 1.12069368, -2.44445181, 1.18639576, .725809216, -.528634608},
	}
	for _, tt := range tests {
		bin, ex, err := QuantizeSplitSingle(vector, centroid, 6, tt.metric, FasterConfig{TConst: 15.75})
		if err != nil {
			t.Fatal(err)
		}
		bm, _ := NewBinData(bin, 64)
		em, _ := NewExData(ex, 64, 6)
		extra, _ := em.Values()
		values := make([]uint8, 64)
		for i := range values {
			values[i] = extra[i]
			if binaryBit(bm.Code(), i) {
				values[i] |= 64
			}
		}
		if !bytes.Equal(values, want) {
			t.Fatalf("metric=%d values=%v", tt.metric, values)
		}
		for name, pair := range map[string][2]float32{
			"coarse add": {bm.FAdd(), tt.coarseAdd}, "coarse rescale": {bm.FRescale(), tt.coarseRescale},
			"coarse error": {bm.FError(), tt.coarseError}, "full add": {em.FAdd(), tt.fullAdd},
			"full rescale": {em.FRescale(), tt.fullRescale},
		} {
			if !close32(pair[0], pair[1]) {
				t.Fatalf("%s = %.9g, want %.9g", name, pair[0], pair[1])
			}
		}
	}
}

func TestExcodeIPFunctions(t *testing.T) {
	query := make([]float32, 64)
	values := make([]uint8, 64)
	var want float32
	for i := range query {
		query[i] = float32(i-17) / 9
		values[i] = uint8(i % 251)
	}
	for bits := 0; bits <= 8; bits++ {
		maxCode := uint8(0)
		if bits > 0 {
			maxCode = uint8((1 << bits) - 1)
		}
		for i := range values {
			values[i] &= maxCode
		}
		packed, err := PackExCode(values, bits)
		if err != nil {
			t.Fatal(err)
		}
		want = 0
		for i := range query {
			want += query[i] * float32(values[i])
		}
		fn, err := SelectExcodeIPFunc(bits)
		if err != nil {
			t.Fatal(err)
		}
		got, err := fn(query, packed)
		if err != nil || !close32(got, want) {
			t.Fatalf("bits=%d: got %g, want %g, err=%v", bits, got, want, err)
		}
	}
	if _, err := SelectExcodeIPFunc(9); err == nil {
		t.Fatal("accepted nine extra bits")
	}
}

func TestSplitSingleDistances(t *testing.T) {
	data, centroid, query := fixture(64)
	for _, metric := range []Metric{MetricL2, MetricIP} {
		cfg, _ := NewFasterConfig(64, 7)
		bin, ex, err := QuantizeSplitSingle(data, centroid, 6, metric, cfg)
		if err != nil {
			t.Fatal(err)
		}
		qcfg, _ := NewFasterConfig(64, SplitSingleQueryBits)
		q, err := NewSplitSingleQuery(query, 6, qcfg, metric)
		if err != nil {
			t.Fatal(err)
		}
		norm := float32(math.Sqrt(float64(mustL2(t, query, centroid))))
		ip := mustDot(t, query, centroid)
		q.SetGAdd(norm, ip)
		coarse, err := SplitSingleEstimate(bin, q)
		if err != nil {
			t.Fatal(err)
		}
		full, err := SplitSingleFullDistance(bin, ex, q, 6)
		if err != nil {
			t.Fatal(err)
		}
		boosted, err := splitDistanceBoosting(ex, &q.queryCore, 6, full.IPX0QR, q.GAdd())
		if err != nil {
			t.Fatal(err)
		}
		if !close32(full.Distance, boosted) || full.LowerBound > full.Distance || coarse.LowerBound > coarse.Distance {
			t.Fatalf("metric=%d coarse=%+v full=%+v boost=%g", metric, coarse, full, boosted)
		}
	}
}

func TestSplitBatchHighAccuracyLUT(t *testing.T) {
	_, centroid, query := fixture(64)
	vectors := make([][]float32, 7)
	for i := range vectors {
		vectors[i], _, _ = fixture(64)
		for j := range vectors[i] {
			vectors[i][j] += float32(i) * 0.03
		}
	}
	cfg, _ := NewFasterConfig(64, 5)
	batch, extras, err := QuantizeSplitBatch(vectors, centroid, 4, MetricL2, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(batch) != BatchDataBytes(64) || len(extras) != len(vectors) {
		t.Fatalf("batch sizes = %d, %d", len(batch), len(extras))
	}
	q, err := NewSplitBatchQuery(query, 4, MetricL2)
	if err != nil {
		t.Fatal(err)
	}
	norm := float32(math.Sqrt(float64(mustL2(t, query, centroid))))
	q.SetGAdd(norm, mustDot(t, query, centroid))
	got, err := SplitBatchEstimate(batch, q)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != BatchSize {
		t.Fatalf("got %d lanes", len(got))
	}
	m, err := NewBatchData(batch, len(query))
	if err != nil {
		t.Fatal(err)
	}
	for lane := range vectors {
		var accumulated uint32
		for table := 0; table < len(query)/4; table++ {
			var code int
			for bit := 0; bit < 4; bit++ {
				if batchBinaryBit(m.Code(), len(query), lane, table*4+bit) {
					code |= 1 << (3 - bit)
				}
			}
			accumulated += uint32(q.lut[table*16+code])
		}
		wantIP := q.delta*float32(accumulated) + q.sumVL
		wantDistance := m.FAdd(lane) + q.GAdd() + m.FRescale(lane)*(wantIP+q.k1)
		if !close32(got[lane].IPX0QR, wantIP) || !close32(got[lane].Distance, wantDistance) {
			t.Fatalf("lane %d: got %+v, want ip=%g distance=%g", lane, got[lane], wantIP, wantDistance)
		}
		if boosted, err := SplitDistanceBoosting(extras[lane], q, 4, got[lane].IPX0QR); err != nil || !isFinite(boosted) {
			t.Fatalf("lane %d boost = %g, %v", lane, boosted, err)
		}
	}
}

func TestBatchCodeMatchesFastScanLayout(t *testing.T) {
	centroid := make([]float32, 64)
	vectors := make([][]float32, 25)
	for lane := range vectors {
		vectors[lane] = make([]float32, 64)
		for i := range vectors[lane] {
			vectors[lane][i] = -1
		}
	}
	vectors[0][0] = 1
	vectors[8][1] = 1
	vectors[16][2] = 1
	vectors[24][3] = 1

	batch, _, err := QuantizeSplitBatch(vectors, centroid, 1, MetricL2, FasterConfig{TConst: -1})
	if err != nil {
		t.Fatal(err)
	}
	m, _ := NewBatchData(batch, 64)
	if m.Code()[0] != 0x28 || m.Code()[1] != 0x14 {
		t.Fatalf("first FastScan block bytes = %#x %#x, want 0x28 0x14", m.Code()[0], m.Code()[1])
	}
	for i := 2; i < 32; i++ {
		if m.Code()[i] != 0 {
			t.Fatalf("unexpected first-block byte %d = %#x", i, m.Code()[i])
		}
	}
}

func TestSplitSingleUsesQuantizedWarmupAndExactFullIP(t *testing.T) {
	data, centroid, query := fixture(64)
	cfg := FasterConfig{TConst: 15.75}
	bin, ex, err := QuantizeSplitSingle(data, centroid, 6, MetricL2, cfg)
	if err != nil {
		t.Fatal(err)
	}
	q, err := NewSplitSingleQuery(query, 6, cfg, MetricL2)
	if err != nil {
		t.Fatal(err)
	}

	coarse, err := SplitSingleEstimate(bin, q)
	if err != nil {
		t.Fatal(err)
	}
	bm, _ := NewBinData(bin, len(query))
	var codeSum, count uint32
	var exactIP float32
	for i, code := range q.quantized {
		if binaryBit(bm.Code(), i) {
			codeSum += uint32(code)
			count++
			exactIP += query[i]
		}
	}
	wantWarmup := q.delta*float32(codeSum) + q.vl*float32(count)
	if !close32(coarse.IPX0QR, wantWarmup) {
		t.Fatalf("coarse ip = %g, want quantized warmup %g", coarse.IPX0QR, wantWarmup)
	}

	full, err := SplitSingleFullDistance(bin, ex, q, 6)
	if err != nil {
		t.Fatal(err)
	}
	if !close32(full.IPX0QR, exactIP) {
		t.Fatalf("full ip = %g, want exact mask ip %g", full.IPX0QR, exactIP)
	}
}

func TestQueryResetAndSetGAdd(t *testing.T) {
	_, _, q0 := fixture(64)
	q1 := append([]float32(nil), q0...)
	q1[0] += 3
	q, err := NewSplitBatchQuery(q0, 3, MetricIP)
	if err != nil {
		t.Fatal(err)
	}
	q.SetGAdd(2, 7)
	if q.GAdd() != -7 || q.GError() != 2 {
		t.Fatalf("IP g factors = %g, %g", q.GAdd(), q.GError())
	}
	if err := q.Reset(q1); err != nil {
		t.Fatal(err)
	}
	if q.GAdd() != 0 || q.GError() != 0 || bytes.Equal(float32Bytes(q0), float32Bytes(q.RotatedQuery())) {
		t.Fatal("Reset did not replace query and clear centroid state")
	}
}

func TestQueryEdgeCasesAndImmutableFactors(t *testing.T) {
	query := make([]float32, 64)
	query[0] = 1
	query[1] = -1
	query[2] = 0
	query[3] = math.Float32frombits(1 << 31)
	cfg := FasterConfig{TConst: 15.75}
	q, err := NewSplitSingleQuery(query, 3, cfg, MetricL2)
	if err != nil {
		t.Fatal(err)
	}
	if q.quantized[2] != 0 || q.quantized[3] != 0 {
		t.Fatalf("zero query codes = %d, %d; want 0, 0", q.quantized[2], q.quantized[3])
	}

	data := make([]float32, 64)
	for i := range data {
		data[i] = float32(i%7) - 3
	}
	bin, ex, err := QuantizeSplitSingle(data, make([]float32, 64), 3, MetricL2, cfg)
	if err != nil {
		t.Fatal(err)
	}
	first, err := SplitSingleEstimateWithFactors(bin, q, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	second, err := SplitSingleEstimateWithFactors(bin, q, 7, 11)
	if err != nil {
		t.Fatal(err)
	}
	if !close32(second.Distance-first.Distance, 5) || q.GAdd() != 0 || q.GError() != 0 {
		t.Fatalf("immutable factors changed query or distance: first=%+v second=%+v", first, second)
	}
	if _, err := SplitSingleFullDistanceWithFactors(bin, ex, q, 3, 2, 3); err != nil {
		t.Fatal(err)
	}

	var nilQuery *SplitBatchQuery
	if _, err := SplitDistanceBoosting(ex, nilQuery, 3, 0); err == nil {
		t.Fatal("accepted typed nil query")
	}
	bad := FasterConfig{TConst: math.Inf(1)}
	if _, err := NewSplitSingleQuery(query, 3, bad, MetricL2); err == nil {
		t.Fatal("accepted infinite TConst")
	}
	if _, _, err := QuantizeSplitSingle(data, make([]float32, 64), 3, MetricL2, FasterConfig{TConst: math.NaN()}); err == nil {
		t.Fatal("accepted NaN TConst")
	}
	if _, _, err := QuantizeSplitSingle(data, make([]float32, 64), 3, MetricL2, FasterConfig{TConst: math.MaxFloat64}); err != nil {
		t.Fatalf("large finite TConst was not safely clamped: %v", err)
	}
	binOnly, _, err := QuantizeSplitSingle(data, make([]float32, 64), 0, MetricL2, FasterConfig{TConst: -1})
	if err != nil {
		t.Fatal(err)
	}
	qOneBit, err := NewSplitSingleQuery(query, 0, FasterConfig{TConst: -1}, MetricL2)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := SplitSingleFullDistance(binOnly, nil, qOneBit, 0); err == nil {
		t.Fatal("one-bit-only configuration unexpectedly accepted full-distance refinement")
	}
}

func TestSplitSingleQueryPinnedDelta(t *testing.T) {
	_, _, query := fixture(64)
	q, err := NewSplitSingleQuery(query, 3, FasterConfig{TConst: 15.75}, MetricL2)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := math.Float32bits(q.Delta()), uint32(0x3eaf1d9f); got != want {
		t.Fatalf("delta bits = %#08x, want %#08x", got, want)
	}
	if got, want := math.Float32bits(q.VL()), uint32(0xc0242bc5); got != want {
		t.Fatalf("vl bits = %#08x, want %#08x", got, want)
	}
}

func TestQuantizeSplitBatchFlat(t *testing.T) {
	_, centroid, _ := fixture(64)
	vectors := make([][]float32, 3)
	flat := make([]float32, 0, 3*64)
	for lane := range vectors {
		vectors[lane], _, _ = fixture(64)
		vectors[lane][lane] += float32(lane + 1)
		flat = append(flat, vectors[lane]...)
	}
	cfg := FasterConfig{TConst: 15.75}
	wantBatch, wantRecords, err := QuantizeSplitBatch(vectors, centroid, 5, MetricIP, cfg)
	if err != nil {
		t.Fatal(err)
	}
	gotBatch, gotExtra, err := QuantizeSplitBatchFlat(flat, len(vectors), centroid, 5, MetricIP, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(gotBatch, wantBatch) {
		t.Fatal("flat batch differs from slice batch")
	}
	var wantExtra []byte
	for _, record := range wantRecords {
		wantExtra = append(wantExtra, record...)
	}
	if !bytes.Equal(gotExtra, wantExtra) {
		t.Fatal("flat extra records differ")
	}
}

func mustDot(t *testing.T, a, b []float32) float32 {
	t.Helper()
	v, err := DotProduct(a, b)
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func mustL2(t *testing.T, a, b []float32) float32 {
	t.Helper()
	v, err := EuclideanSquared(a, b)
	if err != nil {
		t.Fatal(err)
	}
	return v
}
