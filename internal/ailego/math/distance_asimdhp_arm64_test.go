//go:build !noasm && arm64

package mathutil

import (
	"fmt"
	"math"
	"testing"

	"github.com/gorse-io/xvec/internal/ailego/utility"
	"golang.org/x/sys/cpu"
)

// The oracle rounds each subtraction/product to binary16 before FP32 sums.
func asimdhpReference(left, right []uint16) (l2, dot, ln, rn float32) {
	half := func(v float32) float32 { return utility.Float16BitsToFloat32(utility.Float32ToFloat16Bits(v)) }
	for i, bits := range left {
		a, b := utility.Float16BitsToFloat32(bits), utility.Float16BitsToFloat32(right[i])
		d := half(a - b)
		l2 += half(d * d)
		dot += half(a * b)
		ln += half(a * a)
		rn += half(b * b)
	}
	return
}

func testASIMDHPFP16(t *testing.T, l2, dot binaryKernelFP16, products productsKernelFP16) {
	t.Helper()
	cases := []struct {
		name        string
		left, right []uint16
	}{
		{"rounding", []uint16{0x3c01, 0x3555, 0xbc03, 0x3e11}, []uint16{0x3555, 0xbc01, 0x3bff, 0x3001}},
		{"subnormal", []uint16{1, 3, 0x3ff, 0x8001}, []uint16{0x3800, 0x3800, 0x3c01, 0x3800}},
		{"overflow", []uint16{0x7bff}, []uint16{0xfbff}},
		{"product_overflow", []uint16{utility.Float32ToFloat16Bits(300)}, []uint16{utility.Float32ToFloat16Bits(300)}},
		{"wide_accumulation", []uint16{utility.Float32ToFloat16Bits(10)}, []uint16{utility.Float32ToFloat16Bits(10)}},
		{"zero", []uint16{0, 0x8000}, []uint16{0x8000, 0}},
	}
	sizes := []int{}
	for n := 0; n <= 33; n++ {
		sizes = append(sizes, n)
	}
	sizes = append(sizes, 63, 65, 128, 129, 768, 769)
	for _, tc := range cases {
		for _, n := range sizes {
			t.Run(fmt.Sprintf("%s/%d", tc.name, n), func(t *testing.T) {
				// Offset slices also exercise unaligned vector loads.
				a, b := make([]uint16, n+1), make([]uint16, n+1)
				a, b = a[1:], b[1:]
				for i := range a {
					a[i] = tc.left[i%len(tc.left)]
					b[i] = tc.right[i%len(tc.right)]
				}
				wantL2, wantDot, wantLn, wantRn := asimdhpReference(a, b)
				gotDot, gotLn, gotRn := products(a, b)
				for _, v := range []struct {
					name      string
					got, want float32
				}{
					{"l2", l2(a, b), wantL2}, {"dot", dot(a, b), wantDot}, {"products.dot", gotDot, wantDot}, {"products.left", gotLn, wantLn}, {"products.right", gotRn, wantRn},
				} {
					if v.got == v.want {
						continue
					}
					if math.IsNaN(float64(v.got)) && math.IsNaN(float64(v.want)) {
						continue
					}
					if math.IsNaN(float64(v.got)) || math.IsNaN(float64(v.want)) || math.IsInf(float64(v.got), 0) || math.IsInf(float64(v.want), 0) || math.Abs(float64(v.got-v.want)) > 1e-6*math.Abs(float64(v.want)) {
						t.Errorf("%s got %g, want half-rounded %g", v.name, v.got, v.want)
					}
				}
			})
		}
	}
}

func TestASIMDHPFP16Direct(t *testing.T) {
	if !cpu.ARM64.HasFPHP || !cpu.ARM64.HasASIMDHP {
		t.Skip("ASIMDHP/FPHP unavailable")
	}
	testASIMDHPFP16(t, squaredEuclideanFP16ASIMDHP, innerProductFP16ASIMDHP, dotNormsFP16ASIMDHP)
}

func TestFP16DispatchFeatures(t *testing.T) {
	// Verify all slots, including fallback CPUs without binary16 arithmetic.
	a, b := []uint16{0x3c01}, []uint16{0x3555}
	var l2, dot, ln, rn float32
	if cpu.ARM64.HasFPHP && cpu.ARM64.HasASIMDHP {
		l2, dot, ln, rn = asimdhpReference(a, b)
	} else {
		l2 = squaredEuclideanFP16Scalar(a, b)
		dot, ln, rn = dotNormsFP16Scalar(a, b)
	}
	gotDot, gotLn, gotRn := kernelsFP16.products(a, b)
	if kernelsFP16.l2(a, b) != l2 || kernelsFP16.dot(a, b) != dot || gotDot != dot || gotLn != ln || gotRn != rn {
		t.Fatal("FP16 dispatch does not match CPU feature semantics")
	}
}

func TestASIMDHPFP16Default(t *testing.T) {
	if !cpu.ARM64.HasFPHP || !cpu.ARM64.HasASIMDHP {
		t.Skip("ASIMDHP/FPHP unavailable")
	}
	testASIMDHPFP16(t, kernelsFP16.l2, kernelsFP16.dot, kernelsFP16.products)
}
