//go:build !noasm && arm64

package mathbatch

import (
	"fmt"
	"math"
	"testing"

	"github.com/gorse-io/xvec/internal/ailego/utility"
	"golang.org/x/sys/cpu"
)

func init() {
	if cpu.ARM64.HasFPHP && cpu.ARM64.HasASIMDHP {
		nativeFP16BatchTest = TestFP16ASIMDHPDefaultBatch
	}
}

func halfRoundBatch(v float32) float32 {
	return utility.Float16BitsToFloat32(utility.Float32ToFloat16Bits(v))
}

// Round each arithmetic operation to binary16, but accumulate in binary32.
func halfBatchOracle(q, v []uint16) (l2, dot, qnorm, vnorm float32) {
	for i := range q {
		a, b := utility.Float16BitsToFloat32(q[i]), utility.Float16BitsToFloat32(v[i])
		d := halfRoundBatch(a - b)
		l2 += halfRoundBatch(d * d)
		dot += halfRoundBatch(a * b)
		qnorm += halfRoundBatch(a * a)
		vnorm += halfRoundBatch(b * b)
	}
	return
}

func checkHalfBatchResult(t *testing.T, want, got float32) {
	t.Helper()
	if math.IsNaN(float64(want)) {
		if !math.IsNaN(float64(got)) {
			t.Fatalf("want NaN, got %g", got)
		}
		return
	}
	if math.IsInf(float64(want), 0) {
		if want != got {
			t.Fatalf("want %g, got %g", want, got)
		}
		return
	}
	if math.IsNaN(float64(got)) || math.Abs(float64(want-got)) > 2e-5*math.Max(1, math.Abs(float64(want))) {
		t.Fatalf("want half-rounding %g, got %g", want, got)
	}
}

func testASIMDHPBatch(t *testing.T, l2, dot, cosine, mips fp16Batch4Kernel) {
	t.Helper()
	if !cpu.ARM64.HasFPHP || !cpu.ARM64.HasASIMDHP {
		t.Skip("native FP16 arithmetic unavailable")
	}
	sizes := make([]int, 34)
	for i := range sizes {
		sizes[i] = i
	}
	sizes = append(sizes, 128, 768)
	for _, n := range sizes {
		for _, mode := range []string{"rounding", "zero-all", "zero-query", "zero-candidate", "wide-sum", "subnormals", "overflow", "nonfinite"} {
			t.Run(fmt.Sprintf("%d/%s", n, mode), func(t *testing.T) {
				var v [5][]uint16
				for j := range v {
					// Offset slices exercise unaligned input loads.
					v[j] = make([]uint16, n+1)[1:]
					for i := range v[j] {
						bits := utility.Float32ToFloat16Bits(float32(math.Sin(float64(i*3+j*17))) * 8)
						switch mode {
						case "zero-all":
							bits = uint16(j%2) << 15
						case "zero-query":
							if j == 0 {
								bits = 0
							}
						case "zero-candidate":
							if j == 1 {
								bits = 0
							}
						case "wide-sum":
							bits = utility.Float32ToFloat16Bits(100)
						case "subnormals":
							bits = []uint16{1, 0x8001, 0x3ff, 0x400, 0x3801}[(i+j)%5]
						case "overflow":
							bits = []uint16{0x7bff, 0xfbff, 0x5c00, 0xdc00, 0x3c00}[(i+j)%5]
						case "nonfinite":
							bits = []uint16{0x7c00, 0xfc00, 0x7e01, 0x3c00, 0x8000}[(i+j)%5]
						}
						v[j][i] = bits
					}
				}
				products := fp16ProductsASIMDHP4(v[0], v[1], v[2], v[3], v[4])
				var want [4][4]float32
				for j := range 4 {
					a, b, c, d := halfBatchOracle(v[0], v[j+1])
					checkHalfBatchResult(t, b, products[j])
					checkHalfBatchResult(t, c, products[4])
					checkHalfBatchResult(t, d, products[5+j])
					want[0][j], want[1][j] = a, b
					want[2][j] = cosineDistanceFromProduct(b, float32(math.Sqrt(float64(c))), float32(math.Sqrt(float64(d))))
					if denominator := max(c, d); denominator != 0 {
						want[3][j] = 2 - 2*b/denominator
					}
				}
				for k, kernel := range []fp16Batch4Kernel{l2, dot, cosine, mips} {
					out := [5]float32{99, 99, 99, 99, 42}
					kernel(v[0], v[1], v[2], v[3], v[4], out[:4])
					for j := range 4 {
						t.Run(fmt.Sprintf("kernel=%d/candidate=%d", k, j), func(t *testing.T) { checkHalfBatchResult(t, want[k][j], out[j]) })
					}
					if out[4] != 42 {
						t.Fatal("output overrun")
					}
				}
			})
		}
	}
}

func TestFP16ASIMDHPBatchAllocs(t *testing.T) {
	if !cpu.ARM64.HasFPHP || !cpu.ARM64.HasASIMDHP {
		t.Skip("native FP16 arithmetic unavailable")
	}
	if FP16CosineCacheCompatible() {
		t.Fatal("batch and single-pair FP16 reduction orders differ")
	}
	for _, n := range []int{0, 1, 7, 8, 9, 128, 768} {
		v := fp16BatchFixture(n)
		var out [4]float32
		for _, kernel := range []fp16Batch4Kernel{fp16L2ASIMDHP4, fp16DotASIMDHP4, fp16CosineASIMDHP4, fp16MIPSASIMDHP4} {
			if allocs := testing.AllocsPerRun(10, func() { kernel(v[0], v[1], v[2], v[3], v[4], out[:]) }); allocs != 0 {
				t.Fatalf("dimension=%d allocations=%g", n, allocs)
			}
		}
	}
}

func TestFP16ASIMDHPDirectBatch(t *testing.T) {
	testASIMDHPBatch(t, fp16L2ASIMDHP4, fp16DotASIMDHP4, fp16CosineASIMDHP4, fp16MIPSASIMDHP4)
}

func TestFP16ASIMDHPDefaultBatch(t *testing.T) {
	testASIMDHPBatch(t, SquaredEuclideanDistances4FP16, InnerProducts4FP16, CosineDistances4FP16, MIPSL2SquaredDistances4FP16)
}
