//go:build !noasm && linux && arm64

package mathutil

import (
	"testing"
	"unsafe"

	"golang.org/x/sys/cpu"
	"golang.org/x/sys/unix"
)

func TestASIMDHPFP16GuardPage(t *testing.T) {
	if !cpu.ARM64.HasFPHP || !cpu.ARM64.HasASIMDHP {
		t.Skip("ASIMDHP/FPHP unavailable")
	}
	page := unix.Getpagesize()
	guarded := func() []uint16 {
		mapping, err := unix.Mmap(-1, 0, 2*page, unix.PROT_READ|unix.PROT_WRITE, unix.MAP_ANON|unix.MAP_PRIVATE)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := unix.Munmap(mapping); err != nil {
				t.Error(err)
			}
		})
		if err := unix.Mprotect(mapping[page:], unix.PROT_NONE); err != nil {
			t.Fatal(err)
		}
		return unsafe.Slice((*uint16)(unsafe.Pointer(&mapping[0])), page/2)
	}
	a, b := guarded(), guarded()
	for n := 0; n <= 33; n++ {
		left, right := a[len(a)-n:], b[len(b)-n:]
		for i := range left {
			left[i] = 0x3c01
			right[i] = 0x3555
		}
		wantL2, wantDot, wantLn, wantRn := asimdhpReference(left, right)
		gotDot, gotLn, gotRn := dotNormsFP16ASIMDHP(left, right)
		if squaredEuclideanFP16ASIMDHP(left, right) != wantL2 || innerProductFP16ASIMDHP(left, right) != wantDot || gotDot != wantDot || gotLn != wantLn || gotRn != wantRn {
			t.Fatalf("guarded dimension %d: wrong half-rounded results", n)
		}
	}
}
