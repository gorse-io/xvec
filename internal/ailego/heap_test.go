// Copyright 2026-present the zvec-go project
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

package ailego

import "testing"

func TestHeap(t *testing.T) {
	heap := NewHeap(func(a, b int) bool { return a < b })
	for _, value := range []int{5, 1, 9, 2, 7} {
		heap.Push(value)
	}
	if root, ok := heap.Peek(); !ok || root != 1 {
		t.Fatalf("Peek = (%d, %v)", root, ok)
	}
	if replaced, ok := heap.Replace(3); !ok || replaced != 1 {
		t.Fatalf("Replace = (%d, %v)", replaced, ok)
	}
	for index, want := range []int{2, 3, 5, 7, 9} {
		got, ok := heap.Pop()
		if !ok || got != want {
			t.Fatalf("Pop %d = (%d, %v), want %d", index, got, ok, want)
		}
	}
	if _, ok := heap.Pop(); ok {
		t.Fatal("empty heap returned a value")
	}
}
