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

func TestCRC32C(t *testing.T) {
	if got, want := CRC32C([]byte("123456789")), uint32(0xe3069283); got != want {
		t.Fatalf("CRC32C = %#x, want %#x", got, want)
	}
	crc := UpdateCRC32C(0, []byte("1234"))
	crc = UpdateCRC32C(crc, []byte("56789"))
	if got, want := crc, CRC32C([]byte("123456789")); got != want {
		t.Fatalf("incremental CRC32C = %#x, want %#x", got, want)
	}
}
