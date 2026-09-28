//go:build !amd64 || noasm

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

package core

import "runtime"

// Other targets retain the bounded portable cache warming fallback.
func prefetchQuantizedCode(code []byte, lines int) {
	if len(code) == 0 || lines <= 0 {
		return
	}
	lines = min(lines, (len(code)-1)/64+1)
	var touched byte
	for line := 0; line < lines; line++ {
		touched ^= code[line*64]
	}
	runtime.KeepAlive(touched)
}
