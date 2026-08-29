//go:build ignore

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

// postprocess removes unreachable sentinel gotos emitted by ANTLR's Go target.
package main

import (
	"os"
	"strings"
)

const sentinel = "\tgoto errorExit // Trick to prevent compiler error if the label is not used\n"

func main() {
	path := os.Args[len(os.Args)-1]
	content, err := os.ReadFile(path)
	if err != nil {
		panic(err)
	}
	lines := strings.SplitAfter(strings.ReplaceAll(string(content), sentinel, ""), "\n")
	functionStart := 0
	for index, line := range lines {
		if strings.HasPrefix(line, "func ") {
			functionStart = index
		}
		if line != "errorExit:\n" {
			continue
		}
		if !strings.Contains(strings.Join(lines[functionStart:index], ""), "goto errorExit") {
			lines[index] = ""
		}
	}
	if err := os.WriteFile(path, []byte(strings.Join(lines, "")), 0o644); err != nil {
		panic(err)
	}
}
