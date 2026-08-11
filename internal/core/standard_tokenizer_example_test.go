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

package core_test

import (
	"context"
	"fmt"

	"github.com/gorse-io/xvec/internal/core"
)

func ExampleStandardTokenizer() {
	tokenizer, err := core.NewStandardTokenizer(core.DefaultStandardTokenizerOptions())
	if err != nil {
		panic(err)
	}
	tokens, err := tokenizer.Tokenize(context.Background(), "Go向量 search 3.14")
	if err != nil {
		panic(err)
	}
	for _, token := range tokens {
		fmt.Printf("%s@%d ", token.Text, token.Offset)
	}
	// Output: Go@0 向@2 量@5 search@9 3.14@16
}
