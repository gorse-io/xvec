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

package core_test

import (
	"context"
	"fmt"

	"github.com/gorse-io/xvec/internal/core"
)

func ExampleJiebaTokenizer() {
	tokenizer, err := core.NewJiebaTokenizer(context.Background(), core.JiebaTokenizerOptions{
		DictDir: "testdata/jieba", CutMode: core.JiebaCutModeSearch,
	})
	if err != nil {
		panic(err)
	}
	tokens, err := tokenizer.Tokenize(context.Background(), "自然语言处理")
	if err != nil {
		panic(err)
	}
	for _, token := range tokens {
		fmt.Printf("%s@%d ", token.Text, token.Offset)
	}
	// Output: 自然@0 语言@6 自然语言@0 处理@12
}
