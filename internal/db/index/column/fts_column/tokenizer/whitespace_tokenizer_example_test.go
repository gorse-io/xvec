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

package tokenizer_test

import (
	"context"
	"fmt"

	"github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer"
)

func ExampleWhitespaceTokenizer() {
	tokenizer := tokenizer.NewWhitespaceTokenizer()
	tokens, err := tokenizer.Tokenize(context.Background(), "  Go\t向量 search")
	if err != nil {
		panic(err)
	}
	for _, token := range tokens {
		fmt.Printf("%d:%d:%s\n", token.Position, token.Offset, token.Text)
	}
	// Output:
	// 0:2:Go
	// 1:5:向量
	// 2:12:search
}
