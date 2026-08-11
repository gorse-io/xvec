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

func ExampleStemmerTokenFilter() {
	filter, err := core.NewStemmerTokenFilter(core.StemmerTokenFilterOptions{Language: "english"})
	if err != nil {
		panic(err)
	}
	tokens, err := filter.Filter(context.Background(), []core.Token{
		{Text: "running", Offset: 0, Position: 0},
		{Text: "connections", Offset: 8, Position: 1},
	})
	if err != nil {
		panic(err)
	}
	for _, token := range tokens {
		fmt.Printf("%s@%d ", token.Text, token.Offset)
	}
	// Output: run@0 connect@8
}
