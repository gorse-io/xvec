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

package xvec

import (
	"context"
	"fmt"
)

// CallbackReranker adapts a function to the Reranker interface. The callback
// receives candidate batches in sub-query order and the requested final topK.
// Collection.MultiQuery validates the returned documents against its snapshot.
type CallbackReranker func(ctx context.Context, batches []RerankBatch, topK int) ([]Document, error)

// NewCallbackReranker adapts callback to a Reranker. A nil callback remains
// invalid when invoked directly; a nil Reranker on MultiQuery selects RRF.
func NewCallbackReranker(callback func(context.Context, []RerankBatch, int) ([]Document, error)) CallbackReranker {
	return CallbackReranker(callback)
}

// Validate rejects an empty callback.
func (r CallbackReranker) Validate() error {
	if r == nil {
		return invalidArgument("validate callback reranker", "callback is nil")
	}
	return nil
}

// Rerank invokes the callback and converts a panic into a structured internal
// error so caller code cannot unwind through the collection query boundary.
// Callback errors are returned unchanged.
func (r CallbackReranker) Rerank(ctx context.Context, batches []RerankBatch, topK int) (documents []Document, err error) {
	const op = "callback rerank"
	if ctx == nil {
		return nil, invalidArgument(op, "context is nil")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := r.Validate(); err != nil {
		return nil, err
	}
	defer func() {
		if recovered := recover(); recovered != nil {
			documents = nil
			cause, ok := recovered.(error)
			if !ok {
				cause = fmt.Errorf("%v", recovered)
			}
			err = &Error{Code: ErrorCodeInternal, Op: op, Message: "callback panicked", Err: cause}
		}
	}()
	documents, err = r(ctx, batches, topK)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return documents, nil
}

var _ Reranker = CallbackReranker(nil)
