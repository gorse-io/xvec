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

package client

import (
	"context"
	"io"
	"sync"

	"github.com/gorse-io/xvec"
	"github.com/gorse-io/xvec/internal/grpcapi"
	xvecv1 "github.com/gorse-io/xvec/internal/proto/xvec/v1"
	"google.golang.org/grpc"
)

// DocumentIterator lazily consumes a remote document stream.
type DocumentIterator struct {
	nextMu  sync.Mutex
	mu      sync.Mutex
	stream  grpc.ServerStreamingClient[xvecv1.QueryResponse]
	cancel  context.CancelFunc
	pending []*xvecv1.Document
	closed  bool
}

// CreateIterator starts a remote snapshot iterator.
func (c *Collection) CreateIterator(ctx context.Context, options xvec.IteratorOptions) (*DocumentIterator, error) {
	streamContext, cancel := context.WithCancel(ctx)
	stream, err := c.service.Iterate(streamContext, &xvecv1.IterateRequest{Collection: c.name, Projection: grpcapi.ProjectionToProto(options.Projection)})
	if err != nil {
		cancel()
		return nil, grpcapi.ErrorFromStatus(err)
	}
	return &DocumentIterator{stream: stream, cancel: cancel}, nil
}

// Next returns the next document, or io.EOF after exhaustion or Close.
func (i *DocumentIterator) Next() (*xvec.Document, error) {
	if i == nil {
		return nil, io.EOF
	}
	i.nextMu.Lock()
	defer i.nextMu.Unlock()
	for {
		i.mu.Lock()
		if i.closed {
			i.mu.Unlock()
			return nil, io.EOF
		}
		if len(i.pending) > 0 {
			encoded := i.pending[0]
			i.pending = i.pending[1:]
			document, err := grpcapi.DocumentFromProto(encoded)
			if err != nil {
				i.closeLocked()
				i.mu.Unlock()
				return nil, responseError(err)
			}
			i.mu.Unlock()
			return &document, nil
		}
		i.mu.Unlock()

		message, err := i.stream.Recv()
		i.mu.Lock()
		if i.closed {
			i.mu.Unlock()
			return nil, io.EOF
		}
		if err == io.EOF {
			i.closeLocked()
			i.mu.Unlock()
			return nil, io.EOF
		}
		if err != nil {
			i.closeLocked()
			i.mu.Unlock()
			return nil, grpcapi.ErrorFromStatus(err)
		}
		if message == nil {
			i.closeLocked()
			i.mu.Unlock()
			return nil, internalError("iterator stream payload is nil", nil)
		}
		i.pending = append(i.pending, message.Documents...)
		i.mu.Unlock()
	}
}

// Close cancels the remote stream. It is idempotent.
func (i *DocumentIterator) Close() {
	if i == nil {
		return
	}
	i.mu.Lock()
	defer i.mu.Unlock()
	i.closeLocked()
}

func (i *DocumentIterator) closeLocked() {
	if i.closed {
		return
	}
	i.closed = true
	i.pending = nil
	if i.cancel != nil {
		i.cancel()
		i.cancel = nil
	}
}
