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

package server

import (
	"context"
	"sync"
	"sync/atomic"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/stats"
	"google.golang.org/grpc/status"
)

type connectionIDKey struct{}

func withConnectionID(ctx context.Context, id uint64) context.Context {
	return context.WithValue(ctx, connectionIDKey{}, id)
}

func connectionID(ctx context.Context) uint64 {
	id, _ := ctx.Value(connectionIDKey{}).(uint64)
	return id
}

type connectionStatsHandler struct {
	service *Service
	nextID  atomic.Uint64
}

func (h *connectionStatsHandler) TagConn(ctx context.Context, _ *stats.ConnTagInfo) context.Context {
	return withConnectionID(ctx, h.nextID.Add(1))
}

func (*connectionStatsHandler) HandleConn(context.Context, stats.ConnStats) {}
func (*connectionStatsHandler) TagRPC(ctx context.Context, _ *stats.RPCTagInfo) context.Context {
	return ctx
}
func (*connectionStatsHandler) HandleRPC(context.Context, stats.RPCStats) {}

type resourceStream struct {
	grpc.ServerStream
	ctx   context.Context
	mu    sync.Mutex
	idle  *time.Timer
	reset time.Duration
}

func (s *resourceStream) Context() context.Context { return s.ctx }

func (s *resourceStream) SendMsg(message any) error {
	err := s.ServerStream.SendMsg(message)
	if err == nil {
		s.touch()
	}
	return err
}

func (s *resourceStream) touch() {
	if s.reset <= 0 || s.idle == nil {
		return
	}
	s.mu.Lock()
	if !s.idle.Stop() {
		select {
		case <-s.idle.C:
		default:
		}
	}
	s.idle.Reset(s.reset)
	s.mu.Unlock()
}

func (s *resourceStream) stop() {
	if s.idle == nil {
		return
	}
	s.mu.Lock()
	s.idle.Stop()
	s.mu.Unlock()
}

func (s *Service) streamInterceptor() grpc.StreamServerInterceptor {
	return func(server any, stream grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
		id := connectionID(stream.Context())
		if !s.acquireStream(id) {
			return status.Error(codes.ResourceExhausted, "concurrent stream limit exceeded")
		}
		defer s.releaseStream(id)

		ctx := stream.Context()
		var cancel context.CancelFunc
		if s.options.MaxStreamLifetime > 0 {
			ctx, cancel = context.WithTimeout(ctx, s.options.MaxStreamLifetime)
		} else {
			ctx, cancel = context.WithCancel(ctx)
		}
		defer cancel()

		wrapped := &resourceStream{ServerStream: stream, ctx: ctx, reset: s.options.StreamIdleTimeout}
		if wrapped.reset > 0 {
			wrapped.idle = time.AfterFunc(wrapped.reset, cancel)
			defer wrapped.stop()
		}
		err := handler(server, wrapped)
		if err != nil {
			return err
		}
		return ctx.Err()
	}
}

func (s *Service) acquireStream(id uint64) bool {
	s.streamMu.Lock()
	defer s.streamMu.Unlock()
	if s.closed {
		return false
	}
	limit := s.options.MaxConcurrentStreamsPerClient
	if limit > 0 && s.streams[id] >= limit {
		return false
	}
	s.streams[id]++
	return true
}

func (s *Service) releaseStream(id uint64) {
	s.streamMu.Lock()
	if s.streams[id] <= 1 {
		delete(s.streams, id)
	} else {
		s.streams[id]--
	}
	s.streamMu.Unlock()
}
