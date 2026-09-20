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

package main

import (
	"context"
	"errors"
	"fmt"
	"net"
	"time"

	"github.com/gorse-io/xvec"
	"github.com/gorse-io/xvec/internal/server"
	"google.golang.org/grpc"
)

type serviceHandle struct {
	serverOptions []grpc.ServerOption
	register      func(*grpc.Server)
	close         func() error
}

type grpcServer interface {
	Serve(net.Listener) error
	GracefulStop()
	Stop()
}

var listenTCP = func(address string) (net.Listener, error) {
	return net.Listen("tcp", address)
}

var newService = func(dataDir string, options server.Options) (serviceHandle, error) {
	service, err := server.New(dataDir, options)
	if err != nil {
		return serviceHandle{}, err
	}
	return serviceHandle{
		serverOptions: service.GRPCServerOptions(),
		register:      func(grpcServer *grpc.Server) { service.Register(grpcServer) },
		close:         service.Close,
	}, nil
}

func runServer(ctx context.Context, cfg config) (returnErr error) {
	listener, err := listenTCP(cfg.Listen)
	if err != nil {
		return fmt.Errorf("listen on %s: %w", cfg.Listen, err)
	}

	service, err := newService(cfg.DataDir, server.Options{
		CollectionOptions: xvec.CollectionOptions{
			EnableMmap:    cfg.EnableMmap,
			MaxBufferSize: cfg.MaxBufferSize,
			WALSyncEvery:  cfg.WALSyncEvery,
		},
		StreamBatchSize:               cfg.StreamBatchSize,
		MaxStreamLifetime:             cfg.MaxStreamLifetime,
		StreamIdleTimeout:             cfg.StreamIdleTimeout,
		MaxConcurrentStreamsPerClient: cfg.MaxConcurrentStreamsPerClient,
	})
	if err != nil {
		_ = listener.Close()
		return fmt.Errorf("create service: %w", err)
	}
	defer func() {
		if err := service.close(); err != nil {
			returnErr = errors.Join(returnErr, fmt.Errorf("close service: %w", err))
		}
	}()

	grpcServer := grpc.NewServer(service.serverOptions...)
	service.register(grpcServer)
	return serveGRPC(ctx, cfg.ShutdownTimeout, grpcServer, listener)
}

func serveGRPC(ctx context.Context, shutdownTimeout time.Duration, grpcServer grpcServer, listener net.Listener) error {
	serveErr := make(chan error, 1)
	go func() { serveErr <- grpcServer.Serve(listener) }()

	select {
	case err := <-serveErr:
		if err != nil && !errors.Is(err, grpc.ErrServerStopped) {
			return fmt.Errorf("serve gRPC: %w", err)
		}
		return nil
	case <-ctx.Done():
	}

	gracefulDone := make(chan struct{})
	go func() {
		grpcServer.GracefulStop()
		close(gracefulDone)
	}()
	timer := time.NewTimer(shutdownTimeout)
	select {
	case <-gracefulDone:
		if !timer.Stop() {
			<-timer.C
		}
	case <-timer.C:
		grpcServer.Stop()
		<-gracefulDone
	}

	err := <-serveErr
	if err != nil && !errors.Is(err, grpc.ErrServerStopped) {
		return fmt.Errorf("serve gRPC: %w", err)
	}
	return nil
}
