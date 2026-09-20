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
	"net"
	"sync"
	"testing"
	"time"

	"github.com/gorse-io/xvec/internal/server"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
)

type blockingListener struct {
	closed chan struct{}
	once   sync.Once
}

func newBlockingListener() *blockingListener {
	return &blockingListener{closed: make(chan struct{})}
}

func (l *blockingListener) Accept() (net.Conn, error) {
	<-l.closed
	return nil, net.ErrClosed
}
func (l *blockingListener) Close() error {
	l.once.Do(func() { close(l.closed) })
	return nil
}
func (*blockingListener) Addr() net.Addr { return testAddr("test") }

type testAddr string

func (a testAddr) Network() string { return string(a) }
func (a testAddr) String() string  { return string(a) }

func TestRunServerPassesListenerAndServiceOptions(t *testing.T) {
	originalListen, originalNewService := listenTCP, newService
	t.Cleanup(func() { listenTCP, newService = originalListen, originalNewService })

	listener := newBlockingListener()
	var listenedOn, dataDir string
	var gotOptions server.Options
	closed := false
	listenTCP = func(address string) (net.Listener, error) {
		listenedOn = address
		return listener, nil
	}
	newService = func(dir string, options server.Options) (serviceHandle, error) {
		dataDir, gotOptions = dir, options
		return serviceHandle{
			register: func(*grpc.Server) {},
			close: func() error {
				closed = true
				return nil
			},
		}, nil
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	cfg := config{
		Listen:                        "localhost:1234",
		DataDir:                       "store",
		EnableMmap:                    false,
		MaxBufferSize:                 2048,
		WALSyncEvery:                  4,
		StreamBatchSize:               12,
		MaxStreamLifetime:             13 * time.Second,
		StreamIdleTimeout:             14 * time.Second,
		MaxConcurrentStreamsPerClient: 15,
		ShutdownTimeout:               time.Second,
	}

	require.NoError(t, runServer(ctx, cfg))
	require.Equal(t, cfg.Listen, listenedOn)
	require.Equal(t, cfg.DataDir, dataDir)
	require.Equal(t, cfg.EnableMmap, gotOptions.CollectionOptions.EnableMmap)
	require.Equal(t, cfg.MaxBufferSize, gotOptions.CollectionOptions.MaxBufferSize)
	require.Equal(t, cfg.WALSyncEvery, gotOptions.CollectionOptions.WALSyncEvery)
	require.Equal(t, cfg.StreamBatchSize, gotOptions.StreamBatchSize)
	require.Equal(t, cfg.MaxStreamLifetime, gotOptions.MaxStreamLifetime)
	require.Equal(t, cfg.StreamIdleTimeout, gotOptions.StreamIdleTimeout)
	require.Equal(t, cfg.MaxConcurrentStreamsPerClient, gotOptions.MaxConcurrentStreamsPerClient)
	require.True(t, closed)
}

func TestRunServerSurfacesListenError(t *testing.T) {
	originalListen := listenTCP
	t.Cleanup(func() { listenTCP = originalListen })
	listenTCP = func(string) (net.Listener, error) { return nil, errors.New("address unavailable") }

	err := runServer(context.Background(), config{Listen: "bad"})
	require.EqualError(t, err, "listen on bad: address unavailable")
}

func TestRunServerClosesListenerWhenServiceSetupFails(t *testing.T) {
	originalListen, originalNewService := listenTCP, newService
	t.Cleanup(func() { listenTCP, newService = originalListen, originalNewService })
	listener := newBlockingListener()
	listenTCP = func(string) (net.Listener, error) { return listener, nil }
	newService = func(string, server.Options) (serviceHandle, error) {
		return serviceHandle{}, errors.New("broken data directory")
	}

	err := runServer(context.Background(), config{Listen: "addr", DataDir: "data"})
	require.EqualError(t, err, "create service: broken data directory")
	select {
	case <-listener.closed:
	default:
		t.Fatal("listener was not closed")
	}
}

type stuckGRPCServer struct {
	gracefulStarted chan struct{}
	stopCalled      chan struct{}
}

func (s *stuckGRPCServer) Serve(net.Listener) error {
	<-s.stopCalled
	return grpc.ErrServerStopped
}
func (s *stuckGRPCServer) GracefulStop() {
	close(s.gracefulStarted)
	<-s.stopCalled
}
func (s *stuckGRPCServer) Stop() { close(s.stopCalled) }

type failedGRPCServer struct {
	stopCalled bool
}

func (*failedGRPCServer) Serve(net.Listener) error { return errors.New("accept failed") }
func (*failedGRPCServer) GracefulStop()            {}
func (s *failedGRPCServer) Stop()                  { s.stopCalled = true }

func TestServeFailureStopsActiveTransports(t *testing.T) {
	grpcServer := &failedGRPCServer{}
	err := serveGRPC(context.Background(), time.Second, grpcServer, newBlockingListener())
	require.EqualError(t, err, "serve gRPC: accept failed")
	require.True(t, grpcServer.stopCalled)
}

func TestShutdownForcesStopAtDeadline(t *testing.T) {
	server := &stuckGRPCServer{gracefulStarted: make(chan struct{}), stopCalled: make(chan struct{})}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	require.NoError(t, serveGRPC(ctx, 10*time.Millisecond, server, newBlockingListener()))
	select {
	case <-server.gracefulStarted:
	default:
		t.Fatal("graceful shutdown was not attempted")
	}
	select {
	case <-server.stopCalled:
	default:
		t.Fatal("forced stop was not called")
	}
}
