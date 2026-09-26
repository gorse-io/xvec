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
	"net"
	"testing"

	"github.com/gorse-io/xvec"
	"github.com/gorse-io/xvec/internal/server"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"
)

func TestClientServerEndToEnd(t *testing.T) {
	ctx := context.Background()
	service, err := server.New(t.TempDir(), server.Options{StreamBatchSize: 1})
	require.NoError(t, err)

	listener := bufconn.Listen(1 << 20)
	grpcServer := grpc.NewServer(service.GRPCServerOptions()...)
	service.Register(grpcServer)
	serveDone := make(chan error, 1)
	go func() { serveDone <- grpcServer.Serve(listener) }()

	// DialContext keeps this test's bufconn setup aligned with the public Dial path.
	//nolint:staticcheck // Supported throughout gRPC 1.x; grpc.NewClient has different semantics.
	connection, err := grpc.DialContext(ctx, "bufconn",
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) { return listener.Dial() }),
	)
	require.NoError(t, err)
	remote := NewClient(connection)
	t.Cleanup(func() {
		require.NoError(t, connection.Close())
		grpcServer.Stop()
		if serveErr := <-serveDone; serveErr != nil {
			require.ErrorIs(t, serveErr, grpc.ErrServerStopped)
		}
		require.NoError(t, service.Close())
		require.NoError(t, listener.Close())
	})

	schema := xvec.NewCollectionSchema("books",
		xvec.NewField("title", xvec.DataTypeString),
		xvec.NewField("category", xvec.DataTypeString),
	)
	collection, err := remote.CreateAndOpen(ctx, schema, CreateCollectionOptions{})
	require.NoError(t, err)
	require.Equal(t, "books", collection.Name())

	results, err := collection.Insert(ctx, []xvec.Document{
		{PrimaryKey: "a", Fields: map[string]any{"title": "Alpha", "category": "one"}},
		{PrimaryKey: "b", Fields: map[string]any{"title": "Beta", "category": "two"}},
	})
	require.NoError(t, err)
	require.Len(t, results, 2)

	fetched, err := collection.Fetch(ctx, []string{"b", "missing", "a"}, xvec.Projection{})
	require.NoError(t, err)
	require.Equal(t, "b", fetched[0].PrimaryKey)
	require.Nil(t, fetched[1])
	require.Equal(t, "a", fetched[2].PrimaryKey)

	documents, err := collection.Query(ctx, xvec.VectorQuery{TopK: 10})
	require.NoError(t, err)
	require.Len(t, documents, 2)

	iterator, err := collection.CreateIterator(ctx, xvec.IteratorOptions{})
	require.NoError(t, err)
	var iterated []string
	for {
		document, nextErr := iterator.Next()
		if nextErr == io.EOF {
			break
		}
		require.NoError(t, nextErr)
		iterated = append(iterated, document.PrimaryKey)
	}
	require.ElementsMatch(t, []string{"a", "b"}, iterated)
	iterator.Close()

	require.NoError(t, collection.Destroy(ctx))
	_, err = remote.Open(ctx, "books")
	require.ErrorIs(t, err, xvec.ErrNotFound)
}
