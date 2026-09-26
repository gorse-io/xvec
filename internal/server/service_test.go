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
	"io"
	"path/filepath"
	"testing"
	"time"

	"github.com/gorse-io/xvec"
	"github.com/gorse-io/xvec/internal/grpcapi"
	xvecv1 "github.com/gorse-io/xvec/internal/proto/xvec/v1"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/emptypb"
)

type captureStream[T any] struct {
	ctx      context.Context
	messages []*T
}

func (s *captureStream[T]) Context() context.Context { return s.ctx }
func (s *captureStream[T]) Send(message *T) error {
	s.messages = append(s.messages, message)
	return nil
}
func (*captureStream[T]) SetHeader(metadata.MD) error  { return nil }
func (*captureStream[T]) SendHeader(metadata.MD) error { return nil }
func (*captureStream[T]) SetTrailer(metadata.MD)       {}
func (*captureStream[T]) SendMsg(any) error            { return nil }
func (*captureStream[T]) RecvMsg(any) error            { return io.EOF }

func newTestService(t *testing.T, options Options) *Service {
	t.Helper()
	service, err := New(filepath.Join(t.TempDir(), "data"), options)
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, service.Close()) })
	return service
}

func protoTestSchema(name string) *xvecv1.CollectionSchema {
	return &xvecv1.CollectionSchema{Name: name, MaxDocsPerSegment: 1000, Fields: []*xvecv1.FieldSchema{
		{Name: "title", DataType: xvecv1.DataType_DATA_TYPE_STRING},
		{Name: "category", DataType: xvecv1.DataType_DATA_TYPE_STRING},
	}}
}

func protoDocument(key, title, category string) *xvecv1.Document {
	return &xvecv1.Document{PrimaryKey: key, Fields: []*xvecv1.FieldEntry{
		{Name: "title", Value: &xvecv1.Value{Kind: &xvecv1.Value_StringValue{StringValue: title}}},
		{Name: "category", Value: &xvecv1.Value{Kind: &xvecv1.Value_StringValue{StringValue: category}}},
	}}
}

func TestNewValidatesOptions(t *testing.T) {
	for _, options := range []Options{
		{StreamBatchSize: -1},
		{MaxStreamLifetime: -time.Second},
		{StreamIdleTimeout: -time.Second},
		{MaxConcurrentStreamsPerClient: -1},
	} {
		service, err := New(t.TempDir(), options)
		require.Error(t, err)
		require.Nil(t, service)
	}

	service := newTestService(t, Options{})
	require.NotEmpty(t, service.GRPCServerOptions())
	var registrar recordingRegistrar
	service.Register(&registrar)
	require.Equal(t, "xvec.v1.XvecService", registrar.description.ServiceName)
	require.Same(t, service, registrar.implementation)
}

type recordingRegistrar struct {
	description    *grpc.ServiceDesc
	implementation any
}

func (r *recordingRegistrar) RegisterService(description *grpc.ServiceDesc, implementation any) {
	r.description = description
	r.implementation = implementation
}

func TestServiceLifecycleDDLAndStats(t *testing.T) {
	service := newTestService(t, Options{})
	ctx := context.Background()

	created, err := service.CreateCollection(ctx, &xvecv1.CreateCollectionRequest{Schema: protoTestSchema("books")})
	require.NoError(t, err)
	require.Equal(t, "books", created.Name)

	listed, err := service.ListCollections(ctx, &emptypb.Empty{})
	require.NoError(t, err)
	require.Equal(t, []string{"books"}, listed.Collections)

	schema, err := service.AddColumn(ctx, &xvecv1.AddColumnRequest{
		Collection: "books",
		Field:      &xvecv1.FieldSchema{Name: "year", DataType: xvecv1.DataType_DATA_TYPE_INT64, Nullable: true},
	})
	require.NoError(t, err)
	require.Len(t, schema.Fields, 3)

	schema, err = service.AlterColumn(ctx, &xvecv1.AlterColumnRequest{Collection: "books", Column: "year", Rename: "published"})
	require.NoError(t, err)
	require.Equal(t, "published", schema.Fields[2].Name)

	schema, err = service.DropColumn(ctx, &xvecv1.DropColumnRequest{Collection: "books", Column: "published"})
	require.NoError(t, err)
	require.Len(t, schema.Fields, 2)

	stats, err := service.GetStats(ctx, &xvecv1.CollectionRef{Collection: "books"})
	require.NoError(t, err)
	require.Zero(t, stats.DocumentCount)
	require.NoError(t, mustUnaryEmpty(service.Flush(ctx, &xvecv1.CollectionRef{Collection: "books"})))
	require.NoError(t, mustUnaryEmpty(service.Optimize(ctx, &xvecv1.OptimizeRequest{Collection: "books"})))

	_, err = service.GetSchema(ctx, &xvecv1.CollectionRef{Collection: "../secret"})
	require.Equal(t, codes.InvalidArgument, status.Code(err))
	require.NotContains(t, status.Convert(err).Message(), service.registry.DataDir())

	require.NoError(t, mustUnaryEmpty(service.DropCollection(ctx, &xvecv1.CollectionRef{Collection: "books"})))
	_, err = service.GetSchema(ctx, &xvecv1.CollectionRef{Collection: "books"})
	require.Equal(t, codes.NotFound, status.Code(err))
}

func mustUnaryEmpty(value *emptypb.Empty, err error) error { return err }

func TestServiceWriteDeleteAndStreamingReads(t *testing.T) {
	service := newTestService(t, Options{StreamBatchSize: 2})
	ctx := context.Background()
	_, err := service.CreateCollection(ctx, &xvecv1.CreateCollectionRequest{Schema: protoTestSchema("books")})
	require.NoError(t, err)

	response, err := service.Write(ctx, &xvecv1.WriteRequest{
		Collection: "books",
		Operation:  xvecv1.WriteOperation_WRITE_OPERATION_INSERT,
		Documents: []*xvecv1.Document{
			protoDocument("a", "Alpha", "one"),
			protoDocument("b", "Beta", "one"),
			protoDocument("c", "Gamma", "two"),
		},
	})
	require.NoError(t, err)
	require.Len(t, response.Results, 3)
	require.Nil(t, response.Error)

	duplicate, err := service.Write(ctx, &xvecv1.WriteRequest{Collection: "books", Documents: []*xvecv1.Document{protoDocument("a", "again", "one"), protoDocument("d", "Delta", "two")}})
	require.NoError(t, err, "batch failures belong in the OK response payload")
	require.Len(t, duplicate.Results, 2)
	require.Equal(t, "a", duplicate.Results[0].PrimaryKey)
	require.NotNil(t, duplicate.Results[0].Error)
	require.NotNil(t, duplicate.Error)

	fetch := &captureStream[xvecv1.FetchResult]{ctx: ctx}
	require.NoError(t, service.Fetch(&xvecv1.FetchRequest{Collection: "books", PrimaryKeys: []string{"a", "missing", "c"}}, fetch))
	require.Len(t, fetch.messages, 3)
	require.Equal(t, uint32(0), fetch.messages[0].Index)
	require.True(t, fetch.messages[0].Found)
	require.Equal(t, uint32(1), fetch.messages[1].Index)
	require.False(t, fetch.messages[1].Found)
	require.Nil(t, fetch.messages[1].Document)
	require.Equal(t, uint32(2), fetch.messages[2].Index)

	query := &captureStream[xvecv1.QueryResponse]{ctx: ctx}
	require.NoError(t, service.Query(&xvecv1.QueryRequest{Collection: "books", Query: &xvecv1.VectorQuery{
		Target: &xvecv1.QueryTarget{Kind: &xvecv1.QueryTarget_FilterOnly{FilterOnly: &xvecv1.FilterOnly{}}},
		TopK:   10,
	}}, query))
	require.Len(t, query.messages, 2)
	require.Len(t, query.messages[0].Documents, 2)
	require.Len(t, query.messages[1].Documents, 2)

	iterate := &captureStream[xvecv1.QueryResponse]{ctx: ctx}
	require.NoError(t, service.Iterate(&xvecv1.IterateRequest{Collection: "books"}, iterate))
	require.Len(t, iterate.messages, 2)
	require.Len(t, iterate.messages[0].Documents, 2)
	require.Len(t, iterate.messages[1].Documents, 2)

	deleted, err := service.Delete(ctx, &xvecv1.DeleteRequest{Collection: "books", PrimaryKeys: []string{"a", "missing"}})
	require.NoError(t, err)
	require.Len(t, deleted.Results, 2)
	require.NoError(t, mustUnaryEmpty(service.DeleteByFilter(ctx, &xvecv1.DeleteByFilterRequest{Collection: "books", Filter: "category = 'two'"})))
}

func TestServiceRejectsMalformedRequestsAsStatus(t *testing.T) {
	service := newTestService(t, Options{})
	_, err := service.CreateCollection(context.Background(), &xvecv1.CreateCollectionRequest{})
	require.Equal(t, codes.InvalidArgument, status.Code(err))

	_, err = service.Write(context.Background(), &xvecv1.WriteRequest{Collection: "books", Operation: xvecv1.WriteOperation(99)})
	require.Equal(t, codes.InvalidArgument, status.Code(err))
}

func TestStreamInterceptorEnforcesConnectionQuotaAndTimeouts(t *testing.T) {
	service := newTestService(t, Options{
		MaxConcurrentStreamsPerClient: 1,
		MaxStreamLifetime:             30 * time.Millisecond,
		StreamIdleTimeout:             80 * time.Millisecond,
	})
	connectionContext := withConnectionID(context.Background(), 7)
	first := &captureStream[xvecv1.QueryResponse]{ctx: connectionContext}
	entered := make(chan struct{})
	release := make(chan struct{})
	firstDone := make(chan error, 1)
	go func() {
		firstDone <- service.streamInterceptor()(nil, first, &grpc.StreamServerInfo{IsServerStream: true}, func(_ any, stream grpc.ServerStream) error {
			close(entered)
			select {
			case <-release:
				return nil
			case <-stream.Context().Done():
				return stream.Context().Err()
			}
		})
	}()
	<-entered

	second := &captureStream[xvecv1.QueryResponse]{ctx: connectionContext}
	err := service.streamInterceptor()(nil, second, &grpc.StreamServerInfo{IsServerStream: true}, func(any, grpc.ServerStream) error {
		return nil
	})
	require.Equal(t, codes.ResourceExhausted, status.Code(err))

	require.Eventually(t, func() bool {
		select {
		case err := <-firstDone:
			require.ErrorIs(t, err, context.DeadlineExceeded)
			return true
		default:
			return false
		}
	}, time.Second, time.Millisecond)
	close(release)

	third := &captureStream[xvecv1.QueryResponse]{ctx: connectionContext}
	require.NoError(t, service.streamInterceptor()(nil, third, &grpc.StreamServerInfo{IsServerStream: true}, func(any, grpc.ServerStream) error { return nil }))
}

func TestGroupResponseCodecHelper(t *testing.T) {
	document := xvec.Document{PrimaryKey: "a", Fields: map[string]any{"title": "Alpha"}}
	message, err := grpcapi.DocumentToProto(document)
	require.NoError(t, err)
	require.Equal(t, "a", message.PrimaryKey)
}
