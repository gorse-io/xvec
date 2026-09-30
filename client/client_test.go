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
	"errors"
	"io"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/gorse-io/xvec"
	"github.com/gorse-io/xvec/internal/grpcapi"
	xvecv1 "github.com/gorse-io/xvec/internal/proto/xvec/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/emptypb"
)

type fakeConn struct {
	mu        sync.Mutex
	unary     map[string]proto.Message
	unaryErr  map[string]error
	streams   map[string][]proto.Message
	streamErr map[string]error
	requests  map[string][]proto.Message
	contexts  map[string]context.Context
	blocks    map[string]<-chan struct{}
	started   map[string]chan struct{}
}

func newFakeConn() *fakeConn {
	return &fakeConn{
		unary: make(map[string]proto.Message), unaryErr: make(map[string]error),
		streams: make(map[string][]proto.Message), streamErr: make(map[string]error),
		requests: make(map[string][]proto.Message), contexts: make(map[string]context.Context),
		blocks: make(map[string]<-chan struct{}), started: make(map[string]chan struct{}),
	}
}

func (c *fakeConn) Invoke(_ context.Context, method string, args, reply any, _ ...grpc.CallOption) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.requests[method] = append(c.requests[method], proto.Clone(args.(proto.Message)))
	if err := c.unaryErr[method]; err != nil {
		return err
	}
	if response := c.unary[method]; response != nil {
		proto.Reset(reply.(proto.Message))
		proto.Merge(reply.(proto.Message), response)
	}
	return nil
}

func (c *fakeConn) NewStream(ctx context.Context, _ *grpc.StreamDesc, method string, _ ...grpc.CallOption) (grpc.ClientStream, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.contexts[method] = ctx
	return &fakeStream{ctx: ctx, method: method, conn: c, responses: c.streams[method]}, nil
}

func (c *fakeConn) request(method string) proto.Message {
	c.mu.Lock()
	defer c.mu.Unlock()
	items := c.requests[method]
	if len(items) == 0 {
		return nil
	}
	return items[len(items)-1]
}

type fakeStream struct {
	ctx       context.Context
	method    string
	conn      *fakeConn
	responses []proto.Message
	index     int
}

func (*fakeStream) Header() (metadata.MD, error) { return nil, nil }
func (*fakeStream) Trailer() metadata.MD         { return nil }
func (*fakeStream) CloseSend() error             { return nil }
func (s *fakeStream) Context() context.Context   { return s.ctx }
func (s *fakeStream) SendMsg(message any) error {
	s.conn.mu.Lock()
	defer s.conn.mu.Unlock()
	s.conn.requests[s.method] = append(s.conn.requests[s.method], proto.Clone(message.(proto.Message)))
	return nil
}
func (s *fakeStream) RecvMsg(message any) error {
	if s.index < len(s.responses) {
		proto.Merge(message.(proto.Message), s.responses[s.index])
		s.index++
		return nil
	}
	if err := s.conn.streamErr[s.method]; err != nil {
		return err
	}
	if started := s.conn.started[s.method]; started != nil {
		select {
		case <-started:
		default:
			close(started)
		}
	}
	if block := s.conn.blocks[s.method]; block != nil {
		select {
		case <-block:
		case <-s.ctx.Done():
			return s.ctx.Err()
		}
	}
	return io.EOF
}

func requireContextCanceled(t *testing.T, conn *fakeConn, method string) {
	t.Helper()
	select {
	case <-conn.contexts[method].Done():
	default:
		t.Fatalf("%s stream context was not canceled", method)
	}
}

func document(primaryKey string) *xvecv1.Document {
	return &xvecv1.Document{PrimaryKey: primaryKey, Fields: []*xvecv1.FieldEntry{{
		Name: "value", Value: &xvecv1.Value{Kind: &xvecv1.Value_StringValue{StringValue: primaryKey}},
	}}}
}

func malformedDocument() *xvecv1.Document {
	return &xvecv1.Document{Fields: []*xvecv1.FieldEntry{{Name: "broken"}}}
}

func TestClientCollectionLifecycleAndMetadata(t *testing.T) {
	conn := newFakeConn()
	conn.unary[xvecv1.XvecService_ListCollections_FullMethodName] = &xvecv1.ListCollectionsResponse{Collections: []string{"b", "a"}}
	conn.unary[xvecv1.XvecService_CreateCollection_FullMethodName] = &xvecv1.CollectionSchema{Name: "created"}
	conn.unary[xvecv1.XvecService_GetSchema_FullMethodName] = &xvecv1.CollectionSchema{Name: "opened"}
	conn.unary[xvecv1.XvecService_GetStats_FullMethodName] = &xvecv1.CollectionStats{DocumentCount: 3, IndexCompleteness: map[string]float32{"v": .5}}
	client := NewClient(conn)

	names, err := client.ListCollections(context.Background())
	if err != nil || !reflect.DeepEqual(names, []string{"b", "a"}) {
		t.Fatalf("ListCollections() = %v, %v", names, err)
	}
	enableMmap := false
	created, err := client.CreateAndOpen(context.Background(), xvec.CollectionSchema{Name: "created"}, CreateCollectionOptions{EnableMmap: &enableMmap})
	if err != nil || created.Name() != "created" {
		t.Fatalf("CreateAndOpen() = %v, %v", created, err)
	}
	request := conn.request(xvecv1.XvecService_CreateCollection_FullMethodName).(*xvecv1.CreateCollectionRequest)
	if request.EnableMmap == nil || *request.EnableMmap {
		t.Fatalf("CreateAndOpen enable_mmap = %v", request.EnableMmap)
	}
	_, err = client.CreateAndOpen(context.Background(), xvec.CollectionSchema{Name: "created"}, CreateCollectionOptions{})
	if err != nil {
		t.Fatalf("CreateAndOpen() with server defaults: %v", err)
	}
	request = conn.request(xvecv1.XvecService_CreateCollection_FullMethodName).(*xvecv1.CreateCollectionRequest)
	if request.EnableMmap != nil {
		t.Fatalf("CreateAndOpen default enable_mmap = %v", request.EnableMmap)
	}
	opened, err := client.Open(context.Background(), "opened")
	if err != nil || opened.Name() != "opened" {
		t.Fatalf("Open() = %v, %v", opened, err)
	}
	schema, err := opened.Schema(context.Background())
	if err != nil || schema.Name != "opened" {
		t.Fatalf("Schema() = %#v, %v", schema, err)
	}
	stats, err := opened.Stats(context.Background())
	if err != nil || stats.DocumentCount != 3 || stats.IndexCompleteness["v"] != .5 {
		t.Fatalf("Stats() = %#v, %v", stats, err)
	}
	if err := opened.Flush(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := opened.Destroy(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestCollectionManagementRequests(t *testing.T) {
	conn := newFakeConn()
	collection := NewClient(conn).collection("items")
	ctx := context.Background()
	field := xvec.FieldSchema{Name: "age", DataType: xvec.DataTypeInt32}
	if err := collection.AddColumn(ctx, field, "1", xvec.AddColumnOptions{Concurrency: 2}); err != nil {
		t.Fatal(err)
	}
	if err := collection.AlterColumn(ctx, "age", "years", nil, xvec.AlterColumnOptions{Concurrency: 3}); err != nil {
		t.Fatal(err)
	}
	if err := collection.DropColumn(ctx, "years"); err != nil {
		t.Fatal(err)
	}
	index := xvec.InvertIndexParams{EnableRangeOptimization: true}
	if err := collection.CreateIndex(ctx, "years", index, xvec.CreateIndexOptions{Concurrency: 4}); err != nil {
		t.Fatal(err)
	}
	if err := collection.DropIndex(ctx, "years"); err != nil {
		t.Fatal(err)
	}
	if err := collection.Optimize(ctx, xvec.OptimizeOptions{Concurrency: 5}); err != nil {
		t.Fatal(err)
	}
	if got := conn.request(xvecv1.XvecService_AddColumn_FullMethodName).(*xvecv1.AddColumnRequest); got.Collection != "items" || got.Concurrency != 2 || got.Expression != "1" {
		t.Fatalf("AddColumn request = %#v", got)
	}
	if got := conn.request(xvecv1.XvecService_AlterColumn_FullMethodName).(*xvecv1.AlterColumnRequest); got.Rename != "years" || got.Concurrency != 3 {
		t.Fatalf("AlterColumn request = %#v", got)
	}
	if got := conn.request(xvecv1.XvecService_CreateIndex_FullMethodName).(*xvecv1.CreateIndexRequest); got.Field != "years" || got.Concurrency != 4 || got.Index == nil {
		t.Fatalf("CreateIndex request = %#v", got)
	}
	if got := conn.request(xvecv1.XvecService_Optimize_FullMethodName).(*xvecv1.OptimizeRequest); got.Concurrency != 5 {
		t.Fatalf("Optimize request = %#v", got)
	}
}

func TestWritesPreserveResultsAndAggregateError(t *testing.T) {
	conn := newFakeConn()
	failure := &xvec.Error{Code: xvec.ErrorCodeAlreadyExists, Message: "duplicate"}
	conn.unary[xvecv1.XvecService_Write_FullMethodName] = grpcapi.WriteResponseToProto(
		[]xvec.WriteResult{{PrimaryKey: "a", DocID: 7}, {PrimaryKey: "b", Err: failure}},
		xvec.NewBatchWriteError(failure),
	)
	collection := NewClient(conn).collection("items")
	results, err := collection.Insert(context.Background(), []xvec.Document{{PrimaryKey: "a"}, {PrimaryKey: "b"}})
	if !errors.Is(err, xvec.ErrAlreadyExists) || len(results) != 2 || results[0].DocID != 7 || !errors.Is(results[1].Err, xvec.ErrAlreadyExists) {
		t.Fatalf("Insert() = %#v, %v", results, err)
	}
	request := conn.request(xvecv1.XvecService_Write_FullMethodName).(*xvecv1.WriteRequest)
	if request.Operation != xvecv1.WriteOperation_WRITE_OPERATION_INSERT || request.Documents[0].PrimaryKey != "a" || request.Documents[1].PrimaryKey != "b" {
		t.Fatalf("Write request = %#v", request)
	}
	conn.unary[xvecv1.XvecService_Delete_FullMethodName] = grpcapi.WriteResponseToProto([]xvec.WriteResult{{PrimaryKey: "b"}, {PrimaryKey: "a"}}, nil)
	results, err = collection.Delete(context.Background(), []string{"b", "a"})
	if err != nil || results[0].PrimaryKey != "b" || results[1].PrimaryKey != "a" {
		t.Fatalf("Delete() = %#v, %v", results, err)
	}
	if err := collection.DeleteByFilter(context.Background(), "age > 1"); err != nil {
		t.Fatal(err)
	}
}

func TestTransportErrorsAreConverted(t *testing.T) {
	conn := newFakeConn()
	conn.unaryErr[xvecv1.XvecService_ListCollections_FullMethodName] = status.Error(codes.NotFound, "gone")
	_, err := NewClient(conn).ListCollections(context.Background())
	if !errors.Is(err, xvec.ErrNotFound) {
		t.Fatalf("ListCollections error = %T %v", err, err)
	}
	conn.streamErr[xvecv1.XvecService_Query_FullMethodName] = status.Error(codes.Unavailable, "offline")
	_, err = NewClient(conn).collection("items").Query(context.Background(), xvec.VectorQuery{})
	if !errors.Is(err, xvec.ErrUnavailable) {
		t.Fatalf("Query error = %T %v", err, err)
	}
}

func TestFetchPreservesOrderAndRejectsMalformedResults(t *testing.T) {
	method := xvecv1.XvecService_Fetch_FullMethodName
	t.Run("order and missing", func(t *testing.T) {
		conn := newFakeConn()
		conn.streams[method] = []proto.Message{
			&xvecv1.FetchResult{Index: 2, Found: true, Document: document("c")},
			&xvecv1.FetchResult{Index: 0, Found: true, Document: document("a")},
			&xvecv1.FetchResult{Index: 1, Found: false},
		}
		got, err := NewClient(conn).collection("items").Fetch(context.Background(), []string{"a", "b", "c"}, xvec.Projection{})
		if err != nil || got[0].PrimaryKey != "a" || got[1] != nil || got[2].PrimaryKey != "c" {
			t.Fatalf("Fetch() = %#v, %v", got, err)
		}
	})
	for _, test := range []struct {
		name   string
		values []proto.Message
	}{
		{"missing result", nil},
		{"duplicate", []proto.Message{&xvecv1.FetchResult{Index: 0}, &xvecv1.FetchResult{Index: 0}}},
		{"out of range", []proto.Message{&xvecv1.FetchResult{Index: 1}}},
		{"found without document", []proto.Message{&xvecv1.FetchResult{Index: 0, Found: true}}},
		{"not found with document", []proto.Message{&xvecv1.FetchResult{Index: 0, Document: document("a")}}},
	} {
		t.Run(test.name, func(t *testing.T) {
			conn := newFakeConn()
			conn.streams[method] = test.values
			_, err := NewClient(conn).collection("items").Fetch(context.Background(), []string{"a"}, xvec.Projection{})
			if !errors.Is(err, xvec.ErrInternal) {
				t.Fatalf("Fetch error = %T %v", err, err)
			}
			requireContextCanceled(t, conn, method)
		})
	}
}

func TestQueryStreamsAggregateAndRejectMalformedPayloads(t *testing.T) {
	conn := newFakeConn()
	conn.streams[xvecv1.XvecService_Query_FullMethodName] = []proto.Message{
		&xvecv1.QueryResponse{Documents: []*xvecv1.Document{document("a")}},
		&xvecv1.QueryResponse{Documents: []*xvecv1.Document{document("b")}},
	}
	got, err := NewClient(conn).collection("items").Query(context.Background(), xvec.VectorQuery{PrimaryKey: "seed", Field: "v"})
	if err != nil || len(got) != 2 || got[0].PrimaryKey != "a" || got[1].PrimaryKey != "b" {
		t.Fatalf("Query() = %#v, %v", got, err)
	}

	conn = newFakeConn()
	conn.streams[xvecv1.XvecService_MultiQuery_FullMethodName] = []proto.Message{&xvecv1.QueryResponse{Documents: []*xvecv1.Document{document("m")}}}
	got, err = NewClient(conn).collection("items").MultiQuery(context.Background(), xvec.MultiQuery{Queries: []xvec.SubQuery{{PrimaryKey: "a", Field: "v"}, {PrimaryKey: "b", Field: "v"}}, Reranker: xvec.NewRRFReranker()})
	if err != nil || len(got) != 1 || got[0].PrimaryKey != "m" {
		t.Fatalf("MultiQuery() = %#v, %v", got, err)
	}

	conn = newFakeConn()
	conn.streams[xvecv1.XvecService_GroupByQuery_FullMethodName] = []proto.Message{
		&xvecv1.GroupByQueryResponse{Groups: []*xvecv1.GroupResult{{Value: "x", Documents: []*xvecv1.Document{document("a")}}}},
		&xvecv1.GroupByQueryResponse{Groups: []*xvecv1.GroupResult{{Value: "y", Documents: []*xvecv1.Document{document("b")}}}},
	}
	groups, err := NewClient(conn).collection("items").GroupByQuery(context.Background(), xvec.GroupByVectorQuery{PrimaryKey: "seed", Field: "v"})
	if err != nil || len(groups) != 2 || groups[1].Value != "y" {
		t.Fatalf("GroupByQuery() = %#v, %v", groups, err)
	}

	conn = newFakeConn()
	conn.streams[xvecv1.XvecService_Query_FullMethodName] = []proto.Message{&xvecv1.QueryResponse{Documents: []*xvecv1.Document{malformedDocument()}}}
	_, err = NewClient(conn).collection("items").Query(context.Background(), xvec.VectorQuery{PrimaryKey: "seed", Field: "v"})
	if !errors.Is(err, xvec.ErrInternal) {
		t.Fatalf("malformed Query error = %T %v", err, err)
	}
	requireContextCanceled(t, conn, xvecv1.XvecService_Query_FullMethodName)
}

func TestMultiQueryRejectsCallbackRerankerBeforeRPC(t *testing.T) {
	conn := newFakeConn()
	query := xvec.MultiQuery{
		Queries:  []xvec.SubQuery{{PrimaryKey: "a", Field: "v"}, {PrimaryKey: "b", Field: "v"}},
		Reranker: xvec.NewCallbackReranker(func(context.Context, []xvec.RerankBatch, int) ([]xvec.Document, error) { return nil, nil }),
	}
	_, err := NewClient(conn).collection("items").MultiQuery(context.Background(), query)
	if !errors.Is(err, xvec.ErrNotSupported) {
		t.Fatalf("MultiQuery error = %T %v", err, err)
	}
	if request := conn.request(xvecv1.XvecService_MultiQuery_FullMethodName); request != nil {
		t.Fatalf("unexpected request: %#v", request)
	}
}

func TestIteratorAggregatesChunksAndCloseCancels(t *testing.T) {
	conn := newFakeConn()
	method := xvecv1.XvecService_Iterate_FullMethodName
	conn.streams[method] = []proto.Message{
		&xvecv1.QueryResponse{Documents: []*xvecv1.Document{document("a"), document("b")}},
		&xvecv1.QueryResponse{Documents: []*xvecv1.Document{document("c")}},
	}
	iterator, err := NewClient(conn).collection("items").CreateIterator(context.Background(), xvec.IteratorOptions{})
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"a", "b", "c"} {
		got, err := iterator.Next()
		if err != nil || got.PrimaryKey != want {
			t.Fatalf("Next() = %#v, %v; want %s", got, err, want)
		}
	}
	if got, err := iterator.Next(); got != nil || !errors.Is(err, io.EOF) {
		t.Fatalf("Next at EOF = %#v, %v", got, err)
	}
	iterator.Close()
	iterator.Close()
	select {
	case <-conn.contexts[method].Done():
	default:
		t.Fatal("Close did not cancel iterator stream")
	}
}

func TestIteratorCloseCancelsBlockedNext(t *testing.T) {
	conn := newFakeConn()
	method := xvecv1.XvecService_Iterate_FullMethodName
	conn.blocks[method] = make(chan struct{})
	conn.started[method] = make(chan struct{})
	iterator, err := NewClient(conn).collection("items").CreateIterator(context.Background(), xvec.IteratorOptions{})
	if err != nil {
		t.Fatal(err)
	}
	nextDone := make(chan error, 1)
	go func() {
		_, nextErr := iterator.Next()
		nextDone <- nextErr
	}()
	<-conn.started[method]
	closeDone := make(chan struct{})
	go func() {
		iterator.Close()
		close(closeDone)
	}()
	select {
	case <-closeDone:
	case <-time.After(time.Second):
		t.Fatal("Close blocked behind Next")
	}
	select {
	case nextErr := <-nextDone:
		if !errors.Is(nextErr, io.EOF) {
			t.Fatalf("Next error = %v, want EOF", nextErr)
		}
	case <-time.After(time.Second):
		t.Fatal("Next did not unblock after Close")
	}
}

func TestIteratorConvertsStreamAndPayloadErrors(t *testing.T) {
	conn := newFakeConn()
	method := xvecv1.XvecService_Iterate_FullMethodName
	conn.streams[method] = []proto.Message{&xvecv1.QueryResponse{Documents: []*xvecv1.Document{malformedDocument()}}}
	iterator, err := NewClient(conn).collection("items").CreateIterator(context.Background(), xvec.IteratorOptions{})
	if err != nil {
		t.Fatal(err)
	}
	_, err = iterator.Next()
	if !errors.Is(err, xvec.ErrInternal) {
		t.Fatalf("malformed iterator error = %T %v", err, err)
	}
	iterator.Close()

	conn = newFakeConn()
	conn.streamErr[method] = status.Error(codes.Unavailable, "offline")
	iterator, err = NewClient(conn).collection("items").CreateIterator(context.Background(), xvec.IteratorOptions{})
	if err != nil {
		t.Fatal(err)
	}
	_, err = iterator.Next()
	if !errors.Is(err, xvec.ErrUnavailable) {
		t.Fatalf("iterator stream error = %T %v", err, err)
	}
	iterator.Close()
}

var _ grpc.ClientConnInterface = (*fakeConn)(nil)
var _ = emptypb.Empty{}
