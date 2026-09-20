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
	"errors"
	"fmt"
	"io"
	"math"
	"sync"
	"time"

	"github.com/gorse-io/xvec"
	"github.com/gorse-io/xvec/internal/grpcapi"
	xvecv1 "github.com/gorse-io/xvec/internal/proto/xvec/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/emptypb"
)

const (
	defaultMaxMessageSize                = 64 << 20
	defaultStreamBatchSize               = 128
	defaultMaxStreamLifetime             = 10 * time.Minute
	defaultStreamIdleTimeout             = time.Minute
	defaultMaxConcurrentStreamsPerClient = 16
)

// Options configures collection handles and bounded server streams.
type Options struct {
	CollectionOptions             xvec.CollectionOptions
	StreamBatchSize               int
	MaxStreamLifetime             time.Duration
	StreamIdleTimeout             time.Duration
	MaxConcurrentStreamsPerClient int
}

// Service implements the generated XvecService gRPC contract.
type Service struct {
	xvecv1.UnimplementedXvecServiceServer

	registry *Registry
	options  Options
	stats    *connectionStatsHandler

	streamMu sync.Mutex
	streams  map[uint64]int
	closed   bool
}

var _ xvecv1.XvecServiceServer = (*Service)(nil)

// New creates a service rooted at dataDir.
func New(dataDir string, options Options) (*Service, error) {
	if options.StreamBatchSize < 0 {
		return nil, fmt.Errorf("stream batch size cannot be negative")
	}
	if options.MaxStreamLifetime < 0 {
		return nil, fmt.Errorf("maximum stream lifetime cannot be negative")
	}
	if options.StreamIdleTimeout < 0 {
		return nil, fmt.Errorf("stream idle timeout cannot be negative")
	}
	if options.MaxConcurrentStreamsPerClient < 0 {
		return nil, fmt.Errorf("maximum concurrent streams per client cannot be negative")
	}
	if options.StreamBatchSize == 0 {
		options.StreamBatchSize = defaultStreamBatchSize
	}
	if options.MaxStreamLifetime == 0 {
		options.MaxStreamLifetime = defaultMaxStreamLifetime
	}
	if options.StreamIdleTimeout == 0 {
		options.StreamIdleTimeout = defaultStreamIdleTimeout
	}
	if options.MaxConcurrentStreamsPerClient == 0 {
		options.MaxConcurrentStreamsPerClient = defaultMaxConcurrentStreamsPerClient
	}
	registry, err := NewRegistry(dataDir, options.CollectionOptions)
	if err != nil {
		return nil, err
	}
	service := &Service{registry: registry, options: options, streams: make(map[uint64]int)}
	service.stats = &connectionStatsHandler{service: service}
	return service, nil
}

// Register registers the service implementation with a gRPC registrar.
func (s *Service) Register(registrar grpc.ServiceRegistrar) {
	xvecv1.RegisterXvecServiceServer(registrar, s)
}

// GRPCServerOptions returns the resource-enforcement hooks required by Service.
func (s *Service) GRPCServerOptions() []grpc.ServerOption {
	return []grpc.ServerOption{
		grpc.MaxRecvMsgSize(defaultMaxMessageSize),
		grpc.MaxSendMsgSize(defaultMaxMessageSize),
		grpc.StatsHandler(s.stats),
		grpc.StreamInterceptor(s.streamInterceptor()),
	}
}

// Close stops accepting streams and closes all collection handles after leases drain.
func (s *Service) Close() error {
	if s == nil {
		return nil
	}
	s.streamMu.Lock()
	s.closed = true
	s.streamMu.Unlock()
	return s.registry.Close()
}

func (s *Service) CreateCollection(ctx context.Context, request *xvecv1.CreateCollectionRequest) (*xvecv1.CollectionSchema, error) {
	if request == nil {
		return nil, invalidStatus("create collection request is nil")
	}
	schema, err := grpcapi.SchemaFromProto(request.Schema)
	if err != nil {
		return nil, rpcError(err)
	}
	var mmap *bool
	if request.EnableMmap != nil {
		value := request.GetEnableMmap()
		mmap = &value
	}
	collection, err := s.registry.Create(ctx, schema, CreateOptions{EnableMmap: mmap})
	if err != nil {
		return nil, rpcError(err)
	}
	message, err := grpcapi.SchemaToProto(collection.Schema())
	return message, rpcError(err)
}

func (s *Service) DropCollection(ctx context.Context, request *xvecv1.CollectionRef) (*emptypb.Empty, error) {
	if request == nil {
		return nil, invalidStatus("collection reference is nil")
	}
	if err := s.registry.Destroy(ctx, request.Collection); err != nil {
		return nil, rpcError(err)
	}
	return &emptypb.Empty{}, nil
}

func (s *Service) ListCollections(context.Context, *emptypb.Empty) (*xvecv1.ListCollectionsResponse, error) {
	collections, err := s.registry.List()
	if err != nil {
		return nil, rpcError(err)
	}
	return &xvecv1.ListCollectionsResponse{Collections: collections}, nil
}

func (s *Service) GetSchema(ctx context.Context, request *xvecv1.CollectionRef) (*xvecv1.CollectionSchema, error) {
	if request == nil {
		return nil, invalidStatus("collection reference is nil")
	}
	collection, release, err := s.registry.Acquire(ctx, request.Collection)
	if err != nil {
		return nil, rpcError(err)
	}
	defer release()
	message, err := grpcapi.SchemaToProto(collection.Schema())
	return message, rpcError(err)
}

func (s *Service) GetStats(ctx context.Context, request *xvecv1.CollectionRef) (*xvecv1.CollectionStats, error) {
	if request == nil {
		return nil, invalidStatus("collection reference is nil")
	}
	collection, release, err := s.registry.Acquire(ctx, request.Collection)
	if err != nil {
		return nil, rpcError(err)
	}
	defer release()
	stats := collection.Stats()
	return &xvecv1.CollectionStats{
		DocumentCount: stats.DocumentCount, IndexCompleteness: stats.IndexCompleteness,
		ImmutableSegments: stats.ImmutableSegments, MutableDocuments: stats.MutableDocuments,
		DeletedDocuments: stats.DeletedDocuments, StorageMemoryBytes: stats.StorageMemoryBytes,
	}, nil
}

func (s *Service) Flush(ctx context.Context, request *xvecv1.CollectionRef) (*emptypb.Empty, error) {
	return s.withEmptyCollection(ctx, collectionName(request), func(collection *xvec.Collection) error { return collection.Flush(ctx) })
}

func (s *Service) Optimize(ctx context.Context, request *xvecv1.OptimizeRequest) (*emptypb.Empty, error) {
	if request == nil {
		return nil, invalidStatus("optimize request is nil")
	}
	concurrency, err := protoInt("concurrency", request.Concurrency)
	if err != nil {
		return nil, rpcError(err)
	}
	return s.withEmptyCollection(ctx, request.Collection, func(collection *xvec.Collection) error {
		return collection.Optimize(ctx, xvec.OptimizeOptions{Concurrency: concurrency})
	})
}

func (s *Service) AddColumn(ctx context.Context, request *xvecv1.AddColumnRequest) (*xvecv1.CollectionSchema, error) {
	if request == nil {
		return nil, invalidStatus("add column request is nil")
	}
	field, err := fieldFromProto(request.Field)
	if err != nil {
		return nil, rpcError(err)
	}
	concurrency, err := protoInt("concurrency", request.Concurrency)
	if err != nil {
		return nil, rpcError(err)
	}
	return s.mutateSchema(ctx, request.Collection, func(collection *xvec.Collection) error {
		return collection.AddColumn(ctx, field, request.Expression, xvec.AddColumnOptions{Concurrency: concurrency})
	})
}

func (s *Service) AlterColumn(ctx context.Context, request *xvecv1.AlterColumnRequest) (*xvecv1.CollectionSchema, error) {
	if request == nil {
		return nil, invalidStatus("alter column request is nil")
	}
	var field *xvec.FieldSchema
	if request.Field != nil {
		decoded, err := fieldFromProto(request.Field)
		if err != nil {
			return nil, rpcError(err)
		}
		field = &decoded
	}
	concurrency, err := protoInt("concurrency", request.Concurrency)
	if err != nil {
		return nil, rpcError(err)
	}
	return s.mutateSchema(ctx, request.Collection, func(collection *xvec.Collection) error {
		return collection.AlterColumn(ctx, request.Column, request.Rename, field, xvec.AlterColumnOptions{Concurrency: concurrency})
	})
}

func (s *Service) DropColumn(ctx context.Context, request *xvecv1.DropColumnRequest) (*xvecv1.CollectionSchema, error) {
	if request == nil {
		return nil, invalidStatus("drop column request is nil")
	}
	return s.mutateSchema(ctx, request.Collection, func(collection *xvec.Collection) error {
		return collection.DropColumn(ctx, request.Column)
	})
}

func (s *Service) CreateIndex(ctx context.Context, request *xvecv1.CreateIndexRequest) (*emptypb.Empty, error) {
	if request == nil {
		return nil, invalidStatus("create index request is nil")
	}
	index, err := grpcapi.IndexParamsFromProto(request.Index)
	if err != nil {
		return nil, rpcError(err)
	}
	concurrency, err := protoInt("concurrency", request.Concurrency)
	if err != nil {
		return nil, rpcError(err)
	}
	return s.withEmptyCollection(ctx, request.Collection, func(collection *xvec.Collection) error {
		return collection.CreateIndex(ctx, request.Field, index, xvec.CreateIndexOptions{Concurrency: concurrency})
	})
}

func (s *Service) DropIndex(ctx context.Context, request *xvecv1.DropIndexRequest) (*emptypb.Empty, error) {
	if request == nil {
		return nil, invalidStatus("drop index request is nil")
	}
	return s.withEmptyCollection(ctx, request.Collection, func(collection *xvec.Collection) error {
		return collection.DropIndex(ctx, request.Field)
	})
}

func (s *Service) Write(ctx context.Context, request *xvecv1.WriteRequest) (*xvecv1.WriteResponse, error) {
	if request == nil {
		return nil, invalidStatus("write request is nil")
	}
	if request.Operation != xvecv1.WriteOperation_WRITE_OPERATION_INSERT &&
		request.Operation != xvecv1.WriteOperation_WRITE_OPERATION_UPSERT &&
		request.Operation != xvecv1.WriteOperation_WRITE_OPERATION_UPDATE {
		return nil, invalidStatus("unknown write operation")
	}
	documents := make([]xvec.Document, len(request.Documents))
	for index := range request.Documents {
		document, err := grpcapi.DocumentFromProto(request.Documents[index])
		if err != nil {
			return nil, rpcError(err)
		}
		documents[index] = document
	}
	collection, release, err := s.registry.Acquire(ctx, request.Collection)
	if err != nil {
		return nil, rpcError(err)
	}
	defer release()
	var results []xvec.WriteResult
	switch request.Operation {
	case xvecv1.WriteOperation_WRITE_OPERATION_INSERT:
		results, err = collection.Insert(ctx, documents)
	case xvecv1.WriteOperation_WRITE_OPERATION_UPSERT:
		results, err = collection.Upsert(ctx, documents)
	case xvecv1.WriteOperation_WRITE_OPERATION_UPDATE:
		results, err = collection.Update(ctx, documents)
	}
	return writeResponse(results, err)
}

func (s *Service) Delete(ctx context.Context, request *xvecv1.DeleteRequest) (*xvecv1.WriteResponse, error) {
	if request == nil {
		return nil, invalidStatus("delete request is nil")
	}
	collection, release, err := s.registry.Acquire(ctx, request.Collection)
	if err != nil {
		return nil, rpcError(err)
	}
	defer release()
	results, err := collection.Delete(ctx, request.PrimaryKeys)
	return writeResponse(results, err)
}

func (s *Service) DeleteByFilter(ctx context.Context, request *xvecv1.DeleteByFilterRequest) (*emptypb.Empty, error) {
	if request == nil {
		return nil, invalidStatus("delete-by-filter request is nil")
	}
	return s.withEmptyCollection(ctx, request.Collection, func(collection *xvec.Collection) error {
		return collection.DeleteByFilter(ctx, request.Filter)
	})
}

func (s *Service) Fetch(request *xvecv1.FetchRequest, stream grpc.ServerStreamingServer[xvecv1.FetchResult]) error {
	if request == nil {
		return invalidStatus("fetch request is nil")
	}
	projection, err := grpcapi.ProjectionFromProto(request.Projection)
	if err != nil {
		return rpcError(err)
	}
	collection, release, err := s.registry.Acquire(stream.Context(), request.Collection)
	if err != nil {
		return rpcError(err)
	}
	defer release()
	documents, err := collection.Fetch(stream.Context(), request.PrimaryKeys, projection)
	if err != nil {
		return rpcError(err)
	}
	for index, document := range documents {
		result := &xvecv1.FetchResult{Index: uint32(index), Found: document != nil}
		if document != nil {
			result.Document, err = grpcapi.DocumentToProto(*document)
			if err != nil {
				return rpcError(err)
			}
		}
		if err := stream.Send(result); err != nil {
			return rpcError(err)
		}
	}
	return nil
}

func (s *Service) Query(request *xvecv1.QueryRequest, stream grpc.ServerStreamingServer[xvecv1.QueryResponse]) error {
	collectionName, query, err := grpcapi.QueryRequestFromProto(request)
	if err != nil {
		return rpcError(err)
	}
	collection, release, err := s.registry.Acquire(stream.Context(), collectionName)
	if err != nil {
		return rpcError(err)
	}
	defer release()
	documents, err := collection.Query(stream.Context(), query)
	if err != nil {
		return rpcError(err)
	}
	return s.sendDocuments(documents, stream.Send)
}

func (s *Service) MultiQuery(request *xvecv1.MultiQueryRequest, stream grpc.ServerStreamingServer[xvecv1.QueryResponse]) error {
	collectionName, query, err := grpcapi.MultiQueryRequestFromProto(request)
	if err != nil {
		return rpcError(err)
	}
	collection, release, err := s.registry.Acquire(stream.Context(), collectionName)
	if err != nil {
		return rpcError(err)
	}
	defer release()
	documents, err := collection.MultiQuery(stream.Context(), query)
	if err != nil {
		return rpcError(err)
	}
	return s.sendDocuments(documents, stream.Send)
}

func (s *Service) GroupByQuery(request *xvecv1.GroupByQueryRequest, stream grpc.ServerStreamingServer[xvecv1.GroupByQueryResponse]) error {
	collectionName, query, err := grpcapi.GroupByQueryRequestFromProto(request)
	if err != nil {
		return rpcError(err)
	}
	collection, release, err := s.registry.Acquire(stream.Context(), collectionName)
	if err != nil {
		return rpcError(err)
	}
	defer release()
	groups, err := collection.GroupByQuery(stream.Context(), query)
	if err != nil {
		return rpcError(err)
	}
	for start := 0; start < len(groups); start += s.options.StreamBatchSize {
		end := min(start+s.options.StreamBatchSize, len(groups))
		response := &xvecv1.GroupByQueryResponse{Groups: make([]*xvecv1.GroupResult, end-start)}
		for index := start; index < end; index++ {
			group := &xvecv1.GroupResult{Value: groups[index].Value, Documents: make([]*xvecv1.Document, len(groups[index].Documents))}
			for documentIndex := range groups[index].Documents {
				group.Documents[documentIndex], err = grpcapi.DocumentToProto(groups[index].Documents[documentIndex])
				if err != nil {
					return rpcError(err)
				}
			}
			response.Groups[index-start] = group
		}
		if err := stream.Send(response); err != nil {
			return rpcError(err)
		}
	}
	return nil
}

func (s *Service) Iterate(request *xvecv1.IterateRequest, stream grpc.ServerStreamingServer[xvecv1.QueryResponse]) error {
	if request == nil {
		return invalidStatus("iterate request is nil")
	}
	projection, err := grpcapi.ProjectionFromProto(request.Projection)
	if err != nil {
		return rpcError(err)
	}
	collection, release, err := s.registry.Acquire(stream.Context(), request.Collection)
	if err != nil {
		return rpcError(err)
	}
	defer release()
	iterator, err := collection.CreateIterator(stream.Context(), xvec.IteratorOptions{Projection: projection})
	if err != nil {
		return rpcError(err)
	}
	defer iterator.Close()
	batch := make([]*xvecv1.Document, 0, s.options.StreamBatchSize)
	for {
		if err := stream.Context().Err(); err != nil {
			return rpcError(err)
		}
		document, err := iterator.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return rpcError(err)
		}
		message, err := grpcapi.DocumentToProto(*document)
		if err != nil {
			return rpcError(err)
		}
		batch = append(batch, message)
		if len(batch) == s.options.StreamBatchSize {
			if err := stream.Send(&xvecv1.QueryResponse{Documents: batch}); err != nil {
				return rpcError(err)
			}
			batch = make([]*xvecv1.Document, 0, s.options.StreamBatchSize)
		}
	}
	if len(batch) > 0 {
		if err := stream.Send(&xvecv1.QueryResponse{Documents: batch}); err != nil {
			return rpcError(err)
		}
	}
	return nil
}

func (s *Service) mutateSchema(ctx context.Context, name string, mutate func(*xvec.Collection) error) (*xvecv1.CollectionSchema, error) {
	collection, release, err := s.registry.Acquire(ctx, name)
	if err != nil {
		return nil, rpcError(err)
	}
	defer release()
	if err := mutate(collection); err != nil {
		return nil, rpcError(err)
	}
	message, err := grpcapi.SchemaToProto(collection.Schema())
	return message, rpcError(err)
}

func (s *Service) withEmptyCollection(ctx context.Context, name string, operation func(*xvec.Collection) error) (*emptypb.Empty, error) {
	if name == "" {
		return nil, invalidStatus("collection is empty")
	}
	collection, release, err := s.registry.Acquire(ctx, name)
	if err != nil {
		return nil, rpcError(err)
	}
	defer release()
	if err := operation(collection); err != nil {
		return nil, rpcError(err)
	}
	return &emptypb.Empty{}, nil
}

func (s *Service) sendDocuments(documents []xvec.Document, send func(*xvecv1.QueryResponse) error) error {
	for start := 0; start < len(documents); start += s.options.StreamBatchSize {
		end := min(start+s.options.StreamBatchSize, len(documents))
		response := &xvecv1.QueryResponse{Documents: make([]*xvecv1.Document, end-start)}
		for index := start; index < end; index++ {
			message, err := grpcapi.DocumentToProto(documents[index])
			if err != nil {
				return rpcError(err)
			}
			response.Documents[index-start] = message
		}
		if err := send(response); err != nil {
			return rpcError(err)
		}
	}
	return nil
}

func fieldFromProto(message *xvecv1.FieldSchema) (xvec.FieldSchema, error) {
	if message == nil {
		return xvec.FieldSchema{}, &xvec.Error{Code: xvec.ErrorCodeInvalidArgument, Message: "field schema is nil"}
	}
	index, err := grpcapi.IndexParamsFromProto(message.Index)
	if err != nil {
		return xvec.FieldSchema{}, err
	}
	return xvec.FieldSchema{Name: message.Name, DataType: xvec.DataType(message.DataType), Nullable: message.Nullable, Dimension: message.Dimension, Index: index}, nil
}

func writeResponse(results []xvec.WriteResult, err error) (*xvecv1.WriteResponse, error) {
	if err == nil {
		return grpcapi.WriteResponseToProto(results, nil), nil
	}
	var batch *xvec.BatchWriteError
	if !errors.As(err, &batch) {
		return nil, rpcError(err)
	}
	cleanResults := make([]xvec.WriteResult, len(results))
	copy(cleanResults, results)
	for index := range cleanResults {
		cleanResults[index].Err = sanitizedError(cleanResults[index].Err)
	}
	causes := batch.Causes()
	for index := range causes {
		causes[index] = sanitizedError(causes[index])
	}
	return grpcapi.WriteResponseToProto(cleanResults, xvec.NewBatchWriteError(causes...)), nil
}

func collectionName(reference *xvecv1.CollectionRef) string {
	if reference == nil {
		return ""
	}
	return reference.Collection
}

func protoInt(name string, value int64) (int, error) {
	if value < 0 || (int64(int(value)) != value && (value > math.MaxInt32 || value < math.MinInt32)) {
		return 0, &xvec.Error{Code: xvec.ErrorCodeInvalidArgument, Message: name + " is out of range"}
	}
	return int(value), nil
}

func invalidStatus(message string) error { return status.Error(codes.InvalidArgument, message) }

func sanitizedError(err error) error {
	if err == nil {
		return nil
	}
	var structured *xvec.Error
	if errors.As(err, &structured) && structured != nil {
		clone := *structured
		clone.Path = ""
		clone.Err = nil
		return &clone
	}
	return &xvec.Error{Code: xvec.ErrorCodeUnknown, Message: "operation failed"}
}

func rpcError(err error) error {
	if err == nil {
		return nil
	}
	if _, ok := status.FromError(err); ok {
		return err
	}
	switch {
	case errors.Is(err, ErrInvalidCollectionName):
		return grpcapi.ErrorToStatus(&xvec.Error{Code: xvec.ErrorCodeInvalidArgument, Message: "invalid collection name"})
	case errors.Is(err, ErrCollectionExists):
		return grpcapi.ErrorToStatus(&xvec.Error{Code: xvec.ErrorCodeAlreadyExists, Message: "collection already exists"})
	case errors.Is(err, ErrCollectionUnavailable), errors.Is(err, ErrRegistryClosed):
		return grpcapi.ErrorToStatus(&xvec.Error{Code: xvec.ErrorCodeUnavailable, Message: "collection unavailable"})
	}
	return grpcapi.ErrorToStatus(sanitizedError(err))
}
