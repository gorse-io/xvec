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

	"github.com/gorse-io/xvec"
	"github.com/gorse-io/xvec/internal/grpcapi"
	xvecv1 "github.com/gorse-io/xvec/internal/proto/xvec/v1"
)

// Collection is a logical handle to a remote collection.
type Collection struct {
	name    string
	service xvecv1.XvecServiceClient
}

// Name returns the remote collection name.
func (c *Collection) Name() string {
	if c == nil {
		return ""
	}
	return c.name
}

func (c *Collection) reference() *xvecv1.CollectionRef {
	return &xvecv1.CollectionRef{Collection: c.name}
}

// Schema returns the current remote schema.
func (c *Collection) Schema(ctx context.Context) (xvec.CollectionSchema, error) {
	response, err := c.service.GetSchema(ctx, c.reference())
	if err != nil {
		return xvec.CollectionSchema{}, grpcapi.ErrorFromStatus(err)
	}
	schema, err := grpcapi.SchemaFromProto(response)
	if err != nil {
		return xvec.CollectionSchema{}, responseError(err)
	}
	return schema, nil
}

// Stats returns current remote collection statistics.
func (c *Collection) Stats(ctx context.Context) (xvec.CollectionStats, error) {
	response, err := c.service.GetStats(ctx, c.reference())
	if err != nil {
		return xvec.CollectionStats{}, grpcapi.ErrorFromStatus(err)
	}
	if response == nil {
		return xvec.CollectionStats{}, internalError("collection stats response is nil", nil)
	}
	return xvec.CollectionStats{
		DocumentCount:      response.DocumentCount,
		IndexCompleteness:  cloneMap(response.IndexCompleteness),
		ImmutableSegments:  response.ImmutableSegments,
		MutableDocuments:   response.MutableDocuments,
		DeletedDocuments:   response.DeletedDocuments,
		StorageMemoryBytes: response.StorageMemoryBytes,
	}, nil
}

func cloneMap(values map[string]float32) map[string]float32 {
	cloned := make(map[string]float32, len(values))
	for key, value := range values {
		cloned[key] = value
	}
	return cloned
}

// Flush flushes pending remote writes.
func (c *Collection) Flush(ctx context.Context) error {
	_, err := c.service.Flush(ctx, c.reference())
	return grpcapi.ErrorFromStatus(err)
}

// Destroy drops the remote collection.
func (c *Collection) Destroy(ctx context.Context) error {
	_, err := c.service.DropCollection(ctx, c.reference())
	return grpcapi.ErrorFromStatus(err)
}

// AddColumn adds and backfills a column.
func (c *Collection) AddColumn(ctx context.Context, field xvec.FieldSchema, expression string, options xvec.AddColumnOptions) error {
	schema := xvec.CollectionSchema{Fields: []xvec.FieldSchema{field}}
	message, err := grpcapi.SchemaToProto(schema)
	if err != nil {
		return err
	}
	response, err := c.service.AddColumn(ctx, &xvecv1.AddColumnRequest{Collection: c.name, Field: message.Fields[0], Expression: expression, Concurrency: int64(options.Concurrency)})
	return schemaResponseError(response, err)
}

// AlterColumn renames or replaces a column.
func (c *Collection) AlterColumn(ctx context.Context, column, rename string, field *xvec.FieldSchema, options xvec.AlterColumnOptions) error {
	var encoded *xvecv1.FieldSchema
	if field != nil {
		message, err := grpcapi.SchemaToProto(xvec.CollectionSchema{Fields: []xvec.FieldSchema{*field}})
		if err != nil {
			return err
		}
		encoded = message.Fields[0]
	}
	response, err := c.service.AlterColumn(ctx, &xvecv1.AlterColumnRequest{Collection: c.name, Column: column, Rename: rename, Field: encoded, Concurrency: int64(options.Concurrency)})
	return schemaResponseError(response, err)
}

// DropColumn removes a column.
func (c *Collection) DropColumn(ctx context.Context, column string) error {
	response, err := c.service.DropColumn(ctx, &xvecv1.DropColumnRequest{Collection: c.name, Column: column})
	return schemaResponseError(response, err)
}

func schemaResponseError(response *xvecv1.CollectionSchema, err error) error {
	if err != nil {
		return grpcapi.ErrorFromStatus(err)
	}
	_, err = grpcapi.SchemaFromProto(response)
	return responseError(err)
}

// CreateIndex creates an index on a remote field.
func (c *Collection) CreateIndex(ctx context.Context, field string, index xvec.IndexParams, options xvec.CreateIndexOptions) error {
	message, err := grpcapi.IndexParamsToProto(index)
	if err != nil {
		return err
	}
	_, err = c.service.CreateIndex(ctx, &xvecv1.CreateIndexRequest{Collection: c.name, Field: field, Index: message, Concurrency: int64(options.Concurrency)})
	return grpcapi.ErrorFromStatus(err)
}

// DropIndex removes an index from a remote field.
func (c *Collection) DropIndex(ctx context.Context, field string) error {
	_, err := c.service.DropIndex(ctx, &xvecv1.DropIndexRequest{Collection: c.name, Field: field})
	return grpcapi.ErrorFromStatus(err)
}

// Optimize optimizes remote collection segments.
func (c *Collection) Optimize(ctx context.Context, options xvec.OptimizeOptions) error {
	_, err := c.service.Optimize(ctx, &xvecv1.OptimizeRequest{Collection: c.name, Concurrency: int64(options.Concurrency)})
	return grpcapi.ErrorFromStatus(err)
}

// Insert inserts documents without retrying the write RPC.
func (c *Collection) Insert(ctx context.Context, documents []xvec.Document) ([]xvec.WriteResult, error) {
	return c.write(ctx, xvecv1.WriteOperation_WRITE_OPERATION_INSERT, documents)
}

// Upsert upserts documents without retrying the write RPC.
func (c *Collection) Upsert(ctx context.Context, documents []xvec.Document) ([]xvec.WriteResult, error) {
	return c.write(ctx, xvecv1.WriteOperation_WRITE_OPERATION_UPSERT, documents)
}

// Update updates documents without retrying the write RPC.
func (c *Collection) Update(ctx context.Context, documents []xvec.Document) ([]xvec.WriteResult, error) {
	return c.write(ctx, xvecv1.WriteOperation_WRITE_OPERATION_UPDATE, documents)
}

func (c *Collection) write(ctx context.Context, operation xvecv1.WriteOperation, documents []xvec.Document) ([]xvec.WriteResult, error) {
	request := &xvecv1.WriteRequest{Collection: c.name, Operation: operation, Documents: make([]*xvecv1.Document, len(documents))}
	for index := range documents {
		message, err := grpcapi.DocumentToProto(documents[index])
		if err != nil {
			return nil, err
		}
		request.Documents[index] = message
	}
	response, err := c.service.Write(ctx, request)
	if err != nil {
		return nil, grpcapi.ErrorFromStatus(err)
	}
	if response == nil {
		return nil, internalError("write response is nil", nil)
	}
	results, aggregate := grpcapi.WriteResponseFromProto(response)
	if len(results) != len(documents) {
		return nil, internalError("write result count does not match request", nil)
	}
	return results, aggregate
}

// Delete deletes primary keys without retrying the write RPC.
func (c *Collection) Delete(ctx context.Context, primaryKeys []string) ([]xvec.WriteResult, error) {
	response, err := c.service.Delete(ctx, &xvecv1.DeleteRequest{Collection: c.name, PrimaryKeys: append([]string(nil), primaryKeys...)})
	if err != nil {
		return nil, grpcapi.ErrorFromStatus(err)
	}
	if response == nil {
		return nil, internalError("delete response is nil", nil)
	}
	results, aggregate := grpcapi.WriteResponseFromProto(response)
	if len(results) != len(primaryKeys) {
		return nil, internalError("delete result count does not match request", nil)
	}
	return results, aggregate
}

// DeleteByFilter deletes all matching documents.
func (c *Collection) DeleteByFilter(ctx context.Context, filter string) error {
	_, err := c.service.DeleteByFilter(ctx, &xvecv1.DeleteByFilterRequest{Collection: c.name, Filter: filter})
	return grpcapi.ErrorFromStatus(err)
}

// Fetch returns results in requested-key order with nil entries for misses.
func (c *Collection) Fetch(ctx context.Context, primaryKeys []string, projection xvec.Projection) ([]*xvec.Document, error) {
	streamContext, cancel := context.WithCancel(ctx)
	defer cancel()
	stream, err := c.service.Fetch(streamContext, &xvecv1.FetchRequest{Collection: c.name, PrimaryKeys: append([]string(nil), primaryKeys...), Projection: grpcapi.ProjectionToProto(projection)})
	if err != nil {
		return nil, grpcapi.ErrorFromStatus(err)
	}
	results := make([]*xvec.Document, len(primaryKeys))
	seen := make([]bool, len(primaryKeys))
	for {
		message, err := stream.Recv()
		if err == io.EOF {
			for _, present := range seen {
				if !present {
					return nil, internalError("fetch stream ended before every result was received", nil)
				}
			}
			return results, nil
		}
		if err != nil {
			return nil, grpcapi.ErrorFromStatus(err)
		}
		if message == nil || uint64(message.Index) >= uint64(len(results)) {
			return nil, internalError("fetch result index is out of range", nil)
		}
		index := int(message.Index)
		if seen[index] {
			return nil, internalError("fetch result index is duplicated", nil)
		}
		seen[index] = true
		if message.Found != (message.Document != nil) {
			return nil, internalError("fetch result found flag and document disagree", nil)
		}
		if message.Found {
			document, decodeErr := grpcapi.DocumentFromProto(message.Document)
			if decodeErr != nil {
				return nil, responseError(decodeErr)
			}
			results[index] = &document
		}
	}
}

// Query runs and aggregates a streamed vector query.
func (c *Collection) Query(ctx context.Context, query xvec.VectorQuery) ([]xvec.Document, error) {
	request, err := grpcapi.QueryRequestToProto(c.name, query)
	if err != nil {
		return nil, err
	}
	streamContext, cancel := context.WithCancel(ctx)
	defer cancel()
	stream, err := c.service.Query(streamContext, request)
	if err != nil {
		return nil, grpcapi.ErrorFromStatus(err)
	}
	return receiveDocuments(stream)
}

// MultiQuery runs and aggregates a streamed multi-query.
func (c *Collection) MultiQuery(ctx context.Context, query xvec.MultiQuery) ([]xvec.Document, error) {
	request, err := grpcapi.MultiQueryRequestToProto(c.name, query)
	if err != nil {
		return nil, err
	}
	streamContext, cancel := context.WithCancel(ctx)
	defer cancel()
	stream, err := c.service.MultiQuery(streamContext, request)
	if err != nil {
		return nil, grpcapi.ErrorFromStatus(err)
	}
	return receiveDocuments(stream)
}

type queryReceiver interface {
	Recv() (*xvecv1.QueryResponse, error)
}

func receiveDocuments(stream queryReceiver) ([]xvec.Document, error) {
	var results []xvec.Document
	for {
		message, err := stream.Recv()
		if err == io.EOF {
			return results, nil
		}
		if err != nil {
			return nil, grpcapi.ErrorFromStatus(err)
		}
		if message == nil {
			return nil, internalError("query stream payload is nil", nil)
		}
		for _, encoded := range message.Documents {
			document, decodeErr := grpcapi.DocumentFromProto(encoded)
			if decodeErr != nil {
				return nil, responseError(decodeErr)
			}
			results = append(results, document)
		}
	}
}

// GroupByQuery runs and aggregates a streamed group-by query.
func (c *Collection) GroupByQuery(ctx context.Context, query xvec.GroupByVectorQuery) ([]xvec.GroupResult, error) {
	request, err := grpcapi.GroupByQueryRequestToProto(c.name, query)
	if err != nil {
		return nil, err
	}
	streamContext, cancel := context.WithCancel(ctx)
	defer cancel()
	stream, err := c.service.GroupByQuery(streamContext, request)
	if err != nil {
		return nil, grpcapi.ErrorFromStatus(err)
	}
	var results []xvec.GroupResult
	for {
		message, recvErr := stream.Recv()
		if recvErr == io.EOF {
			return results, nil
		}
		if recvErr != nil {
			return nil, grpcapi.ErrorFromStatus(recvErr)
		}
		if message == nil {
			return nil, internalError("group-by stream payload is nil", nil)
		}
		for _, encoded := range message.Groups {
			if encoded == nil {
				return nil, internalError("group-by result is nil", nil)
			}
			group := xvec.GroupResult{Value: encoded.Value, Documents: make([]xvec.Document, len(encoded.Documents))}
			for index, encodedDocument := range encoded.Documents {
				document, decodeErr := grpcapi.DocumentFromProto(encodedDocument)
				if decodeErr != nil {
					return nil, responseError(decodeErr)
				}
				group.Documents[index] = document
			}
			results = append(results, group)
		}
	}
}
