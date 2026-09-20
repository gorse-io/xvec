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

// Package client provides the remote gRPC client for xvec.
package client

import (
	"context"
	"sync"

	"github.com/gorse-io/xvec"
	"github.com/gorse-io/xvec/internal/grpcapi"
	xvecv1 "github.com/gorse-io/xvec/internal/proto/xvec/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/types/known/emptypb"
)

const defaultMaxMessageSize = 64 << 20

// DialOption is a gRPC dial option. Later options may override the insecure
// transport credentials installed by Dial.
type DialOption = grpc.DialOption

// CreateCollectionOptions controls remote collection creation.
type CreateCollectionOptions struct {
	// EnableMmap overrides the server default when non-nil.
	EnableMmap *bool
}

// Client is a remote xvec service client.
type Client struct {
	conn     grpc.ClientConnInterface
	service  xvecv1.XvecServiceClient
	close    func() error
	once     sync.Once
	closeErr error
}

// Dial connects to target. Version 1 uses an insecure transport by default.
func Dial(ctx context.Context, target string, options ...DialOption) (*Client, error) {
	dialOptions := []grpc.DialOption{
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(defaultMaxMessageSize),
			grpc.MaxCallSendMsgSize(defaultMaxMessageSize),
		),
	}
	dialOptions = append(dialOptions, options...)
	// DialContext is required to preserve context-aware dialing and WithBlock semantics.
	//nolint:staticcheck // Supported throughout gRPC 1.x; grpc.NewClient has different semantics.
	conn, err := grpc.DialContext(ctx, target, dialOptions...)
	if err != nil {
		return nil, grpcapi.ErrorFromStatus(err)
	}
	client := NewClient(conn)
	client.close = conn.Close
	return client, nil
}

// NewClient wraps a caller-owned gRPC connection. Close does not close it.
func NewClient(conn grpc.ClientConnInterface) *Client {
	return &Client{conn: conn, service: xvecv1.NewXvecServiceClient(conn)}
}

// Close closes a connection created by Dial. For NewClient it is a no-op.
func (c *Client) Close() error {
	if c == nil {
		return nil
	}
	c.once.Do(func() {
		if c.close != nil {
			c.closeErr = c.close()
		}
	})
	return c.closeErr
}

// ListCollections lists remote collection names in server order.
func (c *Client) ListCollections(ctx context.Context) ([]string, error) {
	response, err := c.service.ListCollections(ctx, &emptypb.Empty{})
	if err != nil {
		return nil, grpcapi.ErrorFromStatus(err)
	}
	if response == nil {
		return nil, internalError("list collections response is nil", nil)
	}
	return append([]string(nil), response.Collections...), nil
}

// CreateAndOpen creates a remote collection and returns its handle.
func (c *Client) CreateAndOpen(ctx context.Context, schema xvec.CollectionSchema, options CreateCollectionOptions) (*Collection, error) {
	message, err := grpcapi.SchemaToProto(schema)
	if err != nil {
		return nil, err
	}
	response, err := c.service.CreateCollection(ctx, &xvecv1.CreateCollectionRequest{Schema: message, EnableMmap: options.EnableMmap})
	if err != nil {
		return nil, grpcapi.ErrorFromStatus(err)
	}
	decoded, err := grpcapi.SchemaFromProto(response)
	if err != nil {
		return nil, responseError(err)
	}
	return c.collection(decoded.Name), nil
}

// Open opens a logical handle to an existing remote collection. The server is
// contacted to verify that the collection exists.
func (c *Client) Open(ctx context.Context, name string) (*Collection, error) {
	response, err := c.service.GetSchema(ctx, &xvecv1.CollectionRef{Collection: name})
	if err != nil {
		return nil, grpcapi.ErrorFromStatus(err)
	}
	decoded, err := grpcapi.SchemaFromProto(response)
	if err != nil {
		return nil, responseError(err)
	}
	if decoded.Name != "" && decoded.Name != name {
		return nil, internalError("schema name does not match requested collection", nil)
	}
	return c.collection(name), nil
}

func (c *Client) collection(name string) *Collection {
	return &Collection{name: name, service: c.service}
}

func internalError(message string, cause error) error {
	return &xvec.Error{Code: xvec.ErrorCodeInternal, Op: "decode remote response", Message: message, Err: cause}
}

func responseError(err error) error {
	if err == nil {
		return nil
	}
	return internalError("remote response is malformed", err)
}
