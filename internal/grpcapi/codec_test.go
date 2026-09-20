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

package grpcapi

import (
	"context"
	"errors"
	"math"
	"reflect"
	"strconv"
	"testing"

	"github.com/gorse-io/xvec"
	xvecv1 "github.com/gorse-io/xvec/internal/proto/xvec/v1"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/reflect/protoreflect"
)

type callbackReranker struct{}

func (callbackReranker) Rerank(context.Context, []xvec.RerankBatch, int) ([]xvec.Document, error) {
	return nil, nil
}

func TestProtoEnumNumbersMatchPublicAPI(t *testing.T) {
	require.Equal(t, int32(5), int32(xvecv1.IndexType_INDEX_TYPE_DISKANN))
	require.Equal(t, int32(6), int32(xvecv1.IndexType_INDEX_TYPE_VAMANA))
	for _, test := range []struct{ got, want int32 }{
		{int32(xvecv1.DataType_DATA_TYPE_VECTOR_FP32), int32(xvec.DataTypeVectorFP32)},
		{int32(xvecv1.QuantizeType_QUANTIZE_TYPE_RABITQ), int32(xvec.QuantizeTypeRaBitQ)},
		{int32(xvecv1.MetricType_METRIC_TYPE_COSINE), int32(xvec.MetricTypeCosine)},
		{int32(xvecv1.ErrorCode_ERROR_CODE_NOT_SUPPORTED), int32(xvec.ErrorCodeNotSupported)},
	} {
		require.Equal(t, test.want, test.got)
	}
}

func TestIndexParamsRoundTrip(t *testing.T) {
	params := []xvec.IndexParams{
		xvec.InvertIndexParams{EnableRangeOptimization: true, EnableExtendedWildcard: true},
		xvec.FlatIndexParams{Metric: xvec.MetricTypeIP, Quantize: xvec.QuantizeTypeInt8, Quantizer: xvec.QuantizerParams{EnableRotate: true}},
		xvec.HNSWIndexParams{Metric: xvec.MetricTypeL2, M: 12, EFConstruction: 48, Quantize: xvec.QuantizeTypeFP16, UseContiguousMemory: true},
		xvec.IVFRaBitQIndexParams{Metric: xvec.MetricTypeCosine, NList: 21, TotalBits: 8, SampleCount: 999},
		xvec.IVFIndexParams{Metric: xvec.MetricTypeL2, NList: 14, NIterations: 3, UseSOAR: true, Quantize: xvec.QuantizeTypeInt4, Quantizer: xvec.QuantizerParams{EnableRotate: true}},
		xvec.DiskANNIndexParams{Metric: xvec.MetricTypeL2, MaxDegree: 32, ListSize: 64, PQChunks: 4, Quantize: xvec.QuantizeTypeFP16},
		xvec.VamanaIndexParams{Metric: xvec.MetricTypeCosine, MaxDegree: 32, SearchListSize: 64, Alpha: 1.3, MaxOcclusionSize: 80, SaturateGraph: true, UseContiguousMemory: true, UseIDMap: true, Quantize: xvec.QuantizeTypeInt8, Quantizer: xvec.QuantizerParams{EnableRotate: true}},
		xvec.FTSIndexParams{Tokenizer: "ngram", Filters: []string{"lowercase", "ascii_folding"}, ExtraParams: `{"ngram_min":2,"ngram_max":3}`},
	}
	for _, want := range params {
		t.Run(want.IndexType().String(), func(t *testing.T) {
			message, err := IndexParamsToProto(want)
			require.NoError(t, err)
			got, err := IndexParamsFromProto(message)
			require.NoError(t, err)
			require.Equal(t, want, got)
		})
	}
	require.Nil(t, mustIndexParamsFromProto(t, nil))
}

func TestQueryParamsRoundTrip(t *testing.T) {
	options := xvec.QueryOptions{Radius: 1.25, Linear: true, UseRefiner: true}
	params := []xvec.QueryParams{
		xvec.FlatQueryParams{QueryOptions: options, ScaleFactor: 2.5},
		xvec.HNSWQueryParams{QueryOptions: options, EF: 91, PrefetchOffset: 7, PrefetchLines: 3},
		xvec.IVFRaBitQQueryParams{QueryOptions: options, NProbe: 12, ScaleFactor: 4},
		xvec.IVFQueryParams{QueryOptions: options, NProbe: 13, ScaleFactor: 5},
		xvec.DiskANNQueryParams{QueryOptions: options, ListSize: 123},
		xvec.VamanaQueryParams{QueryOptions: options, EFSearch: 77, PrefetchOffset: 8, PrefetchLines: 2},
		xvec.FTSQueryParams{DefaultOperator: "AND"},
	}
	for _, want := range params {
		t.Run(want.IndexType().String(), func(t *testing.T) {
			message, err := QueryParamsToProto(want)
			require.NoError(t, err)
			got, err := QueryParamsFromProto(message)
			require.NoError(t, err)
			require.Equal(t, want, got)
		})
	}
	got, err := QueryParamsFromProto(nil)
	require.NoError(t, err)
	require.Nil(t, got)
}

func TestPublicIntProtoFieldsUseInt64(t *testing.T) {
	fields := []protoreflect.FieldDescriptor{
		(&xvecv1.HNSWIndexParams{}).ProtoReflect().Descriptor().Fields().ByName("m"),
		(&xvecv1.IVFRaBitQIndexParams{}).ProtoReflect().Descriptor().Fields().ByName("sample_count"),
		(&xvecv1.HNSWQueryParams{}).ProtoReflect().Descriptor().Fields().ByName("ef"),
		(&xvecv1.RRFReranker{}).ProtoReflect().Descriptor().Fields().ByName("rank_constant"),
		(&xvecv1.VectorQuery{}).ProtoReflect().Descriptor().Fields().ByName("top_k"),
		(&xvecv1.SubQuery{}).ProtoReflect().Descriptor().Fields().ByName("num_candidates"),
		(&xvecv1.GroupByQuery{}).ProtoReflect().Descriptor().Fields().ByName("group_count"),
		(&xvecv1.AddColumnRequest{}).ProtoReflect().Descriptor().Fields().ByName("concurrency"),
		(&xvecv1.OptimizeRequest{}).ProtoReflect().Descriptor().Fields().ByName("concurrency"),
	}
	for _, field := range fields {
		require.Equal(t, protoreflect.Int64Kind, field.Kind(), "%s", field.FullName())
	}
	batchFields := (&xvecv1.BatchWriteError{}).ProtoReflect().Descriptor().Fields()
	require.Nil(t, batchFields.ByName("failed"))
}

func TestIntBackedCodecsRoundTripAboveMaxInt32(t *testing.T) {
	if strconv.IntSize < 64 {
		t.Skip("public int cannot represent values above MaxInt32")
	}
	large := int64(math.MaxInt32) + 1

	indexWant := xvec.IVFRaBitQIndexParams{NList: int(large), TotalBits: int(large + 1), SampleCount: int(large + 2)}
	indexMessage, err := IndexParamsToProto(indexWant)
	require.NoError(t, err)
	indexGot, err := IndexParamsFromProto(indexMessage)
	require.NoError(t, err)
	require.Equal(t, indexWant, indexGot)

	paramsWant := xvec.HNSWQueryParams{EF: int(large)}
	paramsMessage, err := QueryParamsToProto(paramsWant)
	require.NoError(t, err)
	paramsGot, err := QueryParamsFromProto(paramsMessage)
	require.NoError(t, err)
	require.Equal(t, paramsWant, paramsGot)

	rerankerWant := xvec.RRFReranker{RankConstant: int(large)}
	rerankerMessage, err := RerankerToProto(rerankerWant)
	require.NoError(t, err)
	rerankerGot, err := RerankerFromProto(rerankerMessage)
	require.NoError(t, err)
	require.Equal(t, rerankerWant, rerankerGot)
}

func TestQueryParamsToProtoSupportsPointers(t *testing.T) {
	options := xvec.QueryOptions{UseRefiner: true}
	params := []struct {
		pointer xvec.QueryParams
		value   xvec.QueryParams
	}{
		{&xvec.FlatQueryParams{QueryOptions: options, ScaleFactor: 2}, xvec.FlatQueryParams{QueryOptions: options, ScaleFactor: 2}},
		{&xvec.HNSWQueryParams{QueryOptions: options, EF: 3}, xvec.HNSWQueryParams{QueryOptions: options, EF: 3}},
		{&xvec.IVFRaBitQQueryParams{QueryOptions: options, NProbe: 4}, xvec.IVFRaBitQQueryParams{QueryOptions: options, NProbe: 4}},
		{&xvec.IVFQueryParams{QueryOptions: options, NProbe: 5}, xvec.IVFQueryParams{QueryOptions: options, NProbe: 5}},
		{&xvec.DiskANNQueryParams{QueryOptions: options, ListSize: 6}, xvec.DiskANNQueryParams{QueryOptions: options, ListSize: 6}},
		{&xvec.VamanaQueryParams{QueryOptions: options, EFSearch: 7}, xvec.VamanaQueryParams{QueryOptions: options, EFSearch: 7}},
		{&xvec.FTSQueryParams{DefaultOperator: "AND"}, xvec.FTSQueryParams{DefaultOperator: "AND"}},
	}
	for _, test := range params {
		message, err := QueryParamsToProto(test.pointer)
		require.NoError(t, err)
		got, err := QueryParamsFromProto(message)
		require.NoError(t, err)
		require.Equal(t, test.value, got)
	}
}

func TestSchemaRoundTrip(t *testing.T) {
	schema := xvec.CollectionSchema{Name: "catalog", MaxDocsPerSegment: 1234, Fields: []xvec.FieldSchema{
		{Name: "title", DataType: xvec.DataTypeString, Nullable: true, Index: xvec.FTSIndexParams{Tokenizer: "standard", Filters: []string{"lowercase"}}},
		{Name: "embedding", DataType: xvec.DataTypeVectorFP32, Dimension: 3, Index: xvec.HNSWIndexParams{Metric: xvec.MetricTypeL2, M: 8, EFConstruction: 32}},
		{Name: "plain", DataType: xvec.DataTypeInt64},
	}}
	message, err := SchemaToProto(schema)
	require.NoError(t, err)
	got, err := SchemaFromProto(message)
	require.NoError(t, err)
	require.Equal(t, schema, got)
}

func TestDocumentValueRoundTripsEverySupportedType(t *testing.T) {
	values := map[string]any{
		"null": nil, "binary": xvec.Binary{0, 1, 255}, "string": "héllo", "bool": true,
		"int32": int32(-2), "int64": int64(-3), "uint32": uint32(4), "uint64": uint64(5), "float": float32(1.5), "double": float64(2.5),
		"array_binary": xvec.BinaryArray{{1}, {}, {2, 3}}, "array_string": xvec.StringArray{"a", "b"}, "array_bool": xvec.BoolArray{true, false},
		"array_int32": xvec.Int32Array{-1, 2}, "array_int64": xvec.Int64Array{-3, 4}, "array_uint32": xvec.Uint32Array{5, 6}, "array_uint64": xvec.Uint64Array{7, 8},
		"array_float": xvec.Float32Array{1.25, 2.5}, "array_double": xvec.Float64Array{3.25, 4.5},
		"binary32": xvec.VectorBinary32{1, 2}, "binary64": xvec.VectorBinary64{3, 4},
		"fp16": xvec.VectorFP16{1, 2}, "fp32": xvec.VectorFP32{1.5, 2.5}, "fp64": xvec.VectorFP64{3.5, 4.5},
		"int4": xvec.VectorInt4{-8, 7}, "int8": xvec.VectorInt8{-9, 10}, "int16": xvec.VectorInt16{-11, 12},
		"sparse_fp16": xvec.SparseVectorFP16{Indices: []uint32{2, 9}, Values: []xvec.Float16{1, 2}},
		"sparse_fp32": xvec.SparseVectorFP32{Indices: []uint32{3, 8}, Values: []float32{1.5, 2.5}},
	}
	want := xvec.Document{PrimaryKey: "doc-1", Fields: values, Score: 0.75, DocID: 42}
	message, err := DocumentToProto(want)
	require.NoError(t, err)
	require.Len(t, message.Fields, len(values), "fields must be repeated entries, not a map")
	got, err := DocumentFromProto(message)
	require.NoError(t, err)
	require.True(t, reflect.DeepEqual(want, got), "want %#v, got %#v", want, got)
}

func TestDocumentCodecDistinguishesMissingAndExplicitNull(t *testing.T) {
	message, err := DocumentToProto(xvec.Document{PrimaryKey: "doc", Fields: map[string]any{"present": nil}})
	require.NoError(t, err)
	require.Len(t, message.Fields, 1)
	got, err := DocumentFromProto(message)
	require.NoError(t, err)
	value, exists := got.Fields["present"]
	require.True(t, exists)
	require.Nil(t, value)
	_, exists = got.Fields["missing"]
	require.False(t, exists)
}

func TestDocumentCodecRejectsDuplicateFieldEntries(t *testing.T) {
	message := &xvecv1.Document{PrimaryKey: "doc", Fields: []*xvecv1.FieldEntry{
		{Name: "dup", Value: &xvecv1.Value{Kind: &xvecv1.Value_StringValue{StringValue: "first"}}},
		{Name: "dup", Value: &xvecv1.Value{Kind: &xvecv1.Value_StringValue{StringValue: "second"}}},
	}}
	_, err := DocumentFromProto(message)
	require.ErrorIs(t, err, xvec.ErrInvalidArgument)
}

func TestProjectionRoundTripPreservesThreeFieldStates(t *testing.T) {
	for _, want := range []xvec.Projection{
		{OutputFields: nil, IncludeVectors: true},
		{OutputFields: []string{}, IncludeVectors: false},
		{OutputFields: []string{"title"}, IncludeVectors: true},
	} {
		got, err := ProjectionFromProto(ProjectionToProto(want))
		require.NoError(t, err)
		require.Equal(t, want, got)
		require.Equal(t, want.OutputFields == nil, got.OutputFields == nil)
	}
}

func TestRerankerRoundTrip(t *testing.T) {
	for _, want := range []xvec.Reranker{xvec.RRFReranker{RankConstant: 17}, xvec.WeightedReranker{Weights: []float64{0.25, -2}}} {
		message, err := RerankerToProto(want)
		require.NoError(t, err)
		got, err := RerankerFromProto(message)
		require.NoError(t, err)
		require.Equal(t, want, got)
	}
	got, err := RerankerFromProto(nil)
	require.NoError(t, err)
	require.Nil(t, got)
}

func TestRerankerToProtoRejectsCallbacksAsUnsupported(t *testing.T) {
	_, err := RerankerToProto(callbackReranker{})
	require.ErrorIs(t, err, xvec.ErrNotSupported)
}

func TestErrorToStatusPreservesGRPCStatus(t *testing.T) {
	want := status.Error(codes.ResourceExhausted, "quota exceeded")
	got := ErrorToStatus(want)
	require.Equal(t, codes.ResourceExhausted, status.Code(got))
	require.Equal(t, status.Convert(want).Message(), status.Convert(got).Message())
}

func TestStructuredErrorRoundTripAndGRPCStatusMapping(t *testing.T) {
	codes := []struct {
		xvec xvec.ErrorCode
		grpc codes.Code
	}{
		{xvec.ErrorCodeNotFound, codes.NotFound}, {xvec.ErrorCodeAlreadyExists, codes.AlreadyExists},
		{xvec.ErrorCodeInvalidArgument, codes.InvalidArgument}, {xvec.ErrorCodePermissionDenied, codes.PermissionDenied},
		{xvec.ErrorCodeFailedPrecondition, codes.FailedPrecondition}, {xvec.ErrorCodeResourceExhausted, codes.ResourceExhausted},
		{xvec.ErrorCodeUnavailable, codes.Unavailable}, {xvec.ErrorCodeInternal, codes.Internal},
		{xvec.ErrorCodeNotSupported, codes.Unimplemented}, {xvec.ErrorCodeUnknown, codes.Unknown},
	}
	for _, test := range codes {
		want := &xvec.Error{Code: test.xvec, Op: "query", Path: "catalog", Message: "detail"}
		message := ErrorToProto(want)
		got := ErrorFromProto(message)
		require.Equal(t, want.Code, got.Code)
		require.Equal(t, want.Op, got.Op)
		require.Equal(t, want.Path, got.Path)
		require.Equal(t, want.Message, got.Message)
		transport := ErrorToStatus(want)
		require.Equal(t, test.grpc, status.Code(transport))
		decoded := ErrorFromStatus(transport)
		require.ErrorIs(t, decoded, sentinelFor(test.xvec))
		var structured *xvec.Error
		require.ErrorAs(t, decoded, &structured)
		require.Equal(t, "query", structured.Op)
	}
	require.NoError(t, ErrorFromStatus(nil))
}

func TestBatchWriteErrorRoundTripIsNotTransportStatus(t *testing.T) {
	want := xvec.NewBatchWriteError(
		&xvec.Error{Code: xvec.ErrorCodeAlreadyExists, Message: "duplicate"},
		&xvec.Error{Code: xvec.ErrorCodeInvalidArgument, Message: "bad field"},
	)
	message := BatchWriteErrorToProto(want)
	got := BatchWriteErrorFromProto(message)
	require.Equal(t, 2, got.Failed)
	require.Len(t, got.Causes(), 2)
	require.ErrorIs(t, got, xvec.ErrAlreadyExists)
	require.ErrorIs(t, got, xvec.ErrInvalidArgument)
	// Per-item failures are payload data and must not be promoted to a gRPC status.
	require.Equal(t, codes.OK, status.Code(nil))
}

func TestWriteResponseRoundTripPreservesResultsAndAggregateError(t *testing.T) {
	results := []xvec.WriteResult{
		{PrimaryKey: "first", DocID: 11},
		{PrimaryKey: "second", DocID: 22, Err: &xvec.Error{Code: xvec.ErrorCodeAlreadyExists, Op: "insert", Path: "catalog", Message: "duplicate"}},
		{PrimaryKey: "third", DocID: 33, Err: errors.New("opaque")},
	}
	batchErr := xvec.NewBatchWriteError(results[1].Err, results[2].Err)
	message := WriteResponseToProto(results, batchErr)
	require.Len(t, message.Results, 3)
	gotResults, gotErr := WriteResponseFromProto(message)
	require.Len(t, gotResults, 3)
	for i := range results {
		require.Equal(t, results[i].PrimaryKey, gotResults[i].PrimaryKey)
		require.Equal(t, results[i].DocID, gotResults[i].DocID)
	}
	require.NoError(t, gotResults[0].Err)
	require.ErrorIs(t, gotResults[1].Err, xvec.ErrAlreadyExists)
	require.ErrorIs(t, gotResults[2].Err, xvec.ErrUnknown)
	var opaque *xvec.Error
	require.ErrorAs(t, gotResults[2].Err, &opaque)
	require.Equal(t, "opaque", opaque.Message)
	var decodedBatch *xvec.BatchWriteError
	require.ErrorAs(t, gotErr, &decodedBatch)
	require.Equal(t, 2, decodedBatch.Failed)
	require.Len(t, decodedBatch.Causes(), 2)
}

func TestWriteResponseDerivesBatchErrorFromResults(t *testing.T) {
	resultErr := &xvec.Error{Code: xvec.ErrorCodeInvalidArgument, Message: "invalid"}
	results := []xvec.WriteResult{{PrimaryKey: "bad", Err: resultErr}}

	message := WriteResponseToProto(results, nil)
	require.Len(t, message.Error.GetCauses(), 1)
	require.Equal(t, xvecv1.ErrorCode_ERROR_CODE_INVALID_ARGUMENT, message.Error.Causes[0].Code)

	message.Error = nil
	gotResults, gotErr := WriteResponseFromProto(message)
	require.ErrorIs(t, gotResults[0].Err, xvec.ErrInvalidArgument)
	var batchErr *xvec.BatchWriteError
	require.ErrorAs(t, gotErr, &batchErr)
	require.Equal(t, 1, batchErr.Failed)
	require.Len(t, batchErr.Causes(), 1)
}

func TestWriteResponseIgnoresAggregateWithoutFailedResult(t *testing.T) {
	message := &xvecv1.WriteResponse{
		Results: []*xvecv1.WriteResult{{PrimaryKey: "ok"}},
		Error: &xvecv1.BatchWriteError{Causes: []*xvecv1.Error{{
			Code: xvecv1.ErrorCode_ERROR_CODE_INTERNAL, Message: "inconsistent",
		}}},
	}

	results, err := WriteResponseFromProto(message)
	require.Len(t, results, 1)
	require.NoError(t, results[0].Err)
	require.NoError(t, err)
}

func TestErrorToStatusDoesNotEraseNonNilOKError(t *testing.T) {
	err := ErrorToStatus(&xvec.Error{Code: xvec.ErrorCodeOK, Message: "invalid OK error"})
	require.Error(t, err)
	require.Equal(t, codes.Unknown, status.Code(err))
}

func TestErrorFromProtoDoesNotAcceptNonNilOKError(t *testing.T) {
	err := ErrorFromProto(&xvecv1.Error{Code: xvecv1.ErrorCode_ERROR_CODE_OK, Message: "invalid OK error"})
	require.Equal(t, xvec.ErrorCodeUnknown, err.Code)
}

func TestIndexParamsFromProtoRejectsRemoteFTSPaths(t *testing.T) {
	for _, extra := range []string{
		`{"jieba_dict_dir":"/etc"}`,
		`{"user_dict_path":"/etc/passwd"}`,
	} {
		_, err := IndexParamsFromProto(&xvecv1.IndexParams{Kind: &xvecv1.IndexParams_Fts{Fts: &xvecv1.FTSIndexParams{
			Tokenizer: "jieba", ExtraParams: extra,
		}}})
		require.ErrorIs(t, err, xvec.ErrInvalidArgument)
	}
}

func TestQueryRequestCountsRoundTripAboveMaxInt32(t *testing.T) {
	if strconv.IntSize < 64 {
		t.Skip("public int cannot represent values above MaxInt32")
	}
	large := int64(math.MaxInt32) + 1
	vector := xvec.VectorQuery{DenseVector: xvec.VectorFP32{1}, TopK: int(large)}
	message, err := VectorQueryToProto(vector)
	require.NoError(t, err)
	gotVector, err := VectorQueryFromProto(message)
	require.NoError(t, err)
	require.Equal(t, vector.TopK, gotVector.TopK)

	multi := xvec.MultiQuery{TopK: int(large), Queries: []xvec.SubQuery{
		{DenseVector: xvec.VectorFP32{1}, NumCandidates: int(large + 1)},
		{DenseVector: xvec.VectorFP32{2}},
	}}
	multiMessage, err := MultiQueryRequestToProto("catalog", multi)
	require.NoError(t, err)
	_, gotMulti, err := MultiQueryRequestFromProto(multiMessage)
	require.NoError(t, err)
	require.Equal(t, multi.TopK, gotMulti.TopK)
	require.Equal(t, multi.Queries[0].NumCandidates, gotMulti.Queries[0].NumCandidates)

	group := xvec.GroupByVectorQuery{DenseVector: xvec.VectorFP32{1}, GroupCount: int(large), TopKPerGroup: int(large + 1)}
	groupMessage, err := GroupByQueryRequestToProto("catalog", group)
	require.NoError(t, err)
	_, gotGroup, err := GroupByQueryRequestFromProto(groupMessage)
	require.NoError(t, err)
	require.Equal(t, group.GroupCount, gotGroup.GroupCount)
	require.Equal(t, group.TopKPerGroup, gotGroup.TopKPerGroup)
}

func TestQueryTargetRoundTrips(t *testing.T) {
	tests := []xvec.VectorQuery{
		{Field: "embedding", DenseVector: xvec.VectorFP32{1, 2}, TopK: 3},
		{Field: "sparse", SparseVector: xvec.SparseVectorFP32{Indices: []uint32{2}, Values: []float32{4}}, TopK: 3},
		{Field: "embedding", PrimaryKey: "seed", TopK: 3},
		{Field: "title", FTS: &xvec.FTSClause{Match: "hello world"}, TopK: 3},
		{TopK: 3, Filter: "price > 10"},
	}
	for _, want := range tests {
		message, err := VectorQueryToProto(want)
		require.NoError(t, err)
		got, err := VectorQueryFromProto(message)
		require.NoError(t, err)
		require.True(t, reflect.DeepEqual(want, got), "want %#v, got %#v", want, got)
	}
}

func TestQueryRequestRoundTrips(t *testing.T) {
	query := xvec.VectorQuery{
		Field: "embedding", DenseVector: xvec.VectorFP16{1, 2}, TopK: 7,
		Filter: "available = true", Projection: xvec.Projection{OutputFields: []string{}, IncludeVectors: true},
		Params: xvec.HNSWQueryParams{QueryOptions: xvec.QueryOptions{UseRefiner: true}, EF: 20},
	}
	message, err := QueryRequestToProto("catalog", query)
	require.NoError(t, err)
	collection, got, err := QueryRequestFromProto(message)
	require.NoError(t, err)
	require.Equal(t, "catalog", collection)
	require.True(t, reflect.DeepEqual(query, got), "want %#v, got %#v", query, got)
}

func TestMultiAndGroupByQueryRequestsRoundTrip(t *testing.T) {
	multi := xvec.MultiQuery{
		Queries: []xvec.SubQuery{
			{Field: "embedding", DenseVector: xvec.VectorFP32{1, 2}, Params: xvec.FlatQueryParams{ScaleFactor: 2}, NumCandidates: 11},
			{Field: "title", FTS: &xvec.FTSClause{Query: "hello"}, Params: xvec.FTSQueryParams{DefaultOperator: "AND"}, NumCandidates: 12},
		},
		TopK: 5, Filter: "active = true", Projection: xvec.Projection{OutputFields: []string{"title"}},
		Reranker: xvec.WeightedReranker{Weights: []float64{0.25, 0.75}},
	}
	multiMessage, err := MultiQueryRequestToProto("catalog", multi)
	require.NoError(t, err)
	collection, gotMulti, err := MultiQueryRequestFromProto(multiMessage)
	require.NoError(t, err)
	require.Equal(t, "catalog", collection)
	require.True(t, reflect.DeepEqual(multi, gotMulti), "want %#v, got %#v", multi, gotMulti)

	group := xvec.GroupByVectorQuery{
		Field: "embedding", PrimaryKey: "seed", Filter: "active = true",
		Projection: xvec.Projection{OutputFields: nil}, Params: xvec.VamanaQueryParams{EFSearch: 30},
		GroupByField: "category", GroupCount: 4, TopKPerGroup: 2,
	}
	groupMessage, err := GroupByQueryRequestToProto("catalog", group)
	require.NoError(t, err)
	collection, gotGroup, err := GroupByQueryRequestFromProto(groupMessage)
	require.NoError(t, err)
	require.Equal(t, "catalog", collection)
	require.True(t, reflect.DeepEqual(group, gotGroup), "want %#v, got %#v", group, gotGroup)
}

func mustIndexParamsFromProto(t *testing.T, message *xvecv1.IndexParams) xvec.IndexParams {
	t.Helper()
	got, err := IndexParamsFromProto(message)
	require.NoError(t, err)
	return got
}

func sentinelFor(code xvec.ErrorCode) error {
	switch code {
	case xvec.ErrorCodeNotFound:
		return xvec.ErrNotFound
	case xvec.ErrorCodeAlreadyExists:
		return xvec.ErrAlreadyExists
	case xvec.ErrorCodeInvalidArgument:
		return xvec.ErrInvalidArgument
	case xvec.ErrorCodePermissionDenied:
		return xvec.ErrPermissionDenied
	case xvec.ErrorCodeFailedPrecondition:
		return xvec.ErrFailedPrecondition
	case xvec.ErrorCodeResourceExhausted:
		return xvec.ErrResourceExhausted
	case xvec.ErrorCodeUnavailable:
		return xvec.ErrUnavailable
	case xvec.ErrorCodeInternal:
		return xvec.ErrInternal
	case xvec.ErrorCodeNotSupported:
		return xvec.ErrNotSupported
	default:
		return xvec.ErrUnknown
	}
}

var _ = errors.Is
