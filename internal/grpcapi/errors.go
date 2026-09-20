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
	"errors"

	"github.com/gorse-io/xvec"
	xvecv1 "github.com/gorse-io/xvec/internal/proto/xvec/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func ErrorToProto(err *xvec.Error) *xvecv1.Error {
	if err == nil {
		return nil
	}
	return &xvecv1.Error{Code: xvecv1.ErrorCode(err.Code), Op: err.Op, Path: err.Path, Message: err.Message}
}

func ErrorFromProto(message *xvecv1.Error) *xvec.Error {
	if message == nil {
		return nil
	}
	code := xvec.ErrorCode(message.Code)
	if code == xvec.ErrorCodeOK {
		code = xvec.ErrorCodeUnknown
	}
	return &xvec.Error{Code: code, Op: message.Op, Path: message.Path, Message: message.Message}
}

func ErrorToStatus(err error) error {
	if err == nil {
		return nil
	}
	if _, ok := status.FromError(err); ok {
		return err
	}
	var structured *xvec.Error
	if !errors.As(err, &structured) || structured == nil {
		structured = &xvec.Error{Code: xvec.ErrorCodeUnknown, Message: err.Error(), Err: err}
	} else if structured.Code == xvec.ErrorCodeOK {
		copy := *structured
		copy.Code = xvec.ErrorCodeUnknown
		structured = &copy
	}
	message := structured.Message
	if message == "" {
		message = structured.Code.DefaultMessage()
	}
	transport := status.New(grpcCode(structured.Code), message)
	withDetails, detailErr := transport.WithDetails(ErrorToProto(structured))
	if detailErr != nil {
		return transport.Err()
	}
	return withDetails.Err()
}

func ErrorFromStatus(err error) error {
	if err == nil {
		return nil
	}
	transport, ok := status.FromError(err)
	if !ok {
		return err
	}
	for _, detail := range transport.Details() {
		if message, ok := detail.(*xvecv1.Error); ok {
			structured := ErrorFromProto(message)
			if structured.Message == "" {
				structured.Message = transport.Message()
			}
			return structured
		}
	}
	return &xvec.Error{Code: errorCode(transport.Code()), Message: transport.Message()}
}

func BatchWriteErrorToProto(err *xvec.BatchWriteError) *xvecv1.BatchWriteError {
	if err == nil {
		return nil
	}
	message := &xvecv1.BatchWriteError{}
	for _, cause := range err.Causes() {
		if encoded := errorValueToProto(cause); encoded != nil {
			message.Causes = append(message.Causes, encoded)
		}
	}
	return message
}

func BatchWriteErrorFromProto(message *xvecv1.BatchWriteError) *xvec.BatchWriteError {
	if message == nil {
		return nil
	}
	causes := make([]error, 0, len(message.Causes))
	for _, cause := range message.Causes {
		if decoded := ErrorFromProto(cause); decoded != nil {
			causes = append(causes, decoded)
		}
	}
	return xvec.NewBatchWriteError(causes...)
}

func WriteResultToProto(result xvec.WriteResult) *xvecv1.WriteResult {
	return &xvecv1.WriteResult{
		PrimaryKey: result.PrimaryKey,
		DocId:      result.DocID,
		Error:      errorValueToProto(result.Err),
	}
}

func WriteResultFromProto(message *xvecv1.WriteResult) xvec.WriteResult {
	if message == nil {
		return xvec.WriteResult{}
	}
	result := xvec.WriteResult{
		PrimaryKey: message.PrimaryKey,
		DocID:      message.DocId,
	}
	if message.Error != nil {
		result.Err = ErrorFromProto(message.Error)
	}
	return result
}

func WriteResponseToProto(results []xvec.WriteResult, _ *xvec.BatchWriteError) *xvecv1.WriteResponse {
	message := &xvecv1.WriteResponse{
		Results: make([]*xvecv1.WriteResult, len(results)),
	}
	causes := make([]error, 0)
	for i := range results {
		message.Results[i] = WriteResultToProto(results[i])
		if results[i].Err != nil {
			causes = append(causes, results[i].Err)
		}
	}
	if len(causes) > 0 {
		message.Error = BatchWriteErrorToProto(xvec.NewBatchWriteError(causes...))
	}
	return message
}

func WriteResponseFromProto(message *xvecv1.WriteResponse) ([]xvec.WriteResult, error) {
	if message == nil {
		return nil, nil
	}
	results := make([]xvec.WriteResult, len(message.Results))
	causes := make([]error, 0)
	for i := range message.Results {
		results[i] = WriteResultFromProto(message.Results[i])
		if results[i].Err != nil {
			causes = append(causes, results[i].Err)
		}
	}
	if len(causes) == 0 {
		return results, nil
	}
	return results, xvec.NewBatchWriteError(causes...)
}

func errorValueToProto(err error) *xvecv1.Error {
	if err == nil {
		return nil
	}
	var structured *xvec.Error
	if errors.As(err, &structured) && structured != nil {
		return ErrorToProto(structured)
	}
	return &xvecv1.Error{Code: xvecv1.ErrorCode_ERROR_CODE_UNKNOWN, Message: err.Error()}
}

func grpcCode(code xvec.ErrorCode) codes.Code {
	switch code {
	case xvec.ErrorCodeOK:
		return codes.OK
	case xvec.ErrorCodeNotFound:
		return codes.NotFound
	case xvec.ErrorCodeAlreadyExists:
		return codes.AlreadyExists
	case xvec.ErrorCodeInvalidArgument:
		return codes.InvalidArgument
	case xvec.ErrorCodePermissionDenied:
		return codes.PermissionDenied
	case xvec.ErrorCodeFailedPrecondition:
		return codes.FailedPrecondition
	case xvec.ErrorCodeResourceExhausted:
		return codes.ResourceExhausted
	case xvec.ErrorCodeUnavailable:
		return codes.Unavailable
	case xvec.ErrorCodeInternal:
		return codes.Internal
	case xvec.ErrorCodeNotSupported:
		return codes.Unimplemented
	default:
		return codes.Unknown
	}
}

func errorCode(code codes.Code) xvec.ErrorCode {
	switch code {
	case codes.NotFound:
		return xvec.ErrorCodeNotFound
	case codes.AlreadyExists:
		return xvec.ErrorCodeAlreadyExists
	case codes.InvalidArgument:
		return xvec.ErrorCodeInvalidArgument
	case codes.PermissionDenied:
		return xvec.ErrorCodePermissionDenied
	case codes.FailedPrecondition:
		return xvec.ErrorCodeFailedPrecondition
	case codes.ResourceExhausted:
		return xvec.ErrorCodeResourceExhausted
	case codes.Unavailable, codes.Canceled, codes.DeadlineExceeded:
		return xvec.ErrorCodeUnavailable
	case codes.Internal, codes.DataLoss:
		return xvec.ErrorCodeInternal
	case codes.Unimplemented:
		return xvec.ErrorCodeNotSupported
	default:
		return xvec.ErrorCodeUnknown
	}
}
