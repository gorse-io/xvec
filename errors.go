// Copyright 2026-present the zvec-go project
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

package zvec

import "strings"

// ErrorCode classifies errors returned by zvec operations. The values and
// default messages match StatusCode in the C++ public API at commit 58375ff.
type ErrorCode uint32

const (
	ErrorCodeOK                 ErrorCode = 0
	ErrorCodeNotFound           ErrorCode = 1
	ErrorCodeAlreadyExists      ErrorCode = 2
	ErrorCodeInvalidArgument    ErrorCode = 3
	ErrorCodePermissionDenied   ErrorCode = 4
	ErrorCodeFailedPrecondition ErrorCode = 5
	ErrorCodeResourceExhausted  ErrorCode = 6
	ErrorCodeUnavailable        ErrorCode = 7
	ErrorCodeInternal           ErrorCode = 8
	ErrorCodeNotSupported       ErrorCode = 9
	ErrorCodeUnknown            ErrorCode = 10
)

var errorCodeNames = map[ErrorCode]string{
	ErrorCodeOK:                 "OK",
	ErrorCodeNotFound:           "NOT_FOUND",
	ErrorCodeAlreadyExists:      "ALREADY_EXISTS",
	ErrorCodeInvalidArgument:    "INVALID_ARGUMENT",
	ErrorCodePermissionDenied:   "PERMISSION_DENIED",
	ErrorCodeFailedPrecondition: "FAILED_PRECONDITION",
	ErrorCodeResourceExhausted:  "RESOURCE_EXHAUSTED",
	ErrorCodeUnavailable:        "UNAVAILABLE",
	ErrorCodeInternal:           "INTERNAL_ERROR",
	ErrorCodeNotSupported:       "NOT_SUPPORTED",
	ErrorCodeUnknown:            "UNKNOWN",
}

var errorCodeMessages = map[ErrorCode]string{
	ErrorCodeOK:                 "OK",
	ErrorCodeNotFound:           "Not found",
	ErrorCodeAlreadyExists:      "Already exists",
	ErrorCodeInvalidArgument:    "Invalid argument",
	ErrorCodePermissionDenied:   "Permission denied",
	ErrorCodeFailedPrecondition: "Failed precondition",
	ErrorCodeResourceExhausted:  "Resource exhausted",
	ErrorCodeUnavailable:        "Unavailable",
	ErrorCodeInternal:           "Internal error",
	ErrorCodeNotSupported:       "Not supported",
	ErrorCodeUnknown:            "Unknown error",
}

func (c ErrorCode) String() string { return enumName(errorCodeNames, c, "ErrorCode") }

// Valid reports whether c is a value defined by the public API.
func (c ErrorCode) Valid() bool { return enumValid(errorCodeNames, c) }

// DefaultMessage returns the stable default message for c.
func (c ErrorCode) DefaultMessage() string {
	if message, ok := errorCodeMessages[c]; ok {
		return message
	}
	return errorCodeMessages[ErrorCodeUnknown]
}

// Error is the structured error returned by zvec operations.
//
// Code is suitable for programmatic decisions. Op and Path identify the
// failed operation and collection path when available. Err retains the
// underlying error for errors.Is and errors.As traversal.
type Error struct {
	Code    ErrorCode
	Op      string
	Path    string
	Message string
	Err     error
}

// Error formats the structured context without losing the underlying cause.
func (e *Error) Error() string {
	if e == nil {
		return "<nil>"
	}

	parts := []string{"zvec"}
	if e.Op != "" {
		parts = append(parts, e.Op)
	}
	if e.Path != "" {
		parts = append(parts, e.Path)
	}

	detail := e.Message
	if detail == "" {
		detail = e.Code.DefaultMessage()
	}
	if e.Err != nil && e.Err.Error() != detail {
		detail += ": " + e.Err.Error()
	}
	parts = append(parts, detail)
	return strings.Join(parts, ": ")
}

// Unwrap returns the underlying cause.
func (e *Error) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

// Is makes errors.Is compare zvec errors by ErrorCode. An underlying cause is
// still considered by the standard library through Unwrap.
func (e *Error) Is(target error) bool {
	if e == nil || target == nil {
		return false
	}
	switch target := target.(type) {
	case codeSentinel:
		return e.Code == target.code
	case *Error:
		return target != nil && e.Code == target.Code
	default:
		return false
	}
}

type codeSentinel struct {
	code ErrorCode
}

func (e codeSentinel) Error() string { return "zvec: " + e.code.DefaultMessage() }

// Stable errors.Is targets for each non-success ErrorCode.
var (
	ErrNotFound           error = codeSentinel{ErrorCodeNotFound}
	ErrAlreadyExists      error = codeSentinel{ErrorCodeAlreadyExists}
	ErrInvalidArgument    error = codeSentinel{ErrorCodeInvalidArgument}
	ErrPermissionDenied   error = codeSentinel{ErrorCodePermissionDenied}
	ErrFailedPrecondition error = codeSentinel{ErrorCodeFailedPrecondition}
	ErrResourceExhausted  error = codeSentinel{ErrorCodeResourceExhausted}
	ErrUnavailable        error = codeSentinel{ErrorCodeUnavailable}
	ErrInternal           error = codeSentinel{ErrorCodeInternal}
	ErrNotSupported       error = codeSentinel{ErrorCodeNotSupported}
	ErrUnknown            error = codeSentinel{ErrorCodeUnknown}
)
