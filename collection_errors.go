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

import (
	"context"
	"errors"
	"os"
	"slices"
	"strconv"

	"github.com/gorse-io/zvec/internal/db"
)

// BatchWriteError summarizes per-document write failures. The returned
// WriteResult slice remains authoritative and preserves input order.
type BatchWriteError struct {
	Failed int
	causes []error
}

func (e *BatchWriteError) Error() string {
	if e == nil {
		return "zvec: batch write failed"
	}
	return "zvec: batch write: " + strconv.Itoa(e.Failed) + " document operations failed"
}

// Unwrap exposes every per-document cause to errors.Is and errors.As.
func (e *BatchWriteError) Unwrap() []error {
	if e == nil {
		return nil
	}
	return slices.Clone(e.causes)
}

func (e *BatchWriteError) add(err error) {
	if err == nil {
		return
	}
	e.Failed++
	e.causes = append(e.causes, err)
}

func wrapCollectionError(op, path string, err error) error {
	if err == nil {
		return nil
	}
	var existing *Error
	if errors.As(err, &existing) && existing != nil {
		copy := *existing
		if copy.Op == "" {
			copy.Op = op
		}
		if copy.Path == "" {
			copy.Path = path
		}
		return &copy
	}
	code := ErrorCodeUnknown
	switch {
	case errors.Is(err, db.ErrPrimaryKeyNotFound),
		errors.Is(err, db.ErrDocumentNotFound),
		errors.Is(err, db.ErrManifestNotFound),
		errors.Is(err, os.ErrNotExist):
		code = ErrorCodeNotFound
	case errors.Is(err, db.ErrPrimaryKeyExists),
		errors.Is(err, db.ErrManifestExists),
		errors.Is(err, os.ErrExist):
		code = ErrorCodeAlreadyExists
	case errors.Is(err, db.ErrReadOnly), errors.Is(err, os.ErrPermission):
		code = ErrorCodePermissionDenied
	case errors.Is(err, db.ErrCollectionClosed),
		errors.Is(err, db.ErrWALClosed),
		errors.Is(err, db.ErrWALReadOnly):
		code = ErrorCodeFailedPrecondition
	case errors.Is(err, db.ErrSegmentFull),
		errors.Is(err, db.ErrWALRecordTooLarge):
		code = ErrorCodeResourceExhausted
	case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
		code = ErrorCodeUnavailable
	case errors.Is(err, db.ErrCollectionCorrupt),
		errors.Is(err, db.ErrManifestCorrupt),
		errors.Is(err, db.ErrSegmentCorrupt),
		errors.Is(err, db.ErrWALCorrupt):
		code = ErrorCodeInternal
	}
	return &Error{Code: code, Op: op, Path: path, Err: err}
}

func notSupported(op, path, message string) error {
	return &Error{Code: ErrorCodeNotSupported, Op: op, Path: path, Message: message}
}
