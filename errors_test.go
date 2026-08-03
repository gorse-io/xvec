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
	"errors"
	"fmt"
	"io"
	"testing"
)

func TestErrorSupportsIsAsAndUnwrap(t *testing.T) {
	err := &Error{
		Code:    ErrorCodeNotFound,
		Op:      "fetch",
		Path:    "/collections/books",
		Message: "primary key 42",
		Err:     io.EOF,
	}
	wrapped := fmt.Errorf("request failed: %w", err)

	if !errors.Is(wrapped, ErrNotFound) {
		t.Fatal("errors.Is did not match the error code sentinel")
	}
	if !errors.Is(wrapped, io.EOF) {
		t.Fatal("errors.Is did not traverse the underlying cause")
	}
	if errors.Is(wrapped, ErrAlreadyExists) {
		t.Fatal("errors.Is matched the wrong error code")
	}

	var got *Error
	if !errors.As(wrapped, &got) || got != err {
		t.Fatal("errors.As did not recover the structured error")
	}
	if want := "zvec: fetch: /collections/books: primary key 42: EOF"; err.Error() != want {
		t.Fatalf("Error() = %q, want %q", err.Error(), want)
	}
}

func TestErrorDefaults(t *testing.T) {
	err := &Error{Code: ErrorCodeNotSupported}
	if got, want := err.Error(), "zvec: Not supported"; got != want {
		t.Fatalf("Error() = %q, want %q", got, want)
	}
	if got, want := ErrorCode(99).DefaultMessage(), "Unknown error"; got != want {
		t.Fatalf("DefaultMessage() = %q, want %q", got, want)
	}
	if got := (*Error)(nil).Error(); got != "<nil>" {
		t.Fatalf("nil Error() = %q", got)
	}
}
