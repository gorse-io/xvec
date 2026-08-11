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

package xvec

import (
	"fmt"
	"io"
	"testing"

	"github.com/stretchr/testify/require"
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
	require.ErrorIs(t, wrapped, ErrNotFound,
		"errors.Is did not match the error code sentinel")
	require.ErrorIs(t, wrapped, io.EOF,
		"errors.Is did not traverse the underlying cause")
	require.NotErrorIs(t, wrapped, ErrAlreadyExists,
		"errors.Is matched the wrong error code")

	var got *Error
	require.ErrorAs(t, wrapped, &got,
		"errors.As did not recover the structured error")
	require.Same(t, err, got,
		"errors.As did not recover the structured error")
	{
		want := "xvec: fetch: /collections/books: primary key 42: EOF"
		require.Equal(t, want, err.Error())
	}
}

func TestErrorDefaults(t *testing.T) {
	err := &Error{Code: ErrorCodeNotSupported}
	{
		got, want := err.Error(), "xvec: Not supported"
		require.Equal(t, want, got)
	}
	{
		got, want := ErrorCode(99).DefaultMessage(), "Unknown error"
		require.Equal(t, want, got)
	}
	{
		got := (*Error)(nil).Error()
		require.True(t, got == "<nil>")
	}
}
