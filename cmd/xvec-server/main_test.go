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

package main

import (
	"bytes"
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestRunCLIPassesParsedConfigToServer(t *testing.T) {
	original := runServerCommand
	t.Cleanup(func() { runServerCommand = original })
	var got config
	runServerCommand = func(_ context.Context, cfg config) error {
		got = cfg
		return errors.New("serve failed")
	}

	err := runCLI(context.Background(), []string{"--data-dir", "data", "--listen", "localhost:0"}, &bytes.Buffer{})
	require.EqualError(t, err, "serve failed")
	require.Equal(t, "data", got.DataDir)
	require.Equal(t, "localhost:0", got.Listen)
}

func TestRunCLIHandlesHelpWithoutStartingServer(t *testing.T) {
	original := runServerCommand
	t.Cleanup(func() { runServerCommand = original })
	called := false
	runServerCommand = func(context.Context, config) error {
		called = true
		return nil
	}
	var output bytes.Buffer

	require.NoError(t, runCLI(context.Background(), []string{"--help"}, &output))
	require.False(t, called)
	require.Contains(t, output.String(), "Usage of xvec-server")
}
