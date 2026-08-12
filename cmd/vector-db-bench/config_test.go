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
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestParseConfigVectorDBBenchCases(t *testing.T) {
	config, err := parseConfig([]string{
		backendXvec,
		"--path", t.TempDir(),
		"--case-type", casePerformance768D10M,
		"--num-concurrency", "12,14,16",
		"--concurrency-duration", "3",
		"--serial-cooldown", "0.25",
		"--quantize-type", "int8",
		"--is-using-refiner",
	}, &bytes.Buffer{})
	require.NoError(t, err)
	require.Equal(t, backendXvec, config.Backend)
	require.Equal(t, casePerformance768D10M, config.caseSpec.Name)
	require.Equal(t, 768, config.caseSpec.Dimension)
	require.Equal(t, int64(10_000_000), config.caseSpec.Size)
	require.Len(t, config.caseSpec.TrainFiles, 10)
	require.Equal(t, []int{12, 14, 16}, config.concurrency)
	require.Equal(t, 3*time.Second, config.ConcurrencyDuration)
	require.Equal(t, 250*time.Millisecond, config.SerialCooldown)
	require.True(t, config.UseRefiner)
}

func TestParseConfigCustomAndValidation(t *testing.T) {
	datasetDir := t.TempDir()
	config, err := parseConfig([]string{
		backendZvec,
		"--path", t.TempDir(), "--case-type", caseCustom,
		"--dataset-dir", datasetDir, "--dimension", "3", "--metric", "l2",
		"--train-files", "part-0.parquet,part-1.parquet",
	}, &bytes.Buffer{})
	require.NoError(t, err)
	require.Equal(t, backendZvec, config.Backend)
	require.Equal(t, []string{"part-0.parquet", "part-1.parquet"}, config.caseSpec.TrainFiles)
	require.Equal(t, "l2", config.caseSpec.Metric)

	_, err = parseConfig([]string{backendXvec, "--path", t.TempDir(), "--skip-load"}, &bytes.Buffer{})
	require.ErrorContains(t, err, "skip-load requires skip-drop-old")
	_, err = parseConfig([]string{backendXvec, "--path", t.TempDir(), "--num-concurrency", "1,1"}, &bytes.Buffer{})
	require.ErrorContains(t, err, "duplicate concurrency")
	_, err = parseConfig([]string{backendXvec, "--path", t.TempDir(), "--serial-cooldown", "-1"}, &bytes.Buffer{})
	require.ErrorContains(t, err, "serial-cooldown cannot be negative")
	_, err = parseConfig([]string{"--path", t.TempDir()}, &bytes.Buffer{})
	require.ErrorContains(t, err, "backend is required")
	_, err = parseConfig([]string{"unknown", "--path", t.TempDir()}, &bytes.Buffer{})
	require.ErrorContains(t, err, "unsupported backend")
}

func TestParseFlexibleDuration(t *testing.T) {
	duration, err := parseFlexibleDuration("0.25")
	require.NoError(t, err)
	require.Equal(t, 250*time.Millisecond, duration)
	duration, err = parseFlexibleDuration("250ms")
	require.NoError(t, err)
	require.Equal(t, 250*time.Millisecond, duration)
}
