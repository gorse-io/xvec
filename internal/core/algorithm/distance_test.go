// Copyright 2026-present the xvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");

package core

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func testDenseDistance(t testing.TB, metric Metric, left, right []float32) float32 {
	t.Helper()
	distance, err := metric.Distance()
	require.NoError(t, err)
	return distance(left, right)
}
