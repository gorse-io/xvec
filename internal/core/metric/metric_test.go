// Copyright 2026-present the xvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");

package metric

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestMetricComputeOrderingAndValidation(t *testing.T) {
	t.Parallel()

	left := []float32{0.2, 0.9, -0.4, 0.7}
	right := []float32{0.3, 0.5, 0.8, -0.1}
	for _, value := range []Metric{L2, IP, Cosine, MIPSL2} {
		require.True(t, value.Valid())
		expected, err := value.Compute(left, right)
		require.NoError(t, err)
		distance, err := value.PrevalidatedDistance()
		require.NoError(t, err)
		actual, err := distance(left, right)
		require.NoError(t, err)
		require.Equal(t, expected, actual)
	}

	require.False(t, Metric(0).Valid())
	_, err := Metric(0).Compute(left, right)
	require.Error(t, err)
	_, err = Metric(0).PrevalidatedDistance()
	require.Error(t, err)

	require.True(t, IP.Better(2, 1))
	require.True(t, L2.Better(1, 2))
}
