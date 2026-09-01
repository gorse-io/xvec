// Copyright 2026-present the xvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");

package metric

import (
	"testing"

	"github.com/gorse-io/xvec/internal/ailego/math"
	"github.com/stretchr/testify/require"
)

func TestMetricDistanceAndOrdering(t *testing.T) {
	t.Parallel()

	left := []float32{0.2, 0.9, -0.4, 0.7}
	right := []float32{0.3, 0.5, 0.8, -0.1}
	for _, test := range []struct {
		metric   Metric
		distance mathutil.DenseDistance
	}{
		{metric: L2, distance: mathutil.L2Squared},
		{metric: IP, distance: mathutil.InnerProduct},
		{metric: Cosine, distance: mathutil.CosineDistance},
		{metric: MIPSL2, distance: mathutil.MIPSL2Squared},
	} {
		require.True(t, test.metric.Valid())
		distance, err := test.metric.Distance()
		require.NoError(t, err)
		require.Equal(t, test.distance(left, right), distance(left, right))
	}

	require.False(t, Metric(0).Valid())
	_, err := Metric(0).Distance()
	require.Error(t, err)

	require.True(t, IP.Better(2, 1))
	require.True(t, L2.Better(1, 2))
}
