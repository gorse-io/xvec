// Copyright 2026-present the xvec project
// Licensed under the Apache License, Version 2.0.

package rabitq

import (
	"errors"
	"fmt"
	"math"
)

// Metric is the distance convention used by RaBitQ factors.
type Metric uint8

const (
	MetricL2 Metric = iota
	MetricIP
)

var (
	ErrDimensionMismatch = errors.New("rabitq: dimension mismatch")
	ErrInvalidArgument   = errors.New("rabitq: invalid argument")
	ErrNonFinite         = errors.New("rabitq: non-finite value")
	ErrInvalidLayout     = errors.New("rabitq: invalid data layout")
)

func validMetric(metric Metric) bool { return metric == MetricL2 || metric == MetricIP }
func isFinite(v float32) bool {
	return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0)
}

func validatePair(a, b []float32) error {
	if len(a) == 0 || len(a) != len(b) {
		return ErrDimensionMismatch
	}
	for i := range a {
		if !isFinite(a[i]) || !isFinite(b[i]) {
			return fmt.Errorf("%w at coordinate %d", ErrNonFinite, i)
		}
	}
	return nil
}

// DotProduct computes a float32 inner product using source-compatible float32 accumulation.
func DotProduct(a, b []float32) (float32, error) {
	if err := validatePair(a, b); err != nil {
		return 0, err
	}
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	if !isFinite(sum) {
		return 0, ErrNonFinite
	}
	return sum, nil
}

// EuclideanSquared computes squared Euclidean distance.
func EuclideanSquared(a, b []float32) (float32, error) {
	if err := validatePair(a, b); err != nil {
		return 0, err
	}
	var sum float32
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	if !isFinite(sum) {
		return 0, ErrNonFinite
	}
	return sum, nil
}

func normSquared(a []float32) float32 {
	var sum float32
	for _, v := range a {
		sum += v * v
	}
	return sum
}
