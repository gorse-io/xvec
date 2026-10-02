// SPDX-License-Identifier: Apache-2.0

package core

import (
	"context"
	"fmt"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

// Use exactly representable coordinates and reversed document keys to exercise
// equal distances and position-based construction ties on every SIMD backend.
func TestNativeFP16BatchBuildMatchesSinglePairGraph(t *testing.T) {
	ctx := context.Background()
	for _, dimension := range []int{1, 7, 8, 17, 33, 512} {
		for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine, MetricMIPSL2} {
			t.Run(fmt.Sprintf("dim=%d/metric=%v", dimension, metric), func(t *testing.T) {
				options := DefaultHNSWBuildOptions(metric)
				options.M, options.EFConstruction = 4, 24
				builder, err := NewHNSWBuilderFP16(dimension, options)
				require.NoError(t, err)
				for n := 0; n < 96; n++ {
					v := make([]float32, dimension)
					for d := range v {
						if n > 0 {
							v[d] = float32((n*17+d*3)%7 - 3)
						}
					}
					require.NoError(t, builder.Add(ctx, uint64(1000-n), v))
				}
				index, err := builder.BuildWithWorkers(ctx, 1)
				require.NoError(t, err)
				reference := make([][][]int, len(index.keys))
				for n, level := range index.levels {
					reference[n] = make([][]int, level+1)
				}
				entry, level, err := buildParallelHNSW(ctx, 1, options, index.levels, reference, index.computeDistanceAt)
				require.NoError(t, err)
				require.Equal(t, reference, index.neighbors)
				require.Equal(t, entry, index.entryPoint)
				require.Equal(t, level, index.maxLevel)
				require.True(t, index.fp16)
				require.Empty(t, index.vectors, "builder must retain native FP16 storage")
				path := filepath.Join(t.TempDir(), "hnsw")
				require.NoError(t, index.Save(ctx, path))
				reopened, err := OpenHNSWIndex(ctx, path)
				require.NoError(t, err)
				require.Equal(t, index.neighbors, reopened.neighbors)
				require.Equal(t, index.vectorsFP16, reopened.vectorsFP16)
			})
		}
	}
}
