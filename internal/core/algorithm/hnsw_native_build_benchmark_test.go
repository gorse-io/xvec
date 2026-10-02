// SPDX-License-Identifier: Apache-2.0

package core

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"
)

// Set XVEC_HNSW_FIXTURE to an items API response to compare real embeddings.
// The default corpus keeps this benchmark usable without external data.
func BenchmarkHNSWNativeFP16Build512(b *testing.B) {
	vectors := make([][]float32, 2000)
	if path := os.Getenv("XVEC_HNSW_FIXTURE"); path != "" {
		encoded, err := os.ReadFile(path)
		if err != nil {
			b.Fatal(err)
		}
		var response struct {
			Items []struct {
				Labels struct {
					Embedding []float32 `json:"embedding"`
				}
			}
		}
		if err := json.Unmarshal(encoded, &response); err != nil {
			b.Fatal(err)
		}
		for i := range vectors {
			vectors[i] = response.Items[i].Labels.Embedding
		}
	} else {
		for i := range vectors {
			vectors[i] = make([]float32, 512)
			for j := range vectors[i] {
				vectors[i][j] = float32(math.Sin(float64(i*512 + j)))
			}
		}
	}
	for _, workers := range []int{1, 4} {
		b.Run(fmt.Sprintf("workers=%d", workers), func(b *testing.B) {
			ctx := context.Background()
			b.ReportAllocs()
			for iteration := 0; iteration < b.N; iteration++ {
				b.StopTimer()
				builder, err := NewHNSWBuilderFP16(512, DefaultHNSWBuildOptions(MetricL2))
				if err != nil {
					b.Fatal(err)
				}
				for i, v := range vectors {
					if err := builder.Add(ctx, uint64(i), v); err != nil {
						b.Fatal(err)
					}
				}
				b.StartTimer()
				index, err := builder.BuildWithWorkers(ctx, workers)
				b.StopTimer()
				if err != nil {
					b.Fatal(err)
				}
				if len(index.keys) != len(vectors) {
					b.Fatal("incomplete graph")
				}
			}
		})
	}
}
