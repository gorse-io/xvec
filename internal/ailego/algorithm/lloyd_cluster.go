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

// Package algorithm provides reusable Ailego algorithms that do not depend on
// vector-index or database types.
package algorithm

import (
	"context"
	"slices"

	"github.com/gorse-io/xvec/internal/ailego/math"
)

// EmptyPolicy controls how LloydUpdate handles a centroid with no assignments.
type EmptyPolicy uint8

const (
	// EmptyKeep retains an empty centroid at its previous position.
	EmptyKeep EmptyPolicy = iota + 1
	// EmptyReseedFarthest replaces empty centroids with distinct worst-assigned
	// vectors according to Worse.
	EmptyReseedFarthest
	// EmptyDrop removes empty centroids.
	EmptyDrop
)

// LloydOptions configures a deterministic Lloyd centroid update.
type LloydOptions struct {
	Spherical   bool
	EmptyPolicy EmptyPolicy
	// Worse reports whether candidate is a worse assignment than current.
	Worse func(current, candidate float32) bool
	// Objective computes the lower-is-better objective for assignment scores.
	Objective func(context.Context, []float32) (float64, error)
}

// LloydUpdate computes the next centroids from a deterministic assignment.
// Accumulation follows input order so results remain stable across worker
// counts. It reports whether empty-cluster handling changed the centroid shape
// or reseeded a centroid.
func LloydUpdate(
	ctx context.Context,
	vectors, centroids [][]float32,
	labels []int,
	scores []float32,
	options LloydOptions,
) (float64, [][]float32, bool, error) {
	dimension := len(vectors[0])
	counts := make([]int, len(centroids))
	sums := make([][]float64, len(centroids))
	for index := range sums {
		sums[index] = make([]float64, dimension)
	}
	for index, vector := range vectors {
		if index&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return 0, nil, false, err
			}
		}
		label := labels[index]
		counts[label]++
		for coordinate, value := range vector {
			sums[label][coordinate] += float64(value)
		}
	}

	next := make([][]float32, len(centroids))
	empty := make([]int, 0)
	for cluster := range centroids {
		if cluster&255 == 0 {
			if err := ctx.Err(); err != nil {
				return 0, nil, false, err
			}
		}
		if counts[cluster] == 0 {
			empty = append(empty, cluster)
			next[cluster] = slices.Clone(centroids[cluster])
			continue
		}
		next[cluster] = make([]float32, dimension)
		for coordinate := range dimension {
			next[cluster][coordinate] = float32(sums[cluster][coordinate] / float64(counts[cluster]))
		}
		if options.Spherical {
			mathutil.NormalizeL2(next[cluster])
		}
	}

	changedShape := false
	switch options.EmptyPolicy {
	case EmptyKeep:
	case EmptyDrop:
		if len(empty) > 0 {
			compacted := make([][]float32, 0, len(next)-len(empty))
			for cluster := range next {
				if counts[cluster] != 0 {
					compacted = append(compacted, next[cluster])
				}
			}
			next = compacted
			changedShape = true
		}
	case EmptyReseedFarthest:
		used := make([]bool, len(vectors))
		for _, cluster := range empty {
			selected, err := worstAssignedVector(ctx, scores, used, options.Worse)
			if err != nil {
				return 0, nil, false, err
			}
			if selected < 0 {
				break
			}
			used[selected] = true
			next[cluster] = slices.Clone(vectors[selected])
			if options.Spherical {
				mathutil.NormalizeL2(next[cluster])
			}
			changedShape = true
		}
	}
	cost, err := options.Objective(ctx, scores)
	if err != nil {
		return 0, nil, false, err
	}
	return cost, next, changedShape, nil
}

func worstAssignedVector(
	ctx context.Context,
	scores []float32,
	used []bool,
	worse func(current, candidate float32) bool,
) (int, error) {
	selected := -1
	for index, score := range scores {
		if index&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return -1, err
			}
		}
		if used[index] {
			continue
		}
		if selected < 0 || worse(scores[selected], score) {
			selected = index
		}
	}
	return selected, nil
}
