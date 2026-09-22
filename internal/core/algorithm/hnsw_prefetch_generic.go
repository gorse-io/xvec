//go:build !amd64 || noasm

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

package core

// Prefetch is an optional hint. On platforms without a native implementation,
// leave it to the distance kernel instead of synchronously reading cache lines.
func prefetchDenseHNSWNeighbors(_ []float32, _ int, _ []int, _, _ uint32) {}
