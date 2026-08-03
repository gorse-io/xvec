// Copyright 2026-present the zvec-go project
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

package zvec

// CreateIndexOptions controls index-build concurrency. Zero lets the library
// select an appropriate worker count.
type CreateIndexOptions struct{ Concurrency int }

// AddColumnOptions controls column backfill concurrency.
type AddColumnOptions struct{ Concurrency int }

// AlterColumnOptions controls column migration concurrency.
type AlterColumnOptions struct{ Concurrency int }

// OptimizeOptions controls segment-optimization concurrency.
type OptimizeOptions struct{ Concurrency int }
