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

// Package zvec provides a pure-Go embedded vector database.
//
// The public API follows Go conventions while preserving the data type,
// metric, quantization, and index semantics of zvec commit 58375ff. Go zvec
// uses its own versioned disk format and does not read C++ collection files.
package zvec
