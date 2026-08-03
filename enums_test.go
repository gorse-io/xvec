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

import "testing"

func TestEnumBehavior(t *testing.T) {
	if got := IndexTypeDiskANN.String(); got != "DISKANN" {
		t.Fatalf("IndexTypeDiskANN.String() = %q", got)
	}
	if !IndexTypeDiskANN.Valid() || IndexType(9).Valid() {
		t.Fatal("IndexType.Valid returned an incorrect result")
	}
	if got := IndexType(9).String(); got != "IndexType(9)" {
		t.Fatalf("unknown IndexType.String() = %q", got)
	}
	if !IndexTypeHNSWRaBitQ.IsVector() || IndexTypeInvert.IsVector() {
		t.Fatal("IndexType.IsVector returned an incorrect result")
	}
	if !DataTypeVectorFP32.IsDenseVector() || DataTypeSparseVectorFP32.IsDenseVector() {
		t.Fatal("DataType.IsDenseVector returned an incorrect result")
	}
	if !DataTypeSparseVectorFP16.IsSparseVector() || DataTypeFloat.IsSparseVector() {
		t.Fatal("DataType.IsSparseVector returned an incorrect result")
	}
	if !DataTypeArrayString.IsArray() || DataTypeString.IsArray() {
		t.Fatal("DataType.IsArray returned an incorrect result")
	}
}
