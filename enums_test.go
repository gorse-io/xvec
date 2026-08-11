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

package xvec

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestEnumBehavior(t *testing.T) {
	{
		got := IndexTypeDiskANN.String()
		require.True(t, got == "DISKANN")
	}
	require.True(t, IndexTypeDiskANN.Valid(),
		"IndexType.Valid returned an incorrect result")
	require.False(t, IndexType(9).Valid(),
		"IndexType.Valid returned an incorrect result")
	{
		got := IndexType(9).String()
		require.True(t, got == "IndexType(9)")
	}
	require.True(t, IndexTypeHNSWRaBitQ.IsVector(),
		"IndexType.IsVector returned an incorrect result")
	require.False(t, IndexTypeInvert.IsVector(),
		"IndexType.IsVector returned an incorrect result")
	require.True(t, DataTypeVectorFP32.IsDenseVector(),
		"DataType.IsDenseVector returned an incorrect result")
	require.False(t, DataTypeSparseVectorFP32.IsDenseVector(),
		"DataType.IsDenseVector returned an incorrect result")
	require.True(t, DataTypeSparseVectorFP16.IsSparseVector(),
		"DataType.IsSparseVector returned an incorrect result")
	require.False(t, DataTypeFloat.IsSparseVector(),
		"DataType.IsSparseVector returned an incorrect result")
	require.True(t, DataTypeArrayString.IsArray(),
		"DataType.IsArray returned an incorrect result")
	require.False(t, DataTypeString.IsArray(),
		"DataType.IsArray returned an incorrect result")
}
