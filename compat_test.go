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
	"encoding/json"
	"os"
	"testing"

	"github.com/stretchr/testify/require"
)

type publicAPIFixture struct {
	BaselineCommit string                       `json:"baseline_commit"`
	Enums          map[string]map[string]uint32 `json:"enums"`
}

func loadPublicAPIFixture(t *testing.T) publicAPIFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/cpp_public_api_58375ff.json")
	require.NoError(t, err)

	var fixture publicAPIFixture
	{
		err := json.Unmarshal(data, &fixture)
		require.NoError(t, err)
	}
	require.True(t, fixture.BaselineCommit == "58375ff7b8fdd0d6fc7d234e47567b179777883b")

	return fixture
}

func TestPublicEnumCompatibility(t *testing.T) {
	fixture := loadPublicAPIFixture(t)
	got := map[string]map[string]uint32{
		"IndexType":    enumValues(indexTypeNames),
		"DataType":     enumValues(dataTypeNames),
		"QuantizeType": enumValues(quantizeTypeNames),
		"MetricType":   enumValues(metricTypeNames),
		"Operator":     enumValues(operatorNames),
		"CompareOp":    enumValues(compareOpNames),
		"RelationOp":   enumValues(relationOpNames),
		"BlockType":    enumValues(blockTypeNames),
		"FileFormat":   enumValues(fileFormatNames),
		"ColumnOp":     enumValues(columnOpNames),
		"ErrorCode":    enumValues(errorCodeNames),
	}
	{
		diff := diffEnums(fixture.Enums, got)
		require.True(t, diff == "")
	}
}

func enumValues[T ~uint32](names map[T]string) map[string]uint32 {
	values := make(map[string]uint32, len(names))
	for value, name := range names {
		values[name] = uint32(value)
	}
	return values
}

func diffEnums(want, got map[string]map[string]uint32) string {
	wantJSON, _ := json.MarshalIndent(want, "", "  ")
	gotJSON, _ := json.MarshalIndent(got, "", "  ")
	if string(wantJSON) == string(gotJSON) {
		return ""
	}
	return "public enum compatibility mismatch\nwant: " + string(wantJSON) +
		"\ngot:  " + string(gotJSON)
}
