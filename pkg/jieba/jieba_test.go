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

package jieba

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

// These cases are adapted from gojieba's jieba_test.go for the segmentation
// modes supported by this package.
func TestCutModes(t *testing.T) {
	tests := []struct {
		name string
		mode CutMode
		want []Word
	}{
		{
			name: "full",
			mode: CutModeFull,
			want: []Word{
				{Text: "我", Offset: 0}, {Text: "来", Offset: 3}, {Text: "到", Offset: 6},
				{Text: "北", Offset: 9}, {Text: "京", Offset: 12}, {Text: "清华", Offset: 15},
				{Text: "清华大学", Offset: 15}, {Text: "华大", Offset: 18}, {Text: "大学", Offset: 21},
			},
		},
		{
			name: "mix",
			mode: CutModeMix,
			want: []Word{
				{Text: "我", Offset: 0}, {Text: "来", Offset: 3}, {Text: "到", Offset: 6},
				{Text: "北", Offset: 9}, {Text: "京", Offset: 12}, {Text: "清华大学", Offset: 15},
			},
		},
		{
			name: "search",
			mode: CutModeSearch,
			want: []Word{
				{Text: "我", Offset: 0}, {Text: "来", Offset: 3}, {Text: "到", Offset: 6},
				{Text: "北", Offset: 9}, {Text: "京", Offset: 12}, {Text: "清华", Offset: 15},
				{Text: "华大", Offset: 18}, {Text: "大学", Offset: 21}, {Text: "清华大学", Offset: 15},
			},
		},
		{
			name: "hmm",
			mode: CutModeHMM,
			want: []Word{
				{Text: "我", Offset: 0}, {Text: "来", Offset: 3}, {Text: "到", Offset: 6},
				{Text: "北", Offset: 9}, {Text: "京", Offset: 12}, {Text: "清", Offset: 15},
				{Text: "华", Offset: 18}, {Text: "大", Offset: 21}, {Text: "学", Offset: 24},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			segmenter := newTestSegmenter(t, test.mode, "")
			got, err := segmenter.Cut(context.Background(), "我来到北京清华大学")
			require.NoError(t, err)
			require.Equal(t, test.want, got)
		})
	}
}

func TestSearchCutReturnsByteOffsets(t *testing.T) {
	segmenter := newTestSegmenter(t, CutModeSearch, "")
	got, err := segmenter.Cut(context.Background(), "清华大学")
	require.NoError(t, err)
	require.Equal(t, []Word{
		{Text: "清华", Offset: 0},
		{Text: "华大", Offset: 3},
		{Text: "大学", Offset: 6},
		{Text: "清华大学", Offset: 0},
	}, got)
}

func TestHMMCutDiscoversWordAndKeepsASCII(t *testing.T) {
	segmenter := newTestSegmenter(t, CutModeHMM, "")
	got, err := segmenter.Cut(context.Background(), "杭研abc1.2")
	require.NoError(t, err)
	require.Equal(t, []Word{
		{Text: "杭研", Offset: 0},
		{Text: "abc1.2", Offset: 6},
	}, got)
}

func TestUserDictionary(t *testing.T) {
	withoutUser := newTestSegmenter(t, CutModeFull, "")
	withUser := newTestSegmenter(t, CutModeFull, filepath.Join("testdata", "user.dict.utf8"))

	withoutWords, err := withoutUser.Cut(context.Background(), "他来到了网易杭研大厦")
	require.NoError(t, err)
	require.Equal(t, []Word{
		{Text: "他", Offset: 0}, {Text: "来", Offset: 3}, {Text: "到", Offset: 6},
		{Text: "了", Offset: 9}, {Text: "网", Offset: 12}, {Text: "易", Offset: 15},
		{Text: "杭", Offset: 18}, {Text: "研", Offset: 21}, {Text: "大", Offset: 24},
		{Text: "厦", Offset: 27},
	}, withoutWords)

	withWords, err := withUser.Cut(context.Background(), "他来到了网易杭研大厦")
	require.NoError(t, err)
	require.Equal(t, []Word{
		{Text: "他", Offset: 0}, {Text: "来", Offset: 3}, {Text: "到", Offset: 6},
		{Text: "了", Offset: 9}, {Text: "网", Offset: 12}, {Text: "易", Offset: 15},
		{Text: "杭研", Offset: 18}, {Text: "大", Offset: 24}, {Text: "厦", Offset: 27},
	}, withWords)
}

func TestDefaultModeIsSearch(t *testing.T) {
	segmenter := newTestSegmenter(t, "", "")
	got, err := segmenter.Cut(context.Background(), "清华大学")
	require.NoError(t, err)
	require.Equal(t, []Word{
		{Text: "清华", Offset: 0},
		{Text: "华大", Offset: 3},
		{Text: "大学", Offset: 6},
		{Text: "清华大学", Offset: 0},
	}, got)
}

func newTestSegmenter(t testing.TB, mode CutMode, userDictionaryPath string) *Segmenter {
	t.Helper()
	segmenter, err := New(context.Background(), Options{
		DictionaryPath: filepath.Join("testdata", "jieba.dict.utf8"),
		HMMModelPath:   filepath.Join("testdata", "hmm_model.utf8"),
		UserDictPath:   userDictionaryPath,
		CutMode:        mode,
	})
	require.NoError(t, err)
	return segmenter
}
