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

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

const (
	v010FixtureCommit    = "7005638af6c84b87a196afdb393bc37efd0dd7b8"
	v010FixtureGenerator = "0170611295a0fab98805320cfb3b6eda63b6751545bf19cef8e10c2077708272"
	v010FixtureArchive   = "399b4d5e8c34911356153aff76773dd23398571f72c7a54f17f9a851a6317b1d"
)

type historicalCollectionFixture struct {
	Release         string `json:"release"`
	Commit          string `json:"commit"`
	GeneratorSHA256 string `json:"generator_sha256"`
	ArchiveSHA256   string `json:"archive_sha256"`
	ArchiveBase64   string `json:"archive_base64"`
}

func TestOpenV010CollectionAndPublishCurrentGeneration(t *testing.T) {
	ctx := context.Background()
	path := extractHistoricalCollectionFixture(t, "testdata/native_v0_1_0_collection.json")
	collection, err := Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)
	{
		manifest := collection.store.Manifest()
		require.True(t, manifest.FormatVersion == 1)
		require.True(t, manifest.Generation == 2)
		require.True(t, manifest.WritingSegmentStartDocID == 0)
	}
	{
		schema := collection.Schema()
		require.True(t, schema.Name == "compat_v01")
		require.Equal(t, uint64(MinMaxDocsPerSegment), schema.MaxDocsPerSegment)
		require.Len(t, schema.Fields, 3)
	}
	{
		stats := collection.Stats()
		require.True(t, stats.DocumentCount == 4)
		require.True(t, stats.ImmutableSegments == 1)
		require.True(t, stats.MutableDocuments == 3)
		require.True(t, stats.DeletedDocuments == 3)
	}

	fetched, err := collection.Fetch(ctx, []string{"a", "b", "c", "d", "e", "f"}, Projection{IncludeVectors: true})
	require.NoError(t, err)

	wantIDs := []uint64{4, 1, 0, 3, 0, 6}
	for index, document := range fetched {
		if index == 2 || index == 4 {
			require.Nil(t, document)

			continue
		}
		require.NotNil(t, document)
		require.Equal(t, wantIDs[index], document.DocID)
	}
	require.Equal(t, int32(10), fetched[0].Fields["rating"])
	require.Nil(t, fetched[0].Fields["category"])
	require.Equal(t, VectorFP32{1, 0}, fetched[0].Fields["embedding"])

	results, err := collection.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 10,
		Filter: "rating >= 4", Projection: Projection{OutputFields: []string{"rating"}},
	})
	require.NoError(t, err)
	require.Equal(t, []string{"f", "d", "a"}, documentKeys(results))

	inserted, err := collection.Insert(ctx, []Document{{PrimaryKey: "g", Fields: map[string]any{
		"rating": int32(7), "category": "current", "embedding": VectorFP32{7, 0},
	}}})
	require.NoError(t, err)
	require.True(t, inserted[0].DocID == 7)

	index := NewInvertIndexParams()
	index.EnableRangeOptimization = true
	{
		err := collection.AddColumn(ctx, FieldSchema{
			Name: "derived", DataType: DataTypeInt64, Index: index,
		}, "(rating * 2) + 1", AddColumnOptions{Concurrency: 2})
		require.NoError(t, err)
	}
	{
		err := collection.Optimize(ctx, OptimizeOptions{Concurrency: 2})
		require.NoError(t, err)
	}
	{
		manifest := collection.store.Manifest()
		require.True(t, manifest.Generation > 2)
		require.True(t, manifest.WritingSegmentStartDocID == 8)
	}
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	options := NewCollectionOptions()
	options.ReadOnly = true
	collection, err = Open(ctx, path, options)
	require.NoError(t, err)

	defer collection.Close()
	derived, found := collection.Schema().Field("derived")
	require.True(t, found)
	require.Equal(t, DataTypeInt64, derived.DataType)
	require.Equal(t, IndexTypeInvert, derived.IndexType())

	results, err = collection.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 10,
		Filter: "derived >= 9", Projection: Projection{OutputFields: []string{"derived"}},
	})
	require.NoError(t, err)
	require.Equal(t, []string{"g", "f", "d", "a"}, documentKeys(results))

	wantDerived := map[string]int64{"a": 21, "b": 5, "d": 9, "f": 13, "g": 15}
	for _, result := range results {
		require.Equal(t, wantDerived[result.PrimaryKey], result.Fields["derived"])
	}
}

func extractHistoricalCollectionFixture(t *testing.T, fixturePath string) string {
	t.Helper()
	encoded, err := os.ReadFile(fixturePath)
	require.NoError(t, err)

	var fixture historicalCollectionFixture
	decoder := json.NewDecoder(bytes.NewReader(encoded))
	decoder.DisallowUnknownFields()
	{
		err := decoder.Decode(&fixture)
		require.NoError(t, err)
	}
	require.True(t, fixture.Release == "v0.1.0")
	require.Equal(t, v010FixtureCommit, fixture.Commit)
	require.Equal(t, v010FixtureGenerator, fixture.GeneratorSHA256)
	require.Equal(t, v010FixtureArchive, fixture.ArchiveSHA256)

	archive, err := base64.StdEncoding.DecodeString(fixture.ArchiveBase64)
	require.NoError(t, err)
	{
		digest := fmt.Sprintf("%x", sha256.Sum256(archive))
		require.Equal(t, fixture.ArchiveSHA256, digest)
	}

	root := filepath.Join(t.TempDir(), "collection")
	{
		err := os.MkdirAll(root, 0o755)
		require.NoError(t, err)
	}

	compressed, err := gzip.NewReader(bytes.NewReader(archive))
	require.NoError(t, err)

	defer compressed.Close()
	reader := tar.NewReader(compressed)
	var total int64
	for {
		header, err := reader.Next()
		if err == io.EOF {
			break
		}
		require.NoError(t, err)

		name := strings.TrimPrefix(header.Name, "./")
		if name == "" || name == "." {
			continue
		}
		clean := filepath.Clean(filepath.FromSlash(name))
		require.False(t, filepath.IsAbs(clean))
		require.False(t, clean == "..")
		require.False(t, strings.HasPrefix(clean, ".."+string(os.PathSeparator)))

		target := filepath.Join(root, clean)
		switch header.Typeflag {
		case tar.TypeDir:
			{
				err := os.MkdirAll(target, 0o755)
				require.NoError(t, err)
			}

		case tar.TypeReg:
			require.True(t, header.Size >= 0)
			require.True(t, header.Size <= 1<<20)
			require.True(t, total+header.Size <= 8<<20)

			total += header.Size
			{
				err := os.MkdirAll(filepath.Dir(target), 0o755)
				require.NoError(t, err)
			}

			contents := make([]byte, header.Size)
			{
				_, err := io.ReadFull(reader, contents)
				require.NoError(t, err)
			}
			{
				err := os.WriteFile(target, contents, 0o600)
				require.NoError(t, err)
			}

		default:
			require.FailNowf(t, "unsupported historical archive entry", "%q type %d", header.Name, header.Typeflag)
		}
	}
	return root
}
