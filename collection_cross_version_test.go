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
	"reflect"
	"strings"
	"testing"
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
	if err != nil {
		t.Fatal(err)
	}
	if manifest := collection.store.Manifest(); manifest.FormatVersion != 1 || manifest.Generation != 2 || manifest.WritingSegmentStartDocID != 0 {
		t.Fatalf("v0.1 manifest = %#v", manifest)
	}
	if schema := collection.Schema(); schema.Name != "compat_v01" || schema.MaxDocsPerSegment != MinMaxDocsPerSegment || len(schema.Fields) != 3 {
		t.Fatalf("v0.1 schema = %#v", schema)
	}
	if stats := collection.Stats(); stats.DocumentCount != 4 || stats.ImmutableSegments != 1 || stats.MutableDocuments != 3 || stats.DeletedDocuments != 3 {
		t.Fatalf("v0.1 recovered stats = %#v", stats)
	}

	fetched, err := collection.Fetch(ctx, []string{"a", "b", "c", "d", "e", "f"}, Projection{IncludeVectors: true})
	if err != nil {
		t.Fatal(err)
	}
	wantIDs := []uint64{4, 1, 0, 3, 0, 6}
	for index, document := range fetched {
		if index == 2 || index == 4 {
			if document != nil {
				t.Fatalf("deleted historical document %d = %#v", index, document)
			}
			continue
		}
		if document == nil || document.DocID != wantIDs[index] {
			t.Fatalf("historical document %d = %#v", index, document)
		}
	}
	if fetched[0].Fields["rating"] != int32(10) || fetched[0].Fields["category"] != nil ||
		!reflect.DeepEqual(fetched[0].Fields["embedding"], VectorFP32{1, 0}) {
		t.Fatalf("historical update = %#v", fetched[0])
	}
	results, err := collection.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 10,
		Filter: "rating >= 4", Projection: Projection{OutputFields: []string{"rating"}},
	})
	if err != nil || !reflect.DeepEqual(documentKeys(results), []string{"f", "d", "a"}) {
		t.Fatalf("query historical snapshot = %v, %v", documentKeys(results), err)
	}

	inserted, err := collection.Insert(ctx, []Document{{PrimaryKey: "g", Fields: map[string]any{
		"rating": int32(7), "category": "current", "embedding": VectorFP32{7, 0},
	}}})
	if err != nil || inserted[0].DocID != 7 {
		t.Fatalf("continue historical document IDs = %#v, %v", inserted, err)
	}
	index := NewInvertIndexParams()
	index.EnableRangeOptimization = true
	if err := collection.AddColumn(ctx, FieldSchema{
		Name: "derived", DataType: DataTypeInt64, Index: index,
	}, "(rating * 2) + 1", AddColumnOptions{Concurrency: 2}); err != nil {
		t.Fatal(err)
	}
	if err := collection.Optimize(ctx, OptimizeOptions{Concurrency: 2}); err != nil {
		t.Fatal(err)
	}
	if manifest := collection.store.Manifest(); manifest.Generation <= 2 || manifest.WritingSegmentStartDocID != 8 {
		t.Fatalf("migrated manifest = %#v", manifest)
	}
	if err := collection.Close(); err != nil {
		t.Fatal(err)
	}

	options := NewCollectionOptions()
	options.ReadOnly = true
	collection, err = Open(ctx, path, options)
	if err != nil {
		t.Fatal(err)
	}
	defer collection.Close()
	derived, found := collection.Schema().Field("derived")
	if !found || derived.DataType != DataTypeInt64 || derived.IndexType() != IndexTypeInvert {
		t.Fatalf("migrated field = %#v, %v", derived, found)
	}
	results, err = collection.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 10,
		Filter: "derived >= 9", Projection: Projection{OutputFields: []string{"derived"}},
	})
	if err != nil || !reflect.DeepEqual(documentKeys(results), []string{"g", "f", "d", "a"}) {
		t.Fatalf("query migrated snapshot = %v, %v", documentKeys(results), err)
	}
	wantDerived := map[string]int64{"a": 21, "b": 5, "d": 9, "f": 13, "g": 15}
	for _, result := range results {
		if result.Fields["derived"] != wantDerived[result.PrimaryKey] {
			t.Fatalf("migrated result = %#v", result)
		}
	}
}

func extractHistoricalCollectionFixture(t *testing.T, fixturePath string) string {
	t.Helper()
	encoded, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatal(err)
	}
	var fixture historicalCollectionFixture
	decoder := json.NewDecoder(bytes.NewReader(encoded))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Release != "v0.1.0" || fixture.Commit != v010FixtureCommit ||
		fixture.GeneratorSHA256 != v010FixtureGenerator || fixture.ArchiveSHA256 != v010FixtureArchive {
		t.Fatalf("historical fixture identity = %#v", fixture)
	}
	archive, err := base64.StdEncoding.DecodeString(fixture.ArchiveBase64)
	if err != nil {
		t.Fatal(err)
	}
	if digest := fmt.Sprintf("%x", sha256.Sum256(archive)); digest != fixture.ArchiveSHA256 {
		t.Fatalf("historical archive digest = %s, want %s", digest, fixture.ArchiveSHA256)
	}

	root := filepath.Join(t.TempDir(), "collection")
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatal(err)
	}
	compressed, err := gzip.NewReader(bytes.NewReader(archive))
	if err != nil {
		t.Fatal(err)
	}
	defer compressed.Close()
	reader := tar.NewReader(compressed)
	var total int64
	for {
		header, err := reader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		name := strings.TrimPrefix(header.Name, "./")
		if name == "" || name == "." {
			continue
		}
		clean := filepath.Clean(filepath.FromSlash(name))
		if filepath.IsAbs(clean) || clean == ".." || strings.HasPrefix(clean, ".."+string(os.PathSeparator)) {
			t.Fatalf("unsafe historical archive path %q", header.Name)
		}
		target := filepath.Join(root, clean)
		switch header.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0o755); err != nil {
				t.Fatal(err)
			}
		case tar.TypeReg:
			if header.Size < 0 || header.Size > 1<<20 || total+header.Size > 8<<20 {
				t.Fatalf("oversized historical archive entry %q: %d", header.Name, header.Size)
			}
			total += header.Size
			if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
				t.Fatal(err)
			}
			contents := make([]byte, header.Size)
			if _, err := io.ReadFull(reader, contents); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(target, contents, 0o600); err != nil {
				t.Fatal(err)
			}
		default:
			t.Fatalf("unsupported historical archive entry %q type %d", header.Name, header.Typeflag)
		}
	}
	return root
}
