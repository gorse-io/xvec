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

package db

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/gorse-io/zvec/internal/ailego"
)

func TestManifestRoundTrip(t *testing.T) {
	t.Parallel()

	original := sampleManifest(42)
	encoded, err := MarshalManifest(original)
	if err != nil {
		t.Fatalf("marshal manifest: %v", err)
	}
	decoded, err := UnmarshalManifest(encoded)
	if err != nil {
		t.Fatalf("unmarshal manifest: %v", err)
	}
	if !reflect.DeepEqual(decoded, original) {
		t.Fatalf("decoded = %#v, want %#v", decoded, original)
	}
}

func TestManifestCloneIsIndependent(t *testing.T) {
	t.Parallel()

	original := sampleManifest(1)
	clone := original.Clone()
	clone.Schema[0] = '['
	clone.PersistedSegments[0].Files[0] = "changed"
	clone.WritingSegment.Files[0] = "changed"
	if json.Valid(clone.Schema) || !json.Valid(original.Schema) {
		t.Fatal("schema clone shares storage")
	}
	if original.PersistedSegments[0].Files[0] == "changed" {
		t.Fatal("persisted segment clone shares files")
	}
	if original.WritingSegment.Files[0] == "changed" {
		t.Fatal("writing segment clone shares files")
	}
}

func TestManifestValidation(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		mutate   func(*Manifest)
		expected error
	}{
		{name: "unsupported format", mutate: func(m *Manifest) { m.FormatVersion = 2 }, expected: ErrUnsupportedFormatVersion},
		{name: "zero generation", mutate: func(m *Manifest) { m.Generation = 0 }, expected: ErrManifestCorrupt},
		{name: "invalid schema", mutate: func(m *Manifest) { m.Schema = json.RawMessage(`{`) }, expected: ErrManifestCorrupt},
		{name: "null schema", mutate: func(m *Manifest) { m.Schema = json.RawMessage(`null`) }, expected: ErrManifestCorrupt},
		{name: "array schema", mutate: func(m *Manifest) { m.Schema = json.RawMessage(`[]`) }, expected: ErrManifestCorrupt},
		{name: "zero segment capacity", mutate: func(m *Manifest) { m.SegmentMaxDocuments = 0 }, expected: ErrManifestCorrupt},
		{name: "duplicate segment", mutate: func(m *Manifest) { m.PersistedSegments = append(m.PersistedSegments, m.PersistedSegments[0]) }, expected: ErrManifestCorrupt},
		{name: "writing already persisted", mutate: func(m *Manifest) { m.WritingSegment.ID = m.PersistedSegments[0].ID }, expected: ErrManifestCorrupt},
		{name: "next ID not advanced", mutate: func(m *Manifest) { m.NextSegmentID = m.WritingSegment.ID }, expected: ErrManifestCorrupt},
		{name: "empty segment range", mutate: func(m *Manifest) { m.WritingSegment.DocCount = 0 }, expected: ErrManifestCorrupt},
		{name: "descending range", mutate: func(m *Manifest) { m.WritingSegment.MinDocID = 100 }, expected: ErrManifestCorrupt},
		{name: "count exceeds range", mutate: func(m *Manifest) { m.WritingSegment.DocCount = 100 }, expected: ErrManifestCorrupt},
		{name: "absolute file", mutate: func(m *Manifest) { m.WritingSegment.Files = []string{"/data.seg"} }, expected: ErrManifestCorrupt},
		{name: "parent file", mutate: func(m *Manifest) { m.WritingSegment.Files = []string{"../data.seg"} }, expected: ErrManifestCorrupt},
		{name: "unclean file", mutate: func(m *Manifest) { m.WritingSegment.Files = []string{"segment/../data.seg"} }, expected: ErrManifestCorrupt},
		{name: "windows separator", mutate: func(m *Manifest) { m.WritingSegment.Files = []string{`segment\data.seg`} }, expected: ErrManifestCorrupt},
		{name: "duplicate file", mutate: func(m *Manifest) { m.WritingSegment.Files = []string{"data.seg", "data.seg"} }, expected: ErrManifestCorrupt},
	}

	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			manifest := sampleManifest(1)
			testCase.mutate(&manifest)
			if err := manifest.Validate(); !errors.Is(err, testCase.expected) {
				t.Fatalf("validation error = %v, want %v", err, testCase.expected)
			}
		})
	}

	fullRange := Manifest{
		FormatVersion:       DiskFormatVersion,
		Generation:          1,
		Schema:              json.RawMessage(`{}`),
		SegmentMaxDocuments: 1,
		PersistedSegments: []SegmentMetadata{{
			ID: 1, MinDocID: 0, MaxDocID: math.MaxUint64, DocCount: math.MaxUint64,
		}},
		NextSegmentID: 2,
	}
	if err := fullRange.Validate(); err != nil {
		t.Fatalf("full document range: %v", err)
	}
}

func TestManifestDetectsCorruptionAndTruncation(t *testing.T) {
	t.Parallel()

	encoded, err := MarshalManifest(sampleManifest(7))
	if err != nil {
		t.Fatal(err)
	}
	for length := 0; length < len(encoded); length++ {
		if _, err := UnmarshalManifest(encoded[:length]); err == nil {
			t.Fatalf("truncation at %d bytes succeeded", length)
		}
	}

	tests := []struct {
		name     string
		mutate   func([]byte) []byte
		expected error
	}{
		{name: "magic", mutate: func(data []byte) []byte { data[0] ^= 0xff; return data }, expected: ErrManifestCorrupt},
		{name: "format", mutate: func(data []byte) []byte { binary.LittleEndian.PutUint16(data[8:10], 2); return data }, expected: ErrUnsupportedFormatVersion},
		{name: "header size", mutate: func(data []byte) []byte { binary.LittleEndian.PutUint16(data[10:12], 31); return data }, expected: ErrManifestCorrupt},
		{name: "generation", mutate: func(data []byte) []byte { binary.LittleEndian.PutUint64(data[12:20], 8); return data }, expected: ErrManifestCorrupt},
		{name: "huge length", mutate: func(data []byte) []byte { binary.LittleEndian.PutUint64(data[20:28], math.MaxUint64); return data }, expected: ErrManifestCorrupt},
		{name: "checksum", mutate: func(data []byte) []byte { data[28] ^= 1; return data }, expected: ErrManifestCorrupt},
		{name: "payload", mutate: func(data []byte) []byte { data[len(data)-1] ^= 1; return data }, expected: ErrManifestCorrupt},
		{name: "trailing", mutate: func(data []byte) []byte { return append(data, 0) }, expected: ErrManifestCorrupt},
	}
	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			corrupted := testCase.mutate(append([]byte(nil), encoded...))
			if _, err := UnmarshalManifest(corrupted); !errors.Is(err, testCase.expected) {
				t.Fatalf("error = %v, want %v", err, testCase.expected)
			}
		})
	}
}

func TestManifestRejectsUnknownPayloadField(t *testing.T) {
	t.Parallel()

	encoded, err := MarshalManifest(sampleManifest(1))
	if err != nil {
		t.Fatal(err)
	}
	var payload map[string]any
	if err := json.Unmarshal(encoded[manifestHeaderSize:], &payload); err != nil {
		t.Fatal(err)
	}
	payload["future_field"] = true
	newPayload, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	encoded = append(encoded[:manifestHeaderSize], newPayload...)
	binary.LittleEndian.PutUint64(encoded[20:28], uint64(len(newPayload)))
	binary.LittleEndian.PutUint32(encoded[28:32], ailego.CRC32C(newPayload))
	if _, err := UnmarshalManifest(encoded); !errors.Is(err, ErrManifestCorrupt) {
		t.Fatalf("unknown field error = %v", err)
	}
}

func FuzzUnmarshalManifest(f *testing.F) {
	encoded, err := MarshalManifest(sampleManifest(1))
	if err != nil {
		f.Fatal(err)
	}
	f.Add(encoded)
	f.Add(encoded[:manifestHeaderSize])
	f.Add([]byte("not a manifest"))
	f.Fuzz(func(t *testing.T, data []byte) {
		manifest, err := UnmarshalManifest(data)
		if err == nil {
			if err := manifest.Validate(); err != nil {
				t.Fatalf("decoded invalid manifest: %v", err)
			}
		}
	})
}

func sampleManifest(generation uint64) Manifest {
	return Manifest{
		FormatVersion:       DiskFormatVersion,
		Generation:          generation,
		Schema:              json.RawMessage(`{"name":"books","version":1}`),
		EnableMmap:          true,
		SegmentMaxDocuments: 100,
		PersistedSegments: []SegmentMetadata{{
			ID: 3, MinDocID: 10, MaxDocID: 19, DocCount: 8,
			Files: []string{"segments/3/data.seg", "segments/3/delete.snapshot"},
		}},
		WritingSegment: &SegmentMetadata{
			ID: 4, MinDocID: 20, MaxDocID: 21, DocCount: 2,
			Files: []string{"segments/4/data.wal"},
		},
		IDMapGeneration:          5,
		DeleteSnapshotGeneration: 6,
		NextSegmentID:            5,
	}
}
