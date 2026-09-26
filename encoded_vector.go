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
	"context"
	"encoding/binary"
	"errors"
	"math"
)

// encodedVectorFP32 borrows validated little-endian bytes from a read-only
// collection snapshot. It is never exposed through the public document API.
// Close drains query leases before releasing the backing segment mappings.
// Explicit decoding avoids alignment and host-endianness assumptions.
type encodedVectorFP32 []byte

func (v encodedVectorFP32) DataType() DataType { return DataTypeVectorFP32 }
func (v encodedVectorFP32) Dimension() int     { return len(v) / 4 }
func (encodedVectorFP32) denseVector()         {}

func (v encodedVectorFP32) validate() error {
	if len(v)%4 != 0 {
		return errors.New("invalid encoded FP32 vector length")
	}
	for offset := 0; offset < len(v); offset += 4 {
		if !finiteDocumentFloat(float64(math.Float32frombits(binary.LittleEndian.Uint32(v[offset:])))) {
			return errors.New("non-finite encoded FP32 vector")
		}
	}
	return nil
}

func (v encodedVectorFP32) readInto(destination []float32) error {
	if len(v)%4 != 0 || len(destination) != v.Dimension() {
		return errors.New("encoded FP32 vector dimension mismatch")
	}
	for i := range destination {
		destination[i] = math.Float32frombits(binary.LittleEndian.Uint32(v[i*4:]))
	}
	return nil
}

func (v encodedVectorFP32) decode() (VectorFP32, error) {
	vector := make(VectorFP32, v.Dimension())
	if err := v.readInto(vector); err != nil {
		return nil, err
	}
	return vector, validateFiniteFloat32s(vector)
}

type encodedDenseReader struct{ rows []encodedVectorFP32 }

func (r *encodedDenseReader) ReadVector(position int, destination []float32) error {
	if position < 0 || position >= len(r.rows) {
		return errors.New("encoded vector position out of range")
	}
	return r.rows[position].readInto(destination)
}

func collectionEncodedDenseReader(ctx context.Context, field FieldSchema, documents []Document) (*encodedDenseReader, []uint64, error) {
	// Avoid extra allocation on the ordinary decoded-vector path.
	found := false
	for _, document := range documents {
		value := document.Fields[field.Name]
		if value == nil {
			continue
		}
		if _, ok := value.(encodedVectorFP32); !ok {
			return nil, nil, nil
		}
		found = true
		break
	}
	if !found {
		return nil, nil, nil
	}
	reader := &encodedDenseReader{rows: make([]encodedVectorFP32, 0, len(documents))}
	keys := make([]uint64, 0, len(documents))
	for _, document := range documents {
		if err := ctx.Err(); err != nil {
			return nil, nil, err
		}
		value := document.Fields[field.Name]
		if value == nil {
			continue
		}
		vector, ok := value.(encodedVectorFP32)
		if !ok {
			return nil, nil, nil
		}
		reader.rows = append(reader.rows, vector)
		keys = append(keys, document.DocID)
	}
	return reader, keys, nil
}
