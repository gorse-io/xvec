// SPDX-License-Identifier: Apache-2.0
package xvec

import (
	"encoding/binary"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestFP16DocumentArenaDecodingAndOwnership(t *testing.T) {
	schema := CollectionSchema{Fields: []FieldSchema{{Name: "dense", DataType: DataTypeVectorFP16, Dimension: 3}}}
	arenas := newFP16DocumentArenas(schema, 600)
	var rows []VectorFP16
	for n := 0; n < 600; n++ {
		input := VectorFP16{Float16(n), 0x8000, 1}
		encoded, err := marshalDocumentPayload(map[string]any{"dense": input})
		require.NoError(t, err)
		fields, err := unmarshalDocumentPayloadWithVectorArenas(encoded, nil, arenas)
		require.NoError(t, err)
		require.Equal(t, input, fields["dense"])
		row := fields["dense"].(VectorFP16)
		require.Equal(t, len(row), cap(row), "rows cannot append into another document")
		rows = append(rows, row)
		for i := range encoded {
			encoded[i] = 0
		}
	}
	arenas = nil
	for n, row := range rows {
		require.Equal(t, Float16(n), row[0])
	}
	require.NotSame(t, &rows[0][0], &rows[1][0])
	copy, _, err := cloneDocumentValue(rows[0])
	require.NoError(t, err)
	copy.(VectorFP16)[0] = 999
	require.Zero(t, rows[0][0])
	sparse := schema.Clone()
	sparse.Fields[0].Nullable = true
	require.Nil(t, newFP16DocumentArenas(sparse, 600))
}

func TestFP16DocumentArenaRejectsMalformedRows(t *testing.T) {
	arena := &fp16DocumentArena{dimension: 2, blockRows: 2}
	_, err := arena.decode(1, []byte{0, 0})
	require.Error(t, err)
	_, err = arena.decode(2, []byte{0, 0})
	require.Error(t, err)
	for _, bits := range []uint16{0x7c00, 0xfc00, 0x7c01} {
		encoded := binary.LittleEndian.AppendUint16(nil, bits)
		encoded = binary.LittleEndian.AppendUint16(encoded, 0)
		_, err = arena.decode(2, encoded)
		require.Error(t, err)
	}
	row, err := arena.decode(2, []byte{0, 0, 0, 0})
	require.NoError(t, err)
	require.Equal(t, VectorFP16{0, 0}, row)
	require.Len(t, arena.remaining, 2, "invalid rows must not consume the arena")
}
