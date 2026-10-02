// SPDX-License-Identifier: Apache-2.0
package xvec

import (
	"encoding/binary"
	"errors"
)

// Private snapshots share typed FP16 blocks. Each row retains its block through
// a Go slice, so leases need no manual memory release and published rows never
// move. The last block wastes at most 255 rows per field.
type fp16DocumentArena struct {
	dimension int
	blockRows int
	remaining VectorFP16
}

func newFP16DocumentArenas(schema CollectionSchema, documents int) map[string]*fp16DocumentArena {
	if documents <= 0 {
		return nil
	}
	var arenas map[string]*fp16DocumentArena
	for _, field := range schema.Fields {
		// Sparse nullable columns keep per-row allocation to avoid padding blocks.
		if field.DataType != DataTypeVectorFP16 || field.Nullable {
			continue
		}
		if arenas == nil {
			arenas = make(map[string]*fp16DocumentArena)
		}
		arenas[field.Name] = &fp16DocumentArena{dimension: int(field.Dimension), blockRows: min(256, documents)}
	}
	return arenas
}

func (a *fp16DocumentArena) decode(count uint32, data []byte) (VectorFP16, error) {
	if uint64(count)*2 != uint64(len(data)) || int(count) != a.dimension || a.dimension <= 0 {
		return nil, errors.New("FP16 vector length mismatch")
	}
	if len(a.remaining) < a.dimension {
		if a.blockRows <= 0 || a.dimension > int(^uint(0)>>1)/a.blockRows {
			return nil, errors.New("FP16 arena capacity exceeded")
		}
		a.remaining = make(VectorFP16, a.blockRows*a.dimension)
	}
	row := a.remaining[:a.dimension:a.dimension]
	for n := range row {
		bits := binary.LittleEndian.Uint16(data[n*2:])
		if bits&0x7c00 == 0x7c00 {
			return nil, errors.New("non-finite FP16 vector")
		}
		row[n] = Float16(bits)
	}
	a.remaining = a.remaining[a.dimension:]
	return row, nil
}
