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

package core

import (
	"encoding/binary"
	"fmt"
	"math"
	"slices"

	"github.com/gorse-io/xvec/pkg/rabitq"
)

// RaBitQCode is one immutable split-single code. binData and exData use the
// layouts owned by pkg/rabitq, including their float32 estimator factors.
type RaBitQCode struct {
	modelFingerprint uint64
	cluster          int
	paddedDimension  int
	totalBits        int
	binData          []byte
	exData           []byte
}

func (c RaBitQCode) Cluster() int { return c.cluster }

func (c RaBitQCode) PaddedDimension() int { return c.paddedDimension }

func (c RaBitQCode) TotalBits() int { return c.totalBits }

func (c RaBitQCode) BinaryCode() []byte {
	if c.paddedDimension <= 0 || c.paddedDimension%8 != 0 || len(c.binData) < c.paddedDimension/8 {
		return nil
	}
	return slices.Clone(c.binData[:c.paddedDimension/8])
}

func (c RaBitQCode) ExtraCode() []byte {
	extraBits := c.totalBits - 1
	if extraBits == 0 {
		return []byte{}
	}
	if c.paddedDimension <= 0 || c.paddedDimension%8 != 0 || extraBits < 0 || extraBits > 8 ||
		len(c.exData) < c.paddedDimension*extraBits/8 {
		return nil
	}
	return slices.Clone(c.exData[:c.paddedDimension*extraBits/8])
}

// QuantizedValues expands the split code into one unsigned total-bit value per
// padded coordinate. It is intended for diagnostics and fixtures, not search.
func (c RaBitQCode) QuantizedValues() ([]uint16, error) {
	if err := c.validate(); err != nil {
		return nil, err
	}
	extraBits := c.totalBits - 1
	values := make([]uint16, c.paddedDimension)
	binCode := rabitq.NewBinDataMap(c.binData, c.paddedDimension).BinCode()
	if extraBits > 0 {
		ipFunc, err := rabitq.SelectExcodeIPFunc(extraBits)
		if err != nil {
			return nil, err
		}
		extraCode := rabitq.NewExDataMap(c.exData, c.paddedDimension, extraBits).ExCode()
		basis := make([]float32, c.paddedDimension)
		for index := range values {
			basis[index] = 1
			values[index] = uint16(ipFunc(basis, extraCode))
			basis[index] = 0
		}
	}
	for index := range values {
		word := binary.LittleEndian.Uint64(binCode[index/64*8:])
		if word&(uint64(1)<<uint(63-index%64)) != 0 {
			values[index] |= uint16(1) << uint(extraBits)
		}
	}
	return values, nil
}

func (c RaBitQCode) validate() error {
	if c.modelFingerprint == 0 || c.cluster < 0 || c.paddedDimension < MinRaBitQDimension || c.paddedDimension%64 != 0 ||
		c.totalBits < MinRaBitQTotalBits || c.totalBits > MaxRaBitQTotalBits ||
		len(c.binData) != rabitq.BinDataBytes(c.paddedDimension) ||
		len(c.exData) != rabitq.ExDataBytes(c.paddedDimension, c.totalBits-1) {
		return ErrInvalidRaBitQCode
	}
	bin := rabitq.NewBinDataMap(c.binData, c.paddedDimension)
	factors := []float32{bin.FAdd(), bin.FRescale(), bin.FError()}
	if c.totalBits > 1 {
		ex := rabitq.NewExDataMap(c.exData, c.paddedDimension, c.totalBits-1)
		factors = append(factors, ex.FAddEx(), ex.FRescaleEx())
	}
	for _, factor := range factors {
		if math.IsNaN(float64(factor)) || math.IsInf(float64(factor), 0) {
			return ErrInvalidRaBitQCode
		}
	}
	if bin.FError() < 0 {
		return ErrInvalidRaBitQCode
	}
	return nil
}

func quantizeRaBitQVector(
	vector, centroid []float32,
	cluster, totalBits int,
	extraScale float64,
	innerProduct bool,
) (RaBitQCode, error) {
	if len(vector) == 0 || len(vector) != len(centroid) || len(vector)%64 != 0 || cluster < 0 ||
		totalBits < MinRaBitQTotalBits || totalBits > MaxRaBitQTotalBits {
		return RaBitQCode{}, fmt.Errorf("%w: invalid quantization arguments", ErrInvalidRaBitQCode)
	}
	extraBits := totalBits - 1
	if extraBits > 0 && (extraScale <= 0 || math.IsNaN(extraScale) || math.IsInf(extraScale, 0)) {
		return RaBitQCode{}, fmt.Errorf("%w: invalid extra-code scale", ErrInvalidRaBitQCode)
	}
	metric := rabitq.MetricL2
	if innerProduct {
		metric = rabitq.MetricIP
	}
	code := RaBitQCode{
		cluster: cluster, paddedDimension: len(vector), totalBits: totalBits,
		binData: make([]byte, rabitq.BinDataBytes(len(vector))),
		exData:  make([]byte, rabitq.ExDataBytes(len(vector), extraBits)),
	}
	if err := rabitq.QuantizeSplitSingle(
		vector, centroid, len(vector), extraBits, code.binData, code.exData,
		metric, rabitq.RaBitQConfig{TConst: extraScale},
	); err != nil {
		return RaBitQCode{}, fmt.Errorf("%w: %v", ErrInvalidRaBitQCode, err)
	}
	return code, nil
}
