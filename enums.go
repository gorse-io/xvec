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

import "fmt"

// IndexType identifies the index implementation attached to a field.
//
// The numeric values match the public C++ header at baseline commit 58375ff.
// In particular, DiskANN is 5 and Vamana is 6. The legacy C++ protobuf used
// the opposite values; Go disk codecs must therefore map them explicitly.
type IndexType uint32

const (
	IndexTypeUndefined IndexType = 0
	IndexTypeHNSW      IndexType = 1
	IndexTypeIVF       IndexType = 2
	IndexTypeFlat      IndexType = 3
	IndexTypeDiskANN   IndexType = 5
	IndexTypeVamana    IndexType = 6
	IndexTypeIVFRaBitQ IndexType = 7
	IndexTypeInvert    IndexType = 10
	IndexTypeFTS       IndexType = 11
)

var indexTypeNames = map[IndexType]string{
	IndexTypeUndefined: "UNDEFINED",
	IndexTypeHNSW:      "HNSW",
	IndexTypeIVF:       "IVF",
	IndexTypeFlat:      "FLAT",
	IndexTypeDiskANN:   "DISKANN",
	IndexTypeVamana:    "VAMANA",
	IndexTypeIVFRaBitQ: "IVF_RABITQ",
	IndexTypeInvert:    "INVERT",
	IndexTypeFTS:       "FTS",
}

func (t IndexType) String() string { return enumName(indexTypeNames, t, "IndexType") }

// Valid reports whether t is a value defined by the public API.
func (t IndexType) Valid() bool { return enumValid(indexTypeNames, t) }

// IsVector reports whether t is a vector index type.
func (t IndexType) IsVector() bool {
	switch t {
	case IndexTypeHNSW, IndexTypeIVF, IndexTypeFlat, IndexTypeDiskANN,
		IndexTypeVamana, IndexTypeIVFRaBitQ:
		return true
	default:
		return false
	}
}

// DataType identifies a scalar, array, dense-vector, or sparse-vector field.
// Its numeric values match the public C++ header at baseline commit 58375ff.
type DataType uint32

const (
	DataTypeUndefined DataType = 0

	DataTypeBinary DataType = 1
	DataTypeString DataType = 2
	DataTypeBool   DataType = 3
	DataTypeInt32  DataType = 4
	DataTypeInt64  DataType = 5
	DataTypeUint32 DataType = 6
	DataTypeUint64 DataType = 7
	DataTypeFloat  DataType = 8
	DataTypeDouble DataType = 9

	DataTypeVectorBinary32 DataType = 20
	DataTypeVectorBinary64 DataType = 21
	DataTypeVectorFP16     DataType = 22
	DataTypeVectorFP32     DataType = 23
	DataTypeVectorFP64     DataType = 24
	DataTypeVectorInt4     DataType = 25
	DataTypeVectorInt8     DataType = 26
	DataTypeVectorInt16    DataType = 27

	DataTypeSparseVectorFP16 DataType = 30
	DataTypeSparseVectorFP32 DataType = 31

	DataTypeArrayBinary DataType = 40
	DataTypeArrayString DataType = 41
	DataTypeArrayBool   DataType = 42
	DataTypeArrayInt32  DataType = 43
	DataTypeArrayInt64  DataType = 44
	DataTypeArrayUint32 DataType = 45
	DataTypeArrayUint64 DataType = 46
	DataTypeArrayFloat  DataType = 47
	DataTypeArrayDouble DataType = 48
)

var dataTypeNames = map[DataType]string{
	DataTypeUndefined:        "UNDEFINED",
	DataTypeBinary:           "BINARY",
	DataTypeString:           "STRING",
	DataTypeBool:             "BOOL",
	DataTypeInt32:            "INT32",
	DataTypeInt64:            "INT64",
	DataTypeUint32:           "UINT32",
	DataTypeUint64:           "UINT64",
	DataTypeFloat:            "FLOAT",
	DataTypeDouble:           "DOUBLE",
	DataTypeVectorBinary32:   "VECTOR_BINARY32",
	DataTypeVectorBinary64:   "VECTOR_BINARY64",
	DataTypeVectorFP16:       "VECTOR_FP16",
	DataTypeVectorFP32:       "VECTOR_FP32",
	DataTypeVectorFP64:       "VECTOR_FP64",
	DataTypeVectorInt4:       "VECTOR_INT4",
	DataTypeVectorInt8:       "VECTOR_INT8",
	DataTypeVectorInt16:      "VECTOR_INT16",
	DataTypeSparseVectorFP16: "SPARSE_VECTOR_FP16",
	DataTypeSparseVectorFP32: "SPARSE_VECTOR_FP32",
	DataTypeArrayBinary:      "ARRAY_BINARY",
	DataTypeArrayString:      "ARRAY_STRING",
	DataTypeArrayBool:        "ARRAY_BOOL",
	DataTypeArrayInt32:       "ARRAY_INT32",
	DataTypeArrayInt64:       "ARRAY_INT64",
	DataTypeArrayUint32:      "ARRAY_UINT32",
	DataTypeArrayUint64:      "ARRAY_UINT64",
	DataTypeArrayFloat:       "ARRAY_FLOAT",
	DataTypeArrayDouble:      "ARRAY_DOUBLE",
}

func (t DataType) String() string { return enumName(dataTypeNames, t, "DataType") }

// Valid reports whether t is a value defined by the public API.
func (t DataType) Valid() bool { return enumValid(dataTypeNames, t) }

// IsDenseVector reports whether t stores a dense vector.
func (t DataType) IsDenseVector() bool {
	return t >= DataTypeVectorBinary32 && t <= DataTypeVectorInt16
}

// IsSparseVector reports whether t stores a sparse vector.
func (t DataType) IsSparseVector() bool {
	return t >= DataTypeSparseVectorFP16 && t <= DataTypeSparseVectorFP32
}

// IsVector reports whether t stores either a dense or sparse vector.
func (t DataType) IsVector() bool { return t.IsDenseVector() || t.IsSparseVector() }

// IsArray reports whether t stores an array.
func (t DataType) IsArray() bool {
	return t >= DataTypeArrayBinary && t <= DataTypeArrayDouble
}

// QuantizeType identifies a vector quantization scheme.
type QuantizeType uint32

const (
	QuantizeTypeUndefined QuantizeType = 0
	QuantizeTypeFP16      QuantizeType = 1
	QuantizeTypeInt8      QuantizeType = 2
	QuantizeTypeInt4      QuantizeType = 3
	QuantizeTypeRaBitQ    QuantizeType = 4
)

var quantizeTypeNames = map[QuantizeType]string{
	QuantizeTypeUndefined: "UNDEFINED",
	QuantizeTypeFP16:      "FP16",
	QuantizeTypeInt8:      "INT8",
	QuantizeTypeInt4:      "INT4",
	QuantizeTypeRaBitQ:    "RABITQ",
}

func (t QuantizeType) String() string {
	return enumName(quantizeTypeNames, t, "QuantizeType")
}

// Valid reports whether t is a value defined by the public API.
func (t QuantizeType) Valid() bool { return enumValid(quantizeTypeNames, t) }

// MetricType identifies the distance or similarity function used by a vector
// index. Lower scores rank first for L2, cosine distance, and MIPSL2; higher
// scores rank first for IP.
type MetricType uint32

const (
	MetricTypeUndefined MetricType = 0
	MetricTypeL2        MetricType = 1
	MetricTypeIP        MetricType = 2
	MetricTypeCosine    MetricType = 3
	MetricTypeMIPSL2    MetricType = 4
)

var metricTypeNames = map[MetricType]string{
	MetricTypeUndefined: "UNDEFINED",
	MetricTypeL2:        "L2",
	MetricTypeIP:        "IP",
	MetricTypeCosine:    "COSINE",
	MetricTypeMIPSL2:    "MIPSL2",
}

func (t MetricType) String() string { return enumName(metricTypeNames, t, "MetricType") }

// Valid reports whether t is a value defined by the public API.
func (t MetricType) Valid() bool { return enumValid(metricTypeNames, t) }

// Operator identifies a write operation.
type Operator uint32

const (
	OperatorInsert Operator = 0
	OperatorUpsert Operator = 1
	OperatorUpdate Operator = 2
	OperatorDelete Operator = 3
)

var operatorNames = map[Operator]string{
	OperatorInsert: "INSERT",
	OperatorUpsert: "UPSERT",
	OperatorUpdate: "UPDATE",
	OperatorDelete: "DELETE",
}

func (o Operator) String() string { return enumName(operatorNames, o, "Operator") }

// Valid reports whether o is a value defined by the public API.
func (o Operator) Valid() bool { return enumValid(operatorNames, o) }

// CompareOp identifies a scalar filter predicate.
type CompareOp uint32

const (
	CompareOpNone          CompareOp = 0
	CompareOpEQ            CompareOp = 1
	CompareOpNE            CompareOp = 2
	CompareOpLT            CompareOp = 3
	CompareOpLE            CompareOp = 4
	CompareOpGT            CompareOp = 5
	CompareOpGE            CompareOp = 6
	CompareOpLike          CompareOp = 7
	CompareOpContainAll    CompareOp = 8
	CompareOpContainAny    CompareOp = 9
	CompareOpNotContainAll CompareOp = 10
	CompareOpNotContainAny CompareOp = 11
	CompareOpIsNull        CompareOp = 12
	CompareOpIsNotNull     CompareOp = 13
	CompareOpHasPrefix     CompareOp = 14
	CompareOpHasSuffix     CompareOp = 15
)

var compareOpNames = map[CompareOp]string{
	CompareOpNone:          "NONE",
	CompareOpEQ:            "EQ",
	CompareOpNE:            "NE",
	CompareOpLT:            "LT",
	CompareOpLE:            "LE",
	CompareOpGT:            "GT",
	CompareOpGE:            "GE",
	CompareOpLike:          "LIKE",
	CompareOpContainAll:    "CONTAIN_ALL",
	CompareOpContainAny:    "CONTAIN_ANY",
	CompareOpNotContainAll: "NOT_CONTAIN_ALL",
	CompareOpNotContainAny: "NOT_CONTAIN_ANY",
	CompareOpIsNull:        "IS_NULL",
	CompareOpIsNotNull:     "IS_NOT_NULL",
	CompareOpHasPrefix:     "HAS_PREFIX",
	CompareOpHasSuffix:     "HAS_SUFFIX",
}

func (o CompareOp) String() string { return enumName(compareOpNames, o, "CompareOp") }

// Valid reports whether o is a value defined by the public API.
func (o CompareOp) Valid() bool { return enumValid(compareOpNames, o) }

// RelationOp identifies the boolean relationship between filter expressions.
type RelationOp uint32

const (
	RelationOpNone RelationOp = 0
	RelationOpAnd  RelationOp = 1
	RelationOpOr   RelationOp = 2
)

var relationOpNames = map[RelationOp]string{
	RelationOpNone: "NONE",
	RelationOpAnd:  "AND",
	RelationOpOr:   "OR",
}

func (o RelationOp) String() string { return enumName(relationOpNames, o, "RelationOp") }

// Valid reports whether o is a value defined by the public API.
func (o RelationOp) Valid() bool { return enumValid(relationOpNames, o) }

// BlockType identifies a persisted collection block.
type BlockType uint32

const (
	BlockTypeUndefined           BlockType = 0
	BlockTypeScalar              BlockType = 1
	BlockTypeScalarIndex         BlockType = 2
	BlockTypeVectorIndex         BlockType = 3
	BlockTypeVectorIndexQuantize BlockType = 4
	BlockTypeFTSIndex            BlockType = 5
)

var blockTypeNames = map[BlockType]string{
	BlockTypeUndefined:           "UNDEFINED",
	BlockTypeScalar:              "SCALAR",
	BlockTypeScalarIndex:         "SCALAR_INDEX",
	BlockTypeVectorIndex:         "VECTOR_INDEX",
	BlockTypeVectorIndexQuantize: "VECTOR_INDEX_QUANTIZE",
	BlockTypeFTSIndex:            "FTS_INDEX",
}

func (t BlockType) String() string { return enumName(blockTypeNames, t, "BlockType") }

// Valid reports whether t is a value defined by the public API.
func (t BlockType) Valid() bool { return enumValid(blockTypeNames, t) }

// FileFormat identifies an import or export file format.
type FileFormat uint32

const (
	FileFormatUnknown FileFormat = 0
	FileFormatIPC     FileFormat = 1
	FileFormatParquet FileFormat = 2
)

var fileFormatNames = map[FileFormat]string{
	FileFormatUnknown: "UNKNOWN",
	FileFormatIPC:     "IPC",
	FileFormatParquet: "PARQUET",
}

func (f FileFormat) String() string { return enumName(fileFormatNames, f, "FileFormat") }

// Valid reports whether f is a value defined by the public API.
func (f FileFormat) Valid() bool { return enumValid(fileFormatNames, f) }

// ColumnOp identifies a schema mutation operation.
type ColumnOp uint32

const (
	ColumnOpUndefined ColumnOp = 0
	ColumnOpAdd       ColumnOp = 1
	ColumnOpAlter     ColumnOp = 2
	ColumnOpDrop      ColumnOp = 3
)

var columnOpNames = map[ColumnOp]string{
	ColumnOpUndefined: "UNDEFINED",
	ColumnOpAdd:       "ADD",
	ColumnOpAlter:     "ALTER",
	ColumnOpDrop:      "DROP",
}

func (o ColumnOp) String() string { return enumName(columnOpNames, o, "ColumnOp") }

// Valid reports whether o is a value defined by the public API.
func (o ColumnOp) Valid() bool { return enumValid(columnOpNames, o) }

func enumName[T ~uint32](names map[T]string, value T, typeName string) string {
	if name, ok := names[value]; ok {
		return name
	}
	return fmt.Sprintf("%s(%d)", typeName, uint32(value))
}

func enumValid[T ~uint32](names map[T]string, value T) bool {
	_, ok := names[value]
	return ok
}
