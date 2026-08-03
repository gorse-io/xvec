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
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
)

const schemaCodecVersion uint16 = 1

type diskCollectionSchema struct {
	CodecVersion      uint16            `json:"codec_version"`
	Name              string            `json:"name"`
	Fields            []diskFieldSchema `json:"fields"`
	MaxDocsPerSegment uint64            `json:"max_docs_per_segment"`
}

type diskFieldSchema struct {
	Name      string          `json:"name"`
	DataType  DataType        `json:"data_type"`
	Nullable  bool            `json:"nullable"`
	Dimension uint32          `json:"dimension"`
	IndexType uint32          `json:"index_type"`
	Index     json.RawMessage `json:"index,omitempty"`
}

func marshalCollectionSchema(schema CollectionSchema) ([]byte, error) {
	if err := schema.Validate(); err != nil {
		return nil, err
	}
	disk := diskCollectionSchema{
		CodecVersion: schemaCodecVersion, Name: schema.Name,
		MaxDocsPerSegment: schema.MaxDocsPerSegment,
		Fields:            make([]diskFieldSchema, len(schema.Fields)),
	}
	for index, field := range schema.Fields {
		diskField := diskFieldSchema{
			Name: field.Name, DataType: field.DataType,
			Nullable: field.Nullable, Dimension: field.Dimension,
		}
		if !indexParamsNil(field.Index) {
			code, err := encodeDiskIndexType(field.Index.IndexType())
			if err != nil {
				return nil, err
			}
			diskField.IndexType = code
			diskField.Index, err = json.Marshal(field.Index)
			if err != nil {
				return nil, fmt.Errorf("zvec: encode schema index for field %q: %w", field.Name, err)
			}
		}
		disk.Fields[index] = diskField
	}
	encoded, err := json.Marshal(disk)
	if err != nil {
		return nil, fmt.Errorf("zvec: encode collection schema: %w", err)
	}
	return encoded, nil
}

func unmarshalCollectionSchema(encoded []byte) (CollectionSchema, error) {
	var disk diskCollectionSchema
	if err := decodeStrictJSON(encoded, &disk); err != nil {
		return CollectionSchema{}, fmt.Errorf("decode schema JSON: %w", err)
	}
	if disk.CodecVersion != schemaCodecVersion {
		return CollectionSchema{}, fmt.Errorf("unsupported schema codec version %d", disk.CodecVersion)
	}
	schema := CollectionSchema{
		Name: disk.Name, MaxDocsPerSegment: disk.MaxDocsPerSegment,
		Fields: make([]FieldSchema, len(disk.Fields)),
	}
	for index, diskField := range disk.Fields {
		field := FieldSchema{
			Name: diskField.Name, DataType: diskField.DataType,
			Nullable: diskField.Nullable, Dimension: diskField.Dimension,
		}
		if diskField.IndexType == 0 {
			if len(diskField.Index) != 0 {
				return CollectionSchema{}, fmt.Errorf("field %q has index parameters without an index type", field.Name)
			}
		} else {
			indexType, err := decodeDiskIndexType(diskField.IndexType)
			if err != nil {
				return CollectionSchema{}, fmt.Errorf("field %q: %w", field.Name, err)
			}
			if len(diskField.Index) == 0 {
				return CollectionSchema{}, fmt.Errorf("field %q has index type %s without parameters", field.Name, indexType)
			}
			field.Index, err = decodeDiskIndexParams(indexType, diskField.Index)
			if err != nil {
				return CollectionSchema{}, fmt.Errorf("field %q: %w", field.Name, err)
			}
		}
		schema.Fields[index] = field
	}
	if err := schema.Validate(); err != nil {
		return CollectionSchema{}, fmt.Errorf("decoded schema is invalid: %w", err)
	}
	return schema, nil
}

// encodeDiskIndexType deliberately spells out the public-header mapping. In
// particular DiskANN=5 and Vamana=6 must never inherit the reversed legacy
// protobuf values.
func encodeDiskIndexType(indexType IndexType) (uint32, error) {
	switch indexType {
	case IndexTypeHNSW:
		return 1, nil
	case IndexTypeIVF:
		return 2, nil
	case IndexTypeFlat:
		return 3, nil
	case IndexTypeHNSWRaBitQ:
		return 4, nil
	case IndexTypeDiskANN:
		return 5, nil
	case IndexTypeVamana:
		return 6, nil
	case IndexTypeInvert:
		return 10, nil
	case IndexTypeFTS:
		return 11, nil
	default:
		return 0, fmt.Errorf("unsupported schema index type %d", indexType)
	}
}

func decodeDiskIndexType(code uint32) (IndexType, error) {
	switch code {
	case 1:
		return IndexTypeHNSW, nil
	case 2:
		return IndexTypeIVF, nil
	case 3:
		return IndexTypeFlat, nil
	case 4:
		return IndexTypeHNSWRaBitQ, nil
	case 5:
		return IndexTypeDiskANN, nil
	case 6:
		return IndexTypeVamana, nil
	case 10:
		return IndexTypeInvert, nil
	case 11:
		return IndexTypeFTS, nil
	default:
		return IndexTypeUndefined, fmt.Errorf("unknown disk index type %d", code)
	}
}

func decodeDiskIndexParams(indexType IndexType, encoded []byte) (IndexParams, error) {
	var target IndexParams
	switch indexType {
	case IndexTypeHNSW:
		target = &HNSWIndexParams{}
	case IndexTypeIVF:
		target = &IVFIndexParams{}
	case IndexTypeFlat:
		target = &FlatIndexParams{}
	case IndexTypeHNSWRaBitQ:
		target = &HNSWRaBitQIndexParams{}
	case IndexTypeDiskANN:
		target = &DiskANNIndexParams{}
	case IndexTypeVamana:
		target = &VamanaIndexParams{}
	case IndexTypeInvert:
		target = &InvertIndexParams{}
	case IndexTypeFTS:
		target = &FTSIndexParams{}
	default:
		return nil, fmt.Errorf("unsupported index type %d", indexType)
	}
	if err := decodeStrictJSON(encoded, target); err != nil {
		return nil, fmt.Errorf("decode %s index parameters: %w", indexType, err)
	}
	return target.cloneIndexParams(), nil
}

func decodeStrictJSON(encoded []byte, target any) error {
	decoder := json.NewDecoder(bytes.NewReader(encoded))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return err
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("trailing JSON value")
		}
		return err
	}
	return nil
}
