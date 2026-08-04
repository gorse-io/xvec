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
	"context"
	"errors"
	"fmt"

	"github.com/gorse-io/zvec/internal/core"
	dbsql "github.com/gorse-io/zvec/internal/db/sql"
)

func buildFilterPlan(filter string, schema CollectionSchema) (*dbsql.Plan, error) {
	if filter == "" {
		return nil, nil
	}
	fields := make([]dbsql.Field, len(schema.Fields))
	for index, field := range schema.Fields {
		kind, array, supported := filterValueKind(field.DataType)
		filterable := supported && field.IndexType() != IndexTypeFTS
		indexed, rangeOptimized, extendedWildcard := filterIndexOptions(field, filterable)
		fields[index] = dbsql.Field{
			Name: field.Name, Kind: kind, Array: array, Nullable: field.Nullable, Filterable: filterable,
			Indexed: indexed, RangeOptimized: rangeOptimized, ExtendedWildcard: extendedWildcard,
		}
	}
	filterSchema, err := dbsql.NewSchema(fields)
	if err != nil {
		return nil, fmt.Errorf("build filter schema: %w", err)
	}
	return dbsql.BuildPlan(filter, filterSchema)
}

func filterIndexOptions(field FieldSchema, filterable bool) (indexed, rangeOptimized, extendedWildcard bool) {
	if !filterable || indexParamsNil(field.Index) || field.Index.IndexType() != IndexTypeInvert {
		return false, false, false
	}
	var params InvertIndexParams
	switch value := field.Index.(type) {
	case InvertIndexParams:
		params = value
	case *InvertIndexParams:
		if value == nil {
			return false, false, false
		}
		params = *value
	default:
		return false, false, false
	}
	return true, params.EnableRangeOptimization, params.EnableExtendedWildcard
}

func filterValueKind(dataType DataType) (kind dbsql.ValueKind, array, supported bool) {
	switch dataType {
	case DataTypeBinary:
		return dbsql.ValueBinary, false, true
	case DataTypeString:
		return dbsql.ValueString, false, true
	case DataTypeBool:
		return dbsql.ValueBool, false, true
	case DataTypeInt32:
		return dbsql.ValueInt32, false, true
	case DataTypeInt64:
		return dbsql.ValueInt64, false, true
	case DataTypeUint32:
		return dbsql.ValueUint32, false, true
	case DataTypeUint64:
		return dbsql.ValueUint64, false, true
	case DataTypeFloat:
		return dbsql.ValueFloat32, false, true
	case DataTypeDouble:
		return dbsql.ValueFloat64, false, true
	case DataTypeArrayBinary:
		return dbsql.ValueBinary, true, true
	case DataTypeArrayString:
		return dbsql.ValueString, true, true
	case DataTypeArrayBool:
		return dbsql.ValueBool, true, true
	case DataTypeArrayInt32:
		return dbsql.ValueInt32, true, true
	case DataTypeArrayInt64:
		return dbsql.ValueInt64, true, true
	case DataTypeArrayUint32:
		return dbsql.ValueUint32, true, true
	case DataTypeArrayUint64:
		return dbsql.ValueUint64, true, true
	case DataTypeArrayFloat:
		return dbsql.ValueFloat32, true, true
	case DataTypeArrayDouble:
		return dbsql.ValueFloat64, true, true
	default:
		return 0, false, false
	}
}

type evaluatedFilter struct {
	predicate core.CandidateFilter
	ordinals  []uint32
	matched   uint64
	total     uint64
	present   bool
	usedIndex bool
}

func (f evaluatedFilter) useBruteForce(ratio float32) bool {
	return f.present && f.total > 0 && f.matched <= uint64(float64(f.total)*float64(ratio))
}

func evaluateFilterDocuments(ctx context.Context, plan *dbsql.Plan, documents []Document, invertToForwardRatio float32) (evaluatedFilter, error) {
	if plan == nil {
		return evaluatedFilter{total: uint64(len(documents))}, nil
	}
	matched := make(map[uint64]struct{}, len(documents))
	if plan.AlwaysFalse() {
		return evaluatedFilter{
			predicate: func(uint64) bool { return false }, total: uint64(len(documents)), present: true,
		}, nil
	}
	fields := plan.Fields()
	fieldCount := len(fields)
	indexes := make(dbsql.IndexSet)
	for _, field := range fields {
		if !field.Indexed {
			continue
		}
		if _, exists := indexes[field.Name]; exists {
			continue
		}
		index, err := dbsql.NewInvertedIndex(field)
		if err != nil {
			return evaluatedFilter{}, err
		}
		indexes[field.Name] = index
	}
	if len(indexes) > 0 {
		for row := range documents {
			if err := ctx.Err(); err != nil {
				return evaluatedFilter{}, err
			}
			document := &documents[row]
			for name, index := range indexes {
				field := index.Field()
				raw, found := document.Fields[name]
				value, err := toFilterValue(field, raw, found)
				if err != nil {
					return evaluatedFilter{}, fmt.Errorf("document %d field %q: %w", document.DocID, name, err)
				}
				if err := index.Add(uint64(row), value); err != nil {
					return evaluatedFilter{}, err
				}
			}
		}
		for _, index := range indexes {
			if err := index.Seal(); err != nil {
				return evaluatedFilter{}, err
			}
		}
	}
	candidates, candidatesUsed, _, err := plan.Candidates(indexes, uint64(len(documents)))
	if err != nil {
		return evaluatedFilter{}, err
	}
	if candidatesUsed && len(documents) > 0 &&
		float64(candidates.Count())/float64(len(documents)) >= float64(invertToForwardRatio) {
		candidatesUsed = false
	}
	ordinals := make([]uint32, 0, min(len(documents), 64))
	for index := range documents {
		if err := ctx.Err(); err != nil {
			return evaluatedFilter{}, err
		}
		if candidatesUsed && !candidates.Contains(uint64(index)) {
			continue
		}
		document := &documents[index]
		cache := make(map[string]dbsql.Value, fieldCount)
		match, err := plan.Match(func(field dbsql.Field) (dbsql.Value, error) {
			if value, found := cache[field.Name]; found {
				return value, nil
			}
			raw, found := document.Fields[field.Name]
			value, valueErr := toFilterValue(field, raw, found)
			if valueErr != nil {
				return dbsql.Value{}, fmt.Errorf("document %d field %q: %w", document.DocID, field.Name, valueErr)
			}
			cache[field.Name] = value
			return value, nil
		})
		if err != nil {
			return evaluatedFilter{}, err
		}
		if match {
			matched[document.DocID] = struct{}{}
			ordinals = append(ordinals, uint32(index))
		}
	}
	return evaluatedFilter{
		predicate: func(key uint64) bool {
			_, found := matched[key]
			return found
		},
		ordinals: ordinals, matched: uint64(len(matched)), total: uint64(len(documents)),
		present: true, usedIndex: candidatesUsed,
	}, nil
}

func wrapFilterEvaluationError(op, path string, err error) error {
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return wrapCollectionError(op, path, err)
	}
	return &Error{Code: ErrorCodeInternal, Op: op, Path: path, Message: "evaluate scalar filter", Err: err}
}

func toFilterValue(field dbsql.Field, raw any, found bool) (dbsql.Value, error) {
	if !found || raw == nil {
		return dbsql.NullValue(field.Kind, field.Array)
	}
	if !field.Array {
		switch field.Kind {
		case dbsql.ValueBinary:
			value, ok := raw.(Binary)
			if ok {
				return dbsql.BinaryValue(value), nil
			}
		case dbsql.ValueString:
			value, ok := raw.(string)
			if ok {
				return dbsql.StringValue(value), nil
			}
		case dbsql.ValueBool:
			value, ok := raw.(bool)
			if ok {
				return dbsql.BoolValue(value), nil
			}
		case dbsql.ValueInt32:
			value, ok := raw.(int32)
			if ok {
				return dbsql.Int32Value(value), nil
			}
		case dbsql.ValueInt64:
			value, ok := raw.(int64)
			if ok {
				return dbsql.Int64Value(value), nil
			}
		case dbsql.ValueUint32:
			value, ok := raw.(uint32)
			if ok {
				return dbsql.Uint32Value(value), nil
			}
		case dbsql.ValueUint64:
			value, ok := raw.(uint64)
			if ok {
				return dbsql.Uint64Value(value), nil
			}
		case dbsql.ValueFloat32:
			value, ok := raw.(float32)
			if ok {
				return dbsql.Float32Value(value)
			}
		case dbsql.ValueFloat64:
			value, ok := raw.(float64)
			if ok {
				return dbsql.Float64Value(value)
			}
		}
		return dbsql.Value{}, fmt.Errorf("value %T does not match scalar %s", raw, field.Kind)
	}

	var elements []dbsql.Value
	switch field.Kind {
	case dbsql.ValueBinary:
		value, ok := raw.(BinaryArray)
		if !ok {
			return dbsql.Value{}, fmt.Errorf("value %T does not match ARRAY_BINARY", raw)
		}
		elements = make([]dbsql.Value, len(value))
		for index := range value {
			elements[index] = dbsql.BinaryValue(value[index])
		}
	case dbsql.ValueString:
		value, ok := raw.(StringArray)
		if !ok {
			return dbsql.Value{}, fmt.Errorf("value %T does not match ARRAY_STRING", raw)
		}
		elements = make([]dbsql.Value, len(value))
		for index := range value {
			elements[index] = dbsql.StringValue(value[index])
		}
	case dbsql.ValueBool:
		value, ok := raw.(BoolArray)
		if !ok {
			return dbsql.Value{}, fmt.Errorf("value %T does not match ARRAY_BOOL", raw)
		}
		elements = make([]dbsql.Value, len(value))
		for index := range value {
			elements[index] = dbsql.BoolValue(value[index])
		}
	case dbsql.ValueInt32:
		value, ok := raw.(Int32Array)
		if !ok {
			return dbsql.Value{}, fmt.Errorf("value %T does not match ARRAY_INT32", raw)
		}
		elements = make([]dbsql.Value, len(value))
		for index := range value {
			elements[index] = dbsql.Int32Value(value[index])
		}
	case dbsql.ValueInt64:
		value, ok := raw.(Int64Array)
		if !ok {
			return dbsql.Value{}, fmt.Errorf("value %T does not match ARRAY_INT64", raw)
		}
		elements = make([]dbsql.Value, len(value))
		for index := range value {
			elements[index] = dbsql.Int64Value(value[index])
		}
	case dbsql.ValueUint32:
		value, ok := raw.(Uint32Array)
		if !ok {
			return dbsql.Value{}, fmt.Errorf("value %T does not match ARRAY_UINT32", raw)
		}
		elements = make([]dbsql.Value, len(value))
		for index := range value {
			elements[index] = dbsql.Uint32Value(value[index])
		}
	case dbsql.ValueUint64:
		value, ok := raw.(Uint64Array)
		if !ok {
			return dbsql.Value{}, fmt.Errorf("value %T does not match ARRAY_UINT64", raw)
		}
		elements = make([]dbsql.Value, len(value))
		for index := range value {
			elements[index] = dbsql.Uint64Value(value[index])
		}
	case dbsql.ValueFloat32:
		value, ok := raw.(Float32Array)
		if !ok {
			return dbsql.Value{}, fmt.Errorf("value %T does not match ARRAY_FLOAT", raw)
		}
		elements = make([]dbsql.Value, len(value))
		for index := range value {
			element, err := dbsql.Float32Value(value[index])
			if err != nil {
				return dbsql.Value{}, err
			}
			elements[index] = element
		}
	case dbsql.ValueFloat64:
		value, ok := raw.(Float64Array)
		if !ok {
			return dbsql.Value{}, fmt.Errorf("value %T does not match ARRAY_DOUBLE", raw)
		}
		elements = make([]dbsql.Value, len(value))
		for index := range value {
			element, err := dbsql.Float64Value(value[index])
			if err != nil {
				return dbsql.Value{}, err
			}
			elements[index] = element
		}
	default:
		return dbsql.Value{}, fmt.Errorf("unsupported array element kind %d", field.Kind)
	}
	return dbsql.ArrayValue(field.Kind, elements...)
}
