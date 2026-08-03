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
	"fmt"
	"reflect"
	"strconv"

	"github.com/gorse-io/zvec/internal/core"
)

// VectorQuery describes one exact or approximate vector search. Exactly one
// of DenseVector and SparseVector must be set according to the target field.
type VectorQuery struct {
	Field        string
	DenseVector  DenseVector
	SparseVector SparseVector
	TopK         int
	Filter       string
	Projection   Projection
	Params       QueryParams
}

// GroupByVectorQuery describes a vector search retaining the best documents
// from the best distinct scalar groups.
type GroupByVectorQuery struct {
	Field        string
	DenseVector  DenseVector
	SparseVector SparseVector
	Filter       string
	Projection   Projection
	Params       QueryParams
	GroupByField string
	GroupCount   int
	TopKPerGroup int
}

// GroupResult contains one baseline-compatible string group value and its
// metric-ordered, projected documents.
type GroupResult struct {
	Value     string
	Documents []Document
}

// Query executes exact Flat search over the current live document versions.
// Filter is parsed and schema-bound before its candidate mask is applied.
func (c *Collection) Query(ctx context.Context, query VectorQuery) ([]Document, error) {
	const op = "query"
	if c == nil {
		return nil, invalidArgument(op, "collection is nil")
	}
	if ctx == nil {
		return nil, invalidArgument(op, "context is nil")
	}
	if query.TopK <= 0 {
		return nil, invalidArgument(op, "TopK must be positive")
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	if err := c.requireOpenLocked(op); err != nil {
		return nil, err
	}
	field, found := c.schema.Field(query.Field)
	if !found || !field.DataType.IsVector() {
		return nil, invalidArgument(op, "vector field %q does not exist", query.Field)
	}
	if err := query.Projection.Validate(c.schema); err != nil {
		return nil, err
	}
	flatIndex, err := requireFlatField(field, op, c.path)
	if err != nil {
		return nil, err
	}
	params, err := flatQueryParams(query.Params)
	if err != nil {
		return nil, err
	}
	filterPlan, err := buildFilterPlan(query.Filter, c.schema)
	if err != nil {
		return nil, invalidArgument(op, "invalid filter: %v", err)
	}
	documents, err := c.liveDocumentsLocked(ctx)
	if err != nil {
		return nil, wrapCollectionError(op, c.path, err)
	}
	candidateFilter, err := evaluateFilterDocuments(ctx, filterPlan, documents)
	if err != nil {
		return nil, wrapFilterEvaluationError(op, c.path, err)
	}
	metric, err := toCoreMetric(flatIndex.Metric)
	if err != nil {
		return nil, err
	}
	var results []core.Result
	if field.DataType.IsDenseVector() {
		if query.SparseVector != nil {
			return nil, invalidArgument(op, "dense field %q cannot use a sparse query vector", field.Name)
		}
		queryVector, err := validateDenseQueryVector(field, query.DenseVector)
		if err != nil {
			return nil, err
		}
		index, err := buildDenseFlatIndex(ctx, field, metric, documents)
		if err != nil {
			return nil, wrapCollectionError(op, c.path, err)
		}
		results, err = index.SearchWithOptions(ctx, queryVector, core.SearchOptions{
			TopK: query.TopK, Radius: params.Radius, Filter: candidateFilter,
		})
		if err != nil {
			return nil, wrapCollectionError(op, c.path, err)
		}
	} else {
		if query.DenseVector != nil {
			return nil, invalidArgument(op, "sparse field %q cannot use a dense query vector", field.Name)
		}
		queryVector, err := validateSparseQueryVector(field, query.SparseVector)
		if err != nil {
			return nil, err
		}
		index, err := buildSparseFlatIndex(ctx, field, documents)
		if err != nil {
			return nil, wrapCollectionError(op, c.path, err)
		}
		results, err = index.SearchSparseWithOptions(ctx, queryVector, core.SearchOptions{
			TopK: query.TopK, Radius: params.Radius, Filter: candidateFilter,
		})
		if err != nil {
			return nil, wrapCollectionError(op, c.path, err)
		}
	}
	return c.materializeResults(documents, results, query.Projection)
}

// GroupByQuery executes exact Flat group-by search. Groups are selected only
// after every live candidate has passed scalar and radius filtering.
func (c *Collection) GroupByQuery(ctx context.Context, query GroupByVectorQuery) ([]GroupResult, error) {
	const op = "group-by query"
	if c == nil {
		return nil, invalidArgument(op, "collection is nil")
	}
	if ctx == nil {
		return nil, invalidArgument(op, "context is nil")
	}
	if query.GroupCount <= 0 {
		return nil, invalidArgument(op, "GroupCount must be positive")
	}
	if query.TopKPerGroup <= 0 {
		return nil, invalidArgument(op, "TopKPerGroup must be positive")
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	if err := c.requireOpenLocked(op); err != nil {
		return nil, err
	}
	field, found := c.schema.Field(query.Field)
	if !found || !field.DataType.IsVector() {
		return nil, invalidArgument(op, "vector field %q does not exist", query.Field)
	}
	groupField, found := c.schema.Field(query.GroupByField)
	if !found || !groupDataTypeSupported(groupField.DataType) {
		return nil, invalidArgument(op, "group field %q must be a supported scalar field", query.GroupByField)
	}
	if err := query.Projection.Validate(c.schema); err != nil {
		return nil, err
	}
	flatIndex, err := requireFlatField(field, op, c.path)
	if err != nil {
		return nil, err
	}
	params, err := flatQueryParams(query.Params)
	if err != nil {
		return nil, err
	}
	filterPlan, err := buildFilterPlan(query.Filter, c.schema)
	if err != nil {
		return nil, invalidArgument(op, "invalid filter: %v", err)
	}
	documents, err := c.liveDocumentsLocked(ctx)
	if err != nil {
		return nil, wrapCollectionError(op, c.path, err)
	}
	candidateFilter, err := evaluateFilterDocuments(ctx, filterPlan, documents)
	if err != nil {
		return nil, wrapFilterEvaluationError(op, c.path, err)
	}
	groupValues := make(map[uint64]string, len(documents))
	for _, document := range documents {
		value, valueErr := encodeGroupValue(document.Fields[groupField.Name], groupField.DataType)
		if valueErr != nil {
			return nil, &Error{Code: ErrorCodeInternal, Op: op, Path: c.path, Message: "stored group value is invalid", Err: valueErr}
		}
		groupValues[document.DocID] = value
	}
	resolve := func(key uint64) (string, bool) {
		value, found := groupValues[key]
		return value, found
	}
	options := core.GroupByOptions{
		GroupCount: query.GroupCount, TopKPerGroup: query.TopKPerGroup,
		Radius: params.Radius, Filter: candidateFilter, Resolve: resolve,
	}
	metric, err := toCoreMetric(flatIndex.Metric)
	if err != nil {
		return nil, err
	}
	var groups []core.GroupResult
	if field.DataType.IsDenseVector() {
		if query.SparseVector != nil {
			return nil, invalidArgument(op, "dense field %q cannot use a sparse query vector", field.Name)
		}
		queryVector, err := validateDenseQueryVector(field, query.DenseVector)
		if err != nil {
			return nil, err
		}
		index, err := buildDenseFlatIndex(ctx, field, metric, documents)
		if err != nil {
			return nil, wrapCollectionError(op, c.path, err)
		}
		groups, err = index.SearchGroups(ctx, queryVector, options)
		if err != nil {
			return nil, wrapCollectionError(op, c.path, err)
		}
	} else {
		if query.DenseVector != nil {
			return nil, invalidArgument(op, "sparse field %q cannot use a dense query vector", field.Name)
		}
		queryVector, err := validateSparseQueryVector(field, query.SparseVector)
		if err != nil {
			return nil, err
		}
		index, err := buildSparseFlatIndex(ctx, field, documents)
		if err != nil {
			return nil, wrapCollectionError(op, c.path, err)
		}
		groups, err = index.SearchSparseGroups(ctx, queryVector, options)
		if err != nil {
			return nil, wrapCollectionError(op, c.path, err)
		}
	}
	return c.materializeGroups(documents, groups, query.Projection)
}

func (c *Collection) liveDocumentsLocked(ctx context.Context) ([]Document, error) {
	stored, err := c.store.LiveDocuments(ctx)
	if err != nil {
		return nil, err
	}
	documents := make([]Document, len(stored))
	for index, item := range stored {
		document, err := decodeStoredDocument(item)
		if err != nil {
			return nil, err
		}
		if err := document.Validate(c.schema); err != nil {
			return nil, fmt.Errorf("stored document %d violates schema: %w", item.DocID, err)
		}
		documents[index] = document
	}
	return documents, nil
}

func buildDenseFlatIndex(ctx context.Context, field FieldSchema, metric core.Metric, documents []Document) (*core.DenseFlatIndex, error) {
	index, err := core.NewDenseFlatIndex(int(field.Dimension), metric)
	if err != nil {
		return nil, err
	}
	for _, document := range documents {
		value, found := document.Fields[field.Name]
		if !found || value == nil {
			continue
		}
		vector, err := denseValueToFloat32(value)
		if err != nil {
			return nil, fmt.Errorf("document %d field %q: %w", document.DocID, field.Name, err)
		}
		if err := index.Add(ctx, document.DocID, vector); err != nil {
			return nil, err
		}
	}
	return index, nil
}

func buildSparseFlatIndex(ctx context.Context, field FieldSchema, documents []Document) (*core.SparseFlatIndex, error) {
	index, err := core.NewSparseFlatIndex(core.MetricIP)
	if err != nil {
		return nil, err
	}
	for _, document := range documents {
		value, found := document.Fields[field.Name]
		if !found || value == nil {
			continue
		}
		vector, err := sparseValueToCore(value)
		if err != nil {
			return nil, fmt.Errorf("document %d field %q: %w", document.DocID, field.Name, err)
		}
		if err := index.AddSparse(ctx, document.DocID, vector); err != nil {
			return nil, err
		}
	}
	return index, nil
}

func validateDenseQueryVector(field FieldSchema, vector DenseVector) ([]float32, error) {
	if vector == nil || isNilInterface(vector) {
		return nil, invalidArgument("query", "dense query vector is nil")
	}
	cloned, dataType, err := cloneDocumentValue(vector)
	if err != nil {
		return nil, err
	}
	if dataType != field.DataType {
		return nil, invalidArgument("query", "query vector has type %s, field %q requires %s", dataType, field.Name, field.DataType)
	}
	dense := cloned.(DenseVector)
	if dense.Dimension() != int(field.Dimension) {
		return nil, invalidArgument("query", "query vector dimension is %d, want %d", dense.Dimension(), field.Dimension)
	}
	return denseValueToFloat32(cloned)
}

func validateSparseQueryVector(field FieldSchema, vector SparseVector) (core.SparseVector, error) {
	if vector == nil || isNilInterface(vector) {
		return core.SparseVector{}, invalidArgument("query", "sparse query vector is nil")
	}
	cloned, dataType, err := cloneDocumentValue(vector)
	if err != nil {
		return core.SparseVector{}, err
	}
	if dataType != field.DataType {
		return core.SparseVector{}, invalidArgument("query", "query vector has type %s, field %q requires %s", dataType, field.Name, field.DataType)
	}
	return sparseValueToCore(cloned)
}

func denseValueToFloat32(value any) ([]float32, error) {
	switch value := value.(type) {
	case VectorFP16:
		result := make([]float32, len(value))
		for index := range value {
			result[index] = value[index].Float32()
		}
		return result, nil
	case VectorFP32:
		return append([]float32(nil), value...), nil
	case VectorInt8:
		result := make([]float32, len(value))
		for index := range value {
			result[index] = float32(value[index])
		}
		return result, nil
	default:
		return nil, fmt.Errorf("unsupported Flat dense vector type %T", value)
	}
}

func sparseValueToCore(value any) (core.SparseVector, error) {
	switch value := value.(type) {
	case SparseVectorFP16:
		values := make([]float32, len(value.Values))
		for index := range value.Values {
			values[index] = value.Values[index].Float32()
		}
		return core.SparseVector{Indices: append([]uint32(nil), value.Indices...), Values: values}, nil
	case SparseVectorFP32:
		return core.SparseVector{
			Indices: append([]uint32(nil), value.Indices...),
			Values:  append([]float32(nil), value.Values...),
		}, nil
	default:
		return core.SparseVector{}, fmt.Errorf("unsupported Flat sparse vector type %T", value)
	}
}

func (c *Collection) materializeResults(documents []Document, results []core.Result, projection Projection) ([]Document, error) {
	byID := make(map[uint64]Document, len(documents))
	for _, document := range documents {
		byID[document.DocID] = document
	}
	output := make([]Document, 0, len(results))
	for _, result := range results {
		document, found := byID[result.Key]
		if !found {
			return nil, &Error{Code: ErrorCodeInternal, Op: "materialize query", Path: c.path, Message: fmt.Sprintf("document %d disappeared from snapshot", result.Key)}
		}
		document.Score = result.Score
		projected, err := ProjectDocument(document, c.schema, projection)
		if err != nil {
			return nil, err
		}
		output = append(output, projected)
	}
	return output, nil
}

func (c *Collection) materializeGroups(documents []Document, groups []core.GroupResult, projection Projection) ([]GroupResult, error) {
	output := make([]GroupResult, len(groups))
	for index, group := range groups {
		materialized, err := c.materializeResults(documents, group.Results, projection)
		if err != nil {
			return nil, err
		}
		output[index] = GroupResult{Value: group.Value, Documents: materialized}
	}
	return output, nil
}

func requireFlatField(field FieldSchema, op, path string) (FlatIndexParams, error) {
	index := field.EffectiveIndex()
	var flat FlatIndexParams
	switch value := index.(type) {
	case FlatIndexParams:
		flat = value
	case *FlatIndexParams:
		if value == nil {
			return FlatIndexParams{}, invalidArgument(op, "field %q has nil Flat index parameters", field.Name)
		}
		flat = *value
	default:
		return FlatIndexParams{}, notSupported(op, path, fmt.Sprintf("index %s on field %q is not implemented", index.IndexType(), field.Name))
	}
	if flat.Quantize != QuantizeTypeUndefined || flat.Quantizer.EnableRotate {
		return FlatIndexParams{}, notSupported(op, path, fmt.Sprintf("quantized Flat index on field %q is not implemented", field.Name))
	}
	return flat, nil
}

func flatQueryParams(params QueryParams) (FlatQueryParams, error) {
	if params == nil || isNilInterface(params) {
		value := NewFlatQueryParams()
		return value, nil
	}
	if err := params.Validate(); err != nil {
		return FlatQueryParams{}, err
	}
	switch value := params.(type) {
	case FlatQueryParams:
		return value, nil
	case *FlatQueryParams:
		if value != nil {
			return *value, nil
		}
	}
	return FlatQueryParams{}, notSupported("query", "", fmt.Sprintf("query parameters for %s are not implemented", params.IndexType()))
}

func toCoreMetric(metric MetricType) (core.Metric, error) {
	switch metric {
	case MetricTypeL2:
		return core.MetricL2, nil
	case MetricTypeIP:
		return core.MetricIP, nil
	case MetricTypeCosine:
		return core.MetricCosine, nil
	case MetricTypeMIPSL2:
		return core.MetricMIPSL2, nil
	default:
		return 0, invalidArgument("query", "invalid metric %s", metric)
	}
}

func flatFieldSupported(field FieldSchema) bool {
	if !field.DataType.IsVector() {
		return false
	}
	_, err := requireFlatField(field, "stats", "")
	return err == nil
}

func isNilInterface(value any) bool {
	if value == nil {
		return true
	}
	reflected := reflect.ValueOf(value)
	switch reflected.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return reflected.IsNil()
	default:
		return false
	}
}

func groupDataTypeSupported(dataType DataType) bool {
	switch dataType {
	case DataTypeString, DataTypeBool, DataTypeInt32, DataTypeInt64,
		DataTypeUint32, DataTypeUint64, DataTypeFloat, DataTypeDouble:
		return true
	default:
		return false
	}
}

func encodeGroupValue(value any, dataType DataType) (string, error) {
	if value == nil {
		return "", nil
	}
	switch dataType {
	case DataTypeString:
		result, ok := value.(string)
		if ok {
			return result, nil
		}
	case DataTypeBool:
		result, ok := value.(bool)
		if ok {
			return strconv.FormatBool(result), nil
		}
	case DataTypeInt32:
		result, ok := value.(int32)
		if ok {
			return strconv.FormatInt(int64(result), 10), nil
		}
	case DataTypeInt64:
		result, ok := value.(int64)
		if ok {
			return strconv.FormatInt(result, 10), nil
		}
	case DataTypeUint32:
		result, ok := value.(uint32)
		if ok {
			return strconv.FormatUint(uint64(result), 10), nil
		}
	case DataTypeUint64:
		result, ok := value.(uint64)
		if ok {
			return strconv.FormatUint(result, 10), nil
		}
	case DataTypeFloat:
		result, ok := value.(float32)
		if ok {
			return strconv.FormatFloat(float64(result), 'f', 6, 32), nil
		}
	case DataTypeDouble:
		result, ok := value.(float64)
		if ok {
			return strconv.FormatFloat(result, 'f', 6, 64), nil
		}
	}
	return "", fmt.Errorf("value %T does not match group type %s", value, dataType)
}
