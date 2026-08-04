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
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"sort"
	"strconv"
	"sync"

	"github.com/gorse-io/zvec/internal/ailego"
	"github.com/gorse-io/zvec/internal/core"
	"github.com/gorse-io/zvec/internal/db"
	dbsql "github.com/gorse-io/zvec/internal/db/sql"
)

// DefaultMaxBufferSize is the baseline-compatible DiskANN cache budget.
const DefaultMaxBufferSize uint32 = 64 << 20

// CollectionOptions controls one collection handle. EnableMmap is persisted
// when creating a collection; MaxBufferSize bounds native DiskANN node-cache
// bytes. A zero MaxBufferSize selects DefaultMaxBufferSize.
type CollectionOptions struct {
	ReadOnly      bool
	EnableMmap    bool
	MaxBufferSize uint32
	WALSyncEvery  uint64
}

// NewCollectionOptions returns baseline-compatible handle defaults.
func NewCollectionOptions() CollectionOptions {
	return CollectionOptions{EnableMmap: true, MaxBufferSize: DefaultMaxBufferSize}
}

func (o CollectionOptions) normalized() CollectionOptions {
	if o.MaxBufferSize == 0 {
		o.MaxBufferSize = DefaultMaxBufferSize
	}
	return o
}

// CollectionStats is a point-in-time summary of live and retained in-memory
// collection state. StorageMemoryBytes is a conservative encoded-size estimate,
// not the Go process heap size.
type CollectionStats struct {
	DocumentCount      uint64
	IndexCompleteness  map[string]float32
	ImmutableSegments  uint64
	MutableDocuments   uint64
	DeletedDocuments   uint64
	StorageMemoryBytes uint64
}

// Collection is one open native Go collection. Its methods are safe for
// concurrent use; mutations are serialized so queries see complete versions.
type Collection struct {
	mu      sync.RWMutex
	store   *db.CollectionStore
	path    string
	schema  CollectionSchema
	options CollectionOptions
	runtime *runtimeResources
	closed  bool
}

// CreateAndOpen creates a native Go collection and opens its sole writable
// handle. The format is intentionally incompatible with C++ collections.
func CreateAndOpen(ctx context.Context, path string, schema CollectionSchema, options CollectionOptions) (*Collection, error) {
	if ctx == nil {
		return nil, invalidArgument("create collection", "context is nil")
	}
	if path == "" {
		return nil, invalidArgument("create collection", "path is empty")
	}
	if options.ReadOnly {
		return nil, &Error{Code: ErrorCodeInvalidArgument, Op: "create collection", Path: path, Message: "a collection cannot be created read-only"}
	}
	if err := schema.Validate(); err != nil {
		return nil, err
	}
	resources := currentRuntimeResources()
	absolute, err := filepath.Abs(path)
	if err != nil {
		return nil, wrapCollectionError("create collection", path, err)
	}
	encodedSchema, err := marshalCollectionSchema(schema)
	if err != nil {
		return nil, wrapCollectionError("create collection", absolute, err)
	}
	options = options.normalized()
	store, err := db.CreateCollection(ctx, absolute, encodedSchema, db.CollectionOptions{
		EnableMmap:          options.EnableMmap,
		SegmentMaxDocuments: schema.MaxDocsPerSegment,
		WAL:                 db.WALOptions{SyncEvery: options.WALSyncEvery},
	})
	if err != nil {
		return nil, wrapCollectionError("create collection", absolute, err)
	}
	return &Collection{
		store: store, path: absolute, schema: schema.Clone(), options: options,
		runtime: resources,
	}, nil
}

// Open opens the version named by CURRENT and replays the valid WAL prefix.
func Open(ctx context.Context, path string, options CollectionOptions) (*Collection, error) {
	if ctx == nil {
		return nil, invalidArgument("open collection", "context is nil")
	}
	if path == "" {
		return nil, invalidArgument("open collection", "path is empty")
	}
	resources := currentRuntimeResources()
	absolute, err := filepath.Abs(path)
	if err != nil {
		return nil, wrapCollectionError("open collection", path, err)
	}
	options = options.normalized()
	store, err := db.OpenCollection(ctx, absolute, db.CollectionOptions{
		ReadOnly: options.ReadOnly,
		WAL:      db.WALOptions{SyncEvery: options.WALSyncEvery},
	})
	if err != nil {
		return nil, wrapCollectionError("open collection", absolute, err)
	}
	manifest := store.Manifest()
	schema, err := unmarshalCollectionSchema(manifest.Schema)
	if err != nil {
		_ = store.Close()
		return nil, &Error{Code: ErrorCodeInternal, Op: "open collection", Path: absolute, Message: "collection schema is corrupt", Err: err}
	}
	if schema.MaxDocsPerSegment != manifest.SegmentMaxDocuments {
		_ = store.Close()
		return nil, &Error{Code: ErrorCodeInternal, Op: "open collection", Path: absolute, Message: "schema and manifest segment capacities differ"}
	}
	options.EnableMmap = manifest.EnableMmap
	return &Collection{
		store: store, path: absolute, schema: schema, options: options,
		runtime: resources,
	}, nil
}

// Path returns the absolute collection directory.
func (c *Collection) Path() string {
	if c == nil {
		return ""
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.path
}

// Schema returns an independent schema copy.
func (c *Collection) Schema() CollectionSchema {
	if c == nil {
		return CollectionSchema{}
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.schema.Clone()
}

// Options returns the effective options for this handle.
func (c *Collection) Options() CollectionOptions {
	if c == nil {
		return CollectionOptions{}
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.options
}

// Stats returns document, segment, deletion, retained-memory, and current index
// completeness counters.
func (c *Collection) Stats() CollectionStats {
	if c == nil {
		return CollectionStats{IndexCompleteness: map[string]float32{}}
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	stats := CollectionStats{IndexCompleteness: make(map[string]float32)}
	if c.store != nil {
		storage := c.store.Stats()
		stats.DocumentCount = storage.DocumentCount
		stats.ImmutableSegments = storage.ImmutableSegmentCount
		stats.MutableDocuments = storage.MutableDocumentCount
		stats.DeletedDocuments = storage.DeletedDocumentCount
		stats.StorageMemoryBytes = storage.MemoryUsageBytes
	}
	for _, field := range c.schema.Fields {
		index := field.EffectiveIndex()
		if indexParamsNil(index) {
			continue
		}
		completeness := float32(0)
		if collectionVectorFieldSupported(field) || field.IndexType() == IndexTypeFTS {
			completeness = 1
		}
		stats.IndexCompleteness[field.Name] = completeness
	}
	return stats
}

// Flush atomically publishes the current write segment and rotates its WAL.
func (c *Collection) Flush(ctx context.Context) error {
	if c == nil {
		return invalidArgument("flush collection", "collection is nil")
	}
	if ctx == nil {
		return invalidArgument("flush collection", "context is nil")
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.requireOpenLocked("flush collection"); err != nil {
		return err
	}
	return wrapCollectionError("flush collection", c.path, c.store.Flush(ctx))
}

// Close releases files and the cross-process collection lock. It is
// idempotent; WAL-backed writes remain recoverable without an explicit Flush.
func (c *Collection) Close() error {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil
	}
	c.closed = true
	return wrapCollectionError("close collection", c.path, c.store.Close())
}

// Destroy closes the handle and recursively removes only its validated
// collection directory. Read-only handles cannot destroy collections.
func (c *Collection) Destroy(ctx context.Context) error {
	if c == nil {
		return invalidArgument("destroy collection", "collection is nil")
	}
	if ctx == nil {
		return invalidArgument("destroy collection", "context is nil")
	}
	if err := ctx.Err(); err != nil {
		return wrapCollectionError("destroy collection", c.Path(), err)
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed || c.store == nil {
		return &Error{Code: ErrorCodeFailedPrecondition, Op: "destroy collection", Path: c.path, Message: "collection is closed"}
	}
	if c.options.ReadOnly {
		return &Error{Code: ErrorCodePermissionDenied, Op: "destroy collection", Path: c.path, Message: "read-only collection cannot be destroyed"}
	}
	if !safeDestroyPath(c.path) {
		return &Error{Code: ErrorCodeInvalidArgument, Op: "destroy collection", Path: c.path, Message: "refusing to remove an unsafe collection path"}
	}
	c.closed = true
	closeErr := c.store.Close()
	removeErr := os.RemoveAll(c.path)
	return wrapCollectionError("destroy collection", c.path, errors.Join(closeErr, removeErr))
}

func (c *Collection) requireOpenLocked(op string) error {
	if c.closed || c.store == nil {
		return &Error{Code: ErrorCodeFailedPrecondition, Op: op, Path: c.path, Message: "collection is closed"}
	}
	return nil
}

func safeDestroyPath(path string) bool {
	if path == "" || !filepath.IsAbs(path) {
		return false
	}
	clean := filepath.Clean(path)
	volume := filepath.VolumeName(clean)
	root := volume + string(os.PathSeparator)
	return clean != root && clean != volume && filepath.Dir(clean) != clean
}

// AddColumn atomically adds a basic numeric field and backfills every live
// document with a baseline-compatible arithmetic expression. An empty
// expression is allowed only for nullable fields and writes explicit NULLs.
func (c *Collection) AddColumn(ctx context.Context, field FieldSchema, expression string, options AddColumnOptions) error {
	const op = "add column"
	if c == nil {
		return invalidArgument(op, "collection is nil")
	}
	if ctx == nil {
		return invalidArgument(op, "context is nil")
	}
	if err := ctx.Err(); err != nil {
		return wrapCollectionError(op, c.Path(), err)
	}
	releaseRuntime, err := c.beginRuntimeTask(ctx, runtimeOptimizeTask, op, 8)
	if err != nil {
		return wrapCollectionError(op, c.Path(), err)
	}
	defer releaseRuntime()
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.requireOpenLocked(op); err != nil {
		return err
	}
	if c.options.ReadOnly {
		return &Error{Code: ErrorCodePermissionDenied, Op: op, Path: c.path, Message: "collection is read-only"}
	}
	if options.Concurrency < 0 {
		return invalidArgument(op, "Concurrency cannot be negative")
	}
	if err := field.Validate(); err != nil {
		return err
	}
	if !addColumnDataTypeSupported(field.DataType) {
		return invalidArgument(op, "only basic numeric data types are supported, got %s", field.DataType)
	}
	if _, found := c.schema.Field(field.Name); found {
		return invalidArgument(op, "field %q already exists", field.Name)
	}
	if expression == "" && !field.Nullable {
		return invalidArgument(op, "non-nullable field %q requires a backfill expression", field.Name)
	}

	nextSchema := c.schema.Clone()
	nextSchema.Fields = append(nextSchema.Fields, field.Clone())
	if err := nextSchema.Validate(); err != nil {
		return err
	}
	documents, err := c.liveDocumentsLocked(ctx)
	if err != nil {
		return wrapCollectionError(op, c.path, err)
	}
	if len(documents) == 0 {
		return c.publishSchemaLocked(ctx, op, nextSchema)
	}

	var compiled *columnExpression
	if expression != "" {
		compiled, err = parseColumnExpression(expression, c.schema)
		if err != nil {
			return invalidArgument(op, "invalid expression %q: %v", expression, err)
		}
	}
	return c.rewriteCollectionDocumentsLocked(ctx, op, nextSchema, documents, options.Concurrency, func(document *Document) error {
		var value any
		if compiled != nil {
			evaluated, evaluateErr := compiled.evaluate(document.Fields, field.DataType)
			if evaluateErr != nil {
				return evaluateErr
			}
			value = evaluated
		}
		if value == nil && !field.Nullable {
			return fmt.Errorf("expression evaluates to NULL for non-nullable field %q", field.Name)
		}
		document.Fields[field.Name] = value
		return nil
	})
}

// AlterColumn atomically renames or replaces one basic numeric field. A
// replacement may change its name, numeric type, nullability, and INVERT index
// parameters. Rename and replacement forms are mutually exclusive.
func (c *Collection) AlterColumn(ctx context.Context, column, rename string, field *FieldSchema, options AlterColumnOptions) error {
	const op = "alter column"
	if c == nil {
		return invalidArgument(op, "collection is nil")
	}
	if ctx == nil {
		return invalidArgument(op, "context is nil")
	}
	if err := ctx.Err(); err != nil {
		return wrapCollectionError(op, c.Path(), err)
	}
	releaseRuntime, err := c.beginRuntimeTask(ctx, runtimeOptimizeTask, op, 8)
	if err != nil {
		return wrapCollectionError(op, c.Path(), err)
	}
	defer releaseRuntime()
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.requireOpenLocked(op); err != nil {
		return err
	}
	if c.options.ReadOnly {
		return &Error{Code: ErrorCodePermissionDenied, Op: op, Path: c.path, Message: "collection is read-only"}
	}
	if options.Concurrency < 0 {
		return invalidArgument(op, "Concurrency cannot be negative")
	}
	if column == "" {
		return invalidArgument(op, "column is empty")
	}
	position := -1
	for index := range c.schema.Fields {
		if c.schema.Fields[index].Name == column {
			position = index
			break
		}
	}
	if position < 0 {
		return &Error{Code: ErrorCodeNotFound, Op: op, Path: c.path, Message: fmt.Sprintf("field %q does not exist", column)}
	}
	oldField := c.schema.Fields[position]
	if !addColumnDataTypeSupported(oldField.DataType) {
		return invalidArgument(op, "only basic numeric columns can be altered, got %s", oldField.DataType)
	}
	if rename != "" && field != nil {
		return invalidArgument(op, "rename and replacement field cannot both be specified")
	}

	var nextField FieldSchema
	if rename != "" {
		if _, exists := c.schema.Field(rename); exists {
			return invalidArgument(op, "field %q already exists", rename)
		}
		nextField = oldField.Clone()
		nextField.Name = rename
	} else {
		if field == nil {
			return invalidArgument(op, "replacement field is nil")
		}
		nextField = field.Clone()
		if nextField.Name != column {
			if _, exists := c.schema.Field(nextField.Name); exists {
				return invalidArgument(op, "field %q already exists", nextField.Name)
			}
		}
		if oldField.Nullable && !nextField.Nullable {
			return invalidArgument(op, "nullable field %q cannot be changed to non-nullable", column)
		}
	}
	if err := nextField.Validate(); err != nil {
		return err
	}
	if !addColumnDataTypeSupported(nextField.DataType) {
		return invalidArgument(op, "only basic numeric data types are supported, got %s", nextField.DataType)
	}
	if equalFieldSchema(oldField, nextField) {
		return nil
	}

	nextSchema := c.schema.Clone()
	nextSchema.Fields[position] = nextField.Clone()
	if err := nextSchema.Validate(); err != nil {
		return err
	}
	documents, err := c.liveDocumentsLocked(ctx)
	if err != nil {
		return wrapCollectionError(op, c.path, err)
	}
	if len(documents) == 0 {
		return c.publishSchemaLocked(ctx, op, nextSchema)
	}
	return c.rewriteCollectionDocumentsLocked(ctx, op, nextSchema, documents, options.Concurrency, func(document *Document) error {
		value, found := document.Fields[column]
		delete(document.Fields, column)
		if !found {
			return nil
		}
		if value == nil {
			document.Fields[nextField.Name] = nil
			return nil
		}
		number, convertErr := columnNumberFromValue(value)
		if convertErr != nil {
			return convertErr
		}
		converted, convertErr := number.cast(nextField.DataType)
		if convertErr != nil {
			return convertErr
		}
		document.Fields[nextField.Name] = converted
		return nil
	})
}

func equalFieldSchema(left, right FieldSchema) bool {
	return left.Name == right.Name && left.DataType == right.DataType && left.Nullable == right.Nullable &&
		left.Dimension == right.Dimension && equalIndexParams(left.Index, right.Index)
}

type collectionVectorIndex struct {
	indexType IndexType
	metric    core.Metric
	quantize  QuantizeType
	rotate    bool
	flat      FlatIndexParams
	hnsw      HNSWIndexParams
	rabitq    HNSWRaBitQIndexParams
	ivf       IVFIndexParams
	diskann   DiskANNIndexParams
	vamana    VamanaIndexParams
}

type collectionQueryConfig struct {
	options        QueryOptions
	scaleFactor    float32
	ef             int
	nprobe         int
	listSize       int
	prefetchOffset uint32
	prefetchLines  uint32
}

func resolveCollectionVectorIndex(field FieldSchema, op, path string) (collectionVectorIndex, error) {
	index := field.EffectiveIndex()
	if indexParamsNil(index) || !field.DataType.IsVector() {
		return collectionVectorIndex{}, invalidArgument(op, "field %q does not have a vector index", field.Name)
	}
	if err := index.Validate(); err != nil {
		return collectionVectorIndex{}, err
	}
	spec := collectionVectorIndex{indexType: index.IndexType()}
	switch value := index.(type) {
	case FlatIndexParams:
		spec.flat = value
	case *FlatIndexParams:
		if value == nil {
			return collectionVectorIndex{}, invalidArgument(op, "field %q has nil Flat index parameters", field.Name)
		}
		spec.flat = *value
	case HNSWIndexParams:
		spec.hnsw = value
	case *HNSWIndexParams:
		if value == nil {
			return collectionVectorIndex{}, invalidArgument(op, "field %q has nil HNSW index parameters", field.Name)
		}
		spec.hnsw = *value
	case HNSWRaBitQIndexParams:
		spec.rabitq = value
	case *HNSWRaBitQIndexParams:
		if value == nil {
			return collectionVectorIndex{}, invalidArgument(op, "field %q has nil HNSW-RaBitQ index parameters", field.Name)
		}
		spec.rabitq = *value
	case IVFIndexParams:
		spec.ivf = value
	case *IVFIndexParams:
		if value == nil {
			return collectionVectorIndex{}, invalidArgument(op, "field %q has nil IVF index parameters", field.Name)
		}
		spec.ivf = *value
	case DiskANNIndexParams:
		spec.diskann = value
	case *DiskANNIndexParams:
		if value == nil {
			return collectionVectorIndex{}, invalidArgument(op, "field %q has nil DiskANN index parameters", field.Name)
		}
		spec.diskann = *value
	case VamanaIndexParams:
		spec.vamana = value
	case *VamanaIndexParams:
		if value == nil {
			return collectionVectorIndex{}, invalidArgument(op, "field %q has nil Vamana index parameters", field.Name)
		}
		spec.vamana = *value
	default:
		return collectionVectorIndex{}, notSupported(op, path, fmt.Sprintf("index %s on field %q is not implemented", index.IndexType(), field.Name))
	}

	var metric MetricType
	switch spec.indexType {
	case IndexTypeFlat:
		metric, spec.quantize, spec.rotate = spec.flat.Metric, spec.flat.Quantize, spec.flat.Quantizer.EnableRotate
	case IndexTypeHNSW:
		metric, spec.quantize, spec.rotate = spec.hnsw.Metric, spec.hnsw.Quantize, spec.hnsw.Quantizer.EnableRotate
	case IndexTypeHNSWRaBitQ:
		metric, spec.quantize = spec.rabitq.Metric, QuantizeTypeRaBitQ
	case IndexTypeIVF:
		if field.DataType.IsSparseVector() {
			return collectionVectorIndex{}, invalidArgument(op, "sparse field %q cannot use IVF", field.Name)
		}
		metric, spec.quantize, spec.rotate = spec.ivf.Metric, spec.ivf.Quantize, spec.ivf.Quantizer.EnableRotate
	case IndexTypeDiskANN:
		if field.DataType.IsSparseVector() {
			return collectionVectorIndex{}, invalidArgument(op, "sparse field %q cannot use DiskANN", field.Name)
		}
		metric, spec.quantize, spec.rotate = spec.diskann.Metric, spec.diskann.Quantize, spec.diskann.Quantizer.EnableRotate
	case IndexTypeVamana:
		if field.DataType.IsSparseVector() {
			return collectionVectorIndex{}, invalidArgument(op, "sparse field %q cannot use Vamana", field.Name)
		}
		metric, spec.quantize, spec.rotate = spec.vamana.Metric, spec.vamana.Quantize, spec.vamana.Quantizer.EnableRotate
	default:
		return collectionVectorIndex{}, notSupported(op, path, fmt.Sprintf("index %s on field %q is not implemented", spec.indexType, field.Name))
	}
	converted, err := toCoreMetric(metric)
	if err != nil {
		return collectionVectorIndex{}, err
	}
	spec.metric = converted
	if field.DataType.IsSparseVector() && spec.quantize != QuantizeTypeUndefined && spec.quantize != QuantizeTypeFP16 {
		return collectionVectorIndex{}, notSupported(op, path, fmt.Sprintf("%s sparse quantization on field %q is not implemented", spec.quantize, field.Name))
	}
	return spec, nil
}

func collectionQueryParams(params QueryParams, spec collectionVectorIndex) (collectionQueryConfig, error) {
	if params == nil || isNilInterface(params) {
		switch spec.indexType {
		case IndexTypeFlat:
			value := NewFlatQueryParams()
			params = value
		case IndexTypeHNSW:
			value := NewHNSWQueryParams()
			params = value
		case IndexTypeHNSWRaBitQ:
			value := NewHNSWRaBitQQueryParams()
			params = value
		case IndexTypeIVF:
			value := NewIVFQueryParams()
			params = value
		case IndexTypeDiskANN:
			value := NewDiskANNQueryParams()
			params = value
		case IndexTypeVamana:
			value := NewVamanaQueryParams()
			params = value
		}
	}
	if params.IndexType() != spec.indexType {
		return collectionQueryConfig{}, invalidArgument(
			"query", "query parameters for %s cannot be used with %s", params.IndexType(), spec.indexType,
		)
	}
	if err := params.Validate(); err != nil {
		return collectionQueryConfig{}, err
	}
	switch value := params.(type) {
	case FlatQueryParams:
		return collectionQueryConfig{options: value.QueryOptions, scaleFactor: value.ScaleFactor}, nil
	case *FlatQueryParams:
		if value != nil {
			return collectionQueryConfig{options: value.QueryOptions, scaleFactor: value.ScaleFactor}, nil
		}
	case HNSWQueryParams:
		return collectionQueryConfig{
			options: value.QueryOptions, scaleFactor: 1, ef: value.EF,
			prefetchOffset: value.PrefetchOffset, prefetchLines: value.PrefetchLines,
		}, nil
	case *HNSWQueryParams:
		if value != nil {
			return collectionQueryConfig{
				options: value.QueryOptions, scaleFactor: 1, ef: value.EF,
				prefetchOffset: value.PrefetchOffset, prefetchLines: value.PrefetchLines,
			}, nil
		}
	case HNSWRaBitQQueryParams:
		return collectionQueryConfig{options: value.QueryOptions, scaleFactor: 1, ef: value.EF}, nil
	case *HNSWRaBitQQueryParams:
		if value != nil {
			return collectionQueryConfig{options: value.QueryOptions, scaleFactor: 1, ef: value.EF}, nil
		}
	case IVFQueryParams:
		return collectionQueryConfig{
			options: value.QueryOptions, scaleFactor: value.ScaleFactor, nprobe: value.NProbe,
		}, nil
	case *IVFQueryParams:
		if value != nil {
			return collectionQueryConfig{
				options: value.QueryOptions, scaleFactor: value.ScaleFactor, nprobe: value.NProbe,
			}, nil
		}
	case DiskANNQueryParams:
		return collectionQueryConfig{
			options: value.QueryOptions, scaleFactor: DefaultRefinerScaleFactor, listSize: value.ListSize,
		}, nil
	case *DiskANNQueryParams:
		if value != nil {
			return collectionQueryConfig{
				options: value.QueryOptions, scaleFactor: DefaultRefinerScaleFactor, listSize: value.ListSize,
			}, nil
		}
	case VamanaQueryParams:
		return collectionQueryConfig{
			options: value.QueryOptions, scaleFactor: 1, ef: value.EFSearch,
			prefetchOffset: value.PrefetchOffset, prefetchLines: value.PrefetchLines,
		}, nil
	case *VamanaQueryParams:
		if value != nil {
			return collectionQueryConfig{
				options: value.QueryOptions, scaleFactor: 1, ef: value.EFSearch,
				prefetchOffset: value.PrefetchOffset, prefetchLines: value.PrefetchLines,
			}, nil
		}
	}
	return collectionQueryConfig{}, invalidArgument("query", "invalid %s query parameter value", spec.indexType)
}

type collectionDenseIndex interface {
	core.DenseProvider
	core.DenseQuerySearcher
}

type collectionHNSWIndex interface {
	collectionDenseIndex
	SearchHNSW(ctx context.Context, query []float32, options core.HNSWSearchOptions) ([]core.Result, error)
}

type collectionHNSWGroupIndex interface {
	collectionHNSWIndex
	SearchHNSWGroups(ctx context.Context, query []float32, options core.HNSWGroupSearchOptions) ([]core.GroupResult, error)
}

type collectionIVFIndex interface {
	collectionDenseIndex
	SearchIVF(ctx context.Context, query []float32, options core.IVFSearchOptions) ([]core.Result, error)
}

type collectionVamanaIndex interface {
	collectionDenseIndex
	SearchVamana(ctx context.Context, query []float32, options core.VamanaSearchOptions) ([]core.Result, error)
}

type collectionDiskANNIndex interface {
	collectionDenseIndex
	SearchDiskANN(ctx context.Context, query []float32, options core.DiskANNSearchOptions) ([]core.Result, error)
}

func searchCollectionDense(
	ctx context.Context,
	schemaName string,
	field FieldSchema,
	documents []Document,
	query []float32,
	topK int,
	filter core.CandidateFilter,
	spec collectionVectorIndex,
	config collectionQueryConfig,
	workers int,
	maxBufferSize uint32,
) ([]core.Result, error) {
	final := core.SearchOptions{TopK: topK, Radius: config.options.Radius, Filter: filter}
	if (config.options.Linear && spec.indexType != IndexTypeHNSWRaBitQ) || spec.indexType == IndexTypeFlat {
		index, err := buildCollectionDenseFlat(ctx, schemaName, field, documents, spec)
		if err != nil {
			return nil, err
		}
		return executeCollectionDenseSearch(ctx, index, query, final, config.options.UseRefiner, config.scaleFactor,
			func(options core.SearchOptions) ([]core.Result, error) {
				return index.SearchWithOptions(ctx, query, options)
			})
	}
	switch spec.indexType {
	case IndexTypeHNSW:
		index, err := buildCollectionDenseHNSW(ctx, schemaName, field, documents, spec)
		if err != nil {
			return nil, err
		}
		return executeCollectionDenseSearch(ctx, index, query, final, config.options.UseRefiner, config.scaleFactor,
			func(options core.SearchOptions) ([]core.Result, error) {
				return index.SearchHNSW(ctx, query, core.HNSWSearchOptions{
					SearchOptions: options, EF: config.ef,
					PrefetchOffset: config.prefetchOffset, PrefetchLines: config.prefetchLines,
				})
			})
	case IndexTypeHNSWRaBitQ:
		index, err := buildCollectionDenseHNSWRaBitQ(ctx, field, documents, spec, workers)
		if err != nil {
			return nil, err
		}
		return index.SearchHNSWRaBitQ(ctx, query, core.HNSWRaBitQSearchOptions{
			SearchOptions: final, EF: config.ef, Refine: config.options.UseRefiner,
			Linear: config.options.Linear,
		})
	case IndexTypeIVF:
		index, err := buildCollectionDenseIVF(ctx, schemaName, field, documents, spec, workers)
		if err != nil {
			return nil, err
		}
		return executeCollectionDenseSearch(ctx, index, query, final, config.options.UseRefiner, config.scaleFactor,
			func(options core.SearchOptions) ([]core.Result, error) {
				return index.SearchIVF(ctx, query, core.IVFSearchOptions{SearchOptions: options, NProbe: config.nprobe})
			})
	case IndexTypeDiskANN:
		index, err := buildCollectionDenseDiskANN(ctx, schemaName, field, documents, spec, workers, maxBufferSize)
		if err != nil {
			return nil, err
		}
		return executeCollectionDenseSearch(ctx, index, query, final, config.options.UseRefiner, config.scaleFactor,
			func(options core.SearchOptions) ([]core.Result, error) {
				return index.SearchDiskANN(ctx, query, core.DiskANNSearchOptions{
					SearchOptions: options, ListSize: config.listSize, Linear: config.options.Linear,
				})
			})
	case IndexTypeVamana:
		index, err := buildCollectionDenseVamana(ctx, schemaName, field, documents, spec)
		if err != nil {
			return nil, err
		}
		return executeCollectionDenseSearch(ctx, index, query, final, config.options.UseRefiner, config.scaleFactor,
			func(options core.SearchOptions) ([]core.Result, error) {
				return index.SearchVamana(ctx, query, core.VamanaSearchOptions{
					SearchOptions: options, EFSearch: config.ef,
					PrefetchOffset: config.prefetchOffset, PrefetchLines: config.prefetchLines,
				})
			})
	default:
		return nil, fmt.Errorf("unsupported dense collection index %s", spec.indexType)
	}
}

func executeCollectionDenseSearch(
	ctx context.Context,
	index collectionDenseIndex,
	query []float32,
	options core.SearchOptions,
	useRefiner bool,
	scaleFactor float32,
	baseSearch func(core.SearchOptions) ([]core.Result, error),
) ([]core.Result, error) {
	if !useRefiner {
		return baseSearch(options)
	}
	candidateCount, err := core.RefinementCandidateCount(options.TopK, scaleFactor)
	if err != nil {
		return nil, err
	}
	candidates, err := baseSearch(core.SearchOptions{TopK: candidateCount, Filter: options.Filter})
	if err != nil {
		return nil, fmt.Errorf("base candidate search: %w", err)
	}
	refiner, err := core.NewOriginalVectorRefiner(index, index.Metric())
	if err != nil {
		return nil, err
	}
	return refiner.Refine(ctx, query, candidates, options)
}

func buildCollectionDenseFlat(
	ctx context.Context,
	schemaName string,
	field FieldSchema,
	documents []Document,
	spec collectionVectorIndex,
) (collectionDenseIndex, error) {
	candidates, err := collectionDenseCandidates(ctx, field, documents)
	if err != nil {
		return nil, err
	}
	if spec.quantize == QuantizeTypeUndefined {
		index, err := core.NewDenseFlatIndex(int(field.Dimension), spec.metric)
		if err != nil {
			return nil, err
		}
		for _, candidate := range candidates {
			if err := index.Add(ctx, candidate.Key, candidate.Vector); err != nil {
				return nil, err
			}
		}
		return index, nil
	}
	kind, err := toCoreQuantization(spec.quantize)
	if err != nil {
		return nil, err
	}
	reformer, err := collectionReformer(schemaName, field, spec)
	if err != nil {
		return nil, err
	}
	return core.NewScalarQuantizedFlatIndex(ctx, int(field.Dimension), spec.metric, kind, reformer, candidates)
}

func buildCollectionDenseHNSW(
	ctx context.Context,
	schemaName string,
	field FieldSchema,
	documents []Document,
	spec collectionVectorIndex,
) (collectionHNSWIndex, error) {
	candidates, err := collectionDenseCandidates(ctx, field, documents)
	if err != nil {
		return nil, err
	}
	options := core.DefaultHNSWBuildOptions(spec.metric)
	options.M = spec.hnsw.M
	options.EFConstruction = spec.hnsw.EFConstruction
	builder, err := core.NewHNSWBuilder(int(field.Dimension), options)
	if err != nil {
		return nil, err
	}
	for _, candidate := range candidates {
		if err := builder.Add(ctx, candidate.Key, candidate.Vector); err != nil {
			return nil, err
		}
	}
	base, err := builder.Build(ctx)
	if err != nil {
		return nil, err
	}
	if spec.quantize == QuantizeTypeUndefined {
		return base, nil
	}
	kind, err := toCoreQuantization(spec.quantize)
	if err != nil {
		return nil, err
	}
	reformer, err := collectionReformer(schemaName, field, spec)
	if err != nil {
		return nil, err
	}
	return core.NewScalarQuantizedHNSWIndex(ctx, base, kind, reformer)
}

func buildCollectionDenseHNSWRaBitQ(
	ctx context.Context,
	field FieldSchema,
	documents []Document,
	spec collectionVectorIndex,
	workers int,
) (*core.HNSWRaBitQIndex, error) {
	candidates, err := collectionDenseCandidates(ctx, field, documents)
	if err != nil {
		return nil, err
	}
	options := core.DefaultHNSWRaBitQBuildOptions(spec.metric)
	options.TotalBits = spec.rabitq.TotalBits
	options.Clusters = spec.rabitq.NumClusters
	options.SampleCount = spec.rabitq.SampleCount
	options.M = spec.rabitq.M
	options.EFConstruction = spec.rabitq.EFConstruction
	options.Workers = workers
	builder, err := core.NewHNSWRaBitQBuilder(int(field.Dimension), options)
	if err != nil {
		return nil, err
	}
	for _, candidate := range candidates {
		if err := builder.Add(ctx, candidate.Key, candidate.Vector); err != nil {
			return nil, err
		}
	}
	return builder.Build(ctx)
}

func buildCollectionDenseIVF(
	ctx context.Context,
	schemaName string,
	field FieldSchema,
	documents []Document,
	spec collectionVectorIndex,
	workers int,
) (collectionIVFIndex, error) {
	candidates, err := collectionDenseCandidates(ctx, field, documents)
	if err != nil {
		return nil, err
	}
	options := core.DefaultIVFBuildOptions(spec.metric)
	options.NList = spec.ivf.NList
	options.NIterations = spec.ivf.NIterations
	options.Workers = workers
	builder, err := core.NewIVFBuilder(int(field.Dimension), options)
	if err != nil {
		return nil, err
	}
	for _, candidate := range candidates {
		if err := builder.Add(ctx, candidate.Key, candidate.Vector); err != nil {
			return nil, err
		}
	}
	base, err := builder.Build(ctx)
	if err != nil {
		return nil, err
	}
	if spec.quantize == QuantizeTypeUndefined {
		return base, nil
	}
	kind, err := toCoreQuantization(spec.quantize)
	if err != nil {
		return nil, err
	}
	reformer, err := collectionReformer(schemaName, field, spec)
	if err != nil {
		return nil, err
	}
	return core.NewScalarQuantizedIVFIndex(ctx, base, kind, reformer)
}

func buildCollectionDenseVamana(
	ctx context.Context,
	schemaName string,
	field FieldSchema,
	documents []Document,
	spec collectionVectorIndex,
) (collectionVamanaIndex, error) {
	candidates, err := collectionDenseCandidates(ctx, field, documents)
	if err != nil {
		return nil, err
	}
	options := core.DefaultVamanaBuildOptions(spec.metric)
	options.MaxDegree = spec.vamana.MaxDegree
	options.SearchListSize = spec.vamana.SearchListSize
	options.Alpha = spec.vamana.Alpha
	options.MaxOcclusionSize = spec.vamana.MaxOcclusionSize
	if options.MaxOcclusionSize == 0 {
		options.MaxOcclusionSize = core.DefaultVamanaMaxOcclusionSize
	}
	options.SaturateGraph = spec.vamana.SaturateGraph
	builder, err := core.NewVamanaBuilder(int(field.Dimension), options)
	if err != nil {
		return nil, err
	}
	for _, candidate := range candidates {
		if err := builder.Add(ctx, candidate.Key, candidate.Vector); err != nil {
			return nil, err
		}
	}
	base, err := builder.Build(ctx)
	if err != nil {
		return nil, err
	}
	if spec.quantize == QuantizeTypeUndefined {
		return base, nil
	}
	kind, err := toCoreQuantization(spec.quantize)
	if err != nil {
		return nil, err
	}
	reformer, err := collectionReformer(schemaName, field, spec)
	if err != nil {
		return nil, err
	}
	return core.NewScalarQuantizedVamanaIndex(ctx, base, kind, reformer)
}

func buildCollectionDenseDiskANN(
	ctx context.Context,
	schemaName string,
	field FieldSchema,
	documents []Document,
	spec collectionVectorIndex,
	workers int,
	maxBufferSize uint32,
) (collectionDiskANNIndex, error) {
	candidates, err := collectionDenseCandidates(ctx, field, documents)
	if err != nil {
		return nil, err
	}
	options := core.DefaultDiskANNBuildOptions(spec.metric)
	options.MaxDegree = spec.diskann.MaxDegree
	options.ListSize = spec.diskann.ListSize
	options.PQChunks = spec.diskann.PQChunks
	options.Workers = workers
	options.CacheCapacity = collectionDiskANNCacheCapacity(maxBufferSize, len(candidates))
	if spec.quantize != QuantizeTypeUndefined {
		kind, err := toCoreQuantization(spec.quantize)
		if err != nil {
			return nil, err
		}
		reformer, err := collectionReformer(schemaName, field, spec)
		if err != nil {
			return nil, err
		}
		return core.NewScalarQuantizedDiskANNIndex(ctx, int(field.Dimension), options, kind, reformer, candidates)
	}
	builder, err := core.NewDiskANNBuilder(int(field.Dimension), options)
	if err != nil {
		return nil, err
	}
	for _, candidate := range candidates {
		if err := builder.Add(ctx, candidate.Key, candidate.Vector); err != nil {
			return nil, err
		}
	}
	return builder.Build(ctx)
}

func collectionDiskANNCacheCapacity(maxBufferSize uint32, candidateCount int) int {
	if candidateCount <= 0 {
		return 0
	}
	cacheCapacity := int(uint64(maxBufferSize) / core.DiskANNSectorSize)
	return min(candidateCount, cacheCapacity)
}

func buildCollectionSparseIndex(
	ctx context.Context,
	field FieldSchema,
	documents []Document,
	spec collectionVectorIndex,
	linear bool,
) (core.SparseQuerySearcher, error) {
	if linear || spec.indexType == IndexTypeFlat {
		index, err := core.NewSparseFlatIndex(spec.metric)
		if err != nil {
			return nil, err
		}
		if err := addCollectionSparseDocuments(ctx, index, field, documents, spec.quantize); err != nil {
			return nil, err
		}
		return index, nil
	}
	if spec.indexType != IndexTypeHNSW {
		return nil, fmt.Errorf("unsupported sparse collection index %s", spec.indexType)
	}
	options := core.DefaultSparseHNSWBuildOptions()
	options.M = spec.hnsw.M
	options.EFConstruction = spec.hnsw.EFConstruction
	builder, err := core.NewSparseHNSWBuilder(options)
	if err != nil {
		return nil, err
	}
	if err := addCollectionSparseDocuments(ctx, builder, field, documents, spec.quantize); err != nil {
		return nil, err
	}
	return builder.Build(ctx)
}

type sparseCollectionBuilder interface {
	AddSparse(ctx context.Context, key uint64, vector core.SparseVector) error
}

func addCollectionSparseDocuments(
	ctx context.Context,
	builder sparseCollectionBuilder,
	field FieldSchema,
	documents []Document,
	quantize QuantizeType,
) error {
	for _, document := range documents {
		if err := ctx.Err(); err != nil {
			return err
		}
		value, found := document.Fields[field.Name]
		if !found || value == nil {
			continue
		}
		vector, err := sparseValueToCore(value)
		if err != nil {
			return fmt.Errorf("document %d field %q: %w", document.DocID, field.Name, err)
		}
		if quantize == QuantizeTypeFP16 {
			vector, err = sparseFP16Vector(vector)
			if err != nil {
				return fmt.Errorf("document %d field %q: %w", document.DocID, field.Name, err)
			}
		}
		if err := builder.AddSparse(ctx, document.DocID, vector); err != nil {
			return err
		}
	}
	return nil
}

func collectionDenseCandidates(ctx context.Context, field FieldSchema, documents []Document) ([]core.Candidate, error) {
	candidates := make([]core.Candidate, 0, len(documents))
	for _, document := range documents {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		value, found := document.Fields[field.Name]
		if !found || value == nil {
			continue
		}
		vector, err := denseValueToFloat32(value)
		if err != nil {
			return nil, fmt.Errorf("document %d field %q: %w", document.DocID, field.Name, err)
		}
		candidates = append(candidates, core.Candidate{Key: document.DocID, Vector: vector})
	}
	return candidates, nil
}

func toCoreQuantization(value QuantizeType) (core.Quantization, error) {
	switch value {
	case QuantizeTypeFP16:
		return core.QuantizationFP16, nil
	case QuantizeTypeInt8:
		return core.QuantizationInt8, nil
	case QuantizeTypeInt4:
		return core.QuantizationInt4, nil
	default:
		return 0, notSupported("query", "", fmt.Sprintf("quantization %s is not implemented", value))
	}
}

func collectionReformer(schemaName string, field FieldSchema, spec collectionVectorIndex) (core.DenseReformer, error) {
	if !spec.rotate {
		return nil, nil
	}
	dimension := int(field.Dimension)
	signs := make([]byte, 4*((dimension+7)/8))
	seed := []byte(fmt.Sprintf("zvec-go-rotation-v1\x00%s\x00%s\x00%d\x00%d\x00%d", schemaName, field.Name, dimension, spec.indexType, spec.quantize))
	for offset, counter := 0, uint64(0); offset < len(signs); counter++ {
		input := make([]byte, len(seed)+8)
		copy(input, seed)
		binary.LittleEndian.PutUint64(input[len(seed):], counter)
		digest := sha256.Sum256(input)
		offset += copy(signs[offset:], digest[:])
	}
	rotator, err := core.NewFHTRotatorFromSigns(dimension, signs)
	if err != nil {
		return nil, err
	}
	return core.NewRotationReformer(rotator)
}

func sparseFP16Vector(vector core.SparseVector) (core.SparseVector, error) {
	result := core.SparseVector{
		Indices: append([]uint32(nil), vector.Indices...),
		Values:  make([]float32, len(vector.Values)),
	}
	for index, value := range vector.Values {
		converted := ailego.Float16BitsToFloat32(ailego.Float32ToFloat16Bits(value))
		if math.IsInf(float64(converted), 0) || math.IsNaN(float64(converted)) {
			return core.SparseVector{}, core.ErrQuantizationOverflow
		}
		result.Values[index] = converted
	}
	return result, nil
}

func validateCollectionVectorRepresentations(ctx context.Context, schema CollectionSchema, document Document) error {
	for _, field := range schema.Fields {
		if err := ctx.Err(); err != nil {
			return err
		}
		if !field.DataType.IsVector() {
			continue
		}
		value, found := document.Fields[field.Name]
		if !found || value == nil {
			continue
		}
		spec, err := resolveCollectionVectorIndex(field, "validate document", "")
		if err != nil {
			if errors.Is(err, ErrNotSupported) {
				continue
			}
			return err
		}
		if spec.quantize == QuantizeTypeUndefined {
			continue
		}
		if field.DataType.IsSparseVector() {
			vector, err := sparseValueToCore(value)
			if err == nil && spec.quantize == QuantizeTypeFP16 {
				_, err = sparseFP16Vector(vector)
			}
			if err != nil {
				return invalidArgument("validate document", "field %q cannot be represented by %s: %v", field.Name, spec.quantize, err)
			}
			continue
		}
		vector, err := denseValueToFloat32(value)
		if err != nil {
			return invalidArgument("validate document", "field %q: %v", field.Name, err)
		}
		if spec.indexType == IndexTypeHNSWRaBitQ {
			continue
		}
		kind, err := toCoreQuantization(spec.quantize)
		if err != nil {
			return err
		}
		reformer, err := collectionReformer(schema.Name, field, spec)
		if err != nil {
			return invalidArgument("validate document", "field %q reformer: %v", field.Name, err)
		}
		_, err = core.NewScalarQuantizedFlatIndex(ctx, int(field.Dimension), spec.metric, kind, reformer, []core.Candidate{{Vector: vector}})
		if err != nil {
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				return err
			}
			return invalidArgument("validate document", "field %q cannot be represented by %s: %v", field.Name, spec.quantize, err)
		}
	}
	return nil
}

// CreateIndex validates and backfills a currently implemented index, then
// atomically publishes the new schema in a manifest generation. At this stage
// Vector, INVERT, and FTS indexes are snapshot-local runtime indexes, so backfill
// validates the complete live snapshot and publication persists their
// parameters.
func (c *Collection) CreateIndex(ctx context.Context, column string, index IndexParams, options CreateIndexOptions) error {
	const op = "create index"
	if c == nil {
		return invalidArgument(op, "collection is nil")
	}
	if ctx == nil {
		return invalidArgument(op, "context is nil")
	}
	if err := ctx.Err(); err != nil {
		return wrapCollectionError(op, c.Path(), err)
	}
	releaseRuntime, err := c.beginRuntimeTask(ctx, runtimeOptimizeTask, op, 8)
	if err != nil {
		return wrapCollectionError(op, c.Path(), err)
	}
	defer releaseRuntime()
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.requireOpenLocked(op); err != nil {
		return err
	}
	if c.options.ReadOnly {
		return &Error{Code: ErrorCodePermissionDenied, Op: op, Path: c.path, Message: "collection is read-only"}
	}
	if column == "" {
		return invalidArgument(op, "column is empty")
	}
	if indexParamsNil(index) {
		return invalidArgument(op, "index parameters are nil")
	}
	if options.Concurrency < 0 {
		return invalidArgument(op, "Concurrency cannot be negative")
	}
	fieldIndex := -1
	for position := range c.schema.Fields {
		if c.schema.Fields[position].Name == column {
			fieldIndex = position
			break
		}
	}
	if fieldIndex < 0 {
		return &Error{Code: ErrorCodeNotFound, Op: op, Path: c.path, Message: fmt.Sprintf("field %q does not exist", column)}
	}
	if err := validateIndexParams(index); err != nil {
		return err
	}

	normalized := index.cloneIndexParams()
	nextSchema := c.schema.Clone()
	oldField := c.schema.Fields[fieldIndex]
	nextSchema.Fields[fieldIndex].Index = normalized
	if err := nextSchema.Validate(); err != nil {
		return err
	}
	if err := supportedCreateIndex(nextSchema.Fields[fieldIndex], normalized, c.path); err != nil {
		return err
	}
	if equalIndexParams(oldField.Index, normalized) {
		return nil
	}
	if !oldField.DataType.IsVector() && !indexParamsNil(oldField.Index) && oldField.Index.IndexType() != normalized.IndexType() {
		return notSupported(op, c.path, fmt.Sprintf(
			"field %q already has %s and cannot also use %s", column, oldField.Index.IndexType(), normalized.IndexType(),
		))
	}
	if err := c.validateIndexBackfillLocked(ctx, nextSchema.Fields[fieldIndex], options.Concurrency); err != nil {
		return wrapCollectionError(op, c.path, err)
	}
	return c.publishSchemaLocked(ctx, op, nextSchema)
}

// DropIndex atomically clears a scalar index or restores a vector field to the
// baseline unquantized Flat/IP definition. Existing documents are unchanged.
func (c *Collection) DropIndex(ctx context.Context, column string) error {
	const op = "drop index"
	if c == nil {
		return invalidArgument(op, "collection is nil")
	}
	if ctx == nil {
		return invalidArgument(op, "context is nil")
	}
	if err := ctx.Err(); err != nil {
		return wrapCollectionError(op, c.Path(), err)
	}
	releaseRuntime, err := c.beginRuntimeTask(ctx, runtimeOptimizeTask, op, 8)
	if err != nil {
		return wrapCollectionError(op, c.Path(), err)
	}
	defer releaseRuntime()
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.requireOpenLocked(op); err != nil {
		return err
	}
	if c.options.ReadOnly {
		return &Error{Code: ErrorCodePermissionDenied, Op: op, Path: c.path, Message: "collection is read-only"}
	}
	if column == "" {
		return invalidArgument(op, "column is empty")
	}
	fieldIndex := -1
	for position := range c.schema.Fields {
		if c.schema.Fields[position].Name == column {
			fieldIndex = position
			break
		}
	}
	if fieldIndex < 0 {
		return &Error{Code: ErrorCodeNotFound, Op: op, Path: c.path, Message: fmt.Sprintf("field %q does not exist", column)}
	}
	oldField := c.schema.Fields[fieldIndex]
	if !oldField.DataType.IsVector() && indexParamsNil(oldField.Index) {
		return nil
	}
	defaultFlat := NewFlatIndexParams(MetricTypeIP)
	if oldField.DataType.IsVector() && equalIndexParams(oldField.EffectiveIndex(), defaultFlat) {
		return nil
	}
	nextSchema := c.schema.Clone()
	if oldField.DataType.IsVector() {
		nextSchema.Fields[fieldIndex].Index = defaultFlat
	} else {
		nextSchema.Fields[fieldIndex].Index = nil
	}
	if err := nextSchema.Validate(); err != nil {
		return err
	}
	if oldField.DataType.IsVector() {
		if err := c.validateIndexBackfillLocked(ctx, nextSchema.Fields[fieldIndex], 0); err != nil {
			return wrapCollectionError(op, c.path, err)
		}
	}
	return c.publishSchemaLocked(ctx, op, nextSchema)
}

func (c *Collection) publishSchemaLocked(ctx context.Context, op string, nextSchema CollectionSchema) error {
	encoded, err := marshalCollectionSchema(nextSchema)
	if err != nil {
		return wrapCollectionError(op, c.path, err)
	}
	committed, publishErr := c.store.PublishSchema(ctx, encoded)
	if committed {
		c.schema = nextSchema
	}
	if publishErr != nil {
		return wrapCollectionError(op, c.path, publishErr)
	}
	if !committed {
		return &Error{Code: ErrorCodeInternal, Op: op, Path: c.path, Message: "schema publication did not commit"}
	}
	return nil
}

func (c *Collection) rewriteCollectionDocumentsLocked(
	ctx context.Context,
	op string,
	nextSchema CollectionSchema,
	documents []Document,
	workers int,
	transform func(*Document) error,
) error {
	if transform == nil {
		return &Error{Code: ErrorCodeInternal, Op: op, Path: c.path, Message: "document rewrite transform is nil"}
	}
	workers = c.optimizeWorkers(workers)
	rewritten := make([]db.StoredDocument, len(documents))
	if err := ailego.ParallelFor(ctx, len(documents), workers, func(_ context.Context, index int) error {
		document := documents[index]
		if transformErr := transform(&document); transformErr != nil {
			return fmt.Errorf("document %d: %w", document.DocID, transformErr)
		}
		if validateErr := document.Validate(nextSchema); validateErr != nil {
			return fmt.Errorf("document %d: %w", document.DocID, validateErr)
		}
		payload, encodeErr := marshalDocumentPayload(document.Fields)
		if encodeErr != nil {
			return fmt.Errorf("document %d: %w", document.DocID, encodeErr)
		}
		rewritten[index] = db.StoredDocument{
			DocID: document.DocID, PrimaryKey: document.PrimaryKey, Payload: payload,
		}
		return nil
	}); err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return wrapCollectionError(op, c.path, err)
		}
		return invalidArgument(op, "document rewrite failed: %v", err)
	}
	encodedSchema, err := marshalCollectionSchema(nextSchema)
	if err != nil {
		return wrapCollectionError(op, c.path, err)
	}
	committed, rewriteErr := c.store.RewriteDocuments(ctx, encodedSchema, rewritten)
	if committed {
		c.schema = nextSchema
	}
	if rewriteErr != nil {
		return wrapCollectionError(op, c.path, rewriteErr)
	}
	if !committed {
		return &Error{Code: ErrorCodeInternal, Op: op, Path: c.path, Message: "document rewrite did not commit"}
	}
	return nil
}

func supportedCreateIndex(nextField FieldSchema, index IndexParams, path string) error {
	const op = "create index"
	switch index.IndexType() {
	case IndexTypeInvert:
		if nextField.DataType.IsVector() {
			return invalidArgument(op, "vector field %q cannot use INVERT", nextField.Name)
		}
		_, _, supported := filterValueKind(nextField.DataType)
		if !supported {
			return notSupported(op, path, fmt.Sprintf("INVERT is not implemented for %s field %q", nextField.DataType, nextField.Name))
		}
		return nil
	case IndexTypeFTS:
		if nextField.DataType != DataTypeString {
			return invalidArgument(op, "FTS field %q must use STRING", nextField.Name)
		}
		return nil
	case IndexTypeFlat, IndexTypeHNSW, IndexTypeHNSWRaBitQ, IndexTypeIVF, IndexTypeDiskANN, IndexTypeVamana:
		if !nextField.DataType.IsVector() {
			return invalidArgument(op, "scalar field %q cannot use %s", nextField.Name, index.IndexType())
		}
		_, err := resolveCollectionVectorIndex(nextField, op, path)
		return err
	default:
		return notSupported(op, path, fmt.Sprintf("index %s on field %q is not implemented", index.IndexType(), nextField.Name))
	}
}

func equalIndexParams(left, right IndexParams) bool {
	if indexParamsNil(left) || indexParamsNil(right) {
		return indexParamsNil(left) && indexParamsNil(right)
	}
	return reflect.DeepEqual(left.cloneIndexParams(), right.cloneIndexParams())
}

func (c *Collection) validateIndexBackfillLocked(ctx context.Context, field FieldSchema, workers int) error {
	workers = c.optimizeWorkers(workers)
	documents, err := c.liveDocumentsLocked(ctx)
	if err != nil {
		return err
	}
	switch field.Index.IndexType() {
	case IndexTypeInvert:
		kind, array, supported := filterValueKind(field.DataType)
		if !supported {
			return fmt.Errorf("unsupported INVERT data type %s", field.DataType)
		}
		params := field.Index.cloneIndexParams().(InvertIndexParams)
		index, err := dbsql.NewInvertedIndex(dbsql.Field{
			Name: field.Name, Kind: kind, Array: array, Nullable: field.Nullable,
			Filterable: true, Indexed: true,
			RangeOptimized: params.EnableRangeOptimization, ExtendedWildcard: params.EnableExtendedWildcard,
		})
		if err != nil {
			return err
		}
		if err := ailego.ParallelFor(ctx, len(documents), workers, func(_ context.Context, position int) error {
			document := &documents[position]
			raw, found := document.Fields[field.Name]
			value, err := toFilterValue(index.Field(), raw, found)
			if err != nil {
				return fmt.Errorf("document %d field %q: %w", document.DocID, field.Name, err)
			}
			return index.Add(uint64(position), value)
		}); err != nil {
			return err
		}
		return index.Seal()
	case IndexTypeFTS:
		_, err := buildCollectionFTSRuntime(ctx, field, documents, nil)
		return err
	case IndexTypeFlat, IndexTypeHNSW, IndexTypeHNSWRaBitQ, IndexTypeIVF, IndexTypeDiskANN, IndexTypeVamana:
		spec, err := resolveCollectionVectorIndex(field, "create index", c.path)
		if err != nil {
			return err
		}
		if field.DataType.IsDenseVector() {
			switch spec.indexType {
			case IndexTypeFlat:
				_, err = buildCollectionDenseFlat(ctx, c.schema.Name, field, documents, spec)
			case IndexTypeHNSW:
				_, err = buildCollectionDenseHNSW(ctx, c.schema.Name, field, documents, spec)
			case IndexTypeHNSWRaBitQ:
				_, err = buildCollectionDenseHNSWRaBitQ(ctx, field, documents, spec, workers)
			case IndexTypeIVF:
				_, err = buildCollectionDenseIVF(ctx, c.schema.Name, field, documents, spec, workers)
			case IndexTypeDiskANN:
				_, err = buildCollectionDenseDiskANN(ctx, c.schema.Name, field, documents, spec, workers, c.options.MaxBufferSize)
			case IndexTypeVamana:
				_, err = buildCollectionDenseVamana(ctx, c.schema.Name, field, documents, spec)
			}
			return err
		}
		_, err = buildCollectionSparseIndex(ctx, field, documents, spec, false)
		return err
	default:
		return fmt.Errorf("unsupported index type %s", field.Index.IndexType())
	}
}

// DropColumn atomically removes one basic numeric field from the schema and
// every live document payload. Nonnumeric fields are reserved for their owning
// index milestones and are rejected until those implementations can migrate
// their auxiliary state safely.
func (c *Collection) DropColumn(ctx context.Context, column string) error {
	const op = "drop column"
	if c == nil {
		return invalidArgument(op, "collection is nil")
	}
	if ctx == nil {
		return invalidArgument(op, "context is nil")
	}
	if err := ctx.Err(); err != nil {
		return wrapCollectionError(op, c.Path(), err)
	}
	releaseRuntime, err := c.beginRuntimeTask(ctx, runtimeOptimizeTask, op, 8)
	if err != nil {
		return wrapCollectionError(op, c.Path(), err)
	}
	defer releaseRuntime()
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.requireOpenLocked(op); err != nil {
		return err
	}
	if c.options.ReadOnly {
		return &Error{Code: ErrorCodePermissionDenied, Op: op, Path: c.path, Message: "collection is read-only"}
	}
	if column == "" {
		return invalidArgument(op, "column is empty")
	}
	position := -1
	for index := range c.schema.Fields {
		if c.schema.Fields[index].Name == column {
			position = index
			break
		}
	}
	if position < 0 {
		return &Error{Code: ErrorCodeNotFound, Op: op, Path: c.path, Message: fmt.Sprintf("field %q does not exist", column)}
	}
	field := c.schema.Fields[position]
	if !addColumnDataTypeSupported(field.DataType) {
		return invalidArgument(op, "only basic numeric columns can be dropped, got %s", field.DataType)
	}

	nextSchema := c.schema.Clone()
	nextSchema.Fields = slices.Delete(nextSchema.Fields, position, position+1)
	if err := nextSchema.Validate(); err != nil {
		return err
	}
	documents, err := c.liveDocumentsLocked(ctx)
	if err != nil {
		return wrapCollectionError(op, c.path, err)
	}
	if len(documents) == 0 {
		return c.publishSchemaLocked(ctx, op, nextSchema)
	}
	return c.rewriteCollectionDocumentsLocked(ctx, op, nextSchema, documents, 0, func(document *Document) error {
		delete(document.Fields, column)
		return nil
	})
}

// BatchWriteError summarizes per-document write failures. The returned
// WriteResult slice remains authoritative and preserves input order.
type BatchWriteError struct {
	Failed int
	causes []error
}

func (e *BatchWriteError) Error() string {
	if e == nil {
		return "zvec: batch write failed"
	}
	return "zvec: batch write: " + strconv.Itoa(e.Failed) + " document operations failed"
}

// Unwrap exposes every per-document cause to errors.Is and errors.As.
func (e *BatchWriteError) Unwrap() []error {
	if e == nil {
		return nil
	}
	return slices.Clone(e.causes)
}

func (e *BatchWriteError) add(err error) {
	if err == nil {
		return
	}
	e.Failed++
	e.causes = append(e.causes, err)
}

func wrapCollectionError(op, path string, err error) error {
	if err == nil {
		return nil
	}
	var existing *Error
	if errors.As(err, &existing) && existing != nil {
		copy := *existing
		if copy.Op == "" {
			copy.Op = op
		}
		if copy.Path == "" {
			copy.Path = path
		}
		return &copy
	}
	code := ErrorCodeUnknown
	switch {
	case errors.Is(err, db.ErrPrimaryKeyNotFound),
		errors.Is(err, db.ErrDocumentNotFound),
		errors.Is(err, db.ErrManifestNotFound),
		errors.Is(err, os.ErrNotExist):
		code = ErrorCodeNotFound
	case errors.Is(err, db.ErrPrimaryKeyExists),
		errors.Is(err, db.ErrManifestExists),
		errors.Is(err, os.ErrExist):
		code = ErrorCodeAlreadyExists
	case errors.Is(err, db.ErrReadOnly), errors.Is(err, os.ErrPermission):
		code = ErrorCodePermissionDenied
	case errors.Is(err, db.ErrCollectionClosed),
		errors.Is(err, db.ErrWALClosed),
		errors.Is(err, db.ErrWALReadOnly):
		code = ErrorCodeFailedPrecondition
	case errors.Is(err, db.ErrSegmentFull),
		errors.Is(err, db.ErrWALRecordTooLarge):
		code = ErrorCodeResourceExhausted
	case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
		code = ErrorCodeUnavailable
	case errors.Is(err, db.ErrCollectionCorrupt),
		errors.Is(err, db.ErrManifestCorrupt),
		errors.Is(err, db.ErrSegmentCorrupt),
		errors.Is(err, db.ErrWALCorrupt):
		code = ErrorCodeInternal
	}
	return &Error{Code: code, Op: op, Path: path, Err: err}
}

func notSupported(op, path, message string) error {
	return &Error{Code: ErrorCodeNotSupported, Op: op, Path: path, Message: message}
}

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

// CreateIndexOptions controls index-build concurrency. Zero lets the library
// select an appropriate worker count.
type CreateIndexOptions struct{ Concurrency int }

// AddColumnOptions controls column backfill concurrency.
type AddColumnOptions struct{ Concurrency int }

// AlterColumnOptions controls column migration concurrency.
type AlterColumnOptions struct{ Concurrency int }

// OptimizeOptions controls segment-optimization concurrency.
type OptimizeOptions struct{ Concurrency int }

const (
	// DefaultMultiQueryTopK is the pinned default final result count.
	DefaultMultiQueryTopK = 10
	// DefaultSubQueryCandidates is the pinned default candidate count per
	// MultiQuery branch.
	DefaultSubQueryCandidates = 10
	// MaxQueryTopK is the pinned upper bound for a MultiQuery final or branch
	// result count.
	MaxQueryTopK = 100_000
)

// FTSClause describes one full-text target. Exactly one of Query and Match
// must be non-empty. Query uses the FTS expression grammar; Match analyzes the
// text as natural language without interpreting operators.
type FTSClause struct {
	Query string
	Match string
}

// SubQuery describes one candidate-producing branch of MultiQuery. Exactly
// one of DenseVector, SparseVector, and FTS must be set. A zero NumCandidates
// selects DefaultSubQueryCandidates.
type SubQuery struct {
	Field         string
	DenseVector   DenseVector
	SparseVector  SparseVector
	FTS           *FTSClause
	Params        QueryParams
	NumCandidates int
}

// RerankBatch contains one projected, score-ordered candidate list and the
// corresponding independent field schema. Batches retain SubQuery order.
// Documents and their field values are owned by the batch and may be changed
// by the reranker without mutating the collection snapshot.
type RerankBatch struct {
	Field     FieldSchema
	Documents []Document
}

// Reranker combines MultiQuery candidate batches. Implementations may adjust
// scores and order but must return distinct documents drawn from the supplied
// batches. Implementations shared by concurrent calls must be concurrency-safe.
type Reranker interface {
	Rerank(ctx context.Context, batches []RerankBatch, topK int) ([]Document, error)
}

// MultiQuery combines vector, sparse-vector, and full-text candidate lists
// through a Reranker. Nil selects NewRRFReranker. At least two sub-queries are
// required. Zero TopK selects DefaultMultiQueryTopK.
type MultiQuery struct {
	Queries    []SubQuery
	TopK       int
	Filter     string
	Projection Projection
	Reranker   Reranker
}

// MultiQuery executes every branch over one live-document snapshot, applies
// one shared scalar filter, and delegates fusion to the configured or default
// Reranker. BM25 corpus statistics include every live document; the scalar
// filter masks FTS candidates without changing IDF or average document length.
func (c *Collection) MultiQuery(ctx context.Context, query MultiQuery) ([]Document, error) {
	const op = "multi query"
	if c == nil {
		return nil, invalidArgument(op, "collection is nil")
	}
	if ctx == nil {
		return nil, invalidArgument(op, "context is nil")
	}
	if len(query.Queries) < 2 {
		return nil, invalidArgument(op, "at least two sub-queries are required")
	}
	topK, err := normalizedMultiQueryCount(query.TopK, DefaultMultiQueryTopK, "TopK")
	if err != nil {
		return nil, err
	}
	releaseRuntime, err := c.beginRuntimeTask(ctx, runtimeQueryTask, op, uint64(8+2*len(query.Queries)))
	if err != nil {
		return nil, wrapCollectionError(op, c.Path(), err)
	}
	defer releaseRuntime()
	reranker := query.Reranker
	if isNilInterface(reranker) {
		value := NewRRFReranker()
		reranker = value
	}
	projection := query.Projection.Clone()

	locked := true
	c.mu.RLock()
	defer func() {
		if locked {
			c.mu.RUnlock()
		}
	}()
	if err := c.requireOpenLocked(op); err != nil {
		return nil, err
	}
	if err := projection.Validate(c.schema); err != nil {
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
	runtimeConfig := c.runtimeConfig()
	candidateFilter, err := evaluateFilterDocuments(ctx, filterPlan, documents, runtimeConfig.InvertToForwardScanRatio)
	if err != nil {
		return nil, wrapFilterEvaluationError(op, c.path, err)
	}

	batches := make([]RerankBatch, len(query.Queries))
	candidateIDs := make(map[uint64]struct{})
	ftsFields := make(map[string]*collectionFTSRuntime)
	for index, subQuery := range query.Queries {
		if err := ctx.Err(); err != nil {
			return nil, wrapCollectionError(op, c.path, err)
		}
		candidateCount, err := normalizedMultiQueryCount(
			subQuery.NumCandidates, DefaultSubQueryCandidates,
			fmt.Sprintf("Queries[%d].NumCandidates", index),
		)
		if err != nil {
			return nil, err
		}
		field, found := c.schema.Field(subQuery.Field)
		if !found {
			return nil, invalidArgument(op, "sub-query %d field %q does not exist", index, subQuery.Field)
		}
		target, err := multiQueryTargetKind(subQuery)
		if err != nil {
			return nil, invalidArgument(op, "sub-query %d: %v", index, err)
		}

		var results []core.Result
		switch target {
		case multiQueryTargetDense, multiQueryTargetSparse:
			if !field.DataType.IsVector() {
				return nil, invalidArgument(op, "sub-query %d field %q is not a vector field", index, field.Name)
			}
			results, err = c.searchVectorSnapshot(
				ctx, op, field, subQuery.DenseVector, subQuery.SparseVector,
				candidateCount, subQuery.Params, documents, candidateFilter,
			)
		case multiQueryTargetFTS:
			runtime := ftsFields[field.Name]
			if runtime == nil {
				runtime, err = buildCollectionFTSRuntime(ctx, field, documents, candidateFilter.predicate)
				if err == nil {
					ftsFields[field.Name] = runtime
				}
			}
			if err == nil {
				results, err = searchCollectionFTS(
					ctx, runtime, subQuery.FTS, subQuery.Params, candidateCount,
					candidateFilter.ordinals, candidateFilter.useBruteForce(runtimeConfig.FTSBruteForceByKeysRatio),
				)
			}
		}
		if err != nil {
			return nil, wrapMultiQueryBranchError(op, c.path, index, err)
		}
		materialized, err := c.materializeResults(documents, results, projection)
		if err != nil {
			return nil, err
		}
		for _, result := range results {
			candidateIDs[result.Key] = struct{}{}
		}
		batches[index] = RerankBatch{Field: field.Clone(), Documents: materialized}
	}

	// Release the collection lock before invoking caller code. The immutable
	// snapshot, schema, and candidate batches remain owned by this call.
	schema := c.schema.Clone()
	path := c.path
	c.mu.RUnlock()
	locked = false
	if err := ctx.Err(); err != nil {
		return nil, wrapCollectionError(op, path, err)
	}
	reranked, err := reranker.Rerank(ctx, batches, topK)
	if err != nil {
		return nil, wrapCollectionError(op, path, err)
	}
	if err := ctx.Err(); err != nil {
		return nil, wrapCollectionError(op, path, err)
	}
	return validateAndProjectReranked(op, path, schema, documents, candidateIDs, projection, reranked, topK)
}

func normalizedMultiQueryCount(value, fallback int, name string) (int, error) {
	if value == 0 {
		return fallback, nil
	}
	if value < 0 || value > MaxQueryTopK {
		return 0, invalidArgument("multi query", "%s must be in [1, %d] or zero for the default", name, MaxQueryTopK)
	}
	return value, nil
}

type multiQueryTarget uint8

const (
	multiQueryTargetDense multiQueryTarget = iota + 1
	multiQueryTargetSparse
	multiQueryTargetFTS
)

func multiQueryTargetKind(query SubQuery) (multiQueryTarget, error) {
	hasDense := !isNilInterface(query.DenseVector)
	hasSparse := !isNilInterface(query.SparseVector)
	hasFTS := !isNilInterface(query.FTS)
	count := 0
	if hasDense {
		count++
	}
	if hasSparse {
		count++
	}
	if hasFTS {
		count++
	}
	if count != 1 {
		return 0, fmt.Errorf("exactly one of DenseVector, SparseVector, and FTS must be set")
	}
	if hasDense {
		return multiQueryTargetDense, nil
	}
	if hasSparse {
		return multiQueryTargetSparse, nil
	}
	return multiQueryTargetFTS, nil
}

type collectionFTSRuntime struct {
	analyzer    core.FTSAnalyzer
	dictionary  *core.FTSTermDictionary
	scorer      *core.BM25Scorer
	deleted     *ailego.Bitmap
	documentIDs []uint64
}

func buildCollectionFTSRuntime(
	ctx context.Context,
	field FieldSchema,
	documents []Document,
	candidateFilter core.CandidateFilter,
) (*collectionFTSRuntime, error) {
	if field.DataType != DataTypeString || field.IndexType() != IndexTypeFTS {
		return nil, invalidArgument("multi query", "field %q is not an FTS-indexed STRING field", field.Name)
	}
	if uint64(len(documents)) > uint64(math.MaxUint32) {
		return nil, &Error{
			Code: ErrorCodeResourceExhausted, Op: "multi query",
			Message: fmt.Sprintf("FTS field %q exceeds the uint32 document domain", field.Name),
		}
	}
	params, err := collectionFTSIndexParams(field)
	if err != nil {
		return nil, err
	}
	analyzer, err := newCollectionFTSAnalyzer(ctx, params)
	if err != nil {
		return nil, err
	}
	builder := core.NewFTSFieldBuilder()
	documentIDs := make([]uint64, len(documents))
	var deleted *ailego.Bitmap
	if candidateFilter != nil {
		deleted = ailego.NewBitmap(uint64(len(documents)))
	}
	for index := range documents {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		document := &documents[index]
		documentIDs[index] = document.DocID
		if deleted != nil && !candidateFilter(document.DocID) {
			deleted.Set(uint64(index))
		}
		var text string
		if raw, found := document.Fields[field.Name]; found && raw != nil {
			var ok bool
			text, ok = raw.(string)
			if !ok {
				return nil, fmt.Errorf("stored document %d field %q is %T, want string", document.DocID, field.Name, raw)
			}
		}
		tokens, err := analyzer.Analyze(ctx, text)
		if err != nil {
			return nil, err
		}
		if err := builder.AddDocument(ctx, uint32(index), tokens); err != nil {
			return nil, err
		}
	}
	dictionary, err := builder.Build(ctx)
	if err != nil {
		return nil, err
	}
	// Shared scalar filtering is deliberately absent from corpus statistics:
	// it affects eligibility, not BM25 IDF or length normalization.
	stats, err := core.AggregateFTSCorpusStats(ctx, []core.FTSSegmentView{{Dictionary: dictionary}})
	if err != nil {
		return nil, err
	}
	scorer, err := core.NewBM25Scorer(core.DefaultBM25Params(), stats)
	if err != nil {
		return nil, err
	}
	return &collectionFTSRuntime{
		analyzer: analyzer, dictionary: dictionary, scorer: scorer,
		deleted: deleted, documentIDs: documentIDs,
	}, nil
}

func searchCollectionFTS(
	ctx context.Context,
	runtime *collectionFTSRuntime,
	clause *FTSClause,
	queryParams QueryParams,
	topK int,
	candidateOrdinals []uint32,
	bruteForceByKeys bool,
) ([]core.Result, error) {
	if runtime == nil || clause == nil {
		return nil, invalidArgument("multi query", "FTS runtime and clause are required")
	}
	hasQuery, hasMatch := clause.Query != "", clause.Match != ""
	if hasQuery == hasMatch {
		return nil, invalidArgument("multi query", "exactly one of FTS Query and Match must be non-empty")
	}
	defaultOperator, err := collectionFTSDefaultOperator(queryParams)
	if err != nil {
		return nil, err
	}
	var node core.FTSQueryNode
	if hasQuery {
		node, err = core.ParseFTSQuery(ctx, clause.Query, runtime.analyzer, defaultOperator)
	} else {
		node, err = core.AnalyzeFTSMatchQuery(ctx, clause.Match, runtime.analyzer, defaultOperator)
	}
	if err != nil {
		return nil, err
	}
	var results []core.FTSResult
	if bruteForceByKeys {
		results, err = searchCollectionFTSCandidates(ctx, runtime, node, candidateOrdinals, topK)
	} else {
		results, err = core.SearchFTS(ctx, runtime.dictionary, node, runtime.scorer, core.FTSSearchOptions{
			TopK:                     topK,
			FTSQueryExecutionOptions: core.FTSQueryExecutionOptions{DeletedDocuments: runtime.deleted},
		})
	}
	if err != nil {
		return nil, err
	}
	output := make([]core.Result, len(results))
	for index, result := range results {
		if uint64(result.DocumentID) >= uint64(len(runtime.documentIDs)) {
			return nil, fmt.Errorf("FTS result document ID %d is outside the snapshot", result.DocumentID)
		}
		output[index] = core.Result{Key: runtime.documentIDs[result.DocumentID], Score: result.Score}
	}
	return output, nil
}

func searchCollectionFTSCandidates(
	ctx context.Context,
	runtime *collectionFTSRuntime,
	node core.FTSQueryNode,
	candidateOrdinals []uint32,
	topK int,
) ([]core.FTSResult, error) {
	iterator, err := core.NewFTSScoredQueryIterator(ctx, runtime.dictionary, node, runtime.scorer, core.FTSQueryExecutionOptions{})
	if err != nil {
		return nil, err
	}
	results := make([]core.FTSResult, 0, min(topK, len(candidateOrdinals)))
	for _, documentID := range candidateOrdinals {
		if !iterator.Advance(ctx, documentID) {
			break
		}
		if iterator.DocumentID() != documentID || iterator.Score() <= 0 {
			continue
		}
		results = append(results, core.FTSResult{DocumentID: documentID, Score: iterator.Score()})
	}
	if err := iterator.Err(); err != nil {
		return nil, err
	}
	sort.Slice(results, func(left, right int) bool {
		if results[left].Score != results[right].Score {
			return results[left].Score > results[right].Score
		}
		return results[left].DocumentID < results[right].DocumentID
	})
	if len(results) > topK {
		results = results[:topK]
	}
	return results, nil
}

func collectionFTSDefaultOperator(params QueryParams) (core.FTSDefaultOperator, error) {
	value := FTSQueryParams{}
	if !isNilInterface(params) {
		if params.IndexType() != IndexTypeFTS {
			return 0, invalidArgument("multi query", "query parameters for %s cannot be used with FTS", params.IndexType())
		}
		if err := params.Validate(); err != nil {
			return 0, err
		}
		switch typed := params.(type) {
		case FTSQueryParams:
			value = typed
		case *FTSQueryParams:
			if typed == nil {
				return 0, invalidArgument("multi query", "FTS query parameters are nil")
			}
			value = *typed
		default:
			return 0, invalidArgument("multi query", "invalid FTS query parameter value")
		}
	}
	operator, err := core.ParseFTSDefaultOperator(value.DefaultOperator)
	if err != nil {
		return 0, invalidArgument("multi query", "invalid FTS default operator: %v", err)
	}
	return operator, nil
}

func collectionFTSIndexParams(field FieldSchema) (FTSIndexParams, error) {
	if indexParamsNil(field.Index) || field.Index.IndexType() != IndexTypeFTS {
		return FTSIndexParams{}, invalidArgument("multi query", "field %q does not have FTS index parameters", field.Name)
	}
	var params FTSIndexParams
	switch value := field.Index.(type) {
	case FTSIndexParams:
		params = value
	case *FTSIndexParams:
		if value == nil {
			return FTSIndexParams{}, invalidArgument("multi query", "field %q has nil FTS index parameters", field.Name)
		}
		params = *value
	default:
		return FTSIndexParams{}, invalidArgument("multi query", "field %q has invalid FTS index parameters", field.Name)
	}
	if err := params.Validate(); err != nil {
		return FTSIndexParams{}, err
	}
	return params, nil
}

func newCollectionFTSAnalyzer(ctx context.Context, params FTSIndexParams) (core.FTSAnalyzer, error) {
	if err := params.Validate(); err != nil {
		return nil, err
	}
	extra, err := decodeFTSExtraParams(params.ExtraParams)
	if err != nil {
		return nil, err
	}
	tokenizerName := params.Tokenizer
	if tokenizerName == "" {
		tokenizerName = "standard"
	}
	var tokenizer core.Tokenizer
	switch tokenizerName {
	case "whitespace":
		tokenizer = core.NewWhitespaceTokenizer()
	case "standard":
		options := core.DefaultStandardTokenizerOptions()
		if value, found := extra["max_token_length"]; found {
			length, _ := jsonPositiveInteger(value)
			options.MaxTokenLength = uint32(length)
		}
		tokenizer, err = core.NewStandardTokenizer(options)
	case "ngram":
		options := core.DefaultNGramTokenizerOptions()
		minimum, _ := ngramSize(extra, "ngram_min", int64(options.Min))
		maximum, _ := ngramSize(extra, "ngram_max", int64(options.Max))
		options.Min, options.Max = uint32(minimum), uint32(maximum)
		if raw, found := extra["token_chars"]; found {
			for _, item := range raw.([]any) {
				switch item.(string) {
				case "letter":
					options.TokenChars |= core.NGramTokenCharLetter
				case "digit":
					options.TokenChars |= core.NGramTokenCharDigit
				case "whitespace":
					options.TokenChars |= core.NGramTokenCharWhitespace
				case "punctuation":
					options.TokenChars |= core.NGramTokenCharPunctuation
				case "symbol":
					options.TokenChars |= core.NGramTokenCharSymbol
				}
			}
		}
		tokenizer, err = core.NewNGramTokenizer(options)
	case "jieba":
		options := core.DefaultJiebaTokenizerOptions()
		if value, found := extra["jieba_dict_dir"]; found {
			options.DictDir = value.(string)
		}
		if value, found := extra["user_dict_path"]; found {
			options.UserDictPath = value.(string)
		}
		if value, found := extra["cut_mode"]; found {
			options.CutMode = core.JiebaCutMode(value.(string))
		}
		tokenizer, err = core.NewJiebaTokenizer(ctx, options)
	default:
		return nil, invalidArgument("multi query", "unknown FTS tokenizer %q", tokenizerName)
	}
	if err != nil {
		return nil, err
	}
	filters := make([]core.TokenFilter, 0, len(params.Filters))
	for _, name := range params.Filters {
		switch name {
		case "lowercase":
			filters = append(filters, core.NewLowercaseTokenFilter())
		case "ascii_folding":
			filters = append(filters, core.NewASCIIFoldingTokenFilter())
		case "stemmer":
			options := core.DefaultStemmerTokenFilterOptions()
			if value, found := extra["stemmer_lang"]; found {
				options.Language = value.(string)
			}
			filter, filterErr := core.NewStemmerTokenFilter(options)
			if filterErr != nil {
				return nil, filterErr
			}
			filters = append(filters, filter)
		default:
			return nil, invalidArgument("multi query", "unknown FTS token filter %q", name)
		}
	}
	return core.NewFTSTokenizerPipeline(tokenizer, filters...)
}

func wrapMultiQueryBranchError(op, path string, index int, err error) error {
	if err == nil {
		return nil
	}
	var structured *Error
	if errors.As(err, &structured) {
		wrapped := wrapCollectionError(op, path, err)
		var result *Error
		if errors.As(wrapped, &result) && result != nil {
			copy := *result
			copy.Op = op
			copy.Path = path
			copy.Message = fmt.Sprintf("sub-query %d: %s", index, errorDetail(result))
			return &copy
		}
		return wrapped
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return wrapCollectionError(op, path, err)
	}
	if errors.Is(err, core.ErrFTSQuerySyntax) ||
		errors.Is(err, core.ErrUnsupportedFTSQuery) ||
		errors.Is(err, core.ErrInvalidFTSQuery) ||
		errors.Is(err, core.ErrFTSQueryTooComplex) ||
		errors.Is(err, core.ErrTokenizerInputTooLarge) {
		return &Error{
			Code: ErrorCodeInvalidArgument, Op: op, Path: path,
			Message: fmt.Sprintf("sub-query %d has an invalid FTS query", index), Err: err,
		}
	}
	return &Error{
		Code: ErrorCodeInternal, Op: op, Path: path,
		Message: fmt.Sprintf("execute sub-query %d", index), Err: err,
	}
}

func errorDetail(err *Error) string {
	if err == nil {
		return "query failed"
	}
	if err.Message != "" {
		return err.Message
	}
	return err.Code.DefaultMessage()
}

func validateAndProjectReranked(
	op, path string,
	schema CollectionSchema,
	documents []Document,
	candidateIDs map[uint64]struct{},
	projection Projection,
	reranked []Document,
	topK int,
) ([]Document, error) {
	if len(reranked) > topK {
		return nil, &Error{
			Code: ErrorCodeInvalidArgument, Op: op, Path: path,
			Message: fmt.Sprintf("reranker returned %d documents, exceeding TopK %d", len(reranked), topK),
		}
	}
	byID := make(map[uint64]Document, len(documents))
	for _, document := range documents {
		byID[document.DocID] = document
	}
	seen := make(map[uint64]struct{}, len(reranked))
	output := make([]Document, len(reranked))
	for index, result := range reranked {
		if _, found := candidateIDs[result.DocID]; !found {
			return nil, invalidRerankerOutput(op, path, "document %d was not returned by any sub-query", result.DocID)
		}
		source, found := byID[result.DocID]
		if !found || source.PrimaryKey != result.PrimaryKey {
			return nil, invalidRerankerOutput(op, path, "document %d has an invalid primary key", result.DocID)
		}
		if _, duplicate := seen[result.DocID]; duplicate {
			return nil, invalidRerankerOutput(op, path, "document %d appears more than once", result.DocID)
		}
		seen[result.DocID] = struct{}{}
		if value := float64(result.Score); math.IsNaN(value) || math.IsInf(value, 0) {
			return nil, invalidRerankerOutput(op, path, "document %d has a non-finite score", result.DocID)
		}
		source.Score = result.Score
		projected, err := ProjectDocument(source, schema, projection)
		if err != nil {
			return nil, err
		}
		output[index] = projected
	}
	return output, nil
}

func invalidRerankerOutput(op, path, format string, args ...any) error {
	return &Error{
		Code: ErrorCodeInvalidArgument, Op: op, Path: path,
		Message: "invalid reranker output: " + fmt.Sprintf(format, args...),
	}
}

// Optimize atomically compacts the current live snapshot into maximally sized
// contiguous-ID segments, reclaims superseded/deleted versions, rebuilds the
// implemented vector/INVERT/FTS runtime state, and removes obsolete storage
// files.
func (c *Collection) Optimize(ctx context.Context, options OptimizeOptions) error {
	const op = "optimize collection"
	if c == nil {
		return invalidArgument(op, "collection is nil")
	}
	if ctx == nil {
		return invalidArgument(op, "context is nil")
	}
	if err := ctx.Err(); err != nil {
		return wrapCollectionError(op, c.Path(), err)
	}
	releaseRuntime, err := c.beginRuntimeTask(ctx, runtimeOptimizeTask, op, 8)
	if err != nil {
		return wrapCollectionError(op, c.Path(), err)
	}
	defer releaseRuntime()
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.requireOpenLocked(op); err != nil {
		return err
	}
	if c.options.ReadOnly {
		return &Error{Code: ErrorCodePermissionDenied, Op: op, Path: c.path, Message: "collection is read-only"}
	}
	if options.Concurrency < 0 {
		return invalidArgument(op, "Concurrency cannot be negative")
	}
	needed, err := c.store.OptimizationNeeded(ctx)
	if err != nil {
		return wrapCollectionError(op, c.path, err)
	}
	if !needed {
		// Publication is the durability boundary. A process can stop after the
		// new manifest becomes current but before obsolete files are removed;
		// a later no-op optimization must finish that safe cleanup.
		return wrapCollectionError(op, c.path, c.store.PruneObsoleteArtifacts(ctx))
	}
	for _, field := range c.schema.Fields {
		if err := optimizableField(field, c.path); err != nil {
			return err
		}
	}
	documents, err := c.liveDocumentsLocked(ctx)
	if err != nil {
		return wrapCollectionError(op, c.path, err)
	}
	if err := c.rewriteCollectionDocumentsLocked(ctx, op, c.schema.Clone(), documents, options.Concurrency, func(*Document) error {
		return nil
	}); err != nil {
		return err
	}
	return wrapCollectionError(op, c.path, c.store.PruneObsoleteArtifacts(ctx))
}

func optimizableField(field FieldSchema, path string) error {
	index := field.EffectiveIndex()
	if indexParamsNil(index) {
		return nil
	}
	switch index.IndexType() {
	case IndexTypeFlat, IndexTypeHNSW, IndexTypeHNSWRaBitQ, IndexTypeIVF, IndexTypeDiskANN, IndexTypeVamana:
		if !field.DataType.IsVector() {
			return invalidArgument("optimize collection", "scalar field %q cannot use %s", field.Name, index.IndexType())
		}
		_, err := resolveCollectionVectorIndex(field, "optimize collection", path)
		return err
	case IndexTypeInvert:
		if field.DataType.IsVector() {
			return invalidArgument("optimize collection", "vector field %q cannot use INVERT", field.Name)
		}
		_, _, supported := filterValueKind(field.DataType)
		if !supported {
			return notSupported("optimize collection", path, fmt.Sprintf("INVERT is not implemented for %s field %q", field.DataType, field.Name))
		}
		return nil
	case IndexTypeFTS:
		if field.DataType != DataTypeString {
			return invalidArgument("optimize collection", "FTS field %q must use STRING", field.Name)
		}
		return nil
	default:
		return notSupported("optimize collection", path, fmt.Sprintf("index %s on field %q is not implemented", index.IndexType(), field.Name))
	}
}

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

// Query executes the field's exact or ANN search over the current live
// document versions. Filter is parsed and schema-bound before its candidate
// mask is applied.
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
	releaseRuntime, err := c.beginRuntimeTask(ctx, runtimeQueryTask, op, 8)
	if err != nil {
		return nil, wrapCollectionError(op, c.Path(), err)
	}
	defer releaseRuntime()
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
	vectorIndex, err := resolveCollectionVectorIndex(field, op, c.path)
	if err != nil {
		return nil, err
	}
	params, err := collectionQueryParams(query.Params, vectorIndex)
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
	runtimeConfig := c.runtimeConfig()
	candidateFilter, err := evaluateFilterDocuments(ctx, filterPlan, documents, runtimeConfig.InvertToForwardScanRatio)
	if err != nil {
		return nil, wrapFilterEvaluationError(op, c.path, err)
	}
	results, err := c.searchVectorSnapshotResolved(
		ctx, op, field, query.DenseVector, query.SparseVector, query.TopK,
		documents, candidateFilter, vectorIndex, params,
	)
	if err != nil {
		return nil, err
	}
	return c.materializeResults(documents, results, query.Projection)
}

// searchVectorSnapshot executes one vector target against documents while the
// caller holds the collection read lock. MultiQuery reuses it so every branch
// observes the same live document and scalar-filter snapshot as Query.
func (c *Collection) searchVectorSnapshot(
	ctx context.Context,
	op string,
	field FieldSchema,
	dense DenseVector,
	sparse SparseVector,
	topK int,
	queryParams QueryParams,
	documents []Document,
	candidateFilter evaluatedFilter,
) ([]core.Result, error) {
	vectorIndex, err := resolveCollectionVectorIndex(field, op, c.path)
	if err != nil {
		return nil, err
	}
	params, err := collectionQueryParams(queryParams, vectorIndex)
	if err != nil {
		return nil, err
	}
	return c.searchVectorSnapshotResolved(
		ctx, op, field, dense, sparse, topK, documents, candidateFilter, vectorIndex, params,
	)
}

func (c *Collection) searchVectorSnapshotResolved(
	ctx context.Context,
	op string,
	field FieldSchema,
	dense DenseVector,
	sparse SparseVector,
	topK int,
	documents []Document,
	candidateFilter evaluatedFilter,
	vectorIndex collectionVectorIndex,
	params collectionQueryConfig,
) ([]core.Result, error) {
	if candidateFilter.useBruteForce(c.runtimeConfig().BruteForceByKeysRatio) {
		params.options.Linear = true
	}
	if field.DataType.IsDenseVector() {
		if sparse != nil {
			return nil, invalidArgument(op, "dense field %q cannot use a sparse query vector", field.Name)
		}
		queryVector, err := validateDenseQueryVector(field, dense)
		if err != nil {
			return nil, err
		}
		results, err := searchCollectionDense(
			ctx, c.schema.Name, field, documents, queryVector, topK, candidateFilter.predicate, vectorIndex, params,
			c.queryWorkers(), c.options.MaxBufferSize,
		)
		if err != nil {
			return nil, wrapCollectionError(op, c.path, err)
		}
		return results, nil
	}
	if dense != nil {
		return nil, invalidArgument(op, "sparse field %q cannot use a dense query vector", field.Name)
	}
	queryVector, err := validateSparseQueryVector(field, sparse)
	if err != nil {
		return nil, err
	}
	originalQuery := queryVector
	if vectorIndex.quantize == QuantizeTypeFP16 {
		queryVector, err = sparseFP16Vector(queryVector)
		if err != nil {
			return nil, wrapCollectionError(op, c.path, err)
		}
	}
	index, err := buildCollectionSparseIndex(ctx, field, documents, vectorIndex, params.options.Linear)
	if err != nil {
		return nil, wrapCollectionError(op, c.path, err)
	}
	searchOptions := core.SearchOptions{TopK: topK, Radius: params.options.Radius, Filter: candidateFilter.predicate}
	baseOptions := searchOptions
	if params.options.UseRefiner {
		candidateCount, countErr := core.RefinementCandidateCount(topK, params.scaleFactor)
		if countErr != nil {
			return nil, wrapCollectionError(op, c.path, countErr)
		}
		baseOptions = core.SearchOptions{TopK: candidateCount, Filter: candidateFilter.predicate}
	}
	var results []core.Result
	if vectorIndex.indexType == IndexTypeHNSW && !params.options.Linear {
		hnsw, ok := index.(*core.SparseHNSWIndex)
		if !ok {
			return nil, &Error{Code: ErrorCodeInternal, Op: op, Path: c.path, Message: "sparse HNSW builder returned an incompatible index"}
		}
		results, err = hnsw.SearchSparseHNSW(ctx, queryVector, core.HNSWSearchOptions{
			SearchOptions: baseOptions, EF: params.ef,
			PrefetchOffset: params.prefetchOffset, PrefetchLines: params.prefetchLines,
		})
	} else {
		results, err = index.SearchSparseWithOptions(ctx, queryVector, baseOptions)
	}
	if err != nil {
		return nil, wrapCollectionError(op, c.path, err)
	}
	if params.options.UseRefiner {
		originals, err := buildSparseFlatIndex(ctx, field, documents)
		if err != nil {
			return nil, wrapCollectionError(op, c.path, err)
		}
		refiner, err := core.NewOriginalSparseVectorRefiner(originals)
		if err != nil {
			return nil, wrapCollectionError(op, c.path, err)
		}
		results, err = refiner.RefineSparse(ctx, originalQuery, results, searchOptions)
		if err != nil {
			return nil, wrapCollectionError(op, c.path, err)
		}
	}
	return results, nil
}

// GroupByQuery executes complete Flat/Linear grouping or native HNSW grouping.
// IVF, Vamana, and DiskANN group traversal remain unsupported to match the
// pinned native baseline.
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
	releaseRuntime, err := c.beginRuntimeTask(ctx, runtimeQueryTask, op, 8)
	if err != nil {
		return nil, wrapCollectionError(op, c.Path(), err)
	}
	defer releaseRuntime()
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
	vectorIndex, err := resolveCollectionVectorIndex(field, op, c.path)
	if err != nil {
		return nil, err
	}
	params, err := collectionQueryParams(query.Params, vectorIndex)
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
	runtimeConfig := c.runtimeConfig()
	candidateFilter, err := evaluateFilterDocuments(ctx, filterPlan, documents, runtimeConfig.InvertToForwardScanRatio)
	if err != nil {
		return nil, wrapFilterEvaluationError(op, c.path, err)
	}
	if candidateFilter.useBruteForce(runtimeConfig.BruteForceByKeysRatio) {
		params.options.Linear = true
	}
	if !params.options.Linear && vectorIndex.indexType != IndexTypeFlat {
		if params.options.UseRefiner {
			return nil, notSupported(op, c.path, "HNSW group-by with a refiner requires Linear")
		}
		if vectorIndex.indexType != IndexTypeHNSW && vectorIndex.indexType != IndexTypeHNSWRaBitQ {
			return nil, notSupported(op, c.path, fmt.Sprintf("group-by is not supported for %s graph traversal", vectorIndex.indexType))
		}
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
		Radius: params.options.Radius, Filter: candidateFilter.predicate, Resolve: resolve,
	}
	metric := vectorIndex.metric
	var groups []core.GroupResult
	if field.DataType.IsDenseVector() {
		if query.SparseVector != nil {
			return nil, invalidArgument(op, "dense field %q cannot use a sparse query vector", field.Name)
		}
		queryVector, err := validateDenseQueryVector(field, query.DenseVector)
		if err != nil {
			return nil, err
		}
		var searcher core.DenseGroupSearcher
		if params.options.UseRefiner {
			searcher, err = buildDenseFlatIndex(ctx, field, metric, documents)
		} else if vectorIndex.indexType == IndexTypeHNSWRaBitQ {
			var index *core.HNSWRaBitQIndex
			index, err = buildCollectionDenseHNSWRaBitQ(ctx, field, documents, vectorIndex, c.queryWorkers())
			if err == nil {
				if params.options.Linear {
					groups, err = index.SearchGroups(ctx, queryVector, options)
				} else {
					groups, err = index.SearchHNSWRaBitQGroups(ctx, queryVector, core.HNSWGroupSearchOptions{
						GroupByOptions: options, EF: params.ef,
					})
				}
			}
		} else if vectorIndex.indexType == IndexTypeHNSW && !params.options.Linear {
			var index collectionHNSWIndex
			index, err = buildCollectionDenseHNSW(ctx, c.schema.Name, field, documents, vectorIndex)
			if err == nil {
				groupIndex, compatible := index.(collectionHNSWGroupIndex)
				if !compatible {
					err = fmt.Errorf("dense HNSW builder returned an incompatible group searcher")
				} else {
					groups, err = groupIndex.SearchHNSWGroups(ctx, queryVector, core.HNSWGroupSearchOptions{
						GroupByOptions: options, EF: params.ef,
						PrefetchOffset: params.prefetchOffset, PrefetchLines: params.prefetchLines,
					})
				}
			}
		} else {
			var index collectionDenseIndex
			index, err = buildCollectionDenseFlat(ctx, c.schema.Name, field, documents, vectorIndex)
			if err == nil {
				var compatible bool
				searcher, compatible = index.(core.DenseGroupSearcher)
				if !compatible {
					err = fmt.Errorf("dense Flat builder returned an incompatible group searcher")
				}
			}
		}
		if err != nil {
			return nil, wrapCollectionError(op, c.path, err)
		}
		if searcher != nil {
			groups, err = searcher.SearchGroups(ctx, queryVector, options)
			if err != nil {
				return nil, wrapCollectionError(op, c.path, err)
			}
		}
	} else {
		if query.DenseVector != nil {
			return nil, invalidArgument(op, "sparse field %q cannot use a dense query vector", field.Name)
		}
		queryVector, err := validateSparseQueryVector(field, query.SparseVector)
		if err != nil {
			return nil, err
		}
		var searcher core.SparseGroupSearcher
		if params.options.UseRefiner {
			searcher, err = buildSparseFlatIndex(ctx, field, documents)
		} else {
			if vectorIndex.quantize == QuantizeTypeFP16 {
				queryVector, err = sparseFP16Vector(queryVector)
			}
			if err == nil {
				var index core.SparseQuerySearcher
				index, err = buildCollectionSparseIndex(ctx, field, documents, vectorIndex, params.options.Linear)
				if err == nil {
					if !params.options.Linear {
						hnsw, compatible := index.(*core.SparseHNSWIndex)
						if !compatible {
							err = fmt.Errorf("sparse HNSW builder returned an incompatible group searcher")
						} else {
							groups, err = hnsw.SearchSparseHNSWGroups(ctx, queryVector, core.HNSWGroupSearchOptions{
								GroupByOptions: options, EF: params.ef,
								PrefetchOffset: params.prefetchOffset, PrefetchLines: params.prefetchLines,
							})
						}
					} else {
						var compatible bool
						searcher, compatible = index.(core.SparseGroupSearcher)
						if !compatible {
							err = fmt.Errorf("sparse Flat builder returned an incompatible group searcher")
						}
					}
				}
			}
		}
		if err != nil {
			return nil, wrapCollectionError(op, c.path, err)
		}
		if searcher != nil {
			groups, err = searcher.SearchSparseGroups(ctx, queryVector, options)
			if err != nil {
				return nil, wrapCollectionError(op, c.path, err)
			}
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

func collectionVectorFieldSupported(field FieldSchema) bool {
	if !field.DataType.IsVector() {
		return false
	}
	_, err := resolveCollectionVectorIndex(field, "stats", "")
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

// WriteResult reports one document mutation in input order.
type WriteResult struct {
	PrimaryKey string
	DocID      uint64
	Err        error
}

// Insert durably writes new primary keys. Valid documents in a mixed batch
// are committed even when other entries fail validation or already exist.
func (c *Collection) Insert(ctx context.Context, documents []Document) ([]WriteResult, error) {
	return c.writeDocuments(ctx, OperatorInsert, documents)
}

// Upsert inserts new documents and partially updates existing documents.
func (c *Collection) Upsert(ctx context.Context, documents []Document) ([]WriteResult, error) {
	return c.writeDocuments(ctx, OperatorUpsert, documents)
}

// Update partially replaces fields on existing documents while retaining all
// unspecified fields from the current version.
func (c *Collection) Update(ctx context.Context, documents []Document) ([]WriteResult, error) {
	return c.writeDocuments(ctx, OperatorUpdate, documents)
}

func (c *Collection) writeDocuments(ctx context.Context, operator Operator, documents []Document) ([]WriteResult, error) {
	op := operator.String()
	if c == nil {
		return nil, invalidArgument(op, "collection is nil")
	}
	if ctx == nil {
		return nil, invalidArgument(op, "context is nil")
	}
	if len(documents) == 0 {
		return nil, invalidArgument(op, "document batch is empty")
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.requireOpenLocked(op); err != nil {
		return nil, err
	}
	results := make([]WriteResult, len(documents))
	batchError := &BatchWriteError{}
	for index, document := range documents {
		results[index].PrimaryKey = document.PrimaryKey
		if err := ctx.Err(); err != nil {
			for remaining := index; remaining < len(documents); remaining++ {
				wrapped := wrapCollectionError(op, c.path, err)
				results[remaining] = WriteResult{PrimaryKey: documents[remaining].PrimaryKey, Err: wrapped}
				batchError.add(wrapped)
			}
			break
		}
		prepared, err := c.prepareWriteDocumentLocked(ctx, operator, document)
		if err != nil {
			wrapped := wrapCollectionError(op, c.path, err)
			results[index].Err = wrapped
			batchError.add(wrapped)
			continue
		}
		payload, err := marshalDocumentPayload(prepared.Fields)
		if err != nil {
			wrapped := wrapCollectionError(op, c.path, err)
			results[index].Err = wrapped
			batchError.add(wrapped)
			continue
		}
		if len(payload) > db.MaxDocumentPayloadSize {
			wrapped := &Error{
				Code: ErrorCodeResourceExhausted, Op: op, Path: c.path,
				Message: fmt.Sprintf("document payload is %d bytes, maximum %d", len(payload), db.MaxDocumentPayloadSize),
			}
			results[index].Err = wrapped
			batchError.add(wrapped)
			continue
		}
		dbResults, batchErr := c.callStoreWriteLocked(ctx, operator, db.WriteInput{
			PrimaryKey: prepared.PrimaryKey, Payload: payload,
		})
		var itemErr error
		if len(dbResults) == 1 {
			results[index].DocID = dbResults[0].DocID
			itemErr = dbResults[0].Err
		}
		if itemErr == nil {
			itemErr = batchErr
		}
		if itemErr != nil {
			wrapped := wrapCollectionError(op, c.path, itemErr)
			results[index].Err = wrapped
			batchError.add(wrapped)
		}
	}
	if batchError.Failed != 0 {
		return results, batchError
	}
	return results, nil
}

func (c *Collection) prepareWriteDocumentLocked(ctx context.Context, operator Operator, document Document) (Document, error) {
	clone, err := document.Clone()
	if err != nil {
		return Document{}, err
	}
	switch operator {
	case OperatorInsert:
		if err := clone.Validate(c.schema); err != nil {
			return Document{}, err
		}
		if err := validateCollectionVectorRepresentations(ctx, c.schema, clone); err != nil {
			return Document{}, err
		}
		return clone, nil
	case OperatorUpsert, OperatorUpdate:
		if err := validateDocumentAgainstSchema(clone, c.schema, true); err != nil {
			return Document{}, err
		}
		current, found, err := c.fetchOneLocked(ctx, clone.PrimaryKey)
		if err != nil {
			return Document{}, err
		}
		if !found {
			if operator == OperatorUpdate {
				return Document{}, fmt.Errorf("%w: %q", db.ErrPrimaryKeyNotFound, clone.PrimaryKey)
			}
			if err := clone.Validate(c.schema); err != nil {
				return Document{}, err
			}
			if err := validateCollectionVectorRepresentations(ctx, c.schema, clone); err != nil {
				return Document{}, err
			}
			return clone, nil
		}
		for name, value := range clone.Fields {
			current.Fields[name] = value
		}
		current.Score = 0
		current.DocID = 0
		if err := current.Validate(c.schema); err != nil {
			return Document{}, err
		}
		if err := validateCollectionVectorRepresentations(ctx, c.schema, current); err != nil {
			return Document{}, err
		}
		return current, nil
	default:
		return Document{}, invalidArgument("write documents", "unsupported operator %d", operator)
	}
}

func (c *Collection) callStoreWriteLocked(ctx context.Context, operator Operator, input db.WriteInput) ([]db.WriteResult, error) {
	switch operator {
	case OperatorInsert:
		return c.store.Insert(ctx, []db.WriteInput{input})
	case OperatorUpsert:
		return c.store.Upsert(ctx, []db.WriteInput{input})
	case OperatorUpdate:
		return c.store.Update(ctx, []db.WriteInput{input})
	default:
		return nil, errors.New("zvec: unsupported write operator")
	}
}

// Delete durably removes the current versions of the requested primary keys.
func (c *Collection) Delete(ctx context.Context, primaryKeys []string) ([]WriteResult, error) {
	const op = "DELETE"
	if c == nil {
		return nil, invalidArgument(op, "collection is nil")
	}
	if ctx == nil {
		return nil, invalidArgument(op, "context is nil")
	}
	if len(primaryKeys) == 0 {
		return nil, invalidArgument(op, "primary-key batch is empty")
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.requireOpenLocked(op); err != nil {
		return nil, err
	}
	results := make([]WriteResult, len(primaryKeys))
	batchError := &BatchWriteError{}
	for index, primaryKey := range primaryKeys {
		results[index].PrimaryKey = primaryKey
		if err := ctx.Err(); err != nil {
			for remaining := index; remaining < len(primaryKeys); remaining++ {
				wrapped := wrapCollectionError(op, c.path, err)
				results[remaining] = WriteResult{PrimaryKey: primaryKeys[remaining], Err: wrapped}
				batchError.add(wrapped)
			}
			break
		}
		if _, err := (Document{PrimaryKey: primaryKey}).Clone(); err != nil {
			wrapped := wrapCollectionError(op, c.path, err)
			results[index].Err = wrapped
			batchError.add(wrapped)
			continue
		}
		dbResults, batchErr := c.store.Delete(ctx, []string{primaryKey})
		var itemErr error
		if len(dbResults) == 1 {
			results[index].DocID = dbResults[0].DocID
			itemErr = dbResults[0].Err
		}
		if itemErr == nil {
			itemErr = batchErr
		}
		if itemErr != nil {
			wrapped := wrapCollectionError(op, c.path, itemErr)
			results[index].Err = wrapped
			batchError.add(wrapped)
		}
	}
	if batchError.Failed != 0 {
		return results, batchError
	}
	return results, nil
}

// DeleteByFilter durably removes every live document for which filter
// evaluates to SQL TRUE. Selection and WAL-backed deletion are serialized
// under the collection write lock, so a matched version cannot be replaced
// between those two phases.
func (c *Collection) DeleteByFilter(ctx context.Context, filter string) error {
	const op = "delete by filter"
	if c == nil {
		return invalidArgument(op, "collection is nil")
	}
	if ctx == nil {
		return invalidArgument(op, "context is nil")
	}
	if filter == "" {
		return invalidArgument(op, "filter is empty")
	}
	if err := ctx.Err(); err != nil {
		return wrapCollectionError(op, c.Path(), err)
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.requireOpenLocked(op); err != nil {
		return err
	}
	if c.options.ReadOnly {
		return &Error{Code: ErrorCodePermissionDenied, Op: op, Path: c.path, Message: "collection is read-only"}
	}
	plan, err := buildFilterPlan(filter, c.schema)
	if err != nil {
		return invalidArgument(op, "invalid filter: %v", err)
	}
	documents, err := c.liveDocumentsLocked(ctx)
	if err != nil {
		return wrapCollectionError(op, c.path, err)
	}
	matched, err := evaluateFilterDocuments(ctx, plan, documents, c.runtimeConfig().InvertToForwardScanRatio)
	if err != nil {
		return wrapFilterEvaluationError(op, c.path, err)
	}
	primaryKeys := make([]string, 0, len(documents))
	for _, document := range documents {
		if err := ctx.Err(); err != nil {
			return wrapCollectionError(op, c.path, err)
		}
		if matched.predicate(document.DocID) {
			primaryKeys = append(primaryKeys, document.PrimaryKey)
		}
	}
	if len(primaryKeys) == 0 {
		return nil
	}
	results, err := c.store.Delete(ctx, primaryKeys)
	if err != nil {
		return wrapCollectionError(op, c.path, err)
	}
	if len(results) != len(primaryKeys) {
		return &Error{Code: ErrorCodeInternal, Op: op, Path: c.path, Message: "storage returned an incomplete delete result"}
	}
	for _, result := range results {
		if result.Err != nil {
			return wrapCollectionError(op, c.path, result.Err)
		}
	}
	return nil
}

// Fetch returns one independently owned document pointer per requested key.
// Missing or deleted keys have a nil entry and are not errors.
func (c *Collection) Fetch(ctx context.Context, primaryKeys []string, projection Projection) ([]*Document, error) {
	const op = "fetch"
	if c == nil {
		return nil, invalidArgument(op, "collection is nil")
	}
	if ctx == nil {
		return nil, invalidArgument(op, "context is nil")
	}
	if len(primaryKeys) == 0 {
		return nil, invalidArgument(op, "primary-key batch is empty")
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	if err := c.requireOpenLocked(op); err != nil {
		return nil, err
	}
	if err := projection.Validate(c.schema); err != nil {
		return nil, err
	}
	for _, primaryKey := range primaryKeys {
		if _, err := (Document{PrimaryKey: primaryKey}).Clone(); err != nil {
			return nil, err
		}
	}
	fetched, err := c.store.Fetch(ctx, primaryKeys)
	results := make([]*Document, len(fetched))
	for index, item := range fetched {
		if item.Err != nil || item.Document == nil {
			continue
		}
		document, decodeErr := decodeStoredDocument(*item.Document)
		if decodeErr != nil {
			return results, wrapCollectionError(op, c.path, decodeErr)
		}
		if validateErr := document.Validate(c.schema); validateErr != nil {
			return results, &Error{Code: ErrorCodeInternal, Op: op, Path: c.path, Message: "stored document violates collection schema", Err: validateErr}
		}
		projected, projectErr := ProjectDocument(document, c.schema, projection)
		if projectErr != nil {
			return results, projectErr
		}
		results[index] = &projected
	}
	if err != nil {
		return results, wrapCollectionError(op, c.path, err)
	}
	return results, nil
}

func (c *Collection) fetchOneLocked(ctx context.Context, primaryKey string) (Document, bool, error) {
	results, err := c.store.Fetch(ctx, []string{primaryKey})
	if err != nil {
		return Document{}, false, err
	}
	if len(results) != 1 || results[0].Document == nil {
		return Document{}, false, nil
	}
	document, err := decodeStoredDocument(*results[0].Document)
	return document, err == nil, err
}

func decodeStoredDocument(stored db.StoredDocument) (Document, error) {
	fields, err := unmarshalDocumentPayload(stored.Payload)
	if err != nil {
		return Document{}, fmt.Errorf("decode document %d: %w", stored.DocID, err)
	}
	return Document{
		PrimaryKey: stored.PrimaryKey, Fields: fields, DocID: stored.DocID,
	}, nil
}
