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
	"log/slog"
	"math"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorse-io/zvec/internal/core"
)

const (
	// MinRuntimeMemoryLimit is the pinned minimum explicit process budget.
	MinRuntimeMemoryLimit uint64 = 100 << 20
	// MaxRuntimeConcurrency bounds configured query and optimize admission.
	MaxRuntimeConcurrency = 65_536
)

// LogLevel preserves the pinned public logging severity order.
type LogLevel uint32

const (
	LogLevelDebug LogLevel = iota
	LogLevelInfo
	LogLevelWarn
	LogLevelError
	LogLevelFatal
)

var logLevelNames = map[LogLevel]string{
	LogLevelDebug: "DEBUG",
	LogLevelInfo:  "INFO",
	LogLevelWarn:  "WARN",
	LogLevelError: "ERROR",
	LogLevelFatal: "FATAL",
}

func (l LogLevel) String() string { return enumName(logLevelNames, l, "LogLevel") }

// Valid reports whether l is a public logging level.
func (l LogLevel) Valid() bool { return enumValid(logLevelNames, l) }

// RuntimeConfig controls process-wide query and maintenance resources. Call
// ConfigureRuntime before creating or opening the first collection. A zero
// MemoryLimitBytes leaves heap sizing to the Go runtime; a non-zero value is a
// conservative admission budget for collection query and maintenance scratch.
type RuntimeConfig struct {
	MemoryLimitBytes uint64
	Logger           *slog.Logger
	LogLevel         LogLevel

	QueryConcurrency   int
	QueryThreadBinding bool

	InvertToForwardScanRatio float32
	BruteForceByKeysRatio    float32
	FTSBruteForceByKeysRatio float32

	OptimizeConcurrency   int
	OptimizeThreadBinding bool
	JiebaDictionaryDir    string
}

// NewRuntimeConfig returns native Go defaults aligned with the pinned planner
// ratios and current GOMAXPROCS. Explicit memory admission is disabled.
func NewRuntimeConfig() RuntimeConfig {
	workers := min(runtime.GOMAXPROCS(0), MaxRuntimeConcurrency)
	return RuntimeConfig{
		Logger:                   slog.Default(),
		LogLevel:                 LogLevelWarn,
		QueryConcurrency:         workers,
		InvertToForwardScanRatio: 0.9,
		BruteForceByKeysRatio:    0.1,
		FTSBruteForceByKeysRatio: 0.05,
		OptimizeConcurrency:      workers,
	}
}

// Validate checks process resource limits and planner ratios. CPU thread
// binding is explicit NotSupported because Go schedules goroutines onto its
// own cross-platform worker threads.
func (c RuntimeConfig) Validate() error {
	const op = "validate runtime config"
	if c.MemoryLimitBytes != 0 && c.MemoryLimitBytes < MinRuntimeMemoryLimit {
		return invalidArgument(op, "MemoryLimitBytes must be zero or at least %d", MinRuntimeMemoryLimit)
	}
	if c.QueryConcurrency <= 0 || c.QueryConcurrency > MaxRuntimeConcurrency {
		return invalidArgument(op, "QueryConcurrency must be in [1, %d]", MaxRuntimeConcurrency)
	}
	if c.OptimizeConcurrency <= 0 || c.OptimizeConcurrency > MaxRuntimeConcurrency {
		return invalidArgument(op, "OptimizeConcurrency must be in [1, %d]", MaxRuntimeConcurrency)
	}
	if c.QueryThreadBinding || c.OptimizeThreadBinding {
		return notSupported(op, "", "CPU thread binding is not supported by the Go runtime scheduler")
	}
	if !c.LogLevel.Valid() {
		return invalidArgument(op, "LogLevel is invalid")
	}
	for _, ratio := range []struct {
		name  string
		value float32
	}{
		{name: "InvertToForwardScanRatio", value: c.InvertToForwardScanRatio},
		{name: "BruteForceByKeysRatio", value: c.BruteForceByKeysRatio},
		{name: "FTSBruteForceByKeysRatio", value: c.FTSBruteForceByKeysRatio},
	} {
		if value := float64(ratio.value); math.IsNaN(value) || math.IsInf(value, 0) || ratio.value < 0 || ratio.value > 1 {
			return invalidArgument(op, "%s must be finite and in [0, 1]", ratio.name)
		}
	}
	return nil
}

var globalRuntimeRegistry struct {
	sync.Mutex
	resources *runtimeResources
}

// ConfigureRuntime installs the process runtime configuration once. Like the
// pinned GlobalConfig, later calls are successful no-ops. Configuration must
// therefore happen before the first collection is created or opened.
func ConfigureRuntime(config RuntimeConfig) error {
	if err := config.Validate(); err != nil {
		return err
	}
	globalRuntimeRegistry.Lock()
	defer globalRuntimeRegistry.Unlock()
	if globalRuntimeRegistry.resources != nil {
		return nil
	}
	globalRuntimeRegistry.resources = newRuntimeResources(config)
	if config.JiebaDictionaryDir != "" {
		core.SetDefaultJiebaDictDir(config.JiebaDictionaryDir)
	}
	return nil
}

// CurrentRuntimeConfig returns the configured value or defaults without
// freezing the one-shot configuration lifecycle.
func CurrentRuntimeConfig() RuntimeConfig {
	globalRuntimeRegistry.Lock()
	defer globalRuntimeRegistry.Unlock()
	if globalRuntimeRegistry.resources == nil {
		return NewRuntimeConfig()
	}
	return globalRuntimeRegistry.resources.config
}

// SetDefaultJiebaDictDir sets the process-wide lowest-priority Jieba resource
// directory. Per-field configuration and ZVEC_JIEBA_DICT_DIR take precedence.
func SetDefaultJiebaDictDir(path string) { core.SetDefaultJiebaDictDir(path) }

// DefaultJiebaDictDir returns the current process-wide Jieba fallback.
func DefaultJiebaDictDir() string { return core.DefaultJiebaDictDir() }

// RuntimeStats is a concurrency-safe point-in-time view of process resource
// usage.
type RuntimeStats struct {
	MemoryLimitBytes uint64
	MemoryInUseBytes uint64
	PeakMemoryBytes  uint64
	MemoryWaiters    uint64

	ActiveQueries    uint64
	PeakQueries      uint64
	QueuedQueries    uint64
	CompletedQueries uint64

	ActiveOptimizeTasks    uint64
	PeakOptimizeTasks      uint64
	QueuedOptimizeTasks    uint64
	CompletedOptimizeTasks uint64
}

// CurrentRuntimeStats returns process admission and scratch-budget counters.
// Calling it initializes defaults when ConfigureRuntime has not run yet.
func CurrentRuntimeStats() RuntimeStats { return currentRuntimeResources().stats() }

type runtimeTaskKind uint8

const (
	runtimeQueryTask runtimeTaskKind = iota + 1
	runtimeOptimizeTask
)

type runtimeResources struct {
	config   RuntimeConfig
	queries  *taskLimiter
	optimize *taskLimiter
	memory   *memoryBudget
}

func newRuntimeResources(config RuntimeConfig) *runtimeResources {
	return &runtimeResources{
		config: config, queries: newTaskLimiter(config.QueryConcurrency),
		optimize: newTaskLimiter(config.OptimizeConcurrency),
		memory:   newMemoryBudget(config.MemoryLimitBytes),
	}
}

func currentRuntimeResources() *runtimeResources {
	globalRuntimeRegistry.Lock()
	defer globalRuntimeRegistry.Unlock()
	if globalRuntimeRegistry.resources == nil {
		globalRuntimeRegistry.resources = newRuntimeResources(NewRuntimeConfig())
	}
	return globalRuntimeRegistry.resources
}

func (r *runtimeResources) stats() RuntimeStats {
	if r == nil {
		return RuntimeStats{}
	}
	memoryLimit, memoryUsed, memoryPeak, memoryWaiters := r.memory.stats()
	queryActive, queryPeak, queryQueued, queryCompleted := r.queries.stats()
	optimizeActive, optimizePeak, optimizeQueued, optimizeCompleted := r.optimize.stats()
	return RuntimeStats{
		MemoryLimitBytes: memoryLimit, MemoryInUseBytes: memoryUsed,
		PeakMemoryBytes: memoryPeak, MemoryWaiters: memoryWaiters,
		ActiveQueries: queryActive, PeakQueries: queryPeak,
		QueuedQueries: queryQueued, CompletedQueries: queryCompleted,
		ActiveOptimizeTasks: optimizeActive, PeakOptimizeTasks: optimizePeak,
		QueuedOptimizeTasks: optimizeQueued, CompletedOptimizeTasks: optimizeCompleted,
	}
}

func (r *runtimeResources) begin(ctx context.Context, kind runtimeTaskKind, op, path string, memoryBytes uint64) (func(), error) {
	if r == nil {
		r = currentRuntimeResources()
	}
	limiter := r.queries
	level := LogLevelDebug
	if kind == runtimeOptimizeTask {
		limiter = r.optimize
		level = LogLevelInfo
	}
	releaseTask, err := limiter.acquire(ctx)
	if err != nil {
		return nil, err
	}
	releaseMemory, err := r.memory.acquire(ctx, memoryBytes)
	if err != nil {
		releaseTask()
		if errors.Is(err, errRuntimeMemoryLimit) {
			r.log(ctx, LogLevelWarn, "operation rejected by memory budget", "op", op, "path", path, "bytes", memoryBytes)
			return nil, &Error{
				Code: ErrorCodeResourceExhausted, Op: op, Path: path,
				Message: fmt.Sprintf("estimated scratch memory %d exceeds runtime budget %d", memoryBytes, r.config.MemoryLimitBytes),
				Err:     err,
			}
		}
		return nil, err
	}
	started := time.Now()
	r.log(ctx, level, "operation started", "op", op, "path", path, "estimated_memory_bytes", memoryBytes)
	var once sync.Once
	return func() {
		once.Do(func() {
			releaseMemory()
			releaseTask()
			r.log(ctx, level, "operation completed", "op", op, "path", path, "duration", time.Since(started))
		})
	}, nil
}

func (r *runtimeResources) log(ctx context.Context, level LogLevel, message string, args ...any) {
	if r == nil || r.config.Logger == nil || level < r.config.LogLevel {
		return
	}
	slogLevel := slog.LevelDebug
	switch level {
	case LogLevelInfo:
		slogLevel = slog.LevelInfo
	case LogLevelWarn:
		slogLevel = slog.LevelWarn
	case LogLevelError, LogLevelFatal:
		slogLevel = slog.LevelError
	}
	r.config.Logger.Log(ctx, slogLevel, message, args...)
}

type taskLimiter struct {
	tokens    chan struct{}
	active    atomic.Uint64
	peak      atomic.Uint64
	queued    atomic.Uint64
	completed atomic.Uint64
}

func newTaskLimiter(limit int) *taskLimiter {
	return &taskLimiter{tokens: make(chan struct{}, limit)}
}

func (l *taskLimiter) acquire(ctx context.Context) (func(), error) {
	if ctx == nil {
		return nil, context.Canceled
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	select {
	case l.tokens <- struct{}{}:
	default:
		l.queued.Add(1)
		defer l.queued.Add(^uint64(0))
		select {
		case l.tokens <- struct{}{}:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	active := l.active.Add(1)
	updateAtomicPeak(&l.peak, active)
	var once sync.Once
	return func() {
		once.Do(func() {
			l.active.Add(^uint64(0))
			l.completed.Add(1)
			<-l.tokens
		})
	}, nil
}

func (l *taskLimiter) stats() (active, peak, queued, completed uint64) {
	if l == nil {
		return 0, 0, 0, 0
	}
	return l.active.Load(), l.peak.Load(), l.queued.Load(), l.completed.Load()
}

var errRuntimeMemoryLimit = errors.New("zvec: runtime memory limit")

type memoryBudget struct {
	mu      sync.Mutex
	limit   uint64
	used    uint64
	peak    uint64
	waiters uint64
	changed chan struct{}
}

func newMemoryBudget(limit uint64) *memoryBudget {
	return &memoryBudget{limit: limit, changed: make(chan struct{})}
}

func (b *memoryBudget) acquire(ctx context.Context, amount uint64) (func(), error) {
	if ctx == nil {
		return nil, context.Canceled
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	b.mu.Lock()
	if b.limit != 0 && amount > b.limit {
		b.mu.Unlock()
		return nil, errRuntimeMemoryLimit
	}
	for b.limit != 0 && amount > b.limit-b.used {
		changed := b.changed
		b.waiters++
		b.mu.Unlock()
		select {
		case <-ctx.Done():
			b.mu.Lock()
			b.waiters--
			b.mu.Unlock()
			return nil, ctx.Err()
		case <-changed:
			b.mu.Lock()
			b.waiters--
		}
	}
	if amount > math.MaxUint64-b.used {
		b.mu.Unlock()
		return nil, errRuntimeMemoryLimit
	}
	b.used += amount
	b.peak = max(b.peak, b.used)
	b.mu.Unlock()
	var once sync.Once
	return func() {
		once.Do(func() {
			b.mu.Lock()
			b.used -= amount
			close(b.changed)
			b.changed = make(chan struct{})
			b.mu.Unlock()
		})
	}, nil
}

func (b *memoryBudget) stats() (limit, used, peak, waiters uint64) {
	if b == nil {
		return 0, 0, 0, 0
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.limit, b.used, b.peak, b.waiters
}

func updateAtomicPeak(peak *atomic.Uint64, value uint64) {
	for current := peak.Load(); value > current; current = peak.Load() {
		if peak.CompareAndSwap(current, value) {
			return
		}
	}
}

func saturatingMultiply(value, factor uint64) uint64 {
	if value == 0 || factor == 0 {
		return 0
	}
	if value > math.MaxUint64/factor {
		return math.MaxUint64
	}
	return value * factor
}

func (c *Collection) beginRuntimeTask(ctx context.Context, kind runtimeTaskKind, op string, memoryFactor uint64) (func(), error) {
	if c == nil {
		return nil, invalidArgument(op, "collection is nil")
	}
	c.mu.RLock()
	resources, store, path := c.runtime, c.store, c.path
	c.mu.RUnlock()
	var memoryBytes uint64
	if store != nil {
		memoryBytes = saturatingMultiply(store.Stats().MemoryUsageBytes, memoryFactor)
	}
	if resources == nil {
		resources = currentRuntimeResources()
	}
	return resources.begin(ctx, kind, op, path, memoryBytes)
}

func (c *Collection) queryWorkers() int {
	if c != nil && c.runtime != nil {
		return c.runtime.config.QueryConcurrency
	}
	return currentRuntimeResources().config.QueryConcurrency
}

func (c *Collection) optimizeWorkers(requested int) int {
	limit := currentRuntimeResources().config.OptimizeConcurrency
	if c != nil && c.runtime != nil {
		limit = c.runtime.config.OptimizeConcurrency
	}
	if requested <= 0 || requested > limit {
		return limit
	}
	return requested
}

func (c *Collection) runtimeConfig() RuntimeConfig {
	if c != nil && c.runtime != nil {
		return c.runtime.config
	}
	return CurrentRuntimeConfig()
}
