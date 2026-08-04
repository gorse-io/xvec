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
	"encoding/json"
	"errors"
	"log/slog"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"sync"
	"testing"
	"time"
)

func TestRuntimeConfigDefaultsAndValidation(t *testing.T) {
	config := NewRuntimeConfig()
	if err := config.Validate(); err != nil {
		t.Fatal(err)
	}
	defaultWorkers := min(runtime.GOMAXPROCS(0), MaxRuntimeConcurrency)
	if config.QueryConcurrency != defaultWorkers || config.OptimizeConcurrency != defaultWorkers ||
		config.LogLevel != LogLevelWarn || config.MemoryLimitBytes != 0 ||
		config.InvertToForwardScanRatio != 0.9 || config.BruteForceByKeysRatio != 0.1 ||
		config.FTSBruteForceByKeysRatio != 0.05 {
		t.Fatalf("runtime defaults = %#v", config)
	}
	for level, name := range map[LogLevel]string{
		LogLevelDebug: "DEBUG", LogLevelInfo: "INFO", LogLevelWarn: "WARN",
		LogLevelError: "ERROR", LogLevelFatal: "FATAL",
	} {
		if !level.Valid() || level.String() != name {
			t.Fatalf("log level %d = %q, valid %v", level, level, level.Valid())
		}
	}

	tests := []struct {
		name   string
		mutate func(*RuntimeConfig)
		want   error
	}{
		{name: "small memory", mutate: func(c *RuntimeConfig) { c.MemoryLimitBytes = MinRuntimeMemoryLimit - 1 }, want: ErrInvalidArgument},
		{name: "zero query concurrency", mutate: func(c *RuntimeConfig) { c.QueryConcurrency = 0 }, want: ErrInvalidArgument},
		{name: "large query concurrency", mutate: func(c *RuntimeConfig) { c.QueryConcurrency = MaxRuntimeConcurrency + 1 }, want: ErrInvalidArgument},
		{name: "zero optimize concurrency", mutate: func(c *RuntimeConfig) { c.OptimizeConcurrency = 0 }, want: ErrInvalidArgument},
		{name: "large optimize concurrency", mutate: func(c *RuntimeConfig) { c.OptimizeConcurrency = MaxRuntimeConcurrency + 1 }, want: ErrInvalidArgument},
		{name: "invalid log level", mutate: func(c *RuntimeConfig) { c.LogLevel = 99 }, want: ErrInvalidArgument},
		{name: "negative invert ratio", mutate: func(c *RuntimeConfig) { c.InvertToForwardScanRatio = -0.1 }, want: ErrInvalidArgument},
		{name: "large vector ratio", mutate: func(c *RuntimeConfig) { c.BruteForceByKeysRatio = 1.1 }, want: ErrInvalidArgument},
		{name: "NaN FTS ratio", mutate: func(c *RuntimeConfig) { c.FTSBruteForceByKeysRatio = float32(math.NaN()) }, want: ErrInvalidArgument},
		{name: "query binding", mutate: func(c *RuntimeConfig) { c.QueryThreadBinding = true }, want: ErrNotSupported},
		{name: "optimize binding", mutate: func(c *RuntimeConfig) { c.OptimizeThreadBinding = true }, want: ErrNotSupported},
	}
	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			value := NewRuntimeConfig()
			testCase.mutate(&value)
			if err := value.Validate(); !errors.Is(err, testCase.want) {
				t.Fatalf("validation error = %v, want %v", err, testCase.want)
			}
		})
	}
	config.MemoryLimitBytes = MinRuntimeMemoryLimit
	config.Logger = nil
	if err := config.Validate(); err != nil {
		t.Fatalf("minimum memory config = %v", err)
	}
}

func TestMemoryBudgetLimitsWaitsCancellationAndStats(t *testing.T) {
	budget := newMemoryBudget(10)
	release, err := budget.acquire(context.Background(), 7)
	if err != nil {
		t.Fatal(err)
	}
	waitContext, cancel := context.WithCancel(context.Background())
	waitResult := make(chan error, 1)
	go func() {
		_, waitErr := budget.acquire(waitContext, 4)
		waitResult <- waitErr
	}()
	waitForRuntimeCounter(t, func() uint64 {
		_, _, _, waiters := budget.stats()
		return waiters
	}, 1)
	cancel()
	if err := <-waitResult; !errors.Is(err, context.Canceled) {
		t.Fatalf("wait error = %v", err)
	}
	if _, err := budget.acquire(context.Background(), 11); !errors.Is(err, errRuntimeMemoryLimit) {
		t.Fatalf("oversized reservation = %v", err)
	}
	release()
	releaseAll, err := budget.acquire(context.Background(), 10)
	if err != nil {
		t.Fatal(err)
	}
	limit, used, peak, waiters := budget.stats()
	if limit != 10 || used != 10 || peak != 10 || waiters != 0 {
		t.Fatalf("memory stats = %d/%d/%d/%d", limit, used, peak, waiters)
	}
	releaseAll()
	_, used, peak, _ = budget.stats()
	if used != 0 || peak != 10 {
		t.Fatalf("released memory stats = %d/%d", used, peak)
	}

	unlimited := newMemoryBudget(0)
	releaseUnlimited, err := unlimited.acquire(context.Background(), math.MaxUint32)
	if err != nil {
		t.Fatal(err)
	}
	if limit, used, peak, _ := unlimited.stats(); limit != 0 || used != math.MaxUint32 || peak != math.MaxUint32 {
		t.Fatalf("unlimited stats = %d/%d/%d", limit, used, peak)
	}
	releaseUnlimited()
}

func TestTaskLimiterBoundsQueuesAndCounts(t *testing.T) {
	limiter := newTaskLimiter(1)
	releaseFirst, err := limiter.acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	acquired := make(chan func(), 1)
	go func() {
		release, acquireErr := limiter.acquire(context.Background())
		if acquireErr == nil {
			acquired <- release
		}
	}()
	waitForRuntimeCounter(t, func() uint64 {
		_, _, queued, _ := limiter.stats()
		return queued
	}, 1)
	canceledContext, cancel := context.WithCancel(context.Background())
	canceled := make(chan error, 1)
	go func() {
		_, acquireErr := limiter.acquire(canceledContext)
		canceled <- acquireErr
	}()
	waitForRuntimeCounter(t, func() uint64 {
		_, _, queued, _ := limiter.stats()
		return queued
	}, 2)
	cancel()
	if err := <-canceled; !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled acquire = %v", err)
	}
	waitForRuntimeCounter(t, func() uint64 {
		_, _, queued, _ := limiter.stats()
		return queued
	}, 1)
	releaseFirst()
	releaseSecond := <-acquired
	releaseSecond()
	active, peak, queued, completed := limiter.stats()
	if active != 0 || peak != 1 || queued != 0 || completed != 2 {
		t.Fatalf("task stats = %d/%d/%d/%d", active, peak, queued, completed)
	}
	alreadyCanceled, cancelAlready := context.WithCancel(context.Background())
	cancelAlready()
	if _, err := limiter.acquire(alreadyCanceled); !errors.Is(err, context.Canceled) {
		t.Fatalf("already-canceled acquire = %v", err)
	}
}

func TestRuntimeResourcesLoggingAdmissionAndCollectionStats(t *testing.T) {
	handler := &runtimeTestLogHandler{}
	config := NewRuntimeConfig()
	config.Logger = slog.New(handler)
	config.LogLevel = LogLevelDebug
	config.QueryConcurrency = 1
	config.OptimizeConcurrency = 2
	resources := newRuntimeResources(config)

	ctx := context.Background()
	schema := NewCollectionSchema("runtime", FieldSchema{
		Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2,
		Index: NewFlatIndexParams(MetricTypeIP),
	})
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "runtime"), schema, NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	defer collection.Close()
	collection.runtime = resources
	if _, err := collection.Insert(ctx, []Document{
		{PrimaryKey: "a", Fields: map[string]any{"embedding": VectorFP32{1, 0}}},
		{PrimaryKey: "b", Fields: map[string]any{"embedding": VectorFP32{0.8, 0}}},
		{PrimaryKey: "c", Fields: map[string]any{"embedding": VectorFP32{0.6, 0}}},
		{PrimaryKey: "d", Fields: map[string]any{"embedding": VectorFP32{0.4, 0}}},
	}); err != nil {
		t.Fatal(err)
	}
	stats := collection.Stats()
	if stats.DocumentCount != 4 || stats.MutableDocuments != 4 || stats.ImmutableSegments != 0 || stats.StorageMemoryBytes == 0 {
		t.Fatalf("mutable collection stats = %#v", stats)
	}
	if _, err := collection.Query(ctx, VectorQuery{Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 2}); err != nil {
		t.Fatal(err)
	}
	runtimeStats := resources.stats()
	if runtimeStats.CompletedQueries != 1 || runtimeStats.PeakQueries != 1 || runtimeStats.MemoryInUseBytes != 0 || runtimeStats.PeakMemoryBytes == 0 {
		t.Fatalf("query runtime stats = %#v", runtimeStats)
	}
	if messages := handler.messages(); !reflect.DeepEqual(messages, []string{"operation started", "operation completed"}) {
		t.Fatalf("runtime log messages = %#v", messages)
	}
	if collection.queryWorkers() != 1 || collection.optimizeWorkers(0) != 2 || collection.optimizeWorkers(9) != 2 || collection.optimizeWorkers(1) != 1 {
		t.Fatalf("runtime worker routing = query %d optimize %d/%d/%d", collection.queryWorkers(), collection.optimizeWorkers(0), collection.optimizeWorkers(9), collection.optimizeWorkers(1))
	}

	if err := collection.Flush(ctx); err != nil {
		t.Fatal(err)
	}
	stats = collection.Stats()
	if stats.ImmutableSegments != 1 || stats.MutableDocuments != 0 || stats.StorageMemoryBytes == 0 {
		t.Fatalf("flushed collection stats = %#v", stats)
	}
	beforeDeleteMemory := stats.StorageMemoryBytes
	if _, err := collection.Delete(ctx, []string{"a"}); err != nil {
		t.Fatal(err)
	}
	stats = collection.Stats()
	if stats.DocumentCount != 3 || stats.DeletedDocuments != 1 || stats.StorageMemoryBytes != beforeDeleteMemory+8 {
		t.Fatalf("deleted collection stats = %#v", stats)
	}
	if err := collection.Optimize(ctx, OptimizeOptions{}); err != nil {
		t.Fatal(err)
	}
	runtimeStats = resources.stats()
	if runtimeStats.CompletedOptimizeTasks != 1 || runtimeStats.PeakOptimizeTasks != 1 || runtimeStats.MemoryInUseBytes != 0 {
		t.Fatalf("optimize runtime stats = %#v", runtimeStats)
	}
	if messages := handler.messages(); !reflect.DeepEqual(messages, []string{
		"operation started", "operation completed", "operation started", "operation completed",
	}) {
		t.Fatalf("runtime log messages after optimize = %#v", messages)
	}

	tiny := NewRuntimeConfig()
	tiny.Logger = nil
	tiny.MemoryLimitBytes = 1
	collection.runtime = newRuntimeResources(tiny)
	if _, err := collection.Query(ctx, VectorQuery{Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 1}); !errors.Is(err, ErrResourceExhausted) {
		t.Fatalf("memory admission = %v", err)
	}
}

func TestRuntimePlannerRatiosAndFTSCandidateSeek(t *testing.T) {
	if got := collectionDiskANNCacheCapacity(2*4096, 100); got != 2 {
		t.Fatalf("DiskANN cache capacity = %d", got)
	}
	if got := collectionDiskANNCacheCapacity(4095, 100); got != 0 {
		t.Fatalf("sub-sector DiskANN cache capacity = %d", got)
	}
	if got := collectionDiskANNCacheCapacity(DefaultMaxBufferSize, 1); got != 1 {
		t.Fatalf("bounded DiskANN cache capacity = %d", got)
	}
	documents := testMultiQueryDocuments()
	for index := range documents {
		documents[index].DocID = uint64(index + 1)
	}
	plan, err := buildFilterPlan("category = 'keep'", testMultiQuerySchema())
	if err != nil {
		t.Fatal(err)
	}
	indexed, err := evaluateFilterDocuments(context.Background(), plan, documents, 0.9)
	if err != nil {
		t.Fatal(err)
	}
	forward, err := evaluateFilterDocuments(context.Background(), plan, documents, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	if !indexed.usedIndex || forward.usedIndex || indexed.matched != 3 || indexed.total != 4 ||
		!indexed.useBruteForce(0.75) || indexed.useBruteForce(0.74) {
		t.Fatalf("planner routes = indexed %#v, forward %#v", indexed, forward)
	}

	field, found := testMultiQuerySchema().Field("title")
	if !found {
		t.Fatal("missing title field")
	}
	runtime, err := buildCollectionFTSRuntime(context.Background(), field, documents, indexed.predicate)
	if err != nil {
		t.Fatal(err)
	}
	posting, err := searchCollectionFTS(context.Background(), runtime, &FTSClause{Match: "go"}, nil, 10, indexed.ordinals, false)
	if err != nil {
		t.Fatal(err)
	}
	candidate, err := searchCollectionFTS(context.Background(), runtime, &FTSClause{Match: "go"}, nil, 10, indexed.ordinals, true)
	if err != nil || !reflect.DeepEqual(candidate, posting) {
		t.Fatalf("FTS candidate seek = %#v, %v; posting %#v", candidate, err, posting)
	}
}

func TestConfigureRuntimeOneShotSubprocess(t *testing.T) {
	if os.Getenv("ZVEC_RUNTIME_CONFIG_HELPER") == "1" {
		bad := NewRuntimeConfig()
		bad.QueryConcurrency = 0
		if err := ConfigureRuntime(bad); !errors.Is(err, ErrInvalidArgument) {
			t.Fatalf("invalid initial config = %v", err)
		}
		SetDefaultJiebaDictDir("before-config")
		first := NewRuntimeConfig()
		first.Logger = nil
		first.MemoryLimitBytes = MinRuntimeMemoryLimit
		first.QueryConcurrency = 1
		first.OptimizeConcurrency = 2
		first.LogLevel = LogLevelInfo
		first.JiebaDictionaryDir = "configured"
		if err := ConfigureRuntime(first); err != nil {
			t.Fatal(err)
		}
		second := NewRuntimeConfig()
		second.Logger = nil
		second.QueryConcurrency = 7
		second.JiebaDictionaryDir = "ignored"
		if err := ConfigureRuntime(second); err != nil {
			t.Fatal(err)
		}
		got := CurrentRuntimeConfig()
		if got.QueryConcurrency != 1 || got.OptimizeConcurrency != 2 || got.MemoryLimitBytes != MinRuntimeMemoryLimit || got.LogLevel != LogLevelInfo {
			t.Fatalf("configured runtime = %#v", got)
		}
		if DefaultJiebaDictDir() != "configured" {
			t.Fatalf("Jieba fallback = %q", DefaultJiebaDictDir())
		}
		if stats := CurrentRuntimeStats(); stats.MemoryLimitBytes != MinRuntimeMemoryLimit {
			t.Fatalf("runtime stats = %#v", stats)
		}
		return
	}
	command := exec.Command(os.Args[0], "-test.run=^TestConfigureRuntimeOneShotSubprocess$")
	command.Env = append(os.Environ(), "ZVEC_RUNTIME_CONFIG_HELPER=1")
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("runtime config subprocess: %v\n%s", err, output)
	}
}

func TestRuntimeConfigCompatibilityFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/runtime_config_58375ff.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		BaselineCommit string             `json:"baseline_commit"`
		ConfigHeader   string             `json:"config_header_sha256"`
		ConfigSource   string             `json:"config_source_sha256"`
		OptionsHeader  string             `json:"options_header_sha256"`
		StatsHeader    string             `json:"stats_header_sha256"`
		Defaults       map[string]float64 `json:"planner_ratio_defaults"`
		LogLevels      map[string]int     `json:"log_levels"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.BaselineCommit != "58375ff7b8fdd0d6fc7d234e47567b179777883b" ||
		fixture.ConfigHeader != "e2fdabad1fca4b3ffd647081962c2869b4c376379fc1e5506f1e465c985b1758" ||
		fixture.ConfigSource != "04c9ea1d60b74dd3c5a1fb78bd61251bd11ab54acf2ada944e780f5800f3d929" ||
		fixture.OptionsHeader != "865c50a022754ad5101f9f40a03401e2832c5b008713c92d487ebf125334670d" ||
		fixture.StatsHeader != "791bb777751cb3f76ed79ec8c068a3575068361a717d246fb02f43027fc685af" ||
		!reflect.DeepEqual(fixture.Defaults, map[string]float64{"invert_to_forward": 0.9, "vector_brute_force": 0.1, "fts_brute_force": 0.05}) ||
		!reflect.DeepEqual(fixture.LogLevels, map[string]int{"debug": 0, "info": 1, "warn": 2, "error": 3, "fatal": 4}) {
		t.Fatalf("runtime compatibility fixture mismatch: %#v", fixture)
	}
}

func FuzzRuntimeConfigValidation(f *testing.F) {
	f.Add(uint64(0), int32(1), int32(1), uint32(math.Float32bits(0.9)), uint32(math.Float32bits(0.1)))
	f.Add(uint64(MinRuntimeMemoryLimit), int32(4), int32(2), uint32(math.Float32bits(0)), uint32(math.Float32bits(1)))
	f.Fuzz(func(t *testing.T, memory uint64, queryWorkers, optimizeWorkers int32, invertBits, bruteBits uint32) {
		config := NewRuntimeConfig()
		config.Logger = nil
		config.MemoryLimitBytes = memory
		config.QueryConcurrency = int(queryWorkers)
		config.OptimizeConcurrency = int(optimizeWorkers)
		config.InvertToForwardScanRatio = math.Float32frombits(invertBits)
		config.BruteForceByKeysRatio = math.Float32frombits(bruteBits)
		err := config.Validate()
		if err == nil {
			if config.QueryConcurrency <= 0 || config.QueryConcurrency > MaxRuntimeConcurrency ||
				config.OptimizeConcurrency <= 0 || config.OptimizeConcurrency > MaxRuntimeConcurrency ||
				config.MemoryLimitBytes != 0 && config.MemoryLimitBytes < MinRuntimeMemoryLimit {
				t.Fatalf("invalid config accepted: %#v", config)
			}
			return
		}
		if !errors.Is(err, ErrInvalidArgument) && !errors.Is(err, ErrNotSupported) {
			t.Fatalf("unstructured validation error = %v", err)
		}
	})
}

func BenchmarkRuntimeAdmission(b *testing.B) {
	config := NewRuntimeConfig()
	config.Logger = nil
	resources := newRuntimeResources(config)
	b.ReportAllocs()
	b.ResetTimer()
	for iteration := 0; iteration < b.N; iteration++ {
		release, err := resources.begin(context.Background(), runtimeQueryTask, "benchmark", "", 1024)
		if err != nil {
			b.Fatal(err)
		}
		release()
	}
}

type runtimeTestLogHandler struct {
	mu      sync.Mutex
	records []slog.Record
}

func (*runtimeTestLogHandler) Enabled(context.Context, slog.Level) bool { return true }

func (h *runtimeTestLogHandler) Handle(_ context.Context, record slog.Record) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.records = append(h.records, record.Clone())
	return nil
}

func (h *runtimeTestLogHandler) WithAttrs([]slog.Attr) slog.Handler { return h }

func (h *runtimeTestLogHandler) WithGroup(string) slog.Handler { return h }

func (h *runtimeTestLogHandler) messages() []string {
	h.mu.Lock()
	defer h.mu.Unlock()
	messages := make([]string, len(h.records))
	for index := range h.records {
		messages[index] = h.records[index].Message
	}
	return messages
}

func waitForRuntimeCounter(t *testing.T, read func() uint64, want uint64) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if read() == want {
			return
		}
		runtime.Gosched()
	}
	t.Fatalf("runtime counter = %d, want %d", read(), want)
}
