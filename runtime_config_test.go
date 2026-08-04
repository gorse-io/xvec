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
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestRuntimeConfigDefaultsAndValidation(t *testing.T) {
	config := NewRuntimeConfig()
	{
		err := config.Validate()
		require.NoError(t, err)
	}

	defaultWorkers := min(runtime.GOMAXPROCS(0), MaxRuntimeConcurrency)
	require.Equal(t, defaultWorkers, config.QueryConcurrency)
	require.Equal(t, defaultWorkers, config.OptimizeConcurrency)
	require.Equal(t, LogLevelWarn, config.LogLevel)
	require.True(t, config.MemoryLimitBytes == 0)
	require.True(t, config.InvertToForwardScanRatio == 0.9)
	require.True(t, config.BruteForceByKeysRatio == 0.1)
	require.True(t, config.FTSBruteForceByKeysRatio == 0.05)

	for level, name := range map[LogLevel]string{
		LogLevelDebug: "DEBUG", LogLevelInfo: "INFO", LogLevelWarn: "WARN",
		LogLevelError: "ERROR", LogLevelFatal: "FATAL",
	} {
		require.True(t, level.Valid())
		require.Equal(t, name, level.String())
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
			{
				err := value.Validate()
				require.ErrorIs(t, err, testCase.want)
			}
		})
	}
	config.MemoryLimitBytes = MinRuntimeMemoryLimit
	config.Logger = nil
	{
		err := config.Validate()
		require.NoError(t, err)
	}
}

func TestMemoryBudgetLimitsWaitsCancellationAndStats(t *testing.T) {
	budget := newMemoryBudget(10)
	release, err := budget.acquire(context.Background(), 7)
	require.NoError(t, err)

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
	{
		err := <-waitResult
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		_, err := budget.acquire(context.Background(), 11)
		require.ErrorIs(t, err, errRuntimeMemoryLimit)
	}

	release()
	releaseAll, err := budget.acquire(context.Background(), 10)
	require.NoError(t, err)

	limit, used, peak, waiters := budget.stats()
	require.True(t, limit == 10)
	require.True(t, used == 10)
	require.True(t, peak == 10)
	require.True(t, waiters == 0)

	releaseAll()
	_, used, peak, _ = budget.stats()
	require.True(t, used == 0)
	require.True(t, peak == 10)

	unlimited := newMemoryBudget(0)
	releaseUnlimited, err := unlimited.acquire(context.Background(), math.MaxUint32)
	require.NoError(t, err)
	{
		limit, used, peak, _ := unlimited.stats()
		require.True(t, limit == 0)
		require.Equal(t, uint64(math.MaxUint32), used)
		require.Equal(t, uint64(math.MaxUint32), peak)
	}

	releaseUnlimited()
}

func TestTaskLimiterBoundsQueuesAndCounts(t *testing.T) {
	limiter := newTaskLimiter(1)
	releaseFirst, err := limiter.acquire(context.Background())
	require.NoError(t, err)

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
	{
		err := <-canceled
		require.ErrorIs(t, err, context.Canceled)
	}

	waitForRuntimeCounter(t, func() uint64 {
		_, _, queued, _ := limiter.stats()
		return queued
	}, 1)
	releaseFirst()
	releaseSecond := <-acquired
	releaseSecond()
	active, peak, queued, completed := limiter.stats()
	require.True(t, active == 0)
	require.True(t, peak == 1)
	require.True(t, queued == 0)
	require.True(t, completed == 2)

	alreadyCanceled, cancelAlready := context.WithCancel(context.Background())
	cancelAlready()
	{
		_, err := limiter.acquire(alreadyCanceled)
		require.ErrorIs(t, err, context.Canceled)
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
	require.NoError(t, err)

	defer collection.Close()
	collection.runtime = resources
	{
		_, err := collection.Insert(ctx, []Document{
			{PrimaryKey: "a", Fields: map[string]any{"embedding": VectorFP32{1, 0}}},
			{PrimaryKey: "b", Fields: map[string]any{"embedding": VectorFP32{0.8, 0}}},
			{PrimaryKey: "c", Fields: map[string]any{"embedding": VectorFP32{0.6, 0}}},
			{PrimaryKey: "d", Fields: map[string]any{"embedding": VectorFP32{0.4, 0}}},
		})
		require.NoError(t, err)
	}

	stats := collection.Stats()
	require.True(t, stats.DocumentCount == 4)
	require.True(t, stats.MutableDocuments == 4)
	require.True(t, stats.ImmutableSegments == 0)
	require.False(t, stats.StorageMemoryBytes == 0)
	{
		_, err := collection.Query(ctx, VectorQuery{Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 2})
		require.NoError(t, err)
	}

	runtimeStats := resources.stats()
	require.True(t, runtimeStats.CompletedQueries == 1)
	require.True(t, runtimeStats.PeakQueries == 1)
	require.True(t, runtimeStats.MemoryInUseBytes == 0)
	require.False(t, runtimeStats.PeakMemoryBytes == 0)
	{
		messages := handler.messages()
		require.Equal(t, []string{"operation started", "operation completed"}, messages)
	}
	require.True(t, collection.queryWorkers() == 1)
	require.True(t, collection.optimizeWorkers(0) == 2)
	require.True(t, collection.optimizeWorkers(9) == 2)
	require.True(t, collection.optimizeWorkers(1) == 1)
	{
		err := collection.Flush(ctx)
		require.NoError(t, err)
	}

	stats = collection.Stats()
	require.True(t, stats.ImmutableSegments == 1)
	require.True(t, stats.MutableDocuments == 0)
	require.False(t, stats.StorageMemoryBytes == 0)

	beforeDeleteMemory := stats.StorageMemoryBytes
	{
		_, err := collection.Delete(ctx, []string{"a"})
		require.NoError(t, err)
	}

	stats = collection.Stats()
	require.True(t, stats.DocumentCount == 3)
	require.True(t, stats.DeletedDocuments == 1)
	require.Equal(t, beforeDeleteMemory+8, stats.StorageMemoryBytes)
	{
		err := collection.Optimize(ctx, OptimizeOptions{})
		require.NoError(t, err)
	}

	runtimeStats = resources.stats()
	require.True(t, runtimeStats.CompletedOptimizeTasks == 1)
	require.True(t, runtimeStats.PeakOptimizeTasks == 1)
	require.True(t, runtimeStats.MemoryInUseBytes == 0)
	{
		messages := handler.messages()
		require.Equal(t, []string{
			"operation started", "operation completed", "operation started", "operation completed",
		}, messages)
	}

	tiny := NewRuntimeConfig()
	tiny.Logger = nil
	tiny.MemoryLimitBytes = 1
	collection.runtime = newRuntimeResources(tiny)
	{
		_, err := collection.Query(ctx, VectorQuery{Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 1})
		require.ErrorIs(t, err, ErrResourceExhausted)
	}
}

func TestRuntimePlannerRatiosAndFTSCandidateSeek(t *testing.T) {
	{
		got := collectionDiskANNCacheCapacity(2*4096, 100)
		require.True(t, got == 2)
	}
	{
		got := collectionDiskANNCacheCapacity(4095, 100)
		require.True(t, got == 0)
	}
	{
		got := collectionDiskANNCacheCapacity(DefaultMaxBufferSize, 1)
		require.True(t, got == 1)
	}

	documents := testMultiQueryDocuments()
	for index := range documents {
		documents[index].DocID = uint64(index + 1)
	}
	plan, err := buildFilterPlan("category = 'keep'", testMultiQuerySchema())
	require.NoError(t, err)

	indexed, err := evaluateFilterDocuments(context.Background(), plan, documents, 0.9)
	require.NoError(t, err)

	forward, err := evaluateFilterDocuments(context.Background(), plan, documents, 0.5)
	require.NoError(t, err)
	require.True(t, indexed.usedIndex)
	require.False(t, forward.usedIndex)
	require.True(t, indexed.matched == 3)
	require.True(t, indexed.total == 4)
	require.True(t, indexed.useBruteForce(0.75))
	require.False(t, indexed.useBruteForce(0.74))

	field, found := testMultiQuerySchema().Field("title")
	require.True(t, found,
		"missing title field")

	runtime, err := buildCollectionFTSRuntime(context.Background(), field, documents, indexed.predicate)
	require.NoError(t, err)

	posting, err := searchCollectionFTS(context.Background(), runtime, &FTSClause{Match: "go"}, nil, 10, indexed.ordinals, false)
	require.NoError(t, err)

	candidate, err := searchCollectionFTS(context.Background(), runtime, &FTSClause{Match: "go"}, nil, 10, indexed.ordinals, true)
	require.NoError(t, err)
	require.Equal(t, posting, candidate)
}

func TestConfigureRuntimeOneShotSubprocess(t *testing.T) {
	if os.Getenv("ZVEC_RUNTIME_CONFIG_HELPER") == "1" {
		bad := NewRuntimeConfig()
		bad.QueryConcurrency = 0
		{
			err := ConfigureRuntime(bad)
			require.ErrorIs(t, err, ErrInvalidArgument)
		}

		SetDefaultJiebaDictDir("before-config")
		first := NewRuntimeConfig()
		first.Logger = nil
		first.MemoryLimitBytes = MinRuntimeMemoryLimit
		first.QueryConcurrency = 1
		first.OptimizeConcurrency = 2
		first.LogLevel = LogLevelInfo
		first.JiebaDictionaryDir = "configured"
		{
			err := ConfigureRuntime(first)
			require.NoError(t, err)
		}

		second := NewRuntimeConfig()
		second.Logger = nil
		second.QueryConcurrency = 7
		second.JiebaDictionaryDir = "ignored"
		{
			err := ConfigureRuntime(second)
			require.NoError(t, err)
		}

		got := CurrentRuntimeConfig()
		require.True(t, got.QueryConcurrency == 1)
		require.True(t, got.OptimizeConcurrency == 2)
		require.Equal(t, MinRuntimeMemoryLimit, got.MemoryLimitBytes)
		require.Equal(t, LogLevelInfo, got.LogLevel)
		require.True(t, DefaultJiebaDictDir() == "configured")
		{
			stats := CurrentRuntimeStats()
			require.Equal(t, MinRuntimeMemoryLimit, stats.MemoryLimitBytes)
		}

		return
	}
	command := exec.Command(os.Args[0], "-test.run=^TestConfigureRuntimeOneShotSubprocess$")
	command.Env = append(os.Environ(), "ZVEC_RUNTIME_CONFIG_HELPER=1")
	output, err := command.CombinedOutput()
	require.NoError(t, err, "runtime config subprocess output:\n%s", output)
}

func TestRuntimeConfigCompatibilityFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/runtime_config_58375ff.json")
	require.NoError(t, err)

	var fixture struct {
		BaselineCommit string             `json:"baseline_commit"`
		ConfigHeader   string             `json:"config_header_sha256"`
		ConfigSource   string             `json:"config_source_sha256"`
		OptionsHeader  string             `json:"options_header_sha256"`
		StatsHeader    string             `json:"stats_header_sha256"`
		Defaults       map[string]float64 `json:"planner_ratio_defaults"`
		LogLevels      map[string]int     `json:"log_levels"`
	}
	{
		err := json.Unmarshal(data, &fixture)
		require.NoError(t, err)
	}
	require.True(t, fixture.BaselineCommit == "58375ff7b8fdd0d6fc7d234e47567b179777883b")
	require.True(t, fixture.ConfigHeader == "e2fdabad1fca4b3ffd647081962c2869b4c376379fc1e5506f1e465c985b1758")
	require.True(t, fixture.ConfigSource == "04c9ea1d60b74dd3c5a1fb78bd61251bd11ab54acf2ada944e780f5800f3d929")
	require.True(t, fixture.OptionsHeader == "865c50a022754ad5101f9f40a03401e2832c5b008713c92d487ebf125334670d")
	require.True(t, fixture.StatsHeader == "791bb777751cb3f76ed79ec8c068a3575068361a717d246fb02f43027fc685af")
	require.Equal(t, map[string]float64{"invert_to_forward": 0.9, "vector_brute_force": 0.1, "fts_brute_force": 0.05}, fixture.Defaults)
	require.Equal(t, map[string]int{"debug": 0, "info": 1, "warn": 2, "error": 3, "fatal": 4}, fixture.LogLevels)
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
			require.True(t, config.QueryConcurrency > 0)
			require.True(t, config.QueryConcurrency <= MaxRuntimeConcurrency)
			require.True(t, config.OptimizeConcurrency > 0)
			require.True(t, config.OptimizeConcurrency <= MaxRuntimeConcurrency)
			require.False(t, config.MemoryLimitBytes != 0 && config.MemoryLimitBytes < MinRuntimeMemoryLimit)

			return
		}
		require.False(t, !errors.Is(err, ErrInvalidArgument) && !errors.Is(err, ErrNotSupported))
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
			require.NoError(b, err)
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
	require.Equal(t, want, read())
}
