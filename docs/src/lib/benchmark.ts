// Shared by build-time parsing, CSV metadata, and browser charts.
export const backends = ['xvec', 'zvec'] as const;
export const indexTypes = ['hnsw', 'flat'] as const;
export type IndexType = typeof indexTypes[number];
export const indexLabels: Record<IndexType, string> = { hnsw: 'HNSW', flat: 'Flat' };
export const hnswFields = ['m', 'ef_construction', 'ef_search'] as const;
export const quantizations = ['int4', 'int8', 'fp16', 'fp32'] as const;
export const colors = { xvec: '#b47c00', zvec: '#4977cd' };

export const textFields = [
  'machine', 'backend', 'backend_version', 'case', 'index_type', 'quantize_type',
  'payload_profile', 'gomemlimit', 'cpu_affinity', 'go_version',
] as const;
export const booleanFields = ['rotate', 'use_refiner', 'enable_mmap'] as const;
export const integerFields = [
  'm', 'ef_construction', 'ef_search', 'k', 'batch_size', 'max_docs_per_segment',
  'optimize_concurrency', 'query_concurrency', 'gomaxprocs', 'inserted_count',
  'serial_queries', 'concurrent_queries', 'peak_rss_kib',
] as const;
export const decimalFields = [
  'concurrency_duration_sec', 'serial_cooldown_sec', 'insert_duration_sec',
  'optimize_duration_sec', 'load_duration_sec', 'insert_rows_per_sec', 'serial_qps',
  'recall_at_k_pct', 'serial_latency_avg_ms', 'serial_latency_p95_ms',
  'serial_latency_p99_ms', 'concurrent_qps', 'concurrent_latency_avg_ms',
  'concurrent_latency_p95_ms', 'concurrent_latency_p99_ms', 'peak_rss_mib',
  'wall_seconds', 'user_cpu_seconds', 'system_cpu_seconds',
] as const;
export const requiredFields = [...textFields, ...booleanFields, ...integerFields, ...decimalFields];
export type NumericField = typeof integerFields[number] | typeof decimalFields[number];
export type Benchmark = Record<typeof textFields[number], string>
  & Record<typeof booleanFields[number], boolean>
  & Record<Exclude<NumericField, typeof hnswFields[number]>, number>
  & Partial<Record<typeof hnswFields[number], number>>
  & { backend: typeof backends[number]; quantize_type: typeof quantizations[number]; index_type: IndexType };

// Quantization and rotation define the individual x-axis categories. Every
// other workload/runtime setting must match before records share a panel.
export const suiteFields = [
  'machine', 'case', 'index_type', 'use_refiner', 'enable_mmap', 'm',
  'ef_construction', 'ef_search', 'k', 'batch_size', 'max_docs_per_segment',
  'optimize_concurrency', 'query_concurrency', 'concurrency_duration_sec',
  'serial_cooldown_sec', 'payload_profile', 'gomaxprocs', 'gomemlimit',
  'cpu_affinity', 'go_version', 'inserted_count', 'serial_queries',
] as const satisfies readonly (keyof Benchmark)[];

export interface ComparisonGroup {
  id: string;
  machine: string;
  dataset: string;
  indexType: IndexType;
  records: Benchmark[];
}

export function configurationKey(record: Benchmark): string {
  return JSON.stringify(suiteFields.map((field) =>
    record.index_type === 'flat' && (hnswFields as readonly string[]).includes(field) ? null : record[field]));
}

export function groupBenchmarks(records: Benchmark[]): ComparisonGroup[] {
  const groups = new Map<string, ComparisonGroup>();
  for (const record of records) {
    const key = configurationKey(record);
    let group = groups.get(key);
    if (!group) {
      group = { id: `group-${groups.size + 1}`, machine: record.machine, dataset: record.case, indexType: record.index_type, records: [] };
      groups.set(key, group);
    }
    group.records.push(record);
  }
  return [...groups.values()];
}

export const metrics = {
  concurrent_qps: { label: 'Concurrent QPS', unit: 'queries/s', direction: 'Higher is better' },
  serial_qps: { label: 'Serial QPS', unit: 'queries/s', direction: 'Higher is better' },
  recall_at_k_pct: { label: 'Recall@K', unit: '%', direction: 'Higher is better' },
  concurrent_latency_avg_ms: { label: 'Concurrent average latency', unit: 'ms', direction: 'Lower is better' },
  concurrent_latency_p95_ms: { label: 'Concurrent P95 latency', unit: 'ms', direction: 'Lower is better' },
  concurrent_latency_p99_ms: { label: 'Concurrent P99 latency', unit: 'ms', direction: 'Lower is better' },
  serial_latency_avg_ms: { label: 'Serial average latency', unit: 'ms', direction: 'Lower is better' },
  serial_latency_p95_ms: { label: 'Serial P95 latency', unit: 'ms', direction: 'Lower is better' },
  serial_latency_p99_ms: { label: 'Serial P99 latency', unit: 'ms', direction: 'Lower is better' },
  load_duration_sec: { label: 'Total load time', unit: 's', direction: 'Lower is better' },
  insert_duration_sec: { label: 'Insert time', unit: 's', direction: 'Lower is better' },
  optimize_duration_sec: { label: 'Optimize time', unit: 's', direction: 'Lower is better' },
  peak_rss_kib: { label: 'Peak resident memory', unit: 'B', direction: 'Lower is better' },
} as const satisfies Partial<Record<NumericField, { label: string; unit: string; direction: string }>>;
export type Metric = keyof typeof metrics;
export const metricKeys = Object.keys(metrics) as Metric[];

export function formatValue(value: number): string {
  return value.toLocaleString('en-US', { maximumFractionDigits: 6 });
}

export function formatBytes(value: number): string {
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit++;
  }
  return `${value.toLocaleString('en-US', { maximumFractionDigits: 2 })} ${units[unit]}`;
}

export function categories(records: Benchmark[]) {
  return quantizations.flatMap((quantization) => [true, false].flatMap((rotate) => {
    const matches = records.filter((r) => r.quantize_type === quantization && r.rotate === rotate);
    return matches.length ? [{ quantization, rotate, records: matches,
      label: quantization.toUpperCase() }] : [];
  }));
}

export function seriesFor(records: Benchmark[], metric: Metric) {
  return backends.map((backend) => ({
    name: backend,
    data: categories(records).map((category) => {
      const value = category.records.find((r) => r.backend === backend)?.[metric];
      return value === undefined ? null : metric === 'peak_rss_kib' ? value * 1024 : value;
    }),
  }));
}
