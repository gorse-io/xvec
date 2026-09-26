import type { EChartsOption } from 'echarts';
import { categories, colors, formatBytes, formatValue, metrics, seriesFor, type ComparisonGroup, type Metric } from './benchmark';

export const chartCards = [
  { kind: 'qps', metric: 'concurrent_qps' },
  { kind: 'recall', metric: 'recall_at_k_pct' },
  { kind: 'latency', metric: 'concurrent_latency_avg_ms' },
  { kind: 'latency-p99', metric: 'concurrent_latency_p99_ms' },
  { kind: 'load', metric: 'insert_duration_sec' },
  { kind: 'memory', metric: 'peak_rss_kib' },
] as const satisfies readonly { kind: string; metric: Metric }[];

export function configurationLabel(group: ComparisonGroup): string {
  const record = group.records[0];
  return record.index_type === 'hnsw'
    ? `M ${record.m} · ef ${record.ef_search} · Concurrency ${record.query_concurrency}`
    : `K ${record.k} · Concurrency ${record.query_concurrency}`;
}

// One definition for the live charts and the images produced during a build.
export function chartDefinition(group: ComparisonGroup, kind: string, metric: Metric) {
  const definition = metrics[metric];
  const title = kind === 'load' ? 'Insert + optimize time'
    : metric === 'recall_at_k_pct' ? `Recall@${group.records[0].k}` : definition.label;
  const variants = categories(group.records);
  const labels = variants.map((category) => category.label);
  const series = kind === 'load'
    ? (['xvec', 'zvec'] as const).flatMap((backend) => [
      { phase: 'Insert', field: 'insert_duration_sec' as const, color: colors[backend], stack: backend },
      { phase: 'Optimize', field: 'optimize_duration_sec' as const, color: backend === 'xvec' ? '#e5bc55' : '#97b2e8', stack: backend },
    ].map(({ phase, field, color, stack }) => ({
      name: `${backend} · ${phase}`, stack, color,
      data: variants.map((category) => category.records.find((record) => record.backend === backend)?.[field] ?? null),
    })))
    : seriesFor(group.records, metric).map((entry) => ({ ...entry, color: colors[entry.name as keyof typeof colors] }));
  const displayValue = (value: number) => kind === 'memory' ? formatBytes(value) : `${formatValue(value)} ${definition.unit}`;
  const description = `${title} (${definition.unit}). ${series.map((entry) => `${entry.name}: ${entry.data.map((value, index) => `${labels[index]}: ${value === null ? 'not measured' : displayValue(value)}`).join('; ')}`).join('. ')}`;
  const options: EChartsOption = {
    animation: false,
    color: [colors.xvec, colors.zvec],
    textStyle: { fontFamily: 'system-ui, sans-serif' },
    aria: { enabled: true, label: { description }, decal: { show: true } },
    legend: { top: 4, itemWidth: 10, itemHeight: 10, itemGap: 12, textStyle: { color: '#5c677b', fontSize: 10 }, data: series.map((entry) => entry.name) },
    grid: { left: 62, right: 18, top: kind === 'load' ? 48 : 36, bottom: 58 },
    tooltip: {
      trigger: 'axis', renderMode: 'richText', confine: true,
      axisPointer: { type: 'shadow' },
      valueFormatter: (value: unknown) => typeof value === 'number' ? displayValue(value) : 'Not measured',
    },
    xAxis: { type: 'category', data: labels, axisTick: { show: false }, axisLine: { lineStyle: { color: '#dfe4eb' } }, axisLabel: { color: '#5c677b', fontSize: 11, lineHeight: 18, interval: 0 } },
    yAxis: {
      type: 'value', min: 0, ...(metric === 'recall_at_k_pct' ? { max: 100, interval: 25 } : {}),
      axisLabel: { color: '#6c7789', formatter: (value: number) => metric === 'recall_at_k_pct' ? `${value}%`
        : metric === 'peak_rss_kib' ? formatBytes(value)
        : value.toLocaleString('en-US', { notation: 'compact', maximumFractionDigits: 1 }) },
      splitLine: { lineStyle: { color: '#edf0f4', type: 'dashed' } },
    },
    series: series.map((entry) => ({
      name: entry.name, data: entry.data, type: 'bar',
      ...('stack' in entry ? { stack: entry.stack } : {}),
      barMaxWidth: 40, barGap: '15%',
      itemStyle: { color: entry.color, borderRadius: [4, 4, 0, 0] },
      emphasis: { focus: 'series' },
    })),
  };
  return { title, description, unit: definition.unit, direction: definition.direction, options };
}
