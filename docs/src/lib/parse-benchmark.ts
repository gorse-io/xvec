import { parse } from 'csv-parse/sync';
import {
  backends, quantizations, indexTypes, hnswFields, diskannFields, vamanaFields, vamanaBooleanFields, indexFields, textFields, booleanFields, integerFields, decimalFields,
  requiredFields, configurationKey, type Benchmark,
} from './benchmark';

export function parseBenchmarks(csv: string, source = 'benchmark-hnsw.csv'): Benchmark[] {
  const fail = (message: string): never => { throw new Error(`${source}: ${message}`); };
  let rows: { record: string[]; info: { lines: number } }[];
  try {
    // csv-parse's array overload does not model the info:true wrapper.
    rows = parse(csv, { bom: true, skip_empty_lines: true, trim: true, info: true }) as unknown as typeof rows;
  } catch (error) {
    return fail(`invalid CSV: ${error instanceof Error ? error.message : String(error)}`);
  }
  if (!rows.length) return fail('empty data; expected a header and benchmark records');
  const header = rows[0].record;
  if (new Set(header).size !== header.length) return fail('line 1: duplicate column names');
  const missing = requiredFields.filter((field) => !(indexFields as readonly string[]).includes(field) && !header.includes(field));
  if (missing.length) return fail(`line 1: missing required columns: ${missing.join(', ')}`);
  // Reject new columns until classified, so new configuration cannot be silently
  // ignored when deciding whether two runs are comparable.
  const unknown = header.filter((field) => !requiredFields.includes(field as typeof requiredFields[number]));
  if (unknown.length) return fail(`line 1: unknown columns: ${unknown.join(', ')}; update the schema and grouping rules`);
  if (rows.length === 1) return fail('empty data; no benchmark records after the header');
  const seen = new Map<string, number>();
  return rows.slice(1).map(({ record, info }) => {
    const raw = Object.fromEntries(header.map((field, index) => [field, record[index]]));
    const value: Record<string, string | number | boolean> = {};
    const invalid = (field: string, reason: string): never => fail(`line ${info.lines}, field "${field}": ${reason} (received ${JSON.stringify(raw[field])})`);
    for (const field of textFields) {
      if (!raw[field]?.trim()) invalid(field, 'must not be empty');
      value[field] = raw[field].trim();
    }
    for (const field of booleanFields) {
      if (raw.index_type !== 'vamana' && (vamanaBooleanFields as readonly string[]).includes(field) && raw[field] === undefined) continue;
      if (!['true', 'false'].includes(raw[field])) invalid(field, 'expected true or false');
      value[field] = raw[field] === 'true';
    }
    for (const field of [...integerFields, ...decimalFields]) {
      const irrelevant = (raw.index_type !== 'hnsw' && (hnswFields as readonly string[]).includes(field))
        || (raw.index_type !== 'vamana' && (vamanaFields as readonly string[]).includes(field))
        || (raw.index_type !== 'diskann' && (diskannFields as readonly string[]).includes(field));
      if (irrelevant && raw[field] === undefined) continue;
      const input = raw[field];
      const number = Number(input);
      if (!/^(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(input) || !Number.isFinite(number) || number < 0) {
        invalid(field, 'expected a finite, non-negative decimal number');
      }
      if ((integerFields as readonly string[]).includes(field) && !Number.isSafeInteger(number)) invalid(field, 'expected a safe integer');
      value[field] = number;
    }
    for (const field of ['vamana_max_degree', 'vamana_build_list', 'vamana_query_list', 'vamana_alpha', 'm', 'ef_construction', 'ef_search', 'diskann_max_degree', 'diskann_build_list', 'diskann_query_list', 'k', 'batch_size', 'max_docs_per_segment',
      'optimize_concurrency', 'query_concurrency', 'gomaxprocs', 'inserted_count', 'serial_queries', 'concurrent_queries', 'concurrency_duration_sec']) {
      if (value[field] === 0) invalid(field, 'must be greater than zero');
    }
    if (!(backends as readonly string[]).includes(String(value.backend))) invalid('backend', 'expected xvec or zvec');
    if (!(quantizations as readonly string[]).includes(String(value.quantize_type))) invalid('quantize_type', 'expected int4, int8, fp16, or fp32 (unquantized)');
    if (!(indexTypes as readonly string[]).includes(String(value.index_type))) invalid('index_type', 'expected hnsw, flat, diskann, or vamana');
    if (Number(value.recall_at_k_pct) > 100) invalid('recall_at_k_pct', 'must be between 0 and 100');
    const benchmark = value as Benchmark;
    const key = JSON.stringify([configurationKey(benchmark), benchmark.quantize_type, benchmark.rotate, benchmark.backend]);
    const previous = seen.get(key);
    if (previous !== undefined) return fail(`line ${info.lines}: duplicate configuration for ${benchmark.backend} ${benchmark.quantize_type}; first seen at line ${previous} (versions do not distinguish duplicate runs)`);
    seen.set(key, info.lines);
    return benchmark;
  });
}
