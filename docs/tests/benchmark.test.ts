import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { parseBenchmarks } from '../src/lib/parse-benchmark';
import { categories, formatBytes, groupBenchmarks, metricKeys, seriesFor, suiteFields, type Benchmark } from '../src/lib/benchmark';
import { chartDefinition, configurationLabel } from '../src/lib/chart-options';

const csv = readFileSync(new URL('../benchmark-hnsw.csv', import.meta.url), 'utf8');
const flatCsv = readFileSync(new URL('../benchmark-flat.csv', import.meta.url), 'utf8');
const [header, ...lines] = csv.trim().split('\n');
const columns = header.split(',');
const base = lines[0].split(',');
function fixture(changes: Record<string, string> = {}) {
  return [header, columns.map((field, index) => {
    const value = changes[field] ?? base[index];
    return /[",\n]/.test(value) ? `"${value.replaceAll('"', '""')}"` : value;
  }).join(',')].join('\n');
}

test('current eight runs, four categories, and all 13 chart metrics match the published measurements', () => {
  const records = parseBenchmarks(csv);
  assert.equal(records.length, 8);
  assert.equal(groupBenchmarks(records).length, 1);
  assert.deepEqual(categories(records).map((category) => [category.quantization, category.rotate]), [['int4', true], ['int8', true], ['fp16', false], ['fp32', false]]);
  assert.equal(categories(records).at(-1)?.label, 'FP32');
  assert.ok(records.filter((record) => record.quantize_type === 'fp32').every((record) => !record.rotate && !record.use_refiner));
  // Order follows the shared metric definitions; these are independent values
  // from the published CSV, covering every selector and series.
  const expected = [
    [1835.186707,282.908977,87.984,4.357283,6.054819,7.265314,3.270084,4.524953,5.342606,71.003205,7.32474,63.670982,1507928],
    [2285.309151,273.20439,81.892,3.498585,5.234352,6.952462,3.422636,5.406221,5.921854,44.980547,6.133945,38.8244,582392],
    [1709.966729,251.062994,98.958,4.676369,6.336985,7.53349,3.721647,5.141429,6.114587,79.558466,11.828973,67.720461,1476516],
    [1512.037726,216.490614,98.474,5.28701,8.976525,11.737435,4.250335,5.909212,6.700442,61.840791,5.46147,56.333318,656056],
    [1289.41308,230.398904,99.68,6.202116,8.263139,9.70572,4.095718,5.311647,5.829253,90.289743,5.13713,85.145997,1712984],
    [1733.569829,289.796495,99.507,4.61225,6.468389,8.37713,3.242727,4.615804,5.367489,69.127283,4.938918,64.161354,800604],
    [1161.523517,241.250397,99.714,6.88512,9.130002,10.595211,3.914684,5.036292,5.836045,101.112918,3.551601,97.548204,2436860],
    [1228.124355,300.659152,99.718,6.510617,9.4205,11.695275,3.124387,4.144795,4.675958,93.753838,3.87124,89.854771,777244],
  ];
  records.forEach((record, index) => assert.deepEqual(metricKeys.map((metric) => record[metric]), expected[index]));
  metricKeys.forEach((metric, index) => {
    const displayed = expected.map((values) => metric === 'peak_rss_kib' ? values[index] * 1024 : values[index]);
    assert.deepEqual(seriesFor(records, metric).map((series) => series.data), [
      [displayed[0], displayed[2], displayed[4], displayed[6]],
      [displayed[1], displayed[3], displayed[5], displayed[7]],
    ]);
  });
  assert.notEqual(records[0].backend_version, records[4].backend_version);
});

test('byte values use readable units', () => {
  assert.equal(formatBytes(1544118272), '1.44 GB');
  assert.equal(formatBytes(596369408), '568.74 MB');
});

test('quoted commas, escaped quotes, multiline fields, BOM and CRLF', () => {
  const records = parseBenchmarks('\uFEFF' + fixture({ machine: 'Machine, "A"', backend_version: 'revision\nnotes' }).replaceAll('\n', '\r\n'));
  assert.equal(records[0].machine, 'Machine, "A"');
  assert.equal(records[0].backend_version, 'revision\r\nnotes');
});

test('empty CSV and header-only data fail clearly', () => {
  for (const input of ['', '\n', header]) assert.throws(() => parseBenchmarks(input), /benchmark-hnsw.csv: empty data/);
});

test('missing, unknown, duplicate and malformed columns fail with locations', () => {
  assert.throws(() => parseBenchmarks(header.replace('machine,', '')), /line 1: missing required columns: machine/);
  assert.throws(() => parseBenchmarks(header + ',unknown'), /line 1: unknown columns: unknown/);
  assert.throws(() => parseBenchmarks(header + ',machine'), /line 1: duplicate column/);
  assert.throws(() => parseBenchmarks(header + '\n' + lines[0] + ',extra'), /invalid CSV:.*line 2/);
  assert.throws(() => parseBenchmarks(header + '\n"unterminated'), /invalid CSV:/);
});

test('invalid numeric, integer, boolean and enum values identify the offending field', () => {
  const cases = [
    ['serial_qps', ''], ['serial_qps', 'NaN'], ['serial_qps', 'Infinity'], ['serial_qps', '-1'],
    ['serial_qps', '0x20'], ['serial_qps', '1e999'], ['serial_qps', '12ms'],
    ['m', '1.5'], ['m', '0'], ['m', '9007199254740992'], ['recall_at_k_pct', '100.01'],
    ['rotate', 'yes'], ['backend', 'unknown'], ['quantize_type', 'int3'], ['index_type', 'ivf'], ['machine', ''],
  ];
  for (const [field, value] of cases) assert.throws(() => parseBenchmarks(fixture({ [field]: value })), new RegExp(`line 2, field "${field}"`));
});

test('duplicate configurations fail even when versions differ', () => {
  assert.throws(() => parseBenchmarks(csv + lines[0] + '\n'), /line 10: duplicate configuration.*first seen at line 2/);
  assert.throws(() => parseBenchmarks(fixture() + '\n' + fixture({ backend_version: 'another-version' }).split('\n')[1]), /versions do not distinguish duplicate runs/);
});

test('every suite setting splits incompatible runs, including machine and dataset', () => {
  const record = parseBenchmarks(fixture())[0];
  for (const field of suiteFields) {
    const current = record[field];
    const different = typeof current === 'number' ? current + 1 : typeof current === 'boolean' ? !current : `${current}-other`;
    const changed = { ...record, [field]: different } as Benchmark;
    assert.equal(groupBenchmarks([record, changed]).length, 2, field);
  }
});

test('rotation mismatch creates separate categories and missing results stay null', () => {
  const record = parseBenchmarks(fixture())[0];
  const other = { ...record, backend: 'zvec', rotate: false } as Benchmark;
  assert.equal(groupBenchmarks([record, other]).length, 1);
  assert.deepEqual(seriesFor([record, other], 'concurrent_qps').map((series) => series.data), [[record.concurrent_qps, null], [null, other.concurrent_qps]]);
});

test('Flat measurements omit HNSW fields and remain in a separate comparison group', () => {
  const flat = parseBenchmarks(flatCsv, 'benchmark-flat.csv');
  assert.equal(flat.length, 8);
  assert.ok(flat.every((record) => record.index_type === 'flat' && record.m === undefined && record.ef_search === undefined && record.ef_construction === undefined));
  const groups = groupBenchmarks([...parseBenchmarks(csv), ...flat]);
  assert.deepEqual(groups.map((group) => [group.indexType, group.records.length]), [['hnsw', 8], ['flat', 8]]);
  assert.equal(configurationLabel(groups[0]), 'M 50 · ef 300 · Concurrency 8');
  assert.equal(configurationLabel(groups[1]), 'K 100 · Concurrency 8');
  assert.deepEqual(seriesFor(flat, 'concurrent_qps').map((series) => series.data), [
    [405.722249, 366.685121, 226.458463, 172.955356],
    [687.997656, 511.013183, 270.299993, 179.667649],
  ]);
  assert.deepEqual(seriesFor(flat, 'recall_at_k_pct').map((series) => series.data), [
    [87.353, 99.119, 99.957, 99.999], [81.781, 98.709, 99.743, 100],
  ]);
});

test('HNSW parameters remain required only for HNSW; Flat still validates common fields', () => {
  const missingHnsw = flatCsv.replaceAll(',flat,', ',hnsw,');
  assert.throws(() => parseBenchmarks(missingHnsw), /field "m"/);
  assert.throws(() => parseBenchmarks(flatCsv.split('\n')[0].replace('machine,', ''), 'benchmark-flat.csv'), /benchmark-flat.csv:.*missing required columns: machine/);
  const flat = parseBenchmarks(flatCsv)[0];
  assert.equal(groupBenchmarks([flat, { ...flat, m: 50, ef_construction: 500, ef_search: 300 }]).length, 1);
});

test('memory axis uses MB and GB rather than compact-number billions', () => {
  const group = groupBenchmarks(parseBenchmarks(csv))[0];
  const { options } = chartDefinition(group, 'memory', 'peak_rss_kib');
  const axis = options.yAxis as { axisLabel: { formatter: (value: number) => string } };
  assert.equal(axis.axisLabel.formatter(512 * 1024 ** 2), '512 MB');
  assert.equal(axis.axisLabel.formatter(1024 ** 3), '1 GB');
  assert.equal(axis.axisLabel.formatter(1.5 * 1024 ** 3), '1.5 GB');
});
