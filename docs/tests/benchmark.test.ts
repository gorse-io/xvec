import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { parseBenchmarks } from '../src/lib/parse-benchmark';
import { categories, groupBenchmarks, metricKeys, seriesFor, suiteFields, type Benchmark } from '../src/lib/benchmark';

const csv = readFileSync(new URL('../benchmark-hnsw.csv', import.meta.url), 'utf8');
const [header, ...lines] = csv.trim().split('\n');
const columns = header.split(',');
const base = lines[0].split(',');
function fixture(changes: Record<string, string> = {}) {
  return [header, columns.map((field, index) => {
    const value = changes[field] ?? base[index];
    return /[",\n]/.test(value) ? `"${value.replaceAll('"', '""')}"` : value;
  }).join(',')].join('\n');
}

test('current six runs, three categories, and all 13 chart metrics match the published measurements', () => {
  const records = parseBenchmarks(csv);
  assert.equal(records.length, 6);
  assert.equal(groupBenchmarks(records).length, 1);
  assert.deepEqual(categories(records).map((category) => [category.quantization, category.rotate]), [['int4', true], ['int8', true], ['fp16', false]]);
  // Order follows the shared metric definitions; these are independent values
  // from the published CSV, covering every selector and series.
  const expected = [
    [1835.186707,282.908977,87.984,4.357283,6.054819,7.265314,3.270084,4.524953,5.342606,71.003205,7.32474,63.670982,1472.585938],
    [2285.309151,273.20439,81.892,3.498585,5.234352,6.952462,3.422636,5.406221,5.921854,44.980547,6.133945,38.8244,568.742188],
    [1709.966729,251.062994,98.958,4.676369,6.336985,7.53349,3.721647,5.141429,6.114587,79.558466,11.828973,67.720461,1441.910156],
    [1512.037726,216.490614,98.474,5.28701,8.976525,11.737435,4.250335,5.909212,6.700442,61.840791,5.46147,56.333318,640.679688],
    [1289.41308,230.398904,99.68,6.202116,8.263139,9.70572,4.095718,5.311647,5.829253,90.289743,5.13713,85.145997,1672.835938],
    [1733.569829,289.796495,99.507,4.61225,6.468389,8.37713,3.242727,4.615804,5.367489,69.127283,4.938918,64.161354,781.839844],
  ];
  records.forEach((record, index) => assert.deepEqual(metricKeys.map((metric) => record[metric]), expected[index]));
  metricKeys.forEach((metric, index) => assert.deepEqual(seriesFor(records, metric).map((series) => series.data), [
    [expected[0][index], expected[2][index], expected[4][index]],
    [expected[1][index], expected[3][index], expected[5][index]],
  ]));
  assert.notEqual(records[0].backend_version, records[4].backend_version);
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
    ['rotate', 'yes'], ['backend', 'unknown'], ['quantize_type', 'int3'], ['index_type', 'flat'], ['machine', ''],
  ];
  for (const [field, value] of cases) assert.throws(() => parseBenchmarks(fixture({ [field]: value })), new RegExp(`line 2, field "${field}"`));
});

test('duplicate configurations fail even when versions differ', () => {
  assert.throws(() => parseBenchmarks(csv + lines[0] + '\n'), /line 8: duplicate configuration.*first seen at line 2/);
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
