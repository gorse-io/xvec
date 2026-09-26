import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { groupBenchmarks } from '../src/lib/benchmark';
import { parseBenchmarks } from '../src/lib/parse-benchmark';
import { renderBenchmarkSvg } from '../src/lib/build-svg';
import { svgAsset } from '../src/lib/svg-asset';

const records = parseBenchmarks(readFileSync(new URL('../benchmark-hnsw.csv', import.meta.url), 'utf8'));
const group = groupBenchmarks(records)[0];

test('build SVG is deterministic, self-contained, and contains all six current charts', () => {
  const svg = renderBenchmarkSvg(group);
  assert.equal(renderBenchmarkSvg(group), svg);
  assert.equal([...svg.matchAll(/<svg\b/g)].length, 7);
  for (const title of ['Concurrent QPS', 'Recall@100', 'Concurrent average latency', 'Concurrent P99 latency', 'Insert + optimize time', 'Peak resident memory']) {
    assert.ok(svg.includes(title), title);
  }
  assert.doesNotMatch(svg, /<(script|foreignObject|image)\b|GitHub|Export SVG|Serial|P95/);
  const ids = [...svg.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]);
  assert.equal(new Set(ids).size, ids.length);
  for (const match of svg.matchAll(/url\(#([^)]*)\)/g)) assert.ok(ids.includes(match[1]), match[1]);
});

test('SVG metadata escapes CSV text and asset names distinguish sanitized labels and configurations', () => {
  const special = groupBenchmarks(records.map((record) => ({ ...record, machine: 'A/<script>&"B' })))[0];
  const svg = renderBenchmarkSvg(special);
  assert.ok(svg.includes('A/&lt;script&gt;&amp;&quot;B'));
  assert.doesNotMatch(svg, /<script>/);
  const slash = groupBenchmarks(records.map((record) => ({ ...record, machine: 'A/B' })))[0];
  const dash = groupBenchmarks(records.map((record) => ({ ...record, machine: 'A-B' })))[0];
  const otherConfig = groupBenchmarks(records.map((record) => ({ ...record, m: record.m! + 1 })))[0];
  assert.match(svgAsset(special), /^benchmarks\/[a-zA-Z0-9_-]+\.svg$/);
  assert.notEqual(svgAsset(slash), svgAsset(dash));
  assert.notEqual(svgAsset(group), svgAsset(otherConfig));
  assert.equal(svgAsset(group), svgAsset({ ...group, id: 'different-order' }));
});

test('Flat SVG identifies its index and uses a distinct export asset', () => {
  const flat = groupBenchmarks(parseBenchmarks(readFileSync(new URL('../benchmark-flat.csv', import.meta.url), 'utf8')))[0];
  const svg = renderBenchmarkSvg(flat);
  assert.match(svg, /Flat benchmark results/);
  assert.match(svg, />Index<\/text>/);
  assert.match(svg, />Flat<\/text>/);
  assert.doesNotMatch(svg, /HNSW|M undefined|ef undefined/);
  assert.notEqual(svgAsset(flat), svgAsset(group));
});

test('DiskANN SVG identifies its index and only includes FP16 and FP32', () => {
  const diskann = groupBenchmarks(parseBenchmarks(readFileSync(new URL('../benchmark-diskann.csv', import.meta.url), 'utf8')))[0];
  const svg = renderBenchmarkSvg(diskann);
  assert.match(svg, /DiskANN benchmark results/);
  assert.match(svg, />DiskANN<\/text>/);
  assert.match(svg, /Degree 100 · Search 300 · Concurrency 8/);
  assert.match(svg, />FP16<\/text>/);
  assert.match(svg, />FP32<\/text>/);
  assert.doesNotMatch(svg, /INT4|INT8|HNSW|undefined/);
  assert.notEqual(svgAsset(diskann), svgAsset(group));
});
