import vamanaCsv from '../../benchmark-vamana.csv?raw';
import hnswCsv from '../../benchmark-hnsw.csv?raw';
import flatCsv from '../../benchmark-flat.csv?raw';
import diskannCsv from '../../benchmark-diskann.csv?raw';
import { groupBenchmarks } from './benchmark';
import { parseBenchmarks } from './parse-benchmark';

// HNSW comes first so the page and default README image keep the same default.
export const groups = groupBenchmarks([
  ...parseBenchmarks(hnswCsv, 'benchmark-hnsw.csv'),
  ...parseBenchmarks(flatCsv, 'benchmark-flat.csv'),
  ...parseBenchmarks(diskannCsv, 'benchmark-diskann.csv'),
  ...parseBenchmarks(vamanaCsv, 'benchmark-vamana.csv'),
]);
