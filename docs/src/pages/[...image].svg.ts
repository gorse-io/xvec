import type { APIRoute, GetStaticPaths } from 'astro';
import csv from '../../benchmark-hnsw.csv?raw';
import { groupBenchmarks, type ComparisonGroup } from '../lib/benchmark';
import { parseBenchmarks } from '../lib/parse-benchmark';
import { renderBenchmarkSvg } from '../lib/build-svg';
import { svgAsset } from '../lib/svg-asset';

export const getStaticPaths: GetStaticPaths = () => {
  const groups = groupBenchmarks(parseBenchmarks(csv));
  return [
    { params: { image: 'benchmark-hnsw' }, props: { group: groups[0] } },
    ...groups.map((group) => ({ params: { image: svgAsset(group).slice(0, -4) }, props: { group } })),
  ];
};

// Astro writes these responses as static SVG files during the production build.
// The same endpoint also supports downloads from the development server.
export const GET: APIRoute = ({ props }) => new Response(renderBenchmarkSvg(props.group as ComparisonGroup), {
  headers: { 'Content-Type': 'image/svg+xml; charset=utf-8' },
});
