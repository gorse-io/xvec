import { createHash } from 'node:crypto';
import { configurationKey, type ComparisonGroup } from './benchmark';

/** Stable URL for a configuration, independent of its position in the CSV. */
export function svgAsset(group: ComparisonGroup): string {
  const hash = createHash('sha256').update(configurationKey(group.records[0])).digest('hex').slice(0, 10);
  const slug = `${group.machine}-${group.dataset}`.replace(/[^a-zA-Z0-9_-]/g, '-');
  return `benchmarks/${slug}-${hash}.svg`;
}
