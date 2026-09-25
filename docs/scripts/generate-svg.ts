import { copyFile } from 'node:fs/promises';

const root = new URL('../', import.meta.url);
await copyFile(new URL('dist/benchmark-hnsw.svg', root), new URL('benchmark-hnsw.svg', root));
console.log('[svg] benchmark-hnsw.svg (default configuration, ready for README)');
