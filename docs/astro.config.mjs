import { defineConfig } from 'astro/config';

export default defineConfig({
  site: 'https://gorse-io.github.io',
  base: '/',
  output: 'static',
  outDir: './dist',
  trailingSlash: 'always',
  vite: { build: { assetsInlineLimit: 0 } },
});
