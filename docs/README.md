# xvec benchmark website

An English, static Astro + TypeScript site for the HNSW measurements in
`benchmark-hnsw.csv`. ECharts is bundled locally; no backend or external chart
service is needed. The CSV and `logo.png` stay in this directory and are imported
through Astro/Vite to generate the homepage at build time.

## Development

Use Node.js 22.12+ and pnpm 10.33.0 (the version pinned in `package.json`).

```sh
cd docs
pnpm install --frozen-lockfile
pnpm dev
```

Open the local URL printed by Astro, using the `base` path from
`astro.config.mjs`. This is the only page; there are no subpages.
The repository's Go API and benchmark runner are independent of this project.

## Updating measurements

1. Edit `docs/benchmark-hnsw.csv`, retaining its header and units.
2. Run the checks from `docs`:

   ```sh
   pnpm test
   pnpm check
   pnpm build
   pnpm preview
   ```

3. Review the homepage and the generated SVG. Update the published-measurement fixture
   in `tests/benchmark.test.ts` when intentionally replacing the current six runs.
4. Commit the CSV, source changes, and `pnpm-lock.yaml` when dependencies change.
   Generated SVG files, `node_modules`, `.astro`, and `dist` are ignored and must
   not be committed.

The production output is `docs/dist`. The [Astro configuration](https://docs.astro.build/en/reference/configuration-reference/)
uses static output and site `https://gorse-io.github.io`. Asset imports and home
links respect the configured base path. Use `base: '/xvec/'` for GitHub Pages
project hosting, or `base: '/'` for a root homepage. Update `site` and `base` in
`astro.config.mjs` before building for another host. This project does not
publish or deploy automatically.

## Data contract and fair comparisons

`src/lib/parse-benchmark.ts` uses `csv-parse` at build time. Empty data, missing or
duplicate headers, unknown columns, malformed rows, empty text, invalid booleans,
unsupported backends/indexes/quantization, non-finite or negative numbers,
fractional integer fields, and recall outside 0–100 cause a build failure with
the source filename and line/field where applicable. Core counts and workload
settings must be positive. Unknown columns fail until classified, so a new test
setting cannot silently disappear from comparison grouping.

`src/lib/benchmark.ts` defines the typed schema, shared metric labels and units,
and grouping rules. All fields in `suiteFields` must match: machine, dataset,
document count, HNSW and runtime parameters, payload, concurrency duration,
cooldown, and serial query count. Machine, dataset, and test configuration
selectors expose separate groups as data is added. Single-choice selectors are
disabled. Quantization and rotation define individual chart categories. An xvec
and zvec bar are paired only when both category settings match. Incomplete
pairs remain visible, with missing measurements represented as gaps, not zero.

There must be at most one record per backend, suite, quantization, and rotation.
A different backend version does not permit a duplicate: choose the intended
run explicitly rather than silently combining repeated measurements. Backend
versions are preserved in the source CSV. In the current data, xvec INT4/INT8
and FP16 use different commits, so these are not controlled quantization-only
comparisons.

Recall uses the serial-phase measurement and a fixed 0–100% axis. Concurrent
QPS is shown alongside separate average and P99 latency charts. The loading
chart stacks the recorded Insert and Optimize times. Total load
time can include additional overhead beyond those phases. Memory uses the CSV's
MiB value for peak process RSS. Tooltips expose up to six decimal places. The
source CSV retains every field, including auxiliary timings/counts; it is read
during the build and is not exposed as a page download. The homepage presents
charts; detailed result tables and configuration summaries
are intentionally omitted. Selectors and charts need JavaScript.

## Exporting for GitHub README

`pnpm build` renders the charts to SVG using ECharts on the build machine; no
browser is required. The homepage and static images share the same metric and
chart definitions. The images include the filters and six chart cards in a fixed
two-column layout, excluding the navigation bar.

The build creates SVG files only inside `docs/dist`:

- `docs/dist/benchmark-hnsw.svg`: the default configuration for README embeds.
- `docs/dist/benchmarks/<machine>-<dataset>-<configuration-hash>.svg`: one image for
  each comparison group.

Deploy `docs/dist` as the site's document root. The default image is then served
at `<site><base>/benchmark-hnsw.svg`; `dist` is a build directory, not part of the
public URL. No copy is written into the source directory.

Click the image icon (**Export SVG**) to the left of the GitHub icon to download
the static file for the selected machine, dataset, and test configuration. The
link respects the site's base path and also works in `pnpm dev`. Without
JavaScript, it downloads the default configuration. Resizing the browser or
toggling a chart legend does not change the downloaded image.

The SVG contains text, shapes, and chart paths, with no HTML, scripts, bitmap
screenshots, or external assets. After deploying the site, embed its hosted SVG
in the repository's README. With the current `site` and `base` configuration:

```md
![HNSW benchmark results](https://gorse-io.github.io/benchmark-hnsw.svg)
```

Adjust the URL for the actual deployment host and base path. After changing
measurements, commit the CSV and rebuild/redeploy the site to update the image;
keep generated files out of Git. Deployment is separate from this build.

## Verification

`pnpm test` checks all 78 chart measurements in the current six rows, every metric
series, CSV quoting/BOM/CRLF, malformed data, duplicate records, and separation
of incompatible configurations. SVG tests cover deterministic output, safe text,
self-contained chart references, and configuration-specific asset paths. `pnpm check` checks Astro and TypeScript;
`pnpm build` revalidates the CSV while generating the static page and SVG files.

For a release, preview the production build at the configured base; exercise the
concurrent charts, average and P99 latency, the loading stack, and any available dataset/machine
selectors. Check a narrow mobile viewport and resize it to desktop, and hover
bars. Charts use a ResizeObserver, fixed backend colors, pattern decals,
and accessible descriptions of the current metric's exact values.
