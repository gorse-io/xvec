# xvec benchmark website

An English, static Astro + TypeScript site for the HNSW, Flat, DiskANN, and Vamana measurements in
`benchmark-hnsw.csv`, `benchmark-flat.csv`, `benchmark-diskann.csv`, and `benchmark-vamana.csv`. ECharts is bundled locally; no backend or external chart
service is needed. The CSV files and `logo.png` stay in this directory and are imported
through Astro/Vite to generate the homepage at build time.

`benchmark-flat.csv` contains a separate Flat comparison of xvec and zvec for
INT4, INT8, FP16, and unquantized FP32 on `Performance768D100K`. It uses the
same runtime settings and metric units as the HNSW CSV, omitting the inapplicable
`m`, `ef_construction`, and `ef_search` columns. Both backends enable rotation
for INT4/INT8 and disable it for FP16/FP32; all runs disable refinement.
`v0.7.0+rotate` identifies a local zvec-go v0.7.0 modification that calls the
native `zvec_index_params_set_quantizer_enable_rotate` setter when selecting
INT4/INT8, matching xvec. The native library is unchanged, and rotation is
verified through its parameter getter. The Index selector between Dataset and
Test configuration switches all charts and the SVG export between HNSW
(the default), Flat, DiskANN, and Vamana. Configuration labels show the selected index's parameters.

`benchmark-diskann.csv` contains four runs: xvec and zvec with FP16 scalar
quantization or unquantized FP32. It uses the same `Performance768D100K` dataset
and runtime settings as the other CSVs. Both backends explicitly use maximum
degree 100, construction list size 50, 64 PQ chunks, and query list size 300;
these four columns replace the HNSW parameters. Rotation and refinement are
disabled. FP32 means no scalar quantization; DiskANN still uses the configured
product quantization for graph traversal. Both collections use `enable_mmap=true`;
the benchmark does not flush filesystem caches between queries.

`benchmark-vamana.csv` contains eight runs: xvec and zvec with INT4, INT8,
FP16, and unquantized FP32 on the same Cohere `Performance768D100K` dataset.
Both backends use maximum degree 64, construction list size 100, alpha 1.2,
maximum occlusion size 750, and search list size 200. Graph saturation,
two-pass construction, contiguous-memory mode, ID maps, and refinement are
disabled. INT4/INT8 enable rotation, verified through the native parameter getter.
The Vamana-specific CSV columns record these settings and participate in grouping.

These runs use xvec commit `6a8b120d4284bf16853b2b4ad18465c599e72d82` (merged
PR #91), zvec-go `v0.7.0+rotate`, and the unchanged native zvec library built from
`8321c1314a559fd5f909e92498f43e5194bf9b99`. They use Go 1.27.1 with
`CGO_ENABLED=0`, `GOMAXPROCS=8`, `GOMEMLIMIT=24GiB`, and CPU affinity 0–7
on e2-standard-8. Each run uses a fresh collection, all 100,000 vectors,
1,000 serial queries, K=100, batch size 100, optimize concurrency 8,
30 seconds of 8-worker concurrent queries, and a 3-second serial cooldown.
A representative run (repeat with a fresh path for each backend and precision):

```sh
CGO_ENABLED=0 go build -o /tmp/vector-db-bench ./cmd/vector-db-bench
env GOMAXPROCS=8 GOMEMLIMIT=24GiB taskset -c 0-7 /tmp/vector-db-bench xvec \
  --path /tmp/xvec-vamana-fp32 --case-type Performance768D100K \
  --dataset-dir /path/to/dataset --skip-download --index-type vamana \
  --ef-search 200 --k 100 --batch-size 100 --max-docs-per-segment 10000000 \
  --optimize-concurrency 8 --num-concurrency 8 --concurrency-duration 30s \
  --serial-cooldown 3s --payload-profile ids_only --enable-mmap=true \
  --is-using-refiner=false --output /tmp/xvec-vamana-fp32.json
```

Run from the repository root. For FP16/INT8/INT4, add `--quantize-type fp16`,
`int8`, or `int4`. For zvec, set `ZVEC_LIBRARY_PATH` to the native library and
build with a temporary modfile replacing zvec-go with the local rotation-enabled
binding described above. An unmodified binding does not reproduce the rotated runs.

Peak RSS is the entire benchmark process's high-water mark from `wait4`, including
loading, optimization, reopening, and querying. Measurements are single runs;
QPS should be interpreted together with recall, not as equal-recall comparisons.

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

1. Edit the appropriate `docs/benchmark-*.csv`, retaining its header and units.
2. Run the checks from `docs`:

   ```sh
   pnpm test
   pnpm check
   pnpm build
   pnpm preview
   ```

3. Review the homepage and the generated SVG. Update the published-measurement fixture
   in `tests/benchmark.test.ts` when intentionally updating the measurements.
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
cooldown, and serial query count. HNSW, DiskANN, and Vamana parameters are required only
for their respective indexes; Flat omits all three sets. All applicable index
parameters participate in grouping. Machine, dataset, index, and test configuration
selectors expose separate groups as data is added. Single-choice selectors are
disabled. Quantization and rotation define individual chart categories. An xvec
and zvec bar are paired only when both category settings match. Incomplete
pairs remain visible, with missing measurements represented as gaps, not zero.

There must be at most one record per backend, suite, quantization, and rotation.
A different backend version does not permit a duplicate: choose the intended
run explicitly rather than silently combining repeated measurements. Backend
versions are preserved in the source CSV. The index CSVs were measured at different revisions and times, so comparisons
across index types are not controlled index-only experiments. All four Vamana
precisions use the same xvec revision.

`quantize_type=fp32` denotes unquantized FP32 vectors (`--quantize-type` omitted
in the benchmark runner, whose JSON reports this as `none`). Rotation and
refinement are disabled for these runs. The FP32 pair uses the same
`Performance768D100K` workload and runtime settings as the existing rows:
M=50, construction EF=500, search EF=300, K=100, eight optimize/search workers,
30 seconds of concurrent search, a 3-second serial cooldown, CPU affinity 0–7,
`GOMAXPROCS=8`, and `GOMEMLIMIT=24GiB` on an `e2-standard-8` machine.

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

`pnpm test` checks all 104 HNSW chart measurements, Flat QPS and recall,
DiskANN parameters and its FP16/FP32-only categories, all Vamana settings and four precisions, index separation and configuration labels, every metric
series, CSV quoting/BOM/CRLF, malformed data, duplicate records, and separation
of incompatible configurations. SVG tests cover deterministic output, safe text,
self-contained chart references, and configuration-specific asset paths. `pnpm check` checks Astro and TypeScript;
`pnpm build` revalidates the CSV while generating the static page and SVG files.

For a release, preview the production build at the configured base; exercise the
concurrent charts, average and P99 latency, the loading stack, and any available dataset/machine
selectors. Check a narrow mobile viewport and resize it to desktop, and hover
bars. Charts use a ResizeObserver, fixed backend colors, pattern decals,
and accessible descriptions of the current metric's exact values.
