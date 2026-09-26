import { echarts } from '../lib/chart-runtime';
import { chartDefinition, configurationLabel } from '../lib/chart-options';
import { indexTypes, indexLabels, type ComparisonGroup, type Metric } from '../lib/benchmark';

const panels = [...document.querySelectorAll<HTMLElement>('[data-group]')];
const groups: ComparisonGroup[] = panels.map((panel) => JSON.parse(panel.dataset.group!));
const machine = document.querySelector<HTMLSelectElement>('#machine')!;
const dataset = document.querySelector<HTMLSelectElement>('#dataset')!;
const index = document.querySelector<HTMLSelectElement>('#index')!;
const configuration = document.querySelector<HTMLSelectElement>('#configuration')!;
const charts = new Map<HTMLElement, echarts.ECharts>();
const resizeObserver = new ResizeObserver((entries) => {
  for (const entry of entries) {
    if (entry.contentRect.width > 0) charts.get(entry.target as HTMLElement)?.resize();
  }
});

function draw(card: HTMLElement, group: ComparisonGroup) {
  const metric = (card.dataset.kind === 'load' ? 'insert_duration_sec' : card.dataset.metric) as Metric;
  const { title, options } = chartDefinition(group, card.dataset.kind!, metric);
  card.querySelector('[data-metric-title]')!.textContent = title;
  const element = card.querySelector<HTMLElement>('.chart')!;
  let chart = charts.get(element);
  if (!chart) {
    chart = echarts.init(element, undefined, { renderer: 'svg' });
    charts.set(element, chart);
    resizeObserver.observe(element);
  }
  chart.setOption(options, { notMerge: true });
}

function showGroup() {
  const selected = groups.find((group) => group.id === configuration.value)!;
  panels.forEach((panel) => {
    panel.hidden = panel.id !== selected.id;
    if (!panel.hidden) {
      panel.querySelectorAll<HTMLElement>('.chart-card').forEach((card) => {
        card.querySelectorAll('select').forEach((select) => { select.disabled = false; });
        draw(card, selected);
      });
    }
  });
  const indexLabel = indexLabels[selected.indexType];
  document.querySelector('#selection-status')!.textContent = `${selected.machine}, ${selected.dataset}, ${indexLabel}. ${selected.records.length} runs shown.`;
  document.querySelector('.brand-section')!.textContent = indexLabel;
  document.title = `xvec — ${indexLabel} benchmarks`;
  document.querySelector<HTMLAnchorElement>('#export-svg')!.href = panels.find((panel) => panel.id === selected.id)!.dataset.svgUrl!;
}

function updateConfigurations() {
  const previous = configuration.value;
  const available = groups.filter((group) => group.machine === machine.value && group.dataset === dataset.value && group.indexType === index.value);
  configuration.replaceChildren(...available.map((group) => {
    return new Option(configurationLabel(group), group.id);
  }));
  if (available.some((group) => group.id === previous)) configuration.value = previous;
  configuration.disabled = available.length === 1;
  showGroup();
}

function updateIndexes() {
  const previous = index.value;
  const available = indexTypes.filter((indexType) => groups.some((group) =>
    group.machine === machine.value && group.dataset === dataset.value && group.indexType === indexType));
  index.replaceChildren(...available.map((indexType) => new Option(indexLabels[indexType], indexType)));
  if (available.some((indexType) => indexType === previous)) index.value = previous;
  index.disabled = available.length === 1;
  updateConfigurations();
}

function updateDatasets() {
  const previous = dataset.value;
  const available = [...new Set(groups.filter((group) => group.machine === machine.value).map((group) => group.dataset))];
  dataset.replaceChildren(...available.map((name) => new Option(name, name)));
  if (available.includes(previous)) dataset.value = previous;
  dataset.disabled = available.length === 1;
  updateIndexes();
}

// Honor links to any statically rendered configuration on first load.
const linked = groups.find((group) => location.hash === `#${group.id}`);
if (linked) { machine.value = linked.machine; dataset.value = linked.dataset; index.value = linked.indexType; configuration.value = linked.id; }
machine.disabled = new Set(groups.map((group) => group.machine)).size === 1;
machine.addEventListener('change', updateDatasets);
dataset.addEventListener('change', updateIndexes);
index.addEventListener('change', updateConfigurations);
configuration.addEventListener('change', showGroup);
panels.forEach((panel, index) => panel.querySelectorAll<HTMLElement>('.chart-card').forEach((card) => {
  card.addEventListener('change', () => draw(card, groups[index]));
}));
updateDatasets();
document.querySelector<HTMLElement>('#filters')!.hidden = false;
