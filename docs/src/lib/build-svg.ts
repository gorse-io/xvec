import { echarts } from './chart-runtime';
import { chartCards, chartDefinition, configurationLabel } from './chart-options';
import { categories, indexLabels, type ComparisonGroup } from './benchmark';

function escapeXml(value: string): string {
  return value.replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&apos;' })[character]!);
}

// Replace renderer counters with deterministic, chart-local names. This also
// prevents one chart's patterns or clip paths from referencing another chart.
function scopeDefinitions(svg: string, prefix: string): string {
  // SSR adds hover-only CSS with global renderer counters. Static images need
  // only the presentation attributes, so omit those interactive styles.
  svg = svg.replace(/<style\b[^>]*>[\s\S]*?<\/style>/g, '').replace(/\sclass="[^"]*"/g, '');
  const ids = new Map<string, string>();
  svg = svg.replace(/\bid="([^"]+)"/g, (_, id: string) => {
    const replacement = `${prefix}-def-${ids.size}`;
    ids.set(id, replacement);
    return `id="${replacement}"`;
  });
  svg = svg.replace(/url\(#([^)]*)\)/g, (original, id: string) => ids.has(id) ? `url(#${ids.get(id)})` : original);
  svg = svg.replace(/\b(href|xlink:href)="#([^"]+)"/g, (original, attribute: string, id: string) => ids.has(id) ? `${attribute}="#${ids.get(id)}"` : original);
  return svg;
}

/** Render the default desktop results layout without a browser or DOM. */
export function renderBenchmarkSvg(group: ComparisonGroup): string {
  const width = 1272;
  const cardWidth = 609;
  const cardHeight = 373;
  const gap = 22;
  const incomplete = categories(group.records).some((category) => category.records.length < 2);
  const chartTop = 180 + (incomplete ? 48 : 0);
  const height = chartTop + Math.ceil(chartCards.length / 2) * (cardHeight + gap) - gap + 52;
  const parts: string[] = [];
  const text = (x: number, y: number, value: string, size = 12, color = '#18263c', weight = 400, extra = '') =>
    `<text x="${x}" y="${y}" font-size="${size}" fill="${color}" font-weight="${weight}" ${extra}>${escapeXml(value)}</text>`;
  const box = (x: number, y: number, w: number, h: number, fill = '#ffffff', border = '#e0e5ec', radius = 10) =>
    `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${radius}" fill="${fill}" stroke="${border}"/>`;

  parts.push(`<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="benchmark-title benchmark-description" font-family="system-ui, sans-serif">
<title id="benchmark-title">${indexLabels[group.indexType]} benchmark results</title>
<desc id="benchmark-description">${escapeXml(`${group.machine}. ${group.dataset}. ${indexLabels[group.indexType]}. ${configurationLabel(group)}.`)}</desc>
<rect width="${width}" height="${height}" fill="#f7f8fa"/>`);
  parts.push(box(16, 52, 1240, 96));
  const fields = [
    { label: 'Machine', value: group.machine, x: 40, width: 250 },
    { label: 'Dataset', value: group.dataset, x: 310, width: 300 },
    { label: 'Index', value: indexLabels[group.indexType], x: 630, width: 130 },
    { label: 'Test configuration', value: configurationLabel(group), x: 780, width: 452 },
  ];
  fields.forEach((field, index) => {
    parts.push(text(field.x, 82, field.label, 10, '#627087', 600, 'letter-spacing="0.4"'));
    parts.push(box(field.x, 92, field.width, 34, '#fbfcfd', '#dae0e9', 5));
    parts.push(`<clipPath id="field-${index}"><rect x="${field.x + 10}" y="92" width="${field.width - 38}" height="34"/></clipPath>`);
    parts.push(text(field.x + 10, 113, field.value, 12, '#58667a', 400, `clip-path="url(#field-${index})"`));
    parts.push(`<path d="M${field.x + field.width - 17} 107l4 4 4-4" fill="none" stroke="#58667a"/>`);
  });
  if (incomplete) {
    parts.push(box(16, 160, 1240, 40, '#fbf5e5', '#eee0b9', 6));
    parts.push(text(33, 185, 'Some categories have only one backend result. Missing measurements are left blank, never treated as zero.', 12, '#715920'));
  }

  chartCards.forEach(({ kind, metric }, index) => {
    const { title, unit, direction, description, options } = chartDefinition(group, kind, metric);
    const x = 16 + (index % 2) * (cardWidth + gap);
    const y = chartTop + Math.floor(index / 2) * (cardHeight + gap);
    parts.push(`<g role="img" aria-labelledby="chart-${index}-title"><title id="chart-${index}-title">${escapeXml(description)}</title>`);
    parts.push(box(x, y, cardWidth, cardHeight));
    parts.push(text(x + 25, y + 41, title, 18, '#18263c', 600, 'letter-spacing="-0.45"'));
    const higher = direction === 'Higher is better';
    parts.push(box(x + cardWidth - 117, y + 26, 92, 20, '#f2f6f2', 'none', 4));
    parts.push(text(x + cardWidth - 110, y + 39, `${higher ? '↗' : '↘'} ${direction}`, 9, '#65746a'));
    parts.push(text(x + 25, y + 82, unit, 10, '#627087', 400, 'font-family="monospace"'));
    const chart = echarts.init(null, undefined, { renderer: 'svg', ssr: true, width: cardWidth - 50, height: 255 });
    try {
      chart.setOption({ ...options, tooltip: { show: false } });
      const fragment = scopeDefinitions(chart.renderToSVGString(), `chart-${index}`)
        .replace('<svg ', `<svg x="${x + 25}" y="${y + 98}" `);
      parts.push(fragment);
    } finally {
      chart.dispose();
    }
    parts.push('</g>');
  });
  parts.push('</svg>');
  return parts.join('\n');
}
