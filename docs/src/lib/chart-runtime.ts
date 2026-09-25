import * as echarts from 'echarts/core';
import { BarChart } from 'echarts/charts';
import { GridComponent, TooltipComponent, AriaComponent, LegendComponent } from 'echarts/components';
import { SVGRenderer } from 'echarts/renderers';

echarts.use([BarChart, GridComponent, TooltipComponent, AriaComponent, LegendComponent, SVGRenderer]);

export { echarts };
