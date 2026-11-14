<script setup lang="ts">
import { computed, onMounted, onUnmounted, shallowRef } from 'vue'
import SegmentedControl from '../components/SegmentedControl.vue'
import { num, pct } from './format'
import { areaPath, drawdown, drawdownTicks, linePath, monthTicks, navTicks, rangeAvailable, rangeStart, rebase, tickDigits, yearTicks, type Range } from './navChart'
import type { BacktestDocument, Part } from './types'

const props = defineProps<{ document: BacktestDocument | null; part: Part; showValidation: boolean; loading?: boolean }>()
const emit = defineEmits<{ part: [value: Part] }>()

const range = shallowRef<Range>('all'), log = shallowRef(false)
const box = shallowRef<HTMLElement | null>(null), size = shallowRef({ width: 720, height: 480 })
const hover = shallowRef<number | null>(null)
let observer: ResizeObserver | undefined
onMounted(() => {
  observer = new ResizeObserver(([entry]) => {
    const { width, height } = entry.contentRect
    // 宽度为 0 的读数来自隐藏或截图瞬间，丢弃，不当真值。
    if (width > 0 && height > 0) size.value = { width, height }
  })
  if (box.value) observer.observe(box.value)
})
onUnmounted(() => observer?.disconnect())

const LEFT = 46, RIGHT = 10, TOP = 12
const ranges = computed(() => (['1y', '3y', '5y', 'all'] as Range[])
  .filter(r => props.document && rangeAvailable(props.document.series.dates, r))
  .map(r => ({ value: r, label: r === 'all' ? '全部' : `${r[0]} 年` })))
const view = computed(() => {
  const doc = props.document
  if (!doc) return null
  const s = doc.series
  const active = ranges.value.some(r => r.value === range.value) ? range.value : 'all'
  const start = rangeStart(s.dates, active)
  const dates = s.dates.slice(start)
  const nav = rebase(s.nav, start)
  const bench = s.benchmark ? rebase(s.benchmark, start) : null
  return { dates, nav, bench, dd: drawdown(nav) }
})
const layout = computed(() => {
  const { width, height } = size.value
  const ddHeight = Math.max(80, Math.round(height * 0.24))
  const mainHeight = Math.max(160, height - ddHeight - 8)
  return { width, mainHeight, ddHeight, plotW: Math.max(10, width - LEFT - RIGHT) }
})
const axis = computed(() => {
  const v = view.value
  if (!v) return null
  const values = v.bench ? [v.nav, v.bench] : [v.nav]
  const ticks = navTicks(values, 5)
  const { mainHeight, ddHeight, plotW } = layout.value
  const bottom = mainHeight - 22
  const scale = (x: number) => log.value ? Math.log(x) : x
  const lo = scale(Math.max(ticks.min, 1e-6)), hi = scale(Math.max(ticks.max, 1e-6))
  const y = (value: number) => bottom - (scale(Math.max(value, 1e-6)) - lo) / (hi - lo || 1) * (bottom - TOP)
  const n = v.dates.length
  const x = (i: number) => LEFT + (n > 1 ? i / (n - 1) : 0) * plotW
  const dd = drawdownTicks(v.dd)
  const ddBottom = ddHeight - 22
  const yd = (value: number) => 8 + (value / (dd.min || -1)) * (ddBottom - 8)
  const years = n > 400 ? yearTicks(v.dates, Math.ceil(n * 44 / plotW)) : monthTicks(v.dates)
  return { ticks: ticks.ticks.filter(t => !log.value || t > 0), digits: tickDigits(ticks.step), y, x, bottom, dd, yd, ddBottom, years }
})
const paths = computed(() => {
  const v = view.value, a = axis.value
  if (!v || !a) return null
  return {
    nav: linePath(v.nav, a.x, a.y),
    bench: v.bench ? linePath(v.bench, a.x, a.y) : '',
    dd: areaPath(v.dd, a.x, a.yd),
    ddLine: linePath(v.dd, a.x, a.yd),
  }
})
function move(event: PointerEvent) {
  const v = view.value
  if (!v) return
  const rect = (event.currentTarget as SVGElement).getBoundingClientRect()
  const t = (event.clientX - rect.left - LEFT) / layout.value.plotW
  hover.value = t < 0 || t > 1 ? null : Math.round(t * (v.dates.length - 1))
}
const tip = computed(() => {
  const v = view.value, i = hover.value
  if (!v || i === null || i >= v.dates.length) return null
  return { date: v.dates[i], nav: v.nav[i], bench: v.bench?.[i] ?? null, dd: v.dd[i] }
})
</script>

<template>
  <section class="panel nav-panel">
    <div class="panel-header">
      净值
      <span v-if="document" class="legend"><i class="ink" />策略</span>
      <span v-if="document?.benchmark" class="legend"><i class="blue" />{{ document.benchmark.name }}</span>
      <span v-if="document" class="push tools">
        <SegmentedControl v-if="showValidation" label="窗口" :model-value="part" :options="[{ value: 'is', label: '样本内' }, { value: 'validation', label: '验证' }]" @update:model-value="emit('part', $event as Part)" />
        <SegmentedControl label="纵轴" :model-value="log ? 'log' : 'linear'" :options="[{ value: 'linear', label: '线性' }, { value: 'log', label: '对数' }]" @update:model-value="log = $event === 'log'" />
        <SegmentedControl v-if="ranges.length > 1" label="时间窗" :model-value="ranges.some(r => r.value === range) ? range : 'all'" :options="ranges" @update:model-value="range = $event as Range" />
      </span>
    </div>
    <div ref="box" class="chart-box" :class="{ stale: loading }">
      <template v-if="view && axis && paths">
        <svg class="chart" :width="layout.width" :height="layout.mainHeight" role="img" aria-label="策略与基准净值" @pointermove="move" @pointerleave="hover = null">
          <g v-for="t in axis.ticks" :key="t"><line :x1="LEFT" :x2="layout.width - RIGHT" :y1="axis.y(t)" :y2="axis.y(t)" class="grid" /><text :x="LEFT - 6" :y="axis.y(t) + 3.5" text-anchor="end">{{ t.toFixed(axis.digits) }}</text></g>
          <line :x1="LEFT" :x2="layout.width - RIGHT" :y1="axis.bottom" :y2="axis.bottom" class="base" />
          <text v-for="t in axis.years" :key="t.index" :x="axis.x(t.index)" :y="layout.mainHeight - 6" text-anchor="middle">{{ t.label }}</text>
          <path v-if="paths.bench" :d="paths.bench" class="bench" />
          <path :d="paths.nav" class="nav" />
          <template v-if="tip"><line :x1="axis.x(hover!)" :x2="axis.x(hover!)" :y1="TOP" :y2="axis.bottom" class="cross" /></template>
        </svg>
        <svg class="chart" :width="layout.width" :height="layout.ddHeight" role="img" aria-label="策略回撤" @pointermove="move" @pointerleave="hover = null">
          <g v-for="t in axis.dd.ticks" :key="t"><line :x1="LEFT" :x2="layout.width - RIGHT" :y1="axis.yd(t)" :y2="axis.yd(t)" :class="t === 0 ? 'base' : 'grid'" /><text :x="LEFT - 6" :y="axis.yd(t) + 3.5" text-anchor="end">{{ Math.round(t * 100) }}%</text></g>
          <path :d="paths.dd" class="dd" />
          <path :d="paths.ddLine" class="dd-line" />
          <line :x1="LEFT" :x2="layout.width - RIGHT" :y1="axis.ddBottom" :y2="axis.ddBottom" class="base" />
          <text v-for="t in axis.years" :key="t.index" :x="axis.x(t.index)" :y="layout.ddHeight - 6" text-anchor="middle">{{ t.label }}</text>
          <line v-if="tip" :x1="axis.x(hover!)" :x2="axis.x(hover!)" :y1="8" :y2="axis.ddBottom" class="cross" />
        </svg>
        <div v-if="tip" class="tip" :style="{ left: `${Math.min(axis.x(hover!) + 10, layout.width - 170)}px` }">
          <b class="mono">{{ tip.date }}</b>
          <span>策略<em class="mono">{{ num(tip.nav, 3) }}</em></span>
          <span v-if="view.bench">{{ document?.benchmark?.name }}<em class="mono">{{ num(tip.bench, 3) }}</em></span>
          <span>回撤<em class="mono">{{ pct(tip.dd) }}</em></span>
        </div>
      </template>
    </div>
  </section>
</template>

<style scoped>
.nav-panel{min-height:0}
.legend{display:flex;align-items:center;gap:6px;font-weight:400;color:var(--ink2)}
.legend i{width:12px;height:2.5px;display:block}
.legend .ink{background:var(--ink)}.legend .blue{background:var(--accent)}
.tools{display:flex;gap:10px;align-items:center}
.chart-box{flex:1;min-height:0;position:relative;padding:8px 10px 4px;overflow:hidden;transition:opacity .2s}
.chart-box.stale{opacity:.45}
.chart{display:block;font-size:10.5px;fill:var(--ink3);touch-action:none}
.grid{stroke:var(--grid)}.base{stroke:var(--line)}
.nav{fill:none;stroke:var(--ink);stroke-width:1.5;stroke-linejoin:round}
.bench{fill:none;stroke:var(--accent);stroke-width:1.3;stroke-linejoin:round}
.dd{fill:var(--ink3);opacity:.22}.dd-line{fill:none;stroke:var(--ink2);stroke-width:1}
.cross{stroke:var(--ink3);stroke-dasharray:3 3}
.tip{position:absolute;top:10px;width:160px;padding:7px 9px;background:var(--panel);border:1px solid var(--line);border-radius:4px;box-shadow:0 6px 18px var(--overlay-shadow);font-size:11.5px;display:flex;flex-direction:column;gap:2px;pointer-events:none;color:var(--ink2)}
.tip b{font-weight:500;color:var(--ink);margin-bottom:2px}
.tip span{display:flex;justify-content:space-between}
.tip em{font-style:normal;color:var(--ink)}
</style>
