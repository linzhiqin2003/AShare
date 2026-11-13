<script setup lang="ts">
import { computed, onMounted, onUnmounted, shallowRef, watch } from 'vue'
import { curves, fmt, niceAxis } from '../chart'
import type { ChartData, Line } from '../types'

const props = withDefaults(defineProps<{ data: ChartData; compound?: boolean; raw?: boolean; compact?: boolean; selectedId?: string; drawdownLine?: Line | null }>(), { compound: false, raw: false, compact: false })
const root = shallowRef<HTMLDivElement | null>(null)
const width = shallowRef(1000), height = shallowRef(450)
const start = shallowRef(0), end = shallowRef(0), hover = shallowRef(-1)
const hoverY = shallowRef(0)
const drag = shallowRef<{ kind: string; x: number; start: number; end: number } | null>(null)
let observer: ResizeObserver | undefined
let frame = 0
let pendingMove: (() => void) | undefined
const left = 48
const right = computed(() => props.compact ? 12 : 114)
const plotWidth = computed(() => Math.max(40, width.value - left - right.value))
const plotHeight = computed(() => height.value - (props.compact ? 54 : 100) - (props.raw ? 0 : 12))
const mainHeight = computed(() => Math.max(50, plotHeight.value * (props.raw ? 1 : .8)))
const drawTop = computed(() => mainHeight.value + (props.raw ? 10 : 22))
const drawHeight = computed(() => props.raw ? 0 : Math.max(20, plotHeight.value * .2))
const axisY = computed(() => drawTop.value + drawHeight.value + 12)
const brushY = computed(() => height.value - 51)
const length = computed(() => props.data.dates.length)
const x = (i: number) => left + i / Math.max(1, end.value - start.value) * plotWidth.value
const fullX = (i: number) => left + i / Math.max(1, length.value - 1) * plotWidth.value
const plotted = computed(() => props.data.lines.map(line => ({ ...line, ...(props.raw ? { cumulative: line.values.slice(start.value, end.value + 1), drawdown: [] } : curves(line.values, props.compound, start.value, end.value)) })))
function extent(series: number[][], zero = true): [number, number] {
  let lo = zero ? 0 : Infinity, hi = zero ? 0 : -Infinity
  for (const values of series) for (const value of values) if (Number.isFinite(value)) { lo = Math.min(lo, value); hi = Math.max(hi, value) }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [-0.01, 0.01]
  const span = hi - lo || 0.02
  return [lo - span * 0.07, hi + span * 0.12]
}
const selectedLine = computed(() => props.drawdownLine ?? props.data.lines.find(line => !line.benchmark && line.id.startsWith(props.selectedId ?? '')) ?? props.data.lines.find(line => !line.benchmark))
const selectedDrawdown = computed(() => selectedLine.value ? curves(selectedLine.value.values, props.compound, start.value, end.value).drawdown : [])
const mainAxis = computed(() => niceAxis(plotted.value.map(v => v.cumulative),4,!props.raw,true))
const mainRange = computed(() => [mainAxis.value.min,mainAxis.value.max])
const drawAxis = computed(() => niceAxis([selectedDrawdown.value],2,true))
const drawRange = computed(() => [Math.min(-.01,drawAxis.value.min),0])
const y = (value: number) => 18 + (mainRange.value[1] - value) / (mainRange.value[1] - mainRange.value[0]) * (mainHeight.value - 22)
const dy = (value: number) => drawTop.value + (drawRange.value[1] - value) / (drawRange.value[1] - drawRange.value[0]) * drawHeight.value
function path(values: number[], xf: (i: number) => number, yf: (n: number) => number) {
  // 按像素极值保留折点，缩放窗口仍先用完整逐日收益重算。
  const stride = Math.max(1, Math.floor(values.length / (plotWidth.value * 2)))
  let output = '', drawing = false
  for (let i = 0; i < values.length; i += stride) {
    let low = -1, high = -1
    for (let j = i; j < Math.min(values.length, i + stride); j++) {
      if (!Number.isFinite(values[j])) continue
      if (low < 0 || values[j] < values[low]) low = j
      if (high < 0 || values[j] > values[high]) high = j
    }
    if (low < 0) { drawing = false; continue }
    const points = [...new Set([i, low, high, Math.min(values.length - 1, i + stride - 1)])].sort((a, b) => a - b)
    for (const point of points) {
      if (!Number.isFinite(values[point])) { drawing = false; continue }
      output += `${drawing ? 'L' : 'M'}${xf(point).toFixed(1)},${yf(values[point]).toFixed(1)}`; drawing = true
    }
  }
  return output
}
const mainTicks = computed(() => mainAxis.value.ticks)
const drawTicks = computed(() => drawAxis.value.ticks.filter(tick => tick <= 0))
const validationIndex = computed(() => props.data.dates.findIndex(d => d >= '20250101'))
const validationX = computed(() => validationIndex.value < 0 ? width.value : Math.min(left + plotWidth.value, Math.max(left, x(validationIndex.value - start.value))))
const labels = computed(() => {
  const items: { index: number; text: string }[] = []
  for (let i = start.value; i <= end.value; i++) {
    if (i === start.value || props.data.dates[i]?.slice(0, 4) !== props.data.dates[i - 1]?.slice(0, 4)) items.push({ index: i - start.value, text: props.data.dates[i]?.slice(0, 4) ?? '' })
  }
  return items.filter((_, i) => i % Math.max(1, Math.ceil(items.length / (width.value / 90))) === 0)
})
const mini = computed(() => selectedLine.value ? curves(selectedLine.value.values, props.compound).cumulative : [])
const miniYears = computed(() => props.data.dates.map((date,index) => ({date,index})).filter(({date,index}) => index === 0 || date.slice(0,4) !== props.data.dates[index-1].slice(0,4)).filter((_,i,items) => i % Math.max(1,Math.ceil(items.length / (plotWidth.value/65))) === 0))
const miniRange = computed(() => extent([mini.value]))
const miniY = (v: number) => brushY.value + 4 + (miniRange.value[1] - v) / (miniRange.value[1] - miniRange.value[0]) * 30
const tooltipX = computed(() => Math.max(left, Math.min(x(hover.value) + 12, width.value - 235)))
const lineLabels = computed(() => {
  const items = plotted.value.filter(l => Number.isFinite(l.cumulative.at(-1))).map(l => ({ id: l.id, y: y(l.cumulative.at(-1)!) + 4 })).sort((a, b) => a.y - b.y)
  for (let i = 1; i < items.length; i++) items[i].y = Math.max(items[i].y, items[i - 1].y + 13)
  const excess = Math.max(0, (items.at(-1)?.y ?? 0) - mainHeight.value)
  return new Map(items.map(item => [item.id, item.y - excess]))
})
function move(event: PointerEvent) {
  const rect = (event.currentTarget as SVGElement).getBoundingClientRect()
  const pixel = (event.clientX - rect.left) * width.value / rect.width
  const nextY = Math.max(18, Math.min(mainHeight.value, (event.clientY - rect.top) * height.value / rect.height))
  pendingMove = () => { hoverY.value = nextY
  if (drag.value) {
    const delta = Math.round((pixel - drag.value.x) / plotWidth.value * (length.value - 1))
    const span = drag.value.end - drag.value.start
    if (drag.value.kind === 'left') start.value = Math.max(0, Math.min(end.value - 2, drag.value.start + delta))
    else if (drag.value.kind === 'right') end.value = Math.min(length.value - 1, Math.max(start.value + 2, drag.value.end + delta))
    else { start.value = Math.max(0, Math.min(length.value - 1 - span, drag.value.start + delta)); end.value = start.value + span }
  } else hover.value = Math.max(0, Math.min(end.value - start.value, Math.round((pixel - left) / plotWidth.value * (end.value - start.value))))
  }
  if (!frame) frame = requestAnimationFrame(() => { pendingMove?.(); pendingMove = undefined; frame = 0 })
}
function endDrag() { if (frame) cancelAnimationFrame(frame); pendingMove?.(); pendingMove = undefined; frame = 0; drag.value = null }
function begin(event: PointerEvent, kind: string) {
  const svg = (event.currentTarget as SVGElement).ownerSVGElement!
  const rect = svg.getBoundingClientRect()
  drag.value = { kind, x: (event.clientX - rect.left) * width.value / rect.width, start: start.value, end: end.value }
  svg.setPointerCapture(event.pointerId); event.preventDefault()
}
function adjust(event: KeyboardEvent, edge: 'left' | 'right') {
  const amount = event.key === 'ArrowLeft' ? -10 : event.key === 'ArrowRight' ? 10 : 0
  if (!amount) return
  event.preventDefault()
  if (edge === 'left') start.value = Math.max(0, Math.min(end.value - 2, start.value + amount))
  else end.value = Math.min(length.value - 1, Math.max(start.value + 2, end.value + amount))
}
watch(() => props.data.dates.join(','), () => { start.value = 0; end.value = Math.max(0, length.value - 1); hover.value = -1 }, { immediate: true })
onMounted(() => { observer = new ResizeObserver(entries => { width.value = entries[0].contentRect.width; height.value = Math.max(120, entries[0].contentRect.height) }); if (root.value) observer.observe(root.value) })
onUnmounted(() => { observer?.disconnect(); cancelAnimationFrame(frame) })
</script>
<template>
  <div ref="root" class="result-chart" :class="{ compact }">
    <div v-if="!data.lines.length" class="empty-chart">运行后查看结果</div>
    <svg v-else :viewBox="`0 0 ${width} ${height}`" role="img" aria-label="研究结果曲线与可拖动时间窗" @pointermove="move" @pointerup="endDrag" @pointercancel="endDrag" @pointerleave="!drag && (hover = -1)">
      <rect :x="validationX" y="8" :width="Math.max(0, left + plotWidth - validationX)" :height="axisY - 15" fill="var(--shade)" />
      <g v-if="validationX < left + plotWidth && validationX > left" class="muted">
        <line :x1="validationX" :x2="validationX" y1="8" :y2="axisY - 7" stroke="var(--ink3)" stroke-dasharray="3 3" />
        <text :x="validationX - 5" y="15" text-anchor="end">样本内</text><text :x="validationX + 5" y="15">验证</text>
      </g>
      <g v-for="tick in mainTicks" :key="tick" class="muted main-tick"><line :x1="left" :x2="left + plotWidth" :y1="y(tick)" :y2="y(tick)" stroke="var(--grid)" /><text :x="left - 7" :y="y(tick) + 3" text-anchor="end">{{ fmt(tick, !raw, raw ? 2 : 0) }}</text></g>
      <g v-for="line in plotted" :key="line.id">
        <path :d="path(line.cumulative, x, y)" fill="none" :stroke="line.color" :stroke-width="line.benchmark ? 1.2 : compact ? 1.3 : 1.7" :stroke-dasharray="line.benchmark ? '5 4' : undefined" />
        <text v-if="!compact && Number.isFinite(line.cumulative.at(-1))" :x="left + plotWidth + 6" :y="lineLabels.get(line.id)" :fill="line.color" class="line-label">{{ line.name }}</text>
      </g>
      <g v-if="!raw">
        <g v-for="tick in drawTicks" :key="tick" class="muted draw-tick"><line :x1="left" :x2="left + plotWidth" :y1="dy(tick)" :y2="dy(tick)" stroke="var(--grid)"/><text :x="left - 7" :y="dy(tick) + 3" text-anchor="end">{{ fmt(tick, true, 0) }}</text></g>
        <path v-if="selectedDrawdown.length" :d="`${path(selectedDrawdown,x,dy)} L${x(selectedDrawdown.length-1)},${dy(0)} L${x(0)},${dy(0)} Z`" :fill="selectedLine?.color" fill-opacity=".11" />
        <path :d="path(selectedDrawdown, x, dy)" fill="none" :stroke="selectedLine?.color" stroke-width="1" />
        <text :x="left + 5" :y="drawTop + drawHeight - 4" fill="var(--ink2)">回撤</text>
      </g>
      <text v-for="label in labels" :key="label.index" :x="x(label.index)" :y="axisY + 3" class="muted" text-anchor="middle">{{ label.text }}</text>
      <g v-if="!compact" class="brush">
        <rect :x="left" :y="brushY" :width="plotWidth" height="34" rx="2" fill="var(--chrome)" stroke="var(--line)" />
        <rect v-if="validationIndex >= 0" :x="fullX(validationIndex)" :y="brushY" :width="left + plotWidth - fullX(validationIndex)" height="34" fill="var(--shade)" />
        <path v-if="mini.length" :d="`${path(mini,fullX,miniY)} L${fullX(mini.length-1)},${brushY+34} L${fullX(0)},${brushY+34} Z`" fill="var(--ink3)" fill-opacity=".16" />
        <path :d="path(mini, fullX, miniY)" stroke="var(--ink2)" stroke-width="1.1" fill="none" />
        <rect :x="fullX(start)" :y="brushY" :width="fullX(end) - fullX(start)" height="34" fill="rgba(31,79,191,.035)" stroke="var(--accent)" class="brush-body" @pointerdown="begin($event, 'body')" />
        <g v-for="edge in (['left', 'right'] as const)" :key="edge" tabindex="0" role="slider" :aria-label="edge === 'left' ? '时间窗起点' : '时间窗终点'" :aria-valuemin="0" :aria-valuemax="length - 1" :aria-valuenow="edge === 'left' ? start : end" @keydown="adjust($event, edge)" class="brush-handle" @pointerdown="begin($event, edge)">
          <rect :x="fullX(edge === 'left' ? start : end) - 8" :y="brushY - 4" width="16" height="42" fill="transparent" />
          <rect :x="fullX(edge === 'left' ? start : end) - 3" :y="brushY + 9" width="6" height="16" rx="2" fill="white" stroke="var(--accent)" />
        </g>
        <g v-for="year in miniYears" :key="year.date"><line :x1="fullX(year.index)" :x2="fullX(year.index)" :y1="brushY+34" :y2="brushY+37" stroke="var(--line)"/><text :x="fullX(year.index)" :y="brushY+48" class="muted" text-anchor="middle">{{ year.date.slice(0,4) }}</text></g>
      </g>
      <g v-if="hover >= 0 && !drag" class="hover-tip" pointer-events="none">
        <line :x1="x(hover)" :x2="x(hover)" y1="18" :y2="axisY - 8" stroke="var(--ink3)" />
        <line :x1="left" :x2="left + plotWidth" :y1="hoverY" :y2="hoverY" stroke="var(--ink3)" stroke-dasharray="3 3" />
        <g v-for="line in plotted" :key="line.id"><circle v-if="Number.isFinite(line.cumulative[hover])" :cx="x(hover)" :cy="y(line.cumulative[hover])" r="3" :fill="line.color" stroke="white" /></g>
        <rect :x="tooltipX" y="26" width="220" :height="(raw ? 30 : 52) + plotted.length * 18" rx="3" fill="white" stroke="var(--line)" />
        <text :x="tooltipX + 10" y="44" class="mono">{{ data.dates[start + hover]?.replace(/(\d{4})(\d{2})(\d{2})/, '$1-$2-$3') }}</text>
        <g v-for="(line, i) in plotted" :key="line.id"><text :x="tooltipX + 10" :y="64 + i * 18" :fill="line.color">{{ line.name }}</text><text :x="tooltipX + 210" :y="64 + i * 18" text-anchor="end" class="mono">{{ fmt(line.cumulative[hover], !raw) }}</text></g>
        <text v-if="!raw && selectedLine" :x="tooltipX+10" :y="64+plotted.length*18" :fill="selectedLine.color">回撤</text><text v-if="!raw && selectedLine" :x="tooltipX+210" :y="64+plotted.length*18" text-anchor="end" class="mono">{{ fmt(selectedDrawdown[hover],true) }}</text>
      </g>
    </svg>
  </div>
</template>
<style scoped>
.result-chart{height:100%;min-height:200px;position:relative;overflow:hidden}.result-chart svg{display:block;width:100%;height:100%;font-size:10.5px;touch-action:none}.compact{height:220px;flex:none}.muted{fill:var(--ink3)}.line-label{font-family:var(--mono);font-size:10.5px}.brush-body{cursor:grab}.brush-handle{cursor:ew-resize}.empty-chart{height:100%;display:grid;place-items:center;color:var(--ink3)}.hover-tip{filter:drop-shadow(0 2px 3px #0000000a)}
</style>
