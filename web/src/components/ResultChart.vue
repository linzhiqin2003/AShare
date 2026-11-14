<script setup lang="ts">
import { computed, onMounted, onUnmounted, shallowRef, watch } from 'vue'
import { curves, decimatedPath, fmt, niceAxis } from '../chart'
import TimeBrush from './TimeBrush.vue'
import type { ChartData, Line } from '../types'

const props = withDefaults(defineProps<{ data: ChartData; compound?: boolean; relative?: boolean; raw?: boolean; compact?: boolean; selectedId?: string; drawdownLine?: Line | null }>(), { compound: false, relative: false, raw: false, compact: false })
const emit = defineEmits<{ date:[date:string] }>()
let ignoreClick = false
function resetClick() { ignoreClick = false }
function selectDay(event:MouseEvent) { if (ignoreClick) { ignoreClick = false; return } if (props.compact || (event.target as Element).closest('.brush')) return; const rect = (event.currentTarget as SVGElement).getBoundingClientRect(); const px = (event.clientX-rect.left)*width.value/rect.width; if (px < left || px > left+plotWidth.value) return; const index = start.value+Math.round((px-left)/plotWidth.value*(end.value-start.value)); if (props.data.dates[index]) emit('date',props.data.dates[index]) }
const root = shallowRef<HTMLDivElement | null>(null)
const width = shallowRef(1000), height = shallowRef(450)
const start = shallowRef(0), end = shallowRef(0), hover = shallowRef(-1)
const hoverY = shallowRef(0)
// 刷子拖动期间不显示悬停读数（刷子自己处理拖动，见 TimeBrush）。
const drag = shallowRef(false)
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
const baseline = computed(() => props.relative ? props.data.lines.find(line => line.benchmark)?.values : undefined)
const plotted = computed(() => props.relative && !baseline.value ? [] : props.data.lines.filter(line => !props.relative || !line.benchmark).map(line => ({ ...line, ...(props.raw ? { cumulative: line.values.slice(start.value, end.value + 1), drawdown: [] } : curves(line.values, props.compound, start.value, end.value, baseline.value)) })))
const selectedLine = computed(() => props.drawdownLine ?? props.data.lines.find(line => !line.benchmark && line.id.startsWith(props.selectedId ?? '')) ?? props.data.lines.find(line => !line.benchmark))
const selectedDrawdown = computed(() => selectedLine.value ? curves(selectedLine.value.values, props.compound, start.value, end.value, baseline.value).drawdown : [])
const mainAxis = computed(() => niceAxis(plotted.value.map(v => v.cumulative),props.compact ? 2 : 4,!props.raw,true))
const mainRange = computed(() => [mainAxis.value.min,mainAxis.value.max])
const drawAxis = computed(() => niceAxis([selectedDrawdown.value],2,true))
const drawRange = computed(() => [Math.min(-.01,drawAxis.value.min),0])
const y = (value: number) => 18 + (mainRange.value[1] - value) / (mainRange.value[1] - mainRange.value[0]) * (mainHeight.value - 22)
const dy = (value: number) => drawTop.value + (drawRange.value[1] - value) / (drawRange.value[1] - drawRange.value[0]) * drawHeight.value
const path = (values: number[], xf: (i: number) => number, yf: (n: number) => number) => decimatedPath(values, xf, yf, plotWidth.value)
const mainTicks = computed(() => mainAxis.value.ticks)
const drawTicks = computed(() => props.compact ? [] : drawAxis.value.ticks.filter(tick => tick <= 0))
const drawLabel = computed(() => props.relative ? `超额回撤 · ${selectedLine.value?.kind === 'long' ? '第 1 组' : '多空'}` : '回撤')
const validationIndex = computed(() => props.data.dates.findIndex(d => d >= '20250101'))
const validationX = computed(() => validationIndex.value < 0 ? width.value : Math.min(left + plotWidth.value, Math.max(left, x(validationIndex.value - start.value))))
const labels = computed(() => {
  const items: { index: number; text: string }[] = []
  for (let i = start.value; i <= end.value; i++) {
    if (i === start.value || props.data.dates[i]?.slice(0, 4) !== props.data.dates[i - 1]?.slice(0, 4)) items.push({ index: i - start.value, text: props.data.dates[i]?.slice(0, 4) ?? '' })
  }
  return items.filter((_, i) => i % Math.max(1, Math.ceil(items.length / (width.value / 90))) === 0)
})
const mini = computed(() => selectedLine.value ? curves(selectedLine.value.values, props.compound, 0, length.value - 1, baseline.value).cumulative : [])
const tooltipX = computed(() => Math.max(left, Math.min(x(hover.value) + 12, width.value - 235)))
const labelText = (line:Line) => props.relative ? line.name.replace(/ · (第 1 组|多空)$/, '') : line.name
const labelFits = (line:Line) => [...labelText(line)].reduce((sum,c) => sum + (c.charCodeAt(0)>255 ? 11 : 6.5),0) < right.value-12
// 右侧标签按 y 排开，避免重叠；曲线末端的名字和悬停时的数值共用同一套排法。
function stack(points: { id: string; y: number }[], top = 0) {
  const items = [...points].sort((a, b) => a.y - b.y)
  if (items.length) items[0].y = Math.max(items[0].y, top)
  const gap = Math.min(13, (mainHeight.value - 18) / Math.max(1, items.length - 1))
  for (let i = 1; i < items.length; i++) items[i].y = Math.max(items[i].y, items[i - 1].y + gap)
  // 从下边界往回排，避免多空与多头同时显示时把上方标签整体挤出图外。
  if (items.length) items[items.length - 1].y = Math.min(items[items.length - 1].y, mainHeight.value)
  for (let i = items.length - 2; i >= 0; i--) items[i].y = Math.min(items[i].y, items[i + 1].y - gap)
  return new Map(items.map(item => [item.id, item.y]))
}
const lineLabels = computed(() => stack(plotted.value.filter(l => labelFits(l) && Number.isFinite(l.cumulative.at(-1))).map(l => ({ id: l.id, y: y(l.cumulative.at(-1)!) + 4 }))))
// 悬停读数（2026-11-14 用户：浮动卡片挡住曲线，改成右侧无底色、按线条着色的文字）：
// 每条线的数值画在右侧标签栏、对齐到这条线在悬停日的高度；抽屉里的小图没有标签栏，仍用卡片。
const hovering = computed(() => hover.value >= 0 && !drag.value)
const hoverValues = computed(() => {
  if (!hovering.value || props.compact) return []
  const live = plotted.value.filter(l => Number.isFinite(l.cumulative[hover.value]))
  // 最上面留给日期。
  const at = stack(live.map(l => ({ id: l.id, y: y(l.cumulative[hover.value]) + 4 })), 30)
  return live.map(l => ({ id: l.id, color: l.color, y: at.get(l.id)!, text: fmt(l.cumulative[hover.value], !props.raw) }))
})
const hoverDate = computed(() => (props.data.dates[start.value + hover.value] ?? '').replace(/(\d{4})(\d{2})(\d{2})/, '$1-$2-$3'))
function move(event: PointerEvent) {
  const rect = (event.currentTarget as SVGElement).getBoundingClientRect()
  const pixel = (event.clientX - rect.left) * width.value / rect.width
  const nextY = Math.max(18, Math.min(mainHeight.value, (event.clientY - rect.top) * height.value / rect.height))
  pendingMove = () => { hoverY.value = nextY
  if (!drag.value) hover.value = Math.max(0, Math.min(end.value - start.value, Math.round((pixel - left) / plotWidth.value * (end.value - start.value))))
  }
  if (!frame) frame = requestAnimationFrame(() => { pendingMove?.(); pendingMove = undefined; frame = 0 })
}
function dragging(value: boolean) { drag.value = value; if (value) { ignoreClick = true; hover.value = -1 } }
function setRange(a: number, b: number) { start.value = a; end.value = b }
watch(() => props.data.dates.join(','), () => { start.value = 0; end.value = Math.max(0, length.value - 1); hover.value = -1 }, { immediate: true })
onMounted(() => { observer = new ResizeObserver(entries => { width.value = entries[0].contentRect.width; height.value = Math.max(120, entries[0].contentRect.height) }); if (root.value) observer.observe(root.value) })
onUnmounted(() => { observer?.disconnect(); cancelAnimationFrame(frame) })
</script>
<template>
  <div ref="root" class="result-chart" :class="{ compact }">
    <div v-if="!plotted.length" class="empty-chart">{{ relative && data.lines.length ? baseline ? '请选择参数组' : '基准数据暂不可用' : '运行后查看结果' }}</div>
    <svg v-else :viewBox="`0 0 ${width} ${height}`" role="img" @pointerdown.capture="resetClick" @click="selectDay" aria-label="研究结果曲线与可拖动时间窗" @pointermove="move" @pointerleave="!drag && (hover = -1)">
      <rect :x="validationX" y="8" :width="Math.max(0, left + plotWidth - validationX)" :height="axisY - 15" fill="var(--shade)" />
      <g v-if="validationX < left + plotWidth && validationX > left" class="muted">
        <line :x1="validationX" :x2="validationX" y1="8" :y2="axisY - 7" stroke="var(--ink3)" stroke-dasharray="3 3" />
        <text :x="validationX - 5" y="15" text-anchor="end">样本内</text><text :x="validationX + 5" y="15">验证</text>
      </g>
      <g v-for="tick in mainTicks" :key="tick" class="muted main-tick"><line :x1="left" :x2="left + plotWidth" :y1="y(tick)" :y2="y(tick)" stroke="var(--grid)" /><text :x="left - 7" :y="y(tick) + 3" text-anchor="end">{{ fmt(tick, !raw, raw ? 2 : 0) }}</text></g>
      <g v-for="line in plotted" :key="line.id">
        <path class="return-line" :data-kind="line.benchmark ? 'benchmark' : line.kind" :d="path(line.cumulative, x, y)" fill="none" :stroke="line.color" :stroke-width="line.benchmark ? 1.2 : compact ? 1.3 : 1.7" :stroke-dasharray="line.benchmark ? '5 4' : line.kind === 'long' ? '7 3' : undefined" />
        <text v-if="!compact && !hoverValues.length && labelFits(line) && Number.isFinite(line.cumulative.at(-1))" :x="left + plotWidth + 6" :y="lineLabels.get(line.id)" :fill="line.color" class="line-label">{{ labelText(line) }}</text>
      </g>
      <g v-if="!raw">
        <g v-for="tick in drawTicks" :key="tick" class="muted draw-tick"><line :x1="left" :x2="left + plotWidth" :y1="dy(tick)" :y2="dy(tick)" stroke="var(--grid)"/><text :x="left - 7" :y="dy(tick) + 3" text-anchor="end">{{ fmt(tick, true, 0) }}</text></g>
        <path v-if="selectedDrawdown.length" :d="`${path(selectedDrawdown,x,dy)} L${x(selectedDrawdown.length-1)},${dy(0)} L${x(0)},${dy(0)} Z`" :fill="selectedLine?.color" fill-opacity=".11" />
        <path :d="path(selectedDrawdown, x, dy)" fill="none" :stroke="selectedLine?.color" stroke-width="1" />
        <text :x="left + 5" :y="drawTop + drawHeight - 4" fill="var(--ink2)">{{ drawLabel }}</text>
      </g>
      <text v-for="label in labels" :key="label.index" :x="x(label.index)" :y="axisY + 3" class="muted" text-anchor="middle">{{ label.text }}</text>
      <TimeBrush v-if="!compact" :left="left" :top="brushY" :width="plotWidth" :dates="data.dates" :values="mini" :start="start" :end="end" :validation-index="validationIndex" :baseline="0" @range="setRange" @dragging="dragging" />
      <g v-if="hover >= 0 && !drag" class="hover-tip" pointer-events="none">
        <line :x1="x(hover)" :x2="x(hover)" y1="18" :y2="axisY - 8" stroke="var(--ink3)" />
        <line :x1="left" :x2="left + plotWidth" :y1="hoverY" :y2="hoverY" stroke="var(--ink3)" stroke-dasharray="3 3" />
        <g v-for="line in plotted" :key="line.id"><circle v-if="Number.isFinite(line.cumulative[hover])" :cx="x(hover)" :cy="y(line.cumulative[hover])" r="3" :fill="line.color" stroke="var(--panel)" /></g>
        <template v-if="compact">
          <rect :x="tooltipX" y="26" width="220" :height="(raw ? 30 : 52) + plotted.length * 18" rx="3" fill="var(--panel)" stroke="var(--line)" />
          <text :x="tooltipX + 10" y="44" class="mono" fill="var(--ink)">{{ data.dates[start + hover]?.replace(/(\d{4})(\d{2})(\d{2})/, '$1-$2-$3') }}{{ relative ? ' · 超额' : '' }}</text>
          <g v-for="(line, i) in plotted" :key="line.id"><text :x="tooltipX + 10" :y="64 + i * 18" :fill="line.color">{{ line.name }}</text><text :x="tooltipX + 210" :y="64 + i * 18" text-anchor="end" class="mono" fill="var(--ink)">{{ fmt(line.cumulative[hover], !raw) }}</text></g>
          <text v-if="!raw && selectedLine" :x="tooltipX+10" :y="64+plotted.length*18" :fill="selectedLine.color">{{ drawLabel }}</text><text v-if="!raw && selectedLine" :x="tooltipX+210" :y="64+plotted.length*18" text-anchor="end" class="mono" fill="var(--ink)">{{ fmt(selectedDrawdown[hover],true) }}</text>
        </template>
        <template v-else>
          <text :x="left + plotWidth + 6" y="15" class="mono hover-date" fill="var(--ink3)">{{ hoverDate }}{{ relative ? ' · 超额' : '' }}</text>
          <text v-for="item in hoverValues" :key="item.id" :x="left + plotWidth + 6" :y="item.y" :fill="item.color" class="mono hover-value">{{ item.text }}</text>
          <text v-if="!raw && selectedLine && Number.isFinite(selectedDrawdown[hover])" :x="left + plotWidth + 6" :y="Math.min(dy(selectedDrawdown[hover]) + 4, drawTop + drawHeight)" :fill="selectedLine.color" class="mono hover-value">{{ fmt(selectedDrawdown[hover], true) }}</text>
        </template>
      </g>
    </svg>
  </div>
</template>
<style scoped>
.result-chart{height:100%;min-height:200px;position:relative;overflow:hidden}.result-chart svg{display:block;width:100%;height:100%;font-size:10.5px;touch-action:none}.compact{height:220px;flex:none}.muted{fill:var(--ink3)}.line-label{font-family:inherit;font-size:10.5px;letter-spacing:normal;word-spacing:normal}.empty-chart{height:100%;display:grid;place-items:center;color:var(--ink3)}.hover-tip rect{filter:drop-shadow(0 2px 3px var(--tip-shadow))}.hover-value,.hover-date{font-size:10.5px}
</style>
