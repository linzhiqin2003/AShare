<script setup lang="ts">
import { computed, onUnmounted, shallowRef } from 'vue'
import { decimatedPath, moveWindow } from '../chart'

// 图下方的时间窗刷子（2026-11-14 从研究结果图抽出，策略回测的净值图共用同一个）：缩略曲线 + 可拖动窗口，
// 拖窗口整体平移、拉两端改起止，左右方向键每次 10 个点。画在宿主图的 SVG 里（一个 <g>），窗口用下标表示，含两端。
const props = withDefaults(defineProps<{
  left: number; top: number; width: number
  dates: string[]; values: (number | null)[]
  start: number; end: number
  /** 验证期起点下标；没有就是 -1。 */
  validationIndex?: number
  /** 缩略图纵轴要包含的值（收益曲线是 0）；净值不强求，传 null。 */
  baseline?: number | null
}>(), { validationIndex: -1, baseline: 0 })
const emit = defineEmits<{ range: [start: number, end: number]; dragging: [value: boolean] }>()
const HEIGHT = 34
const length = computed(() => props.dates.length)
const fx = (i: number) => props.left + i / Math.max(1, length.value - 1) * props.width
const extent = computed(() => {
  let lo = props.baseline ?? Infinity, hi = props.baseline ?? -Infinity
  for (const v of props.values) if (typeof v === 'number' && Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v) }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [-0.01, 0.01]
  const span = hi - lo || 0.02
  return [lo - span * 0.07, hi + span * 0.12]
})
const fy = (v: number) => props.top + 4 + (extent.value[1] - v) / (extent.value[1] - extent.value[0]) * 30
const line = computed(() => decimatedPath(props.values, fx, fy, props.width))
const years = computed(() => props.dates.map((date, index) => ({ date, index }))
  .filter(({ date, index }) => index === 0 || date.slice(0, 4) !== props.dates[index - 1].slice(0, 4))
  .filter((_, i, items) => i % Math.max(1, Math.ceil(items.length / (props.width / 65))) === 0))

const drag = shallowRef<{ kind: 'left' | 'right' | 'body'; x: number; scale: number; start: number; end: number } | null>(null)
let frame = 0, pending: (() => void) | undefined
function flush() { if (frame) cancelAnimationFrame(frame); frame = 0; pending?.(); pending = undefined }
function onMove(event: PointerEvent) {
  const d = drag.value
  if (!d) return
  const delta = Math.round((event.clientX - d.x) * d.scale / props.width * (length.value - 1))
  pending = () => { const next = moveWindow(d.kind, d, delta, length.value); if (next.start !== props.start || next.end !== props.end) emit('range', next.start, next.end) }
  if (!frame) frame = requestAnimationFrame(() => { frame = 0; pending?.(); pending = undefined })
}
function stop() {
  flush(); drag.value = null; emit('dragging', false)
  window.removeEventListener('pointermove', onMove); window.removeEventListener('pointerup', stop); window.removeEventListener('pointercancel', stop)
}
function begin(event: PointerEvent, kind: 'left' | 'right' | 'body') {
  const svg = (event.currentTarget as SVGElement).ownerSVGElement
  const rect = svg?.getBoundingClientRect()
  const box = svg?.viewBox?.baseVal
  // 宿主 SVG 用 viewBox 缩放时，屏幕像素换成画布单位。
  const scale = rect && rect.width > 0 && box && box.width > 0 ? box.width / rect.width : 1
  drag.value = { kind, x: event.clientX, scale, start: props.start, end: props.end }
  try { svg?.setPointerCapture?.(event.pointerId) } catch { /* 合成事件没有活动指针 */ }
  window.addEventListener('pointermove', onMove); window.addEventListener('pointerup', stop); window.addEventListener('pointercancel', stop)
  emit('dragging', true); event.preventDefault()
}
function key(event: KeyboardEvent, edge: 'left' | 'right') {
  const amount = event.key === 'ArrowLeft' ? -10 : event.key === 'ArrowRight' ? 10 : 0
  if (!amount) return
  event.preventDefault()
  const next = moveWindow(edge, { start: props.start, end: props.end }, amount, length.value)
  emit('range', next.start, next.end)
}
onUnmounted(() => { if (drag.value) stop() })
</script>
<template>
  <g class="brush">
    <rect :x="left" :y="top" :width="width" :height="HEIGHT" rx="2" fill="var(--chrome)" stroke="var(--line)" />
    <rect v-if="validationIndex >= 0" :x="fx(validationIndex)" :y="top" :width="left + width - fx(validationIndex)" :height="HEIGHT" fill="var(--shade)" />
    <path v-if="line" class="brush-area" :d="`${line} L${fx(values.length - 1)},${top + HEIGHT} L${fx(0)},${top + HEIGHT} Z`" fill="var(--ink3)" fill-opacity=".16" />
    <path class="brush-line" :d="line" stroke="var(--ink2)" stroke-width="1.1" fill="none" />
    <rect :x="fx(start)" :y="top" :width="fx(end) - fx(start)" :height="HEIGHT" fill="color-mix(in srgb, var(--accent) 3.5%, transparent)" stroke="var(--accent)" class="brush-body" @pointerdown="begin($event, 'body')" />
    <g v-for="edge in (['left', 'right'] as const)" :key="edge" tabindex="0" role="slider" :aria-label="edge === 'left' ? '时间窗起点' : '时间窗终点'" :aria-valuemin="0" :aria-valuemax="length - 1" :aria-valuenow="edge === 'left' ? start : end" class="brush-handle" @keydown="key($event, edge)" @pointerdown="begin($event, edge)">
      <rect :x="fx(edge === 'left' ? start : end) - 8" :y="top - 4" width="16" :height="HEIGHT + 8" fill="transparent" />
      <rect :x="fx(edge === 'left' ? start : end) - 3" :y="top + 9" width="6" height="16" rx="2" fill="var(--panel)" stroke="var(--accent)" />
    </g>
    <g v-for="year in years" :key="year.date"><line :x1="fx(year.index)" :x2="fx(year.index)" :y1="top + HEIGHT" :y2="top + HEIGHT + 3" stroke="var(--line)" /><text :x="fx(year.index)" :y="top + HEIGHT + 14" class="brush-year" text-anchor="middle">{{ year.date.slice(0, 4) }}</text></g>
  </g>
</template>
<style scoped>
.brush-body{cursor:grab}.brush-handle{cursor:ew-resize}.brush-year{fill:var(--ink3)}
</style>
