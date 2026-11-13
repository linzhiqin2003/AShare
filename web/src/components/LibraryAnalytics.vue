<script setup lang="ts">
import { computed, onMounted, onUnmounted, shallowRef, watch } from 'vue'
import { api, friendly } from '../api'
import type { Entry } from '../types'
import SplitPane from './SplitPane.vue'
import { niceAxis } from '../chart'
import { placeLabels } from '../scatter'

const props = defineProps<{ entries: Entry[] }>()
const matrixElement = shallowRef<HTMLElement | null>(null)
const matrixWidth = shallowRef(700)
let observer: ResizeObserver | undefined
onMounted(() => { observer = new ResizeObserver(entries => matrixWidth.value = entries[0].contentRect.width); if (matrixElement.value) observer.observe(matrixElement.value) })
const shown = computed(() => props.entries.filter(e => e.latest).slice(0, Math.max(3,Math.min(12,Math.floor((matrixWidth.value - 125)/37)))))
const matrix = shallowRef<(number | null)[][]>([])
const error = shallowRef('')
let controller: AbortController | undefined
watch(() => shown.value.map(e => `${e.id}:${e.latest}`).join(','), async () => {
  controller?.abort(); controller = new AbortController()
  const active = controller
  matrix.value = []; error.value = ''
  if (!shown.value.length) return
  try {
    const result = await api<{ matrix: (number | null)[][] }>(`/library/correlation?${new URLSearchParams({ ids: shown.value.map(e => e.id).join(',') })}`, { signal: active.signal })
    if (!active.signal.aborted) matrix.value = result.matrix
  } catch (e) { if (!active.signal.aborted) error.value = friendly(e).message }
}, { immediate: true })
onUnmounted(() => { controller?.abort(); observer?.disconnect() })
const points = computed(() => props.entries.filter(e => e.sharpe !== null && e.turnover !== null && e.turnover > 0))
const axis = computed(() => niceAxis([[0,2,...points.value.map(e => e.sharpe!)]],4))
const ticks = computed(() => axis.value.ticks)
const minimumTurnover = computed(() => points.value.some(e => e.turnover! < .03) ? .01 : .03)
const turnoverTicks = computed(() => (minimumTurnover.value < .03 ? [.01,.03,.05,.1,.2,.5,1] : [.03,.05,.1,.2,.5,1]))
const px = (turnover: number) => 40 + Math.min(1, Math.max(0, Math.log(turnover/minimumTurnover.value) / Math.log(1/minimumTurnover.value))) * 388
const py = (sharpe: number) => 207 - (sharpe - axis.value.min) / (axis.value.max - axis.value.min) * 175
const scatterElement = shallowRef<SVGElement | null>(null)
const textSizes = shallowRef(new Map<string, { width: number; height: number; baseline: number }>())
const namedPoints = computed(() => points.value.slice(0, 2))
let disposed = false
function measureNames() {
  if (disposed) return
  const sizes = new Map<string, { width: number; height: number; baseline: number }>()
  scatterElement.value?.querySelectorAll<SVGGraphicsElement>('.measure-name').forEach(element => {
    const box = element.getBBox()
    sizes.set(element.dataset.id!, { width: box.width + 2, height: box.height + 2, baseline: 1 - box.y })
  })
  textSizes.value = sizes
}
const labelSize = (entry: Entry) => textSizes.value.get(entry.id) ?? { width: entry.name.length * 9 + 2, height: 14, baseline: 11 }
const namePositions = computed(() => placeLabels(
  points.value.map(e => ({ id: e.id, x: px(e.turnover!), y: py(e.sharpe!), radius: 5 })),
  namedPoints.value.map(e => ({ id: e.id, ...labelSize(e) })),
  { x: 40, y: 22, width: 388, height: 185 },
))
watch(() => namedPoints.value.map(e => `${e.id}:${e.name}`).join('|'), measureNames, { flush: 'post' })
onMounted(() => { measureNames(); void document.fonts?.ready.then(measureNames) })
onUnmounted(() => { disposed = true })
const color = (value: number | null) => value === null ? 'transparent' : value >= 0 ? `rgba(59,104,161,${Math.abs(value) * .20})` : `rgba(178,116,83,${Math.abs(value) * .18})`
</script>
<template>
  <SplitPane class="library-analytics" storage-key="library-analysis" label="调整相关矩阵与散点图宽度" :initial=".69" :min-first="350" :min-second="280"><template #first>
    <section ref="matrixElement" class="panel correlation-panel">
      <div class="panel-header"><b>相关矩阵</b></div>
      <div v-if="error" class="analytics-error" role="alert">{{ error }}</div>
      <div v-if="shown.length" class="matrix-scroll">
        <table class="matrix-table"><thead><tr><th></th><th v-for="(entry, i) in shown" :key="entry.id" :title="entry.name">{{ i + 1 }}</th></tr></thead><tbody>
          <tr v-for="(entry, i) in shown" :key="entry.id"><th :title="entry.name"><span>{{ i + 1 }}</span>{{ entry.name }}</th><td v-for="(other, j) in shown" :key="other.id" :style="{ background: i === j ? 'transparent' : color(matrix[i]?.[j] ?? null) }">{{ i === j ? '·' : matrix[i]?.[j]?.toFixed(2) ?? '' }}</td></tr>
        </tbody></table>
      </div>
      <div v-else class="analytics-empty">暂无结果</div>
    </section>
    </template><template #second><section class="panel scatter-panel">
      <div class="panel-header"><b>Sharpe × 换手</b></div>
      <svg v-if="points.length" ref="scatterElement" viewBox="0 0 450 250" role="img" aria-label="当前页因子的样本内 Sharpe 与日均换手率">
        <g visibility="hidden" aria-hidden="true"><text v-for="entry in namedPoints" :key="entry.id" :data-id="entry.id" class="point-name measure-name">{{ entry.name }}</text></g>
        <g v-for="tick in ticks" :key="tick"><line x1="40" x2="428" :y1="py(tick)" :y2="py(tick)" /><text x="32" :y="py(tick) + 3" text-anchor="end">{{ tick.toFixed(1) }}</text></g>
        <g v-for="tick in turnoverTicks" :key="tick"><line :x1="px(tick)" :x2="px(tick)" y1="32" y2="207" /><text :x="px(tick)" y="225" text-anchor="middle">{{ (tick * 100).toFixed(tick < .01 ? 1 : 0) }}%</text></g>
        <line x1="40" x2="428" :y1="py(1)" :y2="py(1)" class="threshold" />
        <g v-for="(entry,i) in points" :key="entry.id" class="scatter-point">
          <title>{{ entry.name }} · Sharpe {{ entry.sharpe?.toFixed(2) }} · 换手 {{ (entry.turnover! * 100).toFixed(1) }}%</title>
          <circle :cx="px(entry.turnover!)" :cy="py(entry.sharpe!)" r="5" :fill="entry.status === 'pass' ? 'var(--green)' : 'var(--ink3)'" fill-opacity=".7" />
          <text :x="px(entry.turnover!)" :y="py(entry.sharpe!)+2.5" text-anchor="middle" class="point-number">{{ i+1 }}</text>
          <text v-if="namePositions.has(entry.id)" :x="namePositions.get(entry.id)!.x + 1" :y="namePositions.get(entry.id)!.y + labelSize(entry).baseline" class="point-name placed-name">{{ entry.name }}</text>
        </g>
      </svg>
      <div v-else class="analytics-empty">暂无结果</div>
    </section>
  </template></SplitPane>
</template>
<style scoped>
.library-analytics{min-height:0}.correlation-panel,.scatter-panel{min-height:0;overflow:hidden}.matrix-scroll{overflow-y:auto;overflow-x:hidden;padding:8px 12px;flex:1}.matrix-table{width:100%;table-layout:fixed;border-collapse:separate;border-spacing:2px;font-family:var(--mono);font-size:10.5px}.matrix-table th{color:var(--ink3);font-weight:400;text-align:center;height:19px}.matrix-table th:first-child{width:118px}.matrix-table tbody th{text-overflow:ellipsis;overflow:hidden;white-space:nowrap;text-align:left;color:var(--ink2);padding-right:6px}.matrix-table tbody th span{display:inline-block;width:20px;color:var(--ink3)}.matrix-table td{height:19px;text-align:center;color:var(--ink2)}.scatter-panel svg{flex:1;min-height:0;width:100%}.scatter-panel svg line{stroke:var(--grid);stroke-width:1}.scatter-panel svg .threshold{stroke:var(--ink3);stroke-dasharray:3 4;opacity:.5}.scatter-panel svg text{font-family:var(--mono);font-size:10px;fill:var(--ink3)}.scatter-panel svg .point-number{pointer-events:none;fill:white;font-size:7px}.scatter-panel svg .point-name{fill:var(--ink2);font-size:9px}.analytics-empty{display:grid;place-items:center;flex:1;color:var(--ink3)}.analytics-error{padding:8px;color:var(--red)}
</style>
