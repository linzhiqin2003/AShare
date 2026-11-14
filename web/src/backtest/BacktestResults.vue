<script setup lang="ts">
import { computed, shallowRef, watch } from 'vue'
import { LoaderCircle } from 'lucide-vue-next'
import BacktestSide from './BacktestSide.vue'
import BacktestTables from './BacktestTables.vue'
import NavChart from './NavChart.vue'
import { UiSkeleton } from '../components/ui'
import { result, state } from './api'
import { friendly } from '../api'
import { batchProgress, selectedGroup } from './useBacktests'
import { showValidation as validation } from '../validation'
import type { BacktestDocument, BacktestRun, Part } from './types'

// 研究页策略回测视图的结果区：净值、分年与归因等表格、右侧统计、成交约束与持仓（原回测页的中栏和右栏）。
// 选中组由结果区顶栏切换；这里只按选中组取结果文档。
const props = defineProps<{ run: BacktestRun; running: boolean }>()
const emit = defineEmits<{ message: [text: string] }>()
const part = shallowRef<Part>('is')
const doc = shallowRef<BacktestDocument | null>(null), loading = shallowRef(false)
const effectivePart = computed<Part>(() => validation.value ? part.value : 'is')
const group = computed(() => selectedGroup(props.run))
const status = computed(() => group.value?.state ?? null)
const ENDED: Record<string, string> = { failed: '回测失败，请稍后重试', cancelled: '已取消', evicted: '结果已清理，重新运行即可' }
// 已完成各组同一窗口的文档，按「身份:窗口」存；净值图把它们叠在选中组下面。
const groupDocs = shallowRef<Record<string, BacktestDocument>>({})
const chartGroups = computed(() => props.run.groups.map(g => ({
  id: g.id, label: g.label, status: g.state?.status ?? null, document: groupDocs.value[`${g.id}:${effectivePart.value}`] ?? null,
})))
// 首屏骨架：已完成的回测第一次取结果；之后的刷新保留旧内容。
const firstLoad = computed(() => !props.running && loading.value && !doc.value)

let request = 0, docGroup = ''
async function load() {
  const id = ++request, selected = group.value
  void loadOthers(id)
  if (!selected || selected.state?.status !== 'complete') {
    // 重跑期间保留上一次的图；在同一批里切到还没算完或失败的组时不保留别组的图。
    if (!props.running || props.run.groups.some(g => g.id === docGroup)) doc.value = null
    return
  }
  loading.value = true
  try {
    const value = await result(selected.id, effectivePart.value)
    if (id === request) { doc.value = value; docGroup = selected.id }
  } catch (e) {
    if (id !== request) return
    // 结果可能刚被容量清理：刷新状态，改为提示重跑，而不是弹错误。
    try { selected.state = await state(selected.id) } catch { /* 保留原状态 */ }
    if (selected.state?.status === 'complete') emit('message', friendly(e).message)
    else doc.value = null
  }
  finally { if (id === request) loading.value = false }
}
/** 其余已完成组的文档（结果按身份缓存，切换选中组时不再下载）；读不到的组只是不画。 */
async function loadOthers(id: number) {
  const part = effectivePart.value
  const done = props.run.groups.filter(g => g.state?.status === 'complete')
  if (done.length < 2) return
  const loaded = await Promise.all(done.map(g => result(g.id, part).then(d => [`${g.id}:${part}`, d] as const, () => null)))
  if (id !== request) return
  groupDocs.value = Object.fromEntries(loaded.filter(x => x !== null))
}
watch(() => [group.value?.id, props.run.groups.map(g => g.state?.status).join(), effectivePart.value] as const, load, { immediate: true })
</script>

<template>
  <div class="backtest-results">
    <div class="middle">
      <div class="chart-wrap">
        <NavChart :document="doc" :part="effectivePart" :show-validation="validation" :loading="loading" :groups="chartGroups" :selected="group?.id ?? null" hide-groups @part="part = $event" />
        <div v-if="firstLoad" class="overlay chart-skeleton" aria-busy="true"><UiSkeleton shape="block" /></div>
        <div v-else-if="running && !doc" class="overlay muted"><LoaderCircle :size="14" class="spinning" />回测中 {{ batchProgress(run.groups) }}</div>
        <div v-else-if="status && status.status in ENDED && !doc" class="overlay muted">{{ ENDED[status.status] }}</div>
      </div>
      <BacktestTables :document="doc" :backtest="doc ? group?.id ?? null : null" :part="effectivePart" />
    </div>
    <BacktestSide v-if="!firstLoad" :document="doc" />
    <section v-else class="panel side-skeleton" aria-busy="true"><div class="panel-header">统计</div><div class="skeleton-body"><UiSkeleton :lines="10" /></div></section>
  </div>
</template>

<style scoped>
.backtest-results{height:100%;min-height:0;display:grid;grid-template-columns:minmax(0,1fr) clamp(290px,24.5vw,392px);gap:8px}
.middle{display:grid;grid-template-rows:minmax(0,1fr) clamp(220px,36%,336px);grid-template-columns:minmax(0,1fr);gap:8px;min-height:0;min-width:0}
.chart-wrap{position:relative;min-height:0;display:flex}
.chart-wrap>:first-child{flex:1}
.overlay{position:absolute;inset:30px 0 0;display:flex;align-items:center;justify-content:center;gap:8px;font-size:12px;pointer-events:none}
.chart-skeleton{align-items:stretch;padding:12px}.side-skeleton .skeleton-body{padding:12px 10px}
</style>
