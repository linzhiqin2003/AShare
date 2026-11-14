<script setup lang="ts">
import { computed, onMounted, onUnmounted, shallowRef, watch } from 'vue'
import { ArrowLeft, Search, ListChecks, Download } from 'lucide-vue-next'
import { api, friendly } from '../api'
import { colors, fmt, unpack } from '../chart'
import type { ChartData, Entry, LibraryData, Line } from '../types'
import type { CompareData, CompareItem } from '../compare'
import { bestValues, distinctLabels, metricColumns, METRIC_GROUPS, DEFAULT_METRICS, settingsSummary, toCsv, volatilityAligned } from '../compare'
import { showValidation } from '../validation'
import ResultChart from './ResultChart.vue'
import { UiCheckbox, vTip } from './ui'

const props = defineProps<{ selected: Entry[] }>()
const emit = defineEmits<{ toggle: [entry: Entry]; close: [] }>()
const MAX = 8
const STORE_KEY = 'ashare.compare.metrics'

const result = shallowRef<CompareData | null>(null)
const error = shallowRef(''), loading = shallowRef(false)
const search = shallowRef(''), candidates = shallowRef<Entry[]>([])
const align = shallowRef(false), range = shallowRef<'1' | '3' | '5' | 'all'>('all')
const chooserOpen = shallowRef(false), metricSearch = shallowRef('')

function savedMetrics(): string[] {
  try { const value = JSON.parse(localStorage.getItem(STORE_KEY) ?? 'null'); if (Array.isArray(value) && value.every(v => typeof v === 'string')) return value } catch { /* 损坏的视图回到默认。 */ }
  return [...DEFAULT_METRICS]
}
const metrics = shallowRef<string[]>(savedMetrics())
function toggleMetric(key: string) { metrics.value = metrics.value.includes(key) ? metrics.value.filter(m => m !== key) : [...metrics.value, key] }
function saveMetrics() { try { localStorage.setItem(STORE_KEY, JSON.stringify(metrics.value)) } catch { /* 无存储权限时只在本页生效。 */ } chooserOpen.value = false }
function resetMetrics() { metrics.value = [...DEFAULT_METRICS] }

// 颜色按勾选顺序固定，左侧列表、曲线、相关矩阵和指标表一致。
const colorOf = computed(() => new Map(props.selected.map((e, i) => [e.id, colors[i % colors.length]])))
const items = computed<CompareItem[]>(() => result.value?.items ?? [])

let controller: AbortController | undefined
async function load() {
  controller?.abort()
  if (!props.selected.length) { result.value = null; return }
  const active = controller = new AbortController()
  loading.value = true
  try {
    const params = new URLSearchParams({ ids: props.selected.map(e => e.id).join(','), validation: String(showValidation.value) })
    const data = await api<CompareData>(`/compare?${params}`, { signal: active.signal })
    if (!active.signal.aborted) { result.value = data; error.value = '' }
  } catch (e) { if (!active.signal.aborted) error.value = friendly(e).message }
  finally { if (!active.signal.aborted) loading.value = false }
}
watch(() => [props.selected.map(e => e.id).join(','), showValidation.value], load, { immediate: true })

let listController: AbortController | undefined, listTimer: ReturnType<typeof setTimeout> | undefined
async function loadCandidates() {
  listController?.abort(); const active = listController = new AbortController()
  try {
    const params = new URLSearchParams({ search: search.value, sort: 'sharpe', descending: 'true', limit: '50' })
    const data = await api<LibraryData>(`/library?${params}`, { signal: active.signal })
    if (!active.signal.aborted) candidates.value = data.items
  } catch (e) { if (!active.signal.aborted) error.value = friendly(e).message }
}
watch(search, () => { clearTimeout(listTimer); listTimer = setTimeout(loadCandidates, 160) }, { immediate: true })
const list = computed(() => {
  const chosen = new Set(props.selected.map(e => e.id))
  const query = search.value.trim().toLowerCase()
  const pinned = props.selected.filter(e => !query || e.name.toLowerCase().includes(query) || e.source.toLowerCase().includes(query))
  return [...pinned, ...candidates.value.filter(e => !chosen.has(e.id))]
})
// 同名因子附上它们之间不同的配置或表达式取值（2026-11-14 用户要求）。
const listNames = computed(() => distinctLabels(list.value))
const names = computed(() => distinctLabels(items.value))
const nameOf = (item: { id: string; name: string }) => names.value.get(item.id) ?? item.name
const isChosen = (entry: Entry) => props.selected.some(e => e.id === entry.id)

const chart = computed<ChartData>(() => {
  const data = result.value
  if (!data?.dates.length) return { dates: [], lines: [] }
  const days = range.value === 'all' ? data.dates.length : Math.min(data.dates.length, Number(range.value) * 242)
  const start = data.dates.length - days
  let lines: Line[] = data.items.filter(item => data.series[item.id]).map(item => ({ id: item.id, name: nameOf(item), color: colorOf.value.get(item.id) ?? colors[0], values: unpack(data.series[item.id]).slice(start), kind: 'long_short' as const }))
  if (align.value) lines = volatilityAligned(lines)
  return { dates: data.dates.slice(start), lines }
})

const columns = computed(() => metricColumns(metrics.value, items.value, showValidation.value))
const best = computed(() => bestValues(columns.value, items.value))
const headerGroups = computed(() => {
  const groups: { label: string; span: number; first: boolean }[] = []
  for (const column of columns.value) {
    const last = groups.at(-1)
    if (last && last.label === column.group) last.span++
    else groups.push({ label: column.group, span: 1, first: true })
  }
  return groups
})
const groupStarts = computed(() => new Set(columns.value.map((c, i) => i === 0 || columns.value[i - 1].group !== c.group ? c.key : '').filter(Boolean)))
const chooserGroups = computed(() => {
  const query = metricSearch.value.trim()
  return METRIC_GROUPS.map(group => ({ ...group, items: group.items.filter(m => !query || m.label.includes(query) || group.label.includes(query)) })).filter(g => g.items.length)
})
const settingsText = computed(() => settingsSummary(items.value.map(i => i.settings)))

function cellColor(column: { heat?: number }, value: number | null) {
  if (!column.heat || value == null || !Number.isFinite(value)) return undefined
  const alpha = Math.min(1, Math.abs(value) / column.heat) * .24
  return value >= 0 ? `rgba(191,58,46,${alpha.toFixed(3)})` : `rgba(30,123,87,${alpha.toFixed(3)})`
}
const matrixColor = (value: number | null | undefined) => value == null ? 'transparent' : value >= 0 ? `rgba(31,79,191,${(Math.abs(value) * .26).toFixed(3)})` : `rgba(161,74,20,${(Math.abs(value) * .22).toFixed(3)})`

function exportCsv() {
  const blob = new Blob(['﻿' + toCsv(columns.value, items.value, names.value)], { type: 'text/csv;charset=utf-8' })
  const link = document.createElement('a')
  link.href = URL.createObjectURL(blob); link.download = 'factor-compare.csv'
  link.click(); URL.revokeObjectURL(link.href)
}

const chooser = shallowRef<HTMLElement | null>(null)
function outside(event: PointerEvent) { const target = event.target as HTMLElement; if (chooserOpen.value && !chooser.value?.contains(target) && !target.closest('.chooser-toggle')) chooserOpen.value = false }
function escape(event: KeyboardEvent) { if (event.key === 'Escape' && chooserOpen.value) { chooserOpen.value = false; event.stopPropagation() } }
onMounted(() => { document.addEventListener('pointerdown', outside); document.addEventListener('keydown', escape, true) })
onUnmounted(() => { document.removeEventListener('pointerdown', outside); document.removeEventListener('keydown', escape, true); controller?.abort(); listController?.abort(); clearTimeout(listTimer) })
</script>
<template>
  <div class="compare-page">
    <div class="compare-left">
      <section class="panel runs-panel">
        <div class="panel-header"><b>因子</b><span class="push muted head-note">已选 {{ selected.length }} 个，最多 {{ MAX }} 个</span></div>
        <label class="find"><Search :size="13" /><input v-model="search" placeholder="按名字或表达式筛选" aria-label="筛选因子"></label>
        <ul class="runs">
          <li v-for="entry in list" :key="entry.id" :class="{ chosen: isChosen(entry) }">
            <label>
              <UiCheckbox :model-value="isChosen(entry)" :disabled="!isChosen(entry) && selected.length >= MAX" :color="colorOf.get(entry.id)" :aria-label="`对比 ${listNames.get(entry.id) ?? entry.name}`" @update:model-value="emit('toggle', entry)" />
              <span class="run-text"><b class="mono">{{ listNames.get(entry.id) ?? entry.name }}</b><em class="mono">{{ entry.source.replaceAll('\n', ' ') }}</em></span>
              <span class="mono run-sharpe">{{ fmt(entry.sharpe) }}</span>
            </label>
          </li>
        </ul>
      </section>
      <section class="panel correlation-panel">
        <div class="panel-header"><b>日盈亏相关</b></div>
        <div v-if="items.length > 1" class="correlation-body">
          <table class="matrix"><thead><tr><th></th><th v-for="item in items" :key="item.id"><i class="swatch" :style="{ background: colorOf.get(item.id) }"></i></th></tr></thead>
            <tbody><tr v-for="(item, i) in items" :key="item.id"><th v-tip="nameOf(item)"><i class="swatch" :style="{ background: colorOf.get(item.id) }"></i><span class="mono">{{ nameOf(item) }}</span></th>
              <td v-for="(other, j) in items" :key="other.id" class="mono" :style="{ background: i === j ? 'transparent' : matrixColor(result?.correlation[i]?.[j]) }">{{ i === j ? '—' : fmt(result?.correlation[i]?.[j]) }}</td></tr></tbody>
          </table>
        </div>
        <div v-else class="empty muted">至少选两个因子</div>
      </section>
    </div>
    <div class="compare-right">
      <div class="compare-bar">
        <button class="text-button back" @click="emit('close')"><ArrowLeft :size="14" />因子库</button>
        <span class="bar-title">对比 {{ selected.length }} 个</span>
        <span v-if="settingsText" class="push muted settings-text">{{ settingsText }}</span>
        <button class="button small" :class="{ push: !settingsText }" :disabled="!items.length" @click="exportCsv"><Download :size="13" />导出</button>
      </div>
      <section class="panel chart-panel" :aria-busy="loading">
        <div class="panel-header"><b>多空累计收益</b>
          <div class="seg push" role="group" aria-label="曲线口径"><button :class="{ on: !align }" :aria-pressed="!align" @click="align = false">原始</button><button :class="{ on: align }" :aria-pressed="align" @click="align = true">按波动对齐</button></div>
          <div class="seg" role="group" aria-label="时间范围"><button v-for="[key, label] in ([['1','1 年'],['3','3 年'],['5','5 年'],['all','全部']] as const)" :key="key" :class="{ on: range === key }" :aria-pressed="range === key" @click="range = key">{{ label }}</button></div>
        </div>
        <ResultChart :data="chart" :selected-id="items[0]?.id" />
      </section>
      <div class="metrics-wrap">
        <section class="panel metrics-panel">
          <div class="panel-header"><b>指标</b><button class="button small push chooser-toggle" :class="{ on: chooserOpen }" @click="chooserOpen = !chooserOpen"><ListChecks :size="13" />选择指标</button></div>
          <div class="table-scroll">
            <table class="data-table metrics-table">
              <thead>
                <tr class="group-head"><th></th><th v-for="group in headerGroups" :key="group.label" :colspan="group.span" class="group-start">{{ group.label }}</th></tr>
                <tr><th>因子</th><th v-for="column in columns" :key="column.key" :class="{ 'group-start': groupStarts.has(column.key), year: column.year }">{{ column.label }}</th></tr>
              </thead>
              <tbody><tr v-for="item in items" :key="item.id"><td v-tip="nameOf(item)" class="mono factor-cell"><i class="swatch" :style="{ background: colorOf.get(item.id) }"></i>{{ nameOf(item) }}</td>
                <td v-for="column in columns" :key="column.key" class="number" :class="{ best: best.get(column.key) === item.id, 'group-start': groupStarts.has(column.key), year: column.year }" :style="{ background: cellColor(column, column.value(item)) }">{{ fmt(column.value(item), column.percent, column.digits) }}</td></tr></tbody>
            </table>
          </div>
        </section>
        <div v-if="chooserOpen" ref="chooser" class="chooser" role="dialog" aria-label="选择指标">
          <label class="find chooser-find"><Search :size="13" /><input v-model="metricSearch" placeholder="搜索指标" aria-label="搜索指标"></label>
          <div class="chooser-groups">
            <section v-for="group in chooserGroups" :key="group.label"><h5>{{ group.label }}</h5>
              <label v-for="metric in group.items" :key="metric.key" :class="{ disabled: metric.validation && !showValidation }"><UiCheckbox :disabled="metric.validation && !showValidation" :model-value="metrics.includes(metric.key) && (!metric.validation || showValidation)" @update:model-value="toggleMetric(metric.key)" />{{ metric.label }}</label>
            </section>
          </div>
          <footer><button class="button" @click="resetMetrics">恢复默认</button><button class="button primary" @click="saveMetrics">存为视图</button></footer>
        </div>
      </div>
    </div>
    <div v-if="error" class="toast" role="alert">{{ error }}</div>
  </div>
</template>
<style scoped>
.compare-page{height:100%;display:grid;grid-template-columns:320px minmax(0,1fr);gap:8px;min-height:0}
.compare-left{display:grid;grid-template-rows:minmax(0,1fr) auto;gap:8px;min-height:0;min-width:0}
.compare-right{display:grid;grid-template-rows:30px minmax(0,1fr) auto;gap:8px;min-height:0;min-width:0}
.panel-header b{font-weight:600}
.head-note{font-weight:400;font-size:11.5px;white-space:nowrap}
.find{margin:8px 10px 4px;height:26px;border:1px solid var(--line);border-radius:3px;display:flex;align-items:center;gap:7px;padding:0 8px;color:var(--ink3);flex:none}
.find input{border:0;padding:0;min-height:20px;outline:0;width:100%;min-width:0;background:transparent;font-size:12px}
.runs{list-style:none;margin:0;padding:0;overflow-y:auto;flex:1;min-height:0}
.runs li{border-bottom:1px solid var(--grid)}
.runs li.chosen{background:var(--selected)}
.runs label{display:grid;grid-template-columns:13px minmax(0,1fr) auto;gap:9px;align-items:center;padding:6px 10px;cursor:pointer}
.run-text{min-width:0}
.run-text b{display:block;font-weight:500;font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.run-text em{display:block;font-style:normal;color:var(--ink3);font-size:10.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.run-sharpe{font-size:12px}
.correlation-panel{max-height:300px}
.correlation-body{padding:8px 12px 10px;overflow:auto}
.matrix{width:100%;border-collapse:separate;border-spacing:0;table-layout:fixed;font-size:11.5px}
.matrix th{font-weight:400;color:var(--ink2);height:30px;text-align:center}
.matrix thead th{height:22px}
.matrix thead th:first-child{width:38%}.matrix tbody th{text-align:left;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.matrix tbody th span{font-size:11.5px}
.matrix td{text-align:center;height:30px;color:var(--ink2)}
.swatch{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:6px;vertical-align:-1px;flex:none}
.matrix thead .swatch{margin:0}
.empty{display:grid;place-items:center;padding:24px;font-size:12px}
.compare-bar{display:flex;align-items:center;gap:14px;background:white;border:1px solid var(--line);border-radius:3px;padding:0 10px;min-width:0}
.back{color:var(--ink2);font-size:12px;white-space:nowrap}
.back:hover{color:var(--ink)}
.bar-title{height:100%;display:flex;align-items:center;font-weight:600;border-bottom:2px solid var(--accent);white-space:nowrap}
.settings-text{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0;font-size:12px;color:var(--ink2)}
.button.small{height:24px;padding:0 10px;font-size:12px;gap:6px}
.chart-panel{min-height:0}
.chart-panel>.result-chart{flex:1;height:0;min-height:0}
.seg{display:flex;border:1px solid var(--line);border-radius:3px;overflow:hidden;font-weight:400}
.seg button{border:0;border-left:1px solid var(--line);background:white;padding:1px 9px;color:var(--ink3);font-size:12px;height:22px;white-space:nowrap}
.seg button:first-child{border-left:0}
.seg button.on{background:var(--selected);color:var(--ink);font-weight:500}
.seg+.seg{margin-left:-4px}
.metrics-wrap{position:relative;min-width:0;min-height:0}
.metrics-panel{max-height:min(320px,38vh)}
.metrics-table{table-layout:auto}
.metrics-table th{text-align:right;font-size:11.5px;height:24px;padding:0 0 0 12px!important}
.metrics-table th:first-child{text-align:left;padding-left:0!important}
.metrics-table .group-head th{text-align:center;height:20px;color:var(--ink2)}
.metrics-table td{height:27px;font-size:12px;padding:0 0 0 12px}
.metrics-table td.number{padding-left:12px}
.metrics-table .group-start{border-left:1px solid var(--line)}
.metrics-table td.group-start,.metrics-table th.group-start{padding-left:12px}
.metrics-table .year{text-align:center!important;padding:0 4px!important;width:37px;font-size:11.5px}
.metrics-table td.year{border-left:1px solid white}
.metrics-table td.year.group-start{border-left:1px solid var(--line)}
.factor-cell{font-size:12.5px;white-space:nowrap;max-width:190px}
.best{font-weight:600;color:var(--ink)}
.chooser{position:absolute;right:0;bottom:calc(100% + 6px);width:min(640px,100%);background:white;border:1px solid var(--line);border-radius:4px;box-shadow:0 12px 32px #00000024;padding:12px 14px;z-index:20}
.chooser-find{margin:0}
.chooser-groups{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));column-gap:22px;row-gap:12px;margin:12px 0}
.chooser-groups h5{font-size:11.5px;font-weight:600;color:var(--ink3);margin:0 0 4px}
.chooser-groups label{display:flex;align-items:center;gap:8px;height:22px;font-size:12px;cursor:pointer}
.chooser-groups label.disabled{color:var(--ink3);cursor:default}
.chooser footer{display:flex;justify-content:flex-end;gap:8px;border-top:1px solid var(--grid);padding-top:10px}
@media (max-width:1100px){.compare-page{grid-template-columns:260px minmax(0,1fr)}}
</style>
