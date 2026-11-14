<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, reactive, shallowRef, watch } from 'vue'
import { Search, Filter, SlidersHorizontal, ChevronDown, ChevronRight, X, Check, CircleCheck, CircleX, CircleAlert, ArrowUp, ArrowDown, GitCompare, Send } from 'lucide-vue-next'
import { api, friendly, series } from '../api'
import { chartData, configLabel, fmt } from '../chart'
import type { Completion, Entry, LibraryData, PlotPart, Run } from '../types'
import ExpressionEditor from './ExpressionEditor.vue'
import ResultChart from './ResultChart.vue'
import CheckTable from './CheckTable.vue'
import LibraryAnalytics from './LibraryAnalytics.vue'
import SplitPane from './SplitPane.vue'
import SyntaxText from './SyntaxText.vue'
import FactorCompare from './FactorCompare.vue'
import { showValidation, visibleParts } from '../validation'
import { UiCheckbox, UiNumberInput, UiSkeleton, vTip } from './ui'

defineProps<{ completions: Completion[] }>()
const query = reactive({ search: '', status: '', universe: '', neutralize: '', category: '', rebalance: '', metric: 'sharpe', minimum: '', maximum: '', sharpe_min: '', turnover_max: '', t_min: '', sort: 'updated', descending: true, group: 'none', offset: 0 })
const data = shallowRef<LibraryData>({ items: [], total: 0, counts: {}, groups: [], offset: 0, limit: 50 })
const root = shallowRef<HTMLElement | null>(null)
const columnOpen = shallowRef(false)
const tooltip = shallowRef<{source:string;left:number;top:number} | null>(null)
const filterOpen = shallowRef(false), viewOpen = shallowRef(false), groupOpen = shallowRef(false)
// loaded：第一次列表请求有结果（或失败）之前显示骨架；之后翻页、筛选只刷新内容，不再闪回骨架。
const error = shallowRef(''), loading = shallowRef(false), loaded = shallowRef(false)
const detail = shallowRef<Entry | null>(null), detailParts = shallowRef<PlotPart[] | null>(null)
const drawer = shallowRef<HTMLElement | null>(null)
// 详情图只画验证期开关允许的部分（2026-11-14 用户要求默认隐藏验证期）。
const detailChart = computed(() => detail.value?.latest && detailParts.value ? chartData([{ id: detail.value.latest, label: detail.value.name, ordinal: 0 } as Run], [visibleParts(detailParts.value, showValidation.value)], 'returns') : { dates: [], lines: [] })
const detailColumns = computed(() => showValidation.value ? (['is', 'validation'] as const) : (['is'] as const))
// 勾选后进入对比；最多 8 个，跨页、跨筛选保留。
const MAX_COMPARE = 8
const picked = shallowRef<Entry[]>([])
const comparing = shallowRef(false)
function isPicked(entry: Entry) { return picked.value.some(e => e.id === entry.id) }
// 送去回测：把一个条目（表达式与股票池、成交时点、基准）带进回测页的一个 tab，只选中一个因子时可用；
// 回测页一次能跑候选列表展开的多组，但各条目的设置不同，不合并进同一个 tab（接收方见 pages/BacktestPage.vue）。
function sendToBacktest() { if (picked.value.length === 1) location.hash = `#/backtest?entry=${encodeURIComponent(picked.value[0].id)}` }
function togglePick(entry: Entry) { picked.value = isPicked(entry) ? picked.value.filter(e => e.id !== entry.id) : picked.value.length < MAX_COMPARE ? [...picked.value, entry] : picked.value }
const collapsed = reactive(new Set<string>())
const definitions = [
  {key:'status',label:'状态',width:35}, {key:'name',label:'因子',width:150}, {key:'source',label:'表达式',width:310},
  {key:'universe',label:'股票池',width:82}, {key:'neutralize',label:'中性化',width:91}, {key:'decay',label:'衰减',width:42}, {key:'rebalance',label:'调仓',width:42},
  {key:'sharpe',label:'Sharpe',width:63}, {key:'validation',label:'验证',width:63}, {key:'annual_return',label:'年化收益',width:75},
  {key:'drawdown',label:'最大回撤',width:76}, {key:'turnover',label:'换手',width:65}, {key:'holding_days',label:'持仓天数',width:65}, {key:'margin',label:'Margin',width:64},
  {key:'t',label:'t 值',width:49}, {key:'fitness',label:'Fitness',width:65}, {key:'coverage',label:'覆盖率',width:65},
]
// 「验证」默认不显示；验证期开关关闭时这一列和按它排序都不可用。
const defaultColumns = definitions.filter(d => !['fitness','coverage','validation'].includes(d.key)).map(d => d.key)
const columns = shallowRef([...defaultColumns])
const displayed = computed(() => definitions.filter(d => columns.value.includes(d.key) && (d.key !== 'validation' || showValidation.value) && !(query.group === 'expression' && ['name','source'].includes(d.key)) && !(query.group === 'config' && ['universe','neutralize','decay','rebalance'].includes(d.key))))
const SELECT_WIDTH = 28
const columnTotal = computed(() => displayed.value.reduce((sum,d) => sum+d.width,SELECT_WIDTH))
const categories = shallowRef(['自定义','登记因子'])
const filterSections = computed(() => [
  {key:'status',label:'状态',options:[['pass','通过'],['fail','未通过'],['pending','待检验']]},
  {key:'universe',label:'股票池',options:[['TOP1800','TOP1800'],['TOP800','TOP800'],['ALL','全市场']]},
  {key:'neutralize',label:'中性化',options:[['industry','行业'],['cap+industry','市值 + 行业'],['cap','市值'],['none','无']]},
  {key:'rebalance',label:'调仓',options:[...new Set(['1','5','10','20',...Object.keys(data.value.facets?.rebalance ?? {})])].sort((a,b) => Number(a)-Number(b)).map(value => [value,`${value} 天`])},
  {key:'category',label:'类别',options:categories.value.map(c => [c,c])},
])
const asNumber = (value: string) => value === '' || value === '__none__' ? null : Number(value)
const asText = (value: number | null) => value == null ? '' : String(value)
function filterSelected(key: string,value: string) { const selection = query[key as 'status']; return !selection || selection.split(',').includes(value) }
function filterToggle(key: string,value: string,options: string[][]) { const selected = query[key as 'status'] === '__none__' ? [] : query[key as 'status'] ? query[key as 'status'].split(',') : options.map(o => o[0]); const next = selected.includes(value) ? selected.filter(v => v !== value) : [...selected,value]; query[key as 'status'] = next.length === options.length ? '' : next.join(',') || '__none__' }
function showExpression(event: MouseEvent | FocusEvent, source: string) { const rect = (event.currentTarget as HTMLElement).getBoundingClientRect(); tooltip.value = { source, left:Math.max(12,Math.min(rect.left,innerWidth-632)), top:Math.max(42,Math.min(rect.bottom+4,innerHeight-Math.min(340,source.split('\n').length*22+36)-12)) } }
function outside(event: PointerEvent) { if (!(event.target as HTMLElement).closest('.popover-anchor')) { filterOpen.value = false; viewOpen.value = false } }
function escape(event: KeyboardEvent) { if (event.key !== 'Escape') return; tooltip.value = null; if (filterOpen.value || viewOpen.value) { filterOpen.value = false; viewOpen.value = false } else if (detail.value) closeDetail() }
let toastTimer: ReturnType<typeof setTimeout> | undefined
watch(error,value => { clearTimeout(toastTimer); if (value) toastTimer = setTimeout(() => error.value = '',5000) })
onMounted(() => { document.addEventListener('pointerdown',outside); document.addEventListener('keydown',escape) })
let controller: AbortController | undefined, detailController: AbortController | undefined
let timer: ReturnType<typeof setTimeout> | undefined
let detailSequence = 0
const detailCache = new Map<string, Entry>()
function groupKey(entry: Entry) { return query.group === 'rebalance' ? String(entry.settings.rebalance ?? 1) : query.group === 'expression' ? entry.expression_id : JSON.stringify(entry.settings) }
const groups = computed(() => {
  if (query.group === 'none') return [{ key: '', label: '', count: data.value.total, items: data.value.items }]
  const map = new Map<string, { key: string; label: string; count: number; items: Entry[] }>()
  for (const entry of data.value.items) {
    const key = groupKey(entry)
    if (!map.has(key)) map.set(key, { key, label: query.group === 'expression' ? `${entry.name}　${entry.source.replaceAll('\n', ' ')}` : query.group === 'rebalance' ? `调仓 ${entry.settings.rebalance ?? 1} 天` : configLabel(entry.settings), count: data.value.groups.find(g => String(g.key) === key)?.count ?? 0, items: [] })
    map.get(key)!.items.push(entry)
  }
  return [...map.values()]
})
async function load() {
  controller?.abort(); controller = new AbortController()
  const active = controller
  const params = new URLSearchParams(Object.entries(query).filter(([, v]) => v !== '').map(([k, v]) => [k, String(v)]))
  loading.value = true
  try { const result = await api<LibraryData>(`/library?${params}`, { signal: active.signal }); if (!active.signal.aborted) { data.value = result; loaded.value = true; categories.value = [...new Set([...categories.value,...Object.keys(result.facets?.category ?? {})])]; error.value = '' } }
  catch (e) { if (!active.signal.aborted) { error.value = friendly(e).message; loaded.value = true } }
  finally { if (!active.signal.aborted) loading.value = false }
}
watch(() => [query.search, query.status, query.universe, query.neutralize, query.category, query.rebalance, query.metric, query.minimum, query.maximum, query.sharpe_min, query.turnover_max, query.t_min, query.sort, query.descending, query.group], () => { query.offset = 0; collapsed.clear(); clearTimeout(timer); timer = setTimeout(load, 160) }, { immediate: true })
watch(() => query.offset, () => { clearTimeout(timer); timer = setTimeout(load, 0) })
watch(showValidation, show => { if (!show && query.sort === 'validation') { query.sort = 'updated'; query.descending = true } })
function sort(key: string) {
  if (key === 'validation' && !showValidation.value) return
  if (query.sort === key) query.descending = !query.descending
  else { query.sort = key; query.descending = key !== 'name' }
}
function clearFilters() { Object.assign(query, { search: '', status: '', universe: '', neutralize: '', category: '', rebalance: '', metric: 'sharpe', minimum: '', maximum: '', sharpe_min: '', turnover_max: '', t_min: '', offset: 0 }) }
function restore() { columns.value = [...defaultColumns]; query.group = 'none'; query.sort = 'updated'; query.descending = true; collapsed.clear(); viewOpen.value = false }
function toggleColumn(key: string) { if (columns.value.includes(key)) { if (columns.value.length > 1) columns.value = columns.value.filter(c => c !== key) } else columns.value = [...columns.value, key] }
let opener: HTMLElement | null = null
async function openDetail(entry: Entry, event?: Event) {
  detailController?.abort(); detailController = new AbortController()
  const sequence = ++detailSequence
  const wasOpen = !!detail.value
  opener = (event?.currentTarget as HTMLElement | null) ?? (document.activeElement as HTMLElement | null)
  // 已打开时直接换内容，不关再开。
  detail.value = entry; detailParts.value = null
  if (!wasOpen) void nextTick(() => drawer.value?.focus())
  try {
    const key = `${entry.id}:${entry.latest}`
    const value = detailCache.get(key) ?? await api<Entry>(`/library/${entry.id}`, { signal: detailController.signal })
    detailCache.set(key, value)
    if (sequence !== detailSequence) return
    detail.value = value
    if (value.latest) {
      const parts = await series(value.latest, 'returns')
      if (sequence !== detailSequence) return
      detailParts.value = parts
    }
  } catch (e) { if (!detailController.signal.aborted) error.value = friendly(e).message }
}
function closeDetail() {
  const id = detail.value?.id
  detailSequence++; detailController?.abort(); detail.value = null; detailParts.value = null
  // 焦点还给打开它的那一行；行被换页刷掉时找同一个因子的行。
  const target = opener?.isConnected ? opener : id ? root.value?.querySelector<HTMLElement>(`[data-entry="${CSS.escape(id)}"]`) : null
  opener = null
  target?.focus()
}
function statusLabel(status: string) { return status === 'pass' ? '通过' : status === 'fail' ? '未通过' : '待检验' }
function numeric(entry: Entry, key: string) { return fmt(entry[key as keyof Entry] as number | null, ['annual_return', 'turnover', 'drawdown', 'coverage'].includes(key)) }
onUnmounted(() => { document.removeEventListener('pointerdown',outside); document.removeEventListener('keydown',escape); clearTimeout(toastTimer); clearTimeout(timer); controller?.abort(); detailController?.abort() })
</script>
<template>
  <main ref="root" class="library-page">
    <FactorCompare v-if="comparing" :selected="picked" @toggle="togglePick" @close="comparing = false" />
    <div v-show="!comparing" class="library-body">
      <SplitPane storage-key="library-rows" label="调整列表与分析图高度" axis="vertical" :initial=".67" :min-first="280" :min-second="190">
        <template #first><section class="panel library-panel">
          <div class="library-toolbar">
            <label class="search"><Search :size="13" /><input v-model="query.search" placeholder="按名字或表达式搜索" aria-label="搜索因子"></label>
            <div class="popover-anchor"><button class="button toolbar-button" :class="{on:filterOpen}" @click="filterOpen = !filterOpen; viewOpen = false"><Filter :size="13" />筛选</button>
              <div v-if="filterOpen" class="filter-menu" aria-label="筛选因子">
                <section v-for="section in filterSections" :key="section.key"><h5>{{ section.label }}</h5><label v-for="[value,label] in section.options" :key="value" class="filter-option"><UiCheckbox :aria-label="`${section.label} ${label}`" :model-value="filterSelected(section.key,value)" @update:model-value="filterToggle(section.key,value,section.options)" /><span>{{ label }}</span><span class="mono muted push">{{ data.facets?.[section.key]?.[value] ?? 0 }}</span></label></section>
                <section class="metric-filters"><h5>指标</h5><div class="metric-row">Sharpe ≥<UiNumberInput :model-value="asNumber(query.sharpe_min)" :step="0.1" aria-label="最低 Sharpe" @update:model-value="query.sharpe_min = asText($event)" /></div><div class="metric-row">换手 ≤<UiNumberInput :model-value="asNumber(query.turnover_max)" :min="0" :max="1" :step="0.01" aria-label="最高换手率" @update:model-value="query.turnover_max = asText($event)" /></div><div class="metric-row">t 值 ≥<UiNumberInput :model-value="asNumber(query.t_min)" :step="0.1" aria-label="最低 t 值" @update:model-value="query.t_min = asText($event)" /></div></section>
                <footer><button class="text-button" @click="clearFilters">清除</button></footer>
              </div>
            </div>
            <div class="popover-anchor"><button class="button toolbar-button" :class="{on:viewOpen}" @click="viewOpen = !viewOpen; filterOpen = false"><SlidersHorizontal :size="13" />视图</button>
              <div v-if="viewOpen" class="view-menu" role="menu">
                <div class="group-menu-anchor" @mouseenter="groupOpen = true; columnOpen = false" @mouseleave="groupOpen = false"><button class="menu-item" @click="groupOpen = true; columnOpen = false">分组<span class="push muted">{{ {none:'不分组',expression:'表达式',config:'配置',rebalance:'调仓'}[query.group as 'none'] }}</span><ChevronRight :size="13" /></button><div v-if="groupOpen" class="submenu"><button v-for="[key,label] in [['none','不分组'],['expression','表达式'],['config','配置'],['rebalance','调仓']]" :key="key" class="menu-item" @click="query.group = key; viewOpen = false">{{ label }}<Check v-if="query.group === key" :size="12" class="push" /></button></div></div>
                <div class="group-menu-anchor" @mouseenter="columnOpen = true; groupOpen = false" @mouseleave="columnOpen = false"><button class="menu-item" @click="columnOpen = true; groupOpen = false">显示<ChevronRight class="push" :size="13" /></button><div v-if="columnOpen" class="submenu column-menu"><label v-for="column in definitions" :key="column.key" class="menu-item" :class="{disabled: column.key === 'validation' && !showValidation}"><UiCheckbox :disabled="column.key === 'validation' && !showValidation" :model-value="columns.includes(column.key) && (column.key !== 'validation' || showValidation)" @update:model-value="toggleColumn(column.key)" />{{ column.label }}</label></div></div>
                <hr><button class="menu-item" @click="collapsed.clear(); viewOpen = false">全部展开</button><button class="menu-item" @click="restore">恢复默认</button>
              </div>
            </div>
            <span v-if="picked.length" class="picked-count muted push" aria-live="polite">已选 {{ picked.length }} 个</span>
            <button class="button toolbar-button" :class="{push: !picked.length}" :disabled="picked.length < 2" @click="comparing = true; detail && closeDetail()"><GitCompare :size="13" />对比</button>
            <button class="button toolbar-button" :disabled="picked.length !== 1" @click="sendToBacktest"><Send :size="13" />送去回测</button>
          </div>
          <div class="library-scroll" :aria-busy="loading">
            <table class="data-table library-table"><colgroup><col :style="{width:`${SELECT_WIDTH/columnTotal*100}%`}"><col v-for="column in displayed" :key="column.key" :style="{width:`${column.width/columnTotal*100}%`}"></colgroup><thead><tr><th class="select-cell"><span class="sr-only">选择</span></th><th v-for="column in displayed" :key="column.key" v-tip="column.label" :aria-sort="query.sort === column.key ? query.descending ? 'descending' : 'ascending' : 'none'"><button @click="sort(column.key)">{{ column.label }}<component v-if="query.sort === column.key" :is="query.descending ? ArrowDown : ArrowUp" :size="10" /></button></th></tr></thead>
              <tbody v-for="group in groups" :key="group.key"><tr v-if="query.group !== 'none'" class="group-row"><td :colspan="displayed.length + 1"><button @click="collapsed.has(group.key) ? collapsed.delete(group.key) : collapsed.add(group.key)"><component :is="collapsed.has(group.key) ? ChevronRight : ChevronDown" :size="12" /><b v-if="query.group === 'expression'" class="mono">{{ group.items[0].name }}</b><span class="group-title"><SyntaxText v-if="query.group === 'expression'" :source="group.items[0].source" flat /><b v-else>{{ group.label }}</b></span><span class="muted">{{ group.count }} {{ query.group === 'expression' ? '个配置' : '个因子' }}</span></button></td></tr>
                <template v-if="!collapsed.has(group.key)"><tr v-for="entry in group.items" :key="entry.id" class="entry-row" :class="{selected:detail?.id === entry.id}" :data-entry="entry.id" tabindex="0" @click="openDetail(entry,$event)" @keydown.enter.self="openDetail(entry,$event)"><td class="select-cell" @click.stop><UiCheckbox :aria-label="`选择 ${entry.name}`" :model-value="isPicked(entry)" :disabled="!isPicked(entry) && picked.length >= MAX_COMPARE" @update:model-value="togglePick(entry)" /></td><td v-for="column in displayed" :key="column.key" :class="{number:!['status','name','source','universe','neutralize'].includes(column.key)}">
                  <span v-if="column.key === 'status'" class="status-icon" :class="entry.status" v-tip="statusLabel(entry.status)"><component :is="entry.status === 'pass' ? CircleCheck : entry.status === 'fail' ? CircleX : CircleAlert" :size="13" /><span class="sr-only">{{ statusLabel(entry.status) }}</span></span>
                  <div v-else-if="column.key === 'name'" class="factor-name mono" v-tip="entry.name">{{ entry.name }}</div>
                  <div v-else-if="column.key === 'source'" class="expression-cell" tabindex="0" @mouseenter="showExpression($event,entry.source)" @mouseleave="tooltip = null" @focus="showExpression($event,entry.source)" @blur="tooltip = null"><SyntaxText :source="entry.source" flat /></div>
                  <span v-else-if="column.key === 'universe'" class="nowrap muted">{{ entry.settings.universe === 'ALL' ? '全市场' : entry.settings.universe }}</span>
                  <span v-else-if="column.key === 'neutralize'" class="nowrap muted">{{ {none:'无',industry:'行业',cap:'市值','cap+industry':'市值 + 行业'}[String(entry.settings.group_neutralize ?? 'none') as 'none'] }}</span>
                  <template v-else-if="column.key === 'rebalance'">{{ entry.settings.rebalance ?? 1 }}</template><template v-else-if="column.key === 'decay'">{{ entry.settings.decay }}</template><template v-else>{{ numeric(entry,column.key) }}</template>
                </td></tr></template>
              </tbody>
            </table><div v-if="!loaded" class="library-skeleton" aria-busy="true"><UiSkeleton :lines="10" /></div><div v-else-if="!data.total && !loading" class="empty-library">没有符合条件的因子</div>
          </div>
          <div class="pagination"><span class="muted">{{ data.total ? query.offset+1 : 0 }}–{{ Math.min(query.offset+data.items.length,data.total) }} / {{ data.total }}</span><button class="button" :disabled="!query.offset" @click="query.offset = Math.max(0,query.offset-50)">上一页</button><button class="button" :disabled="query.offset+50 >= data.total" @click="query.offset += 50">下一页</button></div>
        </section></template>
        <template #second><LibraryAnalytics :entries="data.items" :pending="!loaded" /></template>
      </SplitPane>
      <Transition name="drawer"><aside v-if="detail" ref="drawer" class="panel detail-drawer" aria-label="因子详情" tabindex="-1">
        <div class="panel-header"><b class="mono">{{ detail.name }}</b><span class="push status" :class="detail.status"><component :is="detail.status === 'pass' ? CircleCheck : detail.status === 'fail' ? CircleX : CircleAlert" :size="13" />{{ statusLabel(detail.status) }}</span><button class="icon-button" aria-label="关闭详情" @click="closeDetail"><X :size="15" /></button></div>
        <div class="drawer-content"><div class="expression-box"><div class="expression-config">{{ configLabel(detail.settings) }}</div><ExpressionEditor :key="detail.id" :model-value="detail.source" :completions="completions" readonly /></div><ResultChart :data="detailChart" compact /><table class="data-table detail-metrics"><thead><tr><th></th><th>样本内</th><th v-if="showValidation">验证</th></tr></thead><tbody><tr v-for="[key,label,percent] in [['sharpe','Sharpe',false],['return','年化收益',true],['maxdrawdown','最大回撤',true],['turnover','日均换手',true],['holding_days','平均持仓天数',false],['margin','Margin (bps)',false],['fitness','Fitness',false],['t','t 值',false],['ic','Rank IC',true]]" :key="String(key)"><td>{{ label }}</td><td v-for="part in detailColumns" :key="part" class="number">{{ fmt(detail.summary?.[part]?.[key as keyof typeof detail.summary.is] as number | null, Boolean(percent)) }}</td></tr></tbody></table><CheckTable v-if="detail.summary" :checks="detail.summary.checks" /></div>
      </aside></Transition>
    </div>
    <div v-if="error" class="toast" role="alert">{{ error }}<button class="icon-button" aria-label="关闭提示" @click="error = ''"><X :size="13" /></button></div>
    <Teleport to="body"><div v-if="tooltip" class="expression-full" :style="{left:`${tooltip.left}px`,top:`${tooltip.top}px`}"><SyntaxText :source="tooltip.source" /></div></Teleport>
  </main>
</template>
<style scoped>
.library-page{height:100%;padding:8px;min-width:0;min-height:0;position:relative;overflow:hidden}.library-body{height:100%;min-height:0}.library-panel{overflow:visible;position:relative}.library-toolbar{height:30px;min-height:30px;display:flex;gap:14px;align-items:center;padding:0 10px;border-bottom:1px solid var(--line);background:var(--chrome);border-radius:3px 3px 0 0}.search{display:flex;gap:6px;align-items:center;width:230px;max-width:28%;height:24px;border:1px solid var(--line);border-radius:3px;padding:0 8px;color:var(--ink3)}.search input{min-width:0;width:100%;border:0;padding:0;outline:0;min-height:20px;background:transparent;font-size:12px}.toolbar-button{height:24px;padding:0 10px;gap:6px}.popover-anchor{position:relative}.filter-menu,.view-menu,.submenu{position:absolute;top:28px;left:-5px;background:var(--panel);border:1px solid var(--line);border-radius:6px;box-shadow:0 12px 32px var(--overlay-shadow);z-index:20;padding:5px;width:232px}.filter-menu{width:min(850px,calc(100vw - 300px));display:grid;grid-template-columns:repeat(5,minmax(0,1fr)) minmax(125px,1.2fr);gap:4px 18px;padding:12px 16px 10px}.filter-menu h5{font-size:11.5px;font-weight:500;color:var(--ink3);margin:0 0 4px}.filter-option{height:26px;display:flex;align-items:center;gap:7px;font-size:12px;white-space:nowrap}.filter-option .muted{font-size:11px}.metric-filters>.metric-row{display:flex;align-items:center;justify-content:space-between;gap:6px;min-height:27px;white-space:nowrap;font-size:11.5px}.metric-filters :deep(.ui-number input){width:44px}.filter-menu footer{grid-column:1/-1;display:flex;justify-content:flex-end;border-top:1px solid var(--grid);padding-top:8px;margin-top:10px;color:var(--ink3)}.menu-item{display:flex;align-items:center;gap:8px;width:100%;padding:5px 8px;text-align:left;min-height:28px;border:0;background:var(--panel);cursor:pointer}.menu-item:hover{background:var(--selected)}.group-menu-anchor{position:relative}.group-menu-anchor:after{content:"";position:absolute;left:100%;top:0;width:6px;height:100%}.submenu{left:calc(100% + 5px);top:-5px;width:150px}.submenu:before{content:'';position:absolute;left:-8px;width:8px;height:100%}.column-menu{max-height:390px;overflow:auto}.view-menu hr{border:0;border-top:1px solid var(--grid);margin:5px 0}.picked-count{font-size:12px;white-space:nowrap}.picked-count+.toolbar-button{margin-left:-4px}.select-cell{text-align:center!important;padding:0!important}.menu-item.disabled{color:var(--ink3);cursor:default}.library-scroll{flex:1;min-height:0;overflow-y:auto;overflow-x:hidden;padding:0 10px}.library-table{table-layout:fixed}.library-table thead{position:sticky;top:0;z-index:2;background:var(--panel)}.library-table th button{display:inline-flex;align-items:center;gap:3px;border:0;background:none;padding:0;color:var(--ink3);white-space:nowrap;font-size:11px}.library-table th,.library-table td{padding:0 5px;height:25px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.library-table td.number{padding-left:2px;font-size:11.5px}.library-table th:first-child,.library-table td:first-child{text-align:center}.library-table tr.entry-row{cursor:pointer}.factor-name,.expression-cell{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.expression-cell{max-width:330px;color:var(--ink2);font-size:11.5px}.factor-name{font-size:12px}.group-row td{background:var(--chrome);height:28px;border-bottom:1px solid var(--line);text-align:left!important}.group-row button{width:100%;display:flex;align-items:center;gap:7px;background:none;border:0;text-align:left;padding:0;font-size:12px}.group-row b{font-weight:600;flex:none}.group-title{flex:1;min-width:0;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}.group-row .muted{font-size:11px;flex:none}.status,.status-icon{display:inline-flex;align-items:center;gap:5px;white-space:nowrap}.pass{color:var(--green)}.fail{color:var(--red)}.pending{color:var(--ink3)}.pagination{display:flex;justify-content:flex-end;align-items:center;gap:10px;padding:5px 10px;border-top:1px solid var(--line);font-size:10.5px;min-height:32px}.pagination button{height:23px;font-size:11px;padding:0 9px}.empty-library{padding:50px;text-align:center;color:var(--ink3)}.detail-drawer{position:absolute;top:8px;right:8px;bottom:8px;z-index:30;width:min(520px,max(45vw,360px));max-width:calc(100% - 16px);box-shadow:-10px 0 30px var(--overlay-shadow);outline:none}.drawer-enter-active,.drawer-leave-active{transition:transform .2s cubic-bezier(.2,.7,.3,1),opacity .2s}.drawer-enter-from,.drawer-leave-to{transform:translateX(calc(100% + 16px));opacity:.6}@media (prefers-reduced-motion:reduce){.drawer-enter-active,.drawer-leave-active{transition:none}}.detail-drawer .panel-header{gap:8px}.detail-drawer .panel-header b{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.drawer-content{overflow-y:auto;overflow-x:hidden;padding:10px;flex:1}.expression-box{border:1px solid var(--line);border-radius:3px;overflow:hidden}.expression-config{padding:6px 9px;border-bottom:1px solid var(--line);color:var(--ink3);font-size:11.5px}.expression-box :deep(.cm-scroller){max-height:200px;min-height:75px}.detail-metrics{margin:8px 0 16px}.detail-metrics th:not(:first-child){text-align:right}.detail-metrics td{height:27px}.expression-full{position:fixed;z-index:80;pointer-events:none;width:620px;max-width:calc(100vw - 24px);max-height:340px;overflow:hidden;padding:10px 12px;background:var(--panel);border:1px solid var(--line);border-radius:5px;box-shadow:0 10px 25px var(--overlay-shadow);white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;line-height:1.8}.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)}
.library-skeleton{padding:14px 5px}
</style>
