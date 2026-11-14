<script setup lang="ts">
import { computed, onMounted, onUnmounted, shallowRef, watch } from 'vue'
import { Search } from 'lucide-vue-next'
import { api, friendly } from '../api'
import type { Completion } from '../types'
import FieldDetailView from '../components/data/FieldDetail.vue'
import { bytes, integer, linePath, matches, partition, percent, stamp, type Dataset, type FieldDetail, type FieldList, type Overview } from '../datastatus'

const props = defineProps<{ completions: Completion[] }>()
type Tab = 'fields' | 'indexes' | 'tables' | 'history' | 'verify' | 'operators'
const TAB_LABELS: Record<Tab, string> = { fields: '字段', indexes: '指数', tables: '原始表', history: '更新记录', verify: '校验', operators: '算子' }

const overview = shallowRef<Overview | null>(null), fields = shallowRef<FieldList | null>(null), error = shallowRef('')
const datasetKey = shallowRef(read('ashare.data.dataset', 'stock')), tab = shallowRef<Tab>('fields'), query = shallowRef('')
const selected = shallowRef(read('ashare.data.field', 'close')), detail = shallowRef<FieldDetail | null>(null)
function read(key: string, fallback: string) { try { return localStorage.getItem(key) ?? fallback } catch { return fallback } }
function remember(key: string, value: string) { try { localStorage.setItem(key, value) } catch { /* 只是记住上次的选择。 */ } }

const dataset = computed<Dataset | null>(() => overview.value?.datasets.find(d => d.key === datasetKey.value) ?? overview.value?.datasets[0] ?? null)
const tabs = computed<Tab[]>(() => {
  const d = dataset.value
  if (!d) return []
  const list: Tab[] = d.key === 'stock' ? ['fields'] : d.key === 'index' ? ['indexes'] : []
  list.push('tables', 'history')
  if (d.verify) list.push('verify')
  if (d.key === 'stock') list.push('operators')
  return list
})
const searchable = computed(() => tab.value === 'fields' || tab.value === 'operators' || tab.value === 'indexes')
const visibleFields = computed(() => (fields.value?.fields ?? []).filter(f => matches(f, query.value)))
const operators = computed(() => props.completions.filter(c => c.kind === 'operator' && (!query.value.trim() || `${c.label} ${c.detail}`.toLowerCase().includes(query.value.trim().toLowerCase()))))
const indexes = computed(() => (dataset.value?.indexes ?? []).filter(i => !query.value.trim() || `${i.symbol} ${i.name}`.toLowerCase().includes(query.value.trim().toLowerCase())))
const showDetail = computed(() => dataset.value?.key === 'stock')
const verifyIssues = computed(() => dataset.value?.verify?.checks.filter(c => (c.failed ?? 0) > 0).length ?? 0)

function summary(d: Dataset) {
  if (d.key === 'index') return d.indexes?.length ? `${d.indexes.length} 个指数` : ''
  if (!d.panel) return '原始库，未构建面板'
  const unit = d.key === 'stock' ? '只' : d.key === 'futures' ? '个合约' : '只'
  return `${d.panel.symbols ?? '—'} ${unit} × ${d.panel.days} 个交易日${d.key === 'stock' ? `，${d.panel.fields} 个字段` : ''}`
}
function latestDate(d: Dataset) { return d.panel?.end ?? d.indexes?.reduce((m, i) => i.end > m ? i.end : m, '') ?? '' }
function earliestDate(d: Dataset) { return d.panel?.start ?? d.indexes?.reduce((m, i) => !m || i.start < m ? i.start : m, '') ?? '' }

function chooseDataset(key: string) { datasetKey.value = key; remember('ashare.data.dataset', key); query.value = '' }
watch(tabs, value => { if (value.length && !value.includes(tab.value)) tab.value = value[0] })
watch(() => dataset.value?.key, () => { if (tabs.value.length) tab.value = tabs.value[0] })
function chooseField(name: string) { selected.value = name; remember('ashare.data.field', name) }

let poll: ReturnType<typeof setTimeout> | undefined, alive = true, detailController: AbortController | undefined
async function loadOverview() { try { overview.value = await api<Overview>('/data/overview'); error.value = '' } catch (e) { error.value = friendly(e).message } }
async function loadFields() {
  clearTimeout(poll)
  try {
    const result = await api<FieldList>('/data/fields')
    if (!alive) return
    fields.value = result
    if (result.state === 'pending') poll = setTimeout(loadFields, 1500)
    else if (result.state === 'ready' && !result.fields.some(f => f.name === selected.value) && result.fields.length) chooseField(result.fields[0].name)
  } catch (e) { if (alive) error.value = friendly(e).message }
}
async function loadDetail() {
  if (fields.value?.state !== 'ready' || !selected.value) return
  detailController?.abort(); const controller = detailController = new AbortController()
  try { const result = await api<FieldDetail>(`/data/fields/${encodeURIComponent(selected.value)}`, { signal: controller.signal }); if (!controller.signal.aborted) detail.value = result }
  catch (e) { if (!controller.signal.aborted) { detail.value = null; error.value = friendly(e).message } }
}
watch([selected, () => fields.value?.state], loadDetail)
onMounted(() => { void loadOverview(); void loadFields() })
onUnmounted(() => { alive = false; clearTimeout(poll); detailController?.abort() })

function datasetList(names: string[]) { return names.length > 3 ? `${names.slice(0, 3).join('、')} 等 ${names.length} 个` : names.join('、') }
function spark(values: (number | null)[]) { return linePath(values, 110, 18, 2) }
function rowKey(event: KeyboardEvent, name: string) { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); chooseField(name) } }
</script>
<template>
  <main class="data-page" :class="{ 'no-detail': !showDetail }">
    <section class="panel datasets" aria-label="数据集">
      <div class="panel-header">数据集</div>
      <template v-if="overview">
        <button v-for="item in overview.datasets" :key="item.key" class="dataset" :class="{ on: dataset?.key === item.key }" :aria-pressed="dataset?.key === item.key" @click="chooseDataset(item.key)"><b>{{ item.name }}</b><em v-if="dataset?.key === item.key && summary(item)">{{ summary(item) }}</em></button>
        <p v-if="!overview.datasets.length" class="empty">本地还没有数据</p>
      </template>
      <div v-else class="skeleton-list"><i v-for="n in 4" :key="n" /></div>
      <div v-if="dataset" class="dataset-meta">
        <h5>更新</h5>
        <div class="kv">
          <span>数据截至</span><b>{{ latestDate(dataset) || '—' }}</b>
          <span>起始</span><b>{{ earliestDate(dataset) || '—' }}</b>
          <template v-if="dataset.panel"><span>面板版本</span><b>{{ dataset.panel.version.slice(0, 12) }}</b><span>占用空间</span><b>{{ bytes(dataset.panel.size) }}</b></template>
          <template v-if="dataset.panel?.built_at"><span>构建于</span><b>{{ stamp(dataset.panel.built_at) }}</b></template>
          <template v-if="dataset.verify"><span>校验</span><b :class="{ stale: !dataset.verify.current }">{{ !dataset.verify.current ? '对应旧版本' : dataset.verify.verdict ? (dataset.verify.verdict === 'pass' ? '通过' : '未通过') : verifyIssues ? `${verifyIssues} 项有差异` : '全部通过' }}</b></template>
        </div>
      </div>
    </section>
    <section class="panel center">
      <div class="panel-header tabs-header">
        <button v-for="item in tabs" :key="item" class="panel-tab" :class="{ active: tab === item }" @click="tab = item">{{ TAB_LABELS[item] }}</button>
        <label v-if="searchable" class="find push"><Search :size="13" /><input v-model="query" :placeholder="tab === 'operators' ? '搜索算子' : tab === 'indexes' ? '搜索指数' : '搜索字段'" :aria-label="tab === 'operators' ? '搜索算子' : '搜索字段'"></label>
      </div>
      <div v-if="error" class="error-banner" role="alert">{{ error }}</div>
      <div class="scroll">
        <template v-if="tab === 'fields'">
        <table class="grid-table fields-table">
          <thead><tr><th class="c-name">字段</th><th class="c-label">名称</th><th class="c-cat">类别</th><th class="c-type">类型</th><th class="r c-start">起始</th><th class="r c-cov">最新覆盖率</th><th class="c-bar"></th><th class="spark-head c-spark">历史覆盖</th><th class="r c-refs">被因子引用</th></tr></thead>
          <tbody v-if="fields?.state === 'ready'">
            <tr v-for="field in visibleFields" :key="field.name" :class="{ selected: selected === field.name }" tabindex="0" :aria-selected="selected === field.name" @click="chooseField(field.name)" @keydown="rowKey($event, field.name)">
              <td class="mono">{{ field.name }}</td><td>{{ field.label }}</td><td class="dim2 c-cat">{{ field.category }}</td><td class="mono dim c-type">{{ field.dtype }}</td><td class="mono r">{{ field.start ?? '—' }}</td><td class="mono r">{{ percent(field.latest) }}</td>
              <td class="bar-cell c-bar"><svg width="70" height="8" aria-hidden="true"><rect width="70" height="8" rx="1.5" class="track" /><rect :width="70 * (field.latest ?? 0)" height="8" rx="1.5" class="fill" /></svg></td>
              <td class="spark-cell"><svg width="110" height="18" aria-hidden="true"><line x1="0" x2="110" y1="16" y2="16" class="base" /><path :d="spark(field.spark)" class="spark" /></svg></td>
              <td class="mono r">{{ field.references || '' }}</td>
            </tr>
          </tbody>
          <tbody v-else-if="fields?.state === 'pending' || !fields" class="skeleton"><tr v-for="n in 14" :key="n"><td v-for="c in 9" :key="c"><i /></td></tr></tbody>
        </table>
        <p v-if="fields?.state === 'pending'" class="state-line">正在统计各字段覆盖，首次约需十秒</p>
        <p v-if="fields?.state === 'failed'" class="state-line">字段统计失败，请稍后刷新重试</p>
        <p v-if="fields?.state === 'missing'" class="state-line">还没有构建股票面板</p>
        </template>

        <table v-else-if="tab === 'indexes'" class="grid-table">
          <thead><tr><th>代码</th><th>名称</th><th class="r">起始</th><th class="r">截至</th></tr></thead>
          <tbody><tr v-for="item in indexes" :key="item.symbol"><td class="mono">{{ item.symbol }}</td><td>{{ item.name }}</td><td class="mono r">{{ item.start }}</td><td class="mono r">{{ item.end }}</td></tr></tbody>
        </table>

        <table v-else-if="tab === 'tables'" class="grid-table raw-table">
          <thead><tr><th>原始表</th><th class="r">分区</th><th class="r">行数</th><th class="r">最早分区</th><th class="r">最新分区</th><th class="r">最近同步</th><th class="r">未完成</th></tr></thead>
          <tbody><tr v-for="table in dataset?.tables ?? []" :key="table.name"><td class="mono">{{ table.name }}</td><td class="mono r">{{ integer(table.partitions) }}</td><td class="mono r">{{ integer(table.rows) }}</td><td class="mono r">{{ partition(table.first) }}</td><td class="mono r">{{ partition(table.last) }}</td><td class="mono r">{{ stamp(table.fetched) }}</td><td class="mono r" :class="{ warn: table.failed }">{{ table.failed || '' }}</td></tr></tbody>
        </table>

        <table v-else-if="tab === 'history'" class="grid-table history-table">
          <thead><tr><th>日期</th><th>时间</th><th>数据集</th><th class="r">分区</th><th class="r">行数</th><th class="r">请求</th></tr></thead>
          <tbody><tr v-for="record in dataset?.history ?? []" :key="record.date"><td class="mono">{{ record.date }}</td><td class="mono dim">{{ stamp(record.first, false) }}–{{ stamp(record.last, false) }}</td><td class="datasets-cell">{{ datasetList(record.datasets) }}</td><td class="mono r">{{ integer(record.partitions) }}</td><td class="mono r">{{ integer(record.rows) }}</td><td class="mono r">{{ integer(record.calls) }}</td></tr></tbody>
        </table>

        <template v-else-if="tab === 'verify' && dataset?.verify">
          <p class="verify-line">{{ stamp(dataset.verify.generated_at) }} 生成<template v-if="!dataset.verify.current">，对应的不是当前面板版本</template></p>
          <table class="grid-table verify-table">
            <thead><tr><th>检查</th><th class="r">检查数</th><th class="r">差异</th><th class="r">{{ dataset.verify.verdict ? '结论' : '通过率' }}</th><th>说明</th></tr></thead>
            <tbody><tr v-for="check in dataset.verify.checks" :key="check.name"><td class="mono">{{ check.name }}</td><td class="mono r">{{ integer(check.checked) }}</td><td class="mono r" :class="{ warn: (check.failed ?? 0) > 0 }">{{ check.failed ? integer(check.failed) : '' }}</td><td class="mono r">{{ dataset.verify.verdict ? check.verdict ?? '—' : percent(check.pass_rate, 2) }}</td><td class="note-cell">{{ check.note }}</td></tr></tbody>
          </table>
        </template>

        <table v-else-if="tab === 'operators'" class="grid-table">
          <thead><tr><th>算子</th><th>签名</th><th>说明</th></tr></thead>
          <tbody><tr v-for="item in operators" :key="item.label"><td class="mono">{{ item.label }}</td><td class="mono dim">{{ item.signature }}</td><td>{{ item.detail }}</td></tr></tbody>
        </table>
      </div>
    </section>
    <section v-if="showDetail" class="panel detail">
      <div class="panel-header"><span class="mono detail-name">{{ detail?.name ?? selected }}</span><span v-if="detail" class="push detail-sub">{{ detail.label }} · {{ detail.category }}</span></div>
      <FieldDetailView v-if="detail && detail.name === selected" :field="detail" />
      <div v-else class="skeleton-list"><i v-for="n in 6" :key="n" /></div>
    </section>
  </main>
</template>
<style scoped>
.data-page{height:100%;padding:8px;display:grid;grid-template-columns:232px minmax(0,1fr) 412px;gap:8px;min-height:0}
.data-page.no-detail{grid-template-columns:232px minmax(0,1fr)}
.datasets .panel-header,.detail .panel-header{font-weight:600}
.dataset{display:block;width:100%;text-align:left;padding:9px 10px;border:0;border-bottom:1px solid var(--grid);background:var(--panel)}
.dataset.on{background:var(--selected)}
.dataset b{display:block;font-weight:500}
.dataset em{font-style:normal;color:var(--ink3);font-size:11.5px}
.dataset:hover b{color:var(--ink)}
.dataset-meta{padding:8px 10px}
h5{font-size:11.5px;font-weight:600;color:var(--ink3);margin:2px 0 5px}
.kv{display:grid;grid-template-columns:1fr auto}
.kv>span{color:var(--ink3);height:24px;display:flex;align-items:center;border-bottom:1px solid var(--grid)}
.kv>b{font-weight:500;height:24px;display:flex;align-items:center;justify-content:flex-end;border-bottom:1px solid var(--grid);font-family:var(--mono);font-size:12px}
.kv>b.stale{color:var(--warn);font-family:inherit}
.empty{padding:10px;color:var(--ink3)}
.tabs-header{gap:14px}
.find{height:22px;width:220px;border:1px solid var(--line);border-radius:3px;display:flex;align-items:center;gap:7px;padding:0 8px;color:var(--ink3);background:var(--panel);font-weight:400}
.find input{border:0;padding:0;min-height:0;height:20px;flex:1;min-width:0;background:transparent;font-size:12px;outline:none}
.scroll{flex:1;min-height:0;overflow:auto;padding:0 10px}
.grid-table{width:100%;border-collapse:collapse;font-size:12.5px;table-layout:fixed}
.grid-table th{font-weight:500;color:var(--ink3);text-align:left;height:24px;border-bottom:1px solid var(--line);font-size:11.5px;white-space:nowrap;position:sticky;top:0;background:var(--panel);z-index:1}
.grid-table td{height:21px;border-bottom:1px solid var(--grid);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.grid-table td.mono{font-size:12px}
.grid-table .r{text-align:right}
.grid-table tr:last-child td{border-bottom:0}
.dim{color:var(--ink3)}.dim2{color:var(--ink2)}.warn{color:var(--warn)}
.fields-table tbody tr{cursor:pointer}
.fields-table tbody tr:hover td{background:var(--chrome)}
.fields-table tbody tr.selected td{background:var(--selected)}
.fields-table tr:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
.c-name{width:auto}.c-label{width:120px}.c-cat{width:76px}.c-type{width:76px}.c-start{width:92px}.c-cov{width:84px}.c-bar{width:84px}.c-spark{width:134px}.c-refs{width:84px}
.bar-cell{padding-left:10px}.bar-cell svg,.spark-cell svg{display:block}
.track{fill:var(--grid)}.fill{fill:var(--ink2)}
.spark-head,.spark-cell{padding-left:14px}
.base{stroke:var(--line)}.spark{fill:none;stroke:var(--ink2);stroke-width:1.1}
.skeleton td i,.skeleton-list i{display:block;height:8px;border-radius:2px;background:var(--grid)}
.skeleton-list{padding:12px 10px;display:grid;gap:14px}
.state-line{color:var(--ink3);padding:10px 0;margin:0}
.history-table th:first-child{width:96px}.history-table th:nth-child(2){width:100px}.history-table th:nth-child(n+4){width:80px}
.verify-line{color:var(--ink3);margin:8px 0 2px;font-size:12px}
.verify-table th:first-child{width:30%}.verify-table th:nth-child(2),.verify-table th:nth-child(3),.verify-table th:nth-child(4){width:12%}
.raw-table th:first-child{width:20%}.raw-table th:nth-child(2){width:9%}.raw-table th:nth-child(3){width:14%}.raw-table th:nth-child(4),.raw-table th:nth-child(5){width:14%}.raw-table th:nth-child(6){width:20%}
.grid-table td.r,.grid-table th.r{padding-left:10px}
.note-cell{color:var(--ink2);padding:3px 0 3px 14px;white-space:normal!important;line-height:1.45}
.verify-table th:last-child{padding-left:14px}
.detail-name{color:var(--ink)}
.detail-sub{font-weight:400;color:var(--ink3)}
.detail{min-width:0}
@media (max-width:1500px){.data-page{grid-template-columns:220px minmax(0,1fr) 360px}.data-page.no-detail{grid-template-columns:220px minmax(0,1fr)}.c-type{display:none}}
@media (max-width:1340px){.c-bar{display:none}.c-cat{display:none}.c-label{width:100px}.c-refs{width:64px}}
</style>
