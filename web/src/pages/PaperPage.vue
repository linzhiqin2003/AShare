<script setup lang="ts">
import { computed, onMounted, onUnmounted, shallowRef, watch } from 'vue'
import { CircleCheck, CircleMinus, Plus } from 'lucide-vue-next'
import { api, friendly } from '../api'
import { niceAxis } from '../chart'
import SegmentedControl from '../components/SegmentedControl.vue'
import { UiSkeleton } from '../components/ui'
import PaperNewAccount from './PaperNewAccount.vue'
import type { PaperAccountItem, PaperCurve, PaperFill, PaperSummary, Range } from './paper'
import { accountLine, count, money, monthTicks, percent, points, pollDelay, rebalanceLabel, relative, signed, slice, statusLabel, tone, universeLabel } from './paper'

const accounts = shallowRef<PaperAccountItem[] | null>(null)
const selected = shallowRef<string | null>(null)
const summary = shallowRef<PaperSummary | null>(null)
const curve = shallowRef<PaperCurve | null>(null)
const error = shallowRef('')
const range = shallowRef<Range>('6m')
const tab = shallowRef<'positions' | 'orders' | 'history'>('positions')
const history = shallowRef<{ items: PaperFill[]; total: number } | null>(null)
const ranges = [{ value: '1m', label: '1 月' }, { value: '6m', label: '6 月' }, { value: '1y', label: '1 年' }, { value: 'all', label: '全部' }]
const sideLabel = { buy: '买入', sell: '卖出' }
const path = (id: string) => `/paper/accounts/${encodeURIComponent(id)}`
const creating = shallowRef(false)
const current = computed(() => accounts.value?.find(a => a.id === selected.value) ?? null)
// 新建、还没开始运行的账户没有账本：只显示策略，定时刷新列表等它开始。
const waiting = computed(() => current.value?.started === false)

let timer: ReturnType<typeof setTimeout> | undefined
let generation = 0
async function refresh() {
  clearTimeout(timer)
  const id = selected.value, mine = ++generation
  if (id && waiting.value && !document.hidden) {
    await loadAccounts()
    if (mine === generation && !document.hidden) timer = setTimeout(refresh, 60_000)
    return
  }
  if (id && !document.hidden) {
    try {
      const next = await api<PaperSummary>(path(id))
      if (mine !== generation) return
      const reload = !curve.value || summary.value?.id !== next.id || summary.value?.day !== next.day || summary.value?.settled !== next.settled
      summary.value = next; error.value = ''
      if (reload) { const c = await api<PaperCurve>(`${path(id)}/curve`); if (mine === generation) curve.value = c }
      if (tab.value === 'history' && reload) void loadHistory(true)
    } catch (e) { if (mine === generation) error.value = friendly(e).message }
  }
  if (mine === generation && !document.hidden) timer = setTimeout(refresh, pollDelay(summary.value))
}
async function loadAccounts() {
  try {
    const data = await api<{ items: PaperAccountItem[] }>('/paper/accounts')
    accounts.value = data.items
    let remembered: string | null = null
    try { remembered = localStorage.getItem('paper-account') } catch { /* 无痕窗口等 */ }
    if (!data.items.some(a => a.id === selected.value)) choose(data.items.find(a => a.id === remembered)?.id ?? data.items[0]?.id ?? null)
  } catch (e) { error.value = friendly(e).message; accounts.value = accounts.value ?? [] }
}
function created(item: PaperAccountItem) {
  creating.value = false
  accounts.value = [...(accounts.value ?? []).filter(a => a.id !== item.id), item].sort((a, b) => a.id.localeCompare(b.id))
  choose(item.id)
}
function choose(id: string | null) {
  if (id === selected.value) return
  selected.value = id; summary.value = null; curve.value = null; history.value = null; error.value = ''
  try { if (id) localStorage.setItem('paper-account', id) } catch { /* 忽略 */ }
  void refresh()
}
async function loadHistory(reset = false) {
  const id = selected.value
  if (!id) return
  const offset = reset ? 0 : history.value?.items.length ?? 0
  try {
    const page = await api<{ items: PaperFill[]; total: number }>(`${path(id)}/fills?offset=${offset}&limit=200`)
    if (id !== selected.value) return
    history.value = { total: page.total, items: reset ? page.items : [...(history.value?.items ?? []), ...page.items] }
  } catch (e) { error.value = friendly(e).message }
}
watch(tab, value => { if (value === 'history' && !history.value) void loadHistory(true) })
function visibility() { if (!document.hidden) void refresh(); else clearTimeout(timer) }
onMounted(() => { document.addEventListener('visibilitychange', visibility); void loadAccounts() })
onUnmounted(() => { generation++; clearTimeout(timer); document.removeEventListener('visibilitychange', visibility); observer?.disconnect() })

// 权益曲线
const chartBox = shallowRef<HTMLDivElement | null>(null)
const width = shallowRef(880), height = 218
let observer: ResizeObserver | undefined
watch(chartBox, element => { observer?.disconnect(); if (element) { observer = new ResizeObserver(() => { width.value = Math.max(320, element.clientWidth) }); observer.observe(element); width.value = Math.max(320, element.clientWidth) } })
const left = 46, right = 10, top = 8, bottom = 200
const visible = computed(() => slice(points(curve.value, summary.value), range.value))
const lines = computed(() => ({ account: relative(visible.value.map(p => p.nav)), benchmark: relative(visible.value.map(p => p.close)) }))
const axis = computed(() => niceAxis([lines.value.account, curve.value?.benchmark ? lines.value.benchmark : []], 4, true))
const px = (i: number) => left + (visible.value.length > 1 ? i / (visible.value.length - 1) : 0) * (width.value - left - right)
const py = (v: number) => bottom - (v - axis.value.min) / ((axis.value.max - axis.value.min) || 1) * (bottom - top)
function line(values: number[]) {
  let d = '', pen = false
  values.forEach((v, i) => { if (Number.isFinite(v)) { d += `${pen ? 'L' : 'M'}${px(i).toFixed(1)},${py(v).toFixed(1)}`; pen = true } else pen = false })
  return d
}
// 相邻月份标签太近（如区间从月末开始）时丢掉前一个，避免重叠。
const ticks = computed(() => {
  const all = monthTicks(visible.value.map(p => p.date), Math.max(3, Math.floor(width.value / 100)))
  return all.filter((t, i) => i === all.length - 1 || px(all[i + 1].index) - px(t.index) >= 72)
})
const anchor = (x: number) => x < left + 40 ? 'start' : x > width.value - right - 40 ? 'end' : 'middle'
const hover = shallowRef(-1)
function move(event: PointerEvent) {
  const rect = (event.currentTarget as SVGElement).getBoundingClientRect()
  const n = visible.value.length
  hover.value = n ? Math.max(0, Math.min(n - 1, Math.round(((event.clientX - rect.left) - left) / (width.value - left - right) * (n - 1)))) : -1
}

const positions = computed(() => summary.value?.positions ?? [])
const latestTrades = computed(() => summary.value?.fills_day === summary.value?.day ? '当日交易' : '上次交易')
const phaseText = computed(() => !summary.value ? '' : summary.value.settled ? '已结算' : summary.value.phase)
</script>

<template>
  <main class="paper-page">
    <div v-if="accounts && !accounts.length" class="empty panel">
      <b>还没有模拟账户</b>
      <span>从回测建立，或用命令行：<code>python -m ashare.paper account create &lt;名称&gt; --expression '&lt;表达式&gt;'</code></span>
      <button type="button" class="button create" @click="creating = true"><Plus :size="14" />从回测新建账户</button>
      <span v-if="error" class="error-text" role="alert">{{ error }}</span>
    </div>
    <template v-else-if="accounts">
      <section class="panel account-panel">
        <div class="panel-header">账户</div>
        <button v-for="item in accounts" :key="item.id" class="account" :class="{ on: item.id === selected }" :aria-pressed="item.id === selected" @click="choose(item.id)">
          <b>{{ item.strategy?.name && item.strategy.name !== item.id ? item.strategy.name : item.id }}</b><em>{{ accountLine(item) }}</em>
        </button>
        <div class="create-row"><button type="button" class="button create" @click="creating = true"><Plus :size="14" />从回测新建账户</button></div>
        <div v-if="summary || current?.strategy" class="side-body">
          <h5>运行</h5>
          <div class="kv">
            <template v-if="summary">
              <span>状态</span>
              <b class="state" :class="{ running: summary.running }"><CircleCheck v-if="summary.running" :size="14" /><CircleMinus v-else :size="14" />{{ summary.running ? '运行中' : '未运行' }}</b>
              <span>时段</span><b class="plain">{{ phaseText }}</b>
              <span>行情更新至</span><b>{{ summary.quote_time ?? '—' }}</b>
            </template>
            <template v-if="current?.strategy">
              <span>股票池</span><b class="plain">{{ universeLabel[current.strategy.universe] ?? current.strategy.universe }} · 前 {{ current.strategy.top_n }} 只</b>
              <span>调仓</span><b class="plain">{{ rebalanceLabel[current.strategy.rebalance] ?? current.strategy.rebalance }}</b>
              <span>上次调仓</span><b>{{ current.plan?.last_execution_day ?? current.plan?.pending_execution_day ?? '—' }}</b>
              <span>下次信号日</span><b>{{ current.plan?.next_signal_day ?? '—' }}</b>
            </template>
          </div>
          <template v-if="summary">
          <h5>集中度</h5>
          <div class="kv">
            <span>最大单只</span><b>{{ percent(summary.concentration.max_weight) }}</b>
            <span>最大行业<template v-if="summary.concentration.max_industry">（{{ summary.concentration.max_industry }}）</template></span><b>{{ percent(summary.concentration.max_industry_weight, 1) }}</b>
            <span>现金占比</span><b>{{ percent(summary.concentration.cash_weight) }}</b>
          </div>
          <h5>风控</h5>
          <div class="kv">
            <span>未成交委托</span><b>{{ summary.risk.open_orders }}</b>
            <template v-for="[reason, n] in summary.risk.rejected" :key="reason"><span>拒单 · {{ reason }}</span><b class="warn">{{ n }}</b></template>
            <template v-if="!summary.risk.rejected.length"><span>当日拒单</span><b>0</b></template>
          </div>
          </template>
          <p class="note">只做纸面记账，不向任何券商发委托。</p>
        </div>
      </section>

      <div class="center">
        <section class="panel">
          <div class="panel-header">
            权益<span class="legend"><i class="ink" />账户</span><span v-if="curve?.benchmark" class="legend"><i class="accent" />{{ curve.benchmark.name }}</span>
            <span class="push"><SegmentedControl :model-value="range" :options="ranges" label="区间" @update:model-value="range = $event as Range" /></span>
          </div>
          <div class="chart-body">
            <div v-if="summary" class="hero">
              <div class="equity mono">¥ {{ money(summary.equity, 0) }}</div>
              <div><span>当日</span><b :class="tone(summary.day_pnl)">{{ signed(summary.day_pnl) }}（{{ percent(summary.day_return, 2, true) }}）</b></div>
              <div><span>本月</span><b>{{ percent(summary.month_return, 2, true) }}</b></div>
              <div><span>今年</span><b>{{ percent(summary.year_return, 2, true) }}</b></div>
              <div><span>持仓</span><b>{{ positions.length }} 只</b></div>
            </div>
            <div v-else-if="!waiting" class="hero-skeleton" aria-busy="true"><UiSkeleton shape="block" height="40px" /></div>
            <div ref="chartBox" class="chart">
              <svg v-if="visible.length > 1" :viewBox="`0 0 ${width} ${height}`" :width="width" :height="height" role="img" aria-label="权益曲线" @pointermove="move" @pointerleave="hover = -1">
                <g v-for="t in axis.ticks" :key="t"><line :x1="left" :x2="width - right" :y1="py(t)" :y2="py(t)" :class="t === 0 ? 'zero' : 'grid'" /><text :x="left - 6" :y="py(t) + 3.5" text-anchor="end">{{ percent(t, 0) }}</text></g>
                <text v-for="t in ticks" :key="t.index" :x="px(t.index)" :y="height - 4" :text-anchor="anchor(px(t.index))">{{ t.label }}</text>
                <line :x1="left" :x2="width - right" :y1="bottom" :y2="bottom" class="zero" />
                <path v-if="curve?.benchmark" :d="line(lines.benchmark)" class="bench" />
                <path :d="line(lines.account)" class="own" />
                <template v-if="hover >= 0">
                  <line :x1="px(hover)" :x2="px(hover)" :y1="top" :y2="bottom" class="cursor" />
                  <text :x="left + 6" :y="top + 10" class="readout">{{ visible[hover].date }}　账户 {{ percent(lines.account[hover], 2, true) }}<tspan v-if="curve?.benchmark">　{{ curve.benchmark.name }} {{ percent(lines.benchmark[hover], 2, true) }}</tspan></text>
                </template>
              </svg>
              <div v-else-if="summary" class="chart-empty">结算一个交易日后显示曲线</div>
              <div v-else-if="waiting" class="chart-empty">下一个交易日开盘后开始运行</div>
              <UiSkeleton v-else-if="selected && !error" shape="block" />
            </div>
          </div>
        </section>

        <section class="panel table-panel">
          <div class="panel-header">
            <button class="panel-tab" :class="{ active: tab === 'positions' }" @click="tab = 'positions'">持仓</button>
            <button class="panel-tab" :class="{ active: tab === 'orders' }" @click="tab = 'orders'">当日委托</button>
            <button class="panel-tab" :class="{ active: tab === 'history' }" @click="tab = 'history'">历史成交</button>
            <span v-if="summary" class="push muted">{{ summary.day }} {{ summary.settled ? '收盘' : '盘中' }}</span>
          </div>
          <div class="scroll">
            <table v-if="tab === 'positions'" class="grid-table">
              <thead><tr><th class="l">代码</th><th class="l">名称</th><th class="l">行业</th><th>股数</th><th class="opt">可卖</th><th class="opt">成本</th><th>现价</th><th>市值（万元）</th><th>权重</th><th>浮动盈亏</th></tr></thead>
              <tbody>
                <tr v-for="p in positions" :key="p.symbol"><td class="mono">{{ p.symbol }}</td><td class="l">{{ p.name }}</td><td class="l dim">{{ p.industry ?? '—' }}</td><td>{{ count(p.total) }}</td><td class="opt" :class="{ dim: p.sellable < p.total }">{{ count(p.sellable) }}</td><td class="opt">{{ money(p.cost, 3) }}</td><td>{{ money(p.mark, 2) }}</td><td>{{ money(p.value / 1e4, 1) }}</td><td>{{ percent(p.weight) }}</td><td :class="tone(p.pnl)">{{ signed(p.pnl) }}</td></tr>
                <tr v-if="summary && !positions.length"><td colspan="10" class="none">空仓</td></tr>
                <template v-if="!summary && selected && !error && !waiting"><tr v-for="n in 6" :key="n" class="skeleton-row"><td v-for="c in 10" :key="c"><UiSkeleton /></td></tr></template>
              </tbody>
            </table>
            <table v-else-if="tab === 'orders'" class="grid-table">
              <thead><tr><th class="l">时间</th><th class="l">代码</th><th class="l">名称</th><th class="l">方向</th><th>委托价</th><th>数量</th><th>成交</th><th class="l status">状态</th></tr></thead>
              <tbody>
                <tr v-for="o in summary?.orders ?? []" :key="o.id"><td class="mono l">{{ o.time ?? '—' }}</td><td class="mono">{{ o.symbol }}</td><td class="l">{{ o.name }}</td><td class="l" :class="o.side === 'buy' ? 'up' : 'down'">{{ sideLabel[o.side] }}</td><td>{{ money(o.price, 3) }}</td><td>{{ count(o.qty) }}</td><td>{{ count(o.filled) }}</td><td class="l status" :class="{ dim: o.status !== 'open' }">{{ o.status === 'open' && o.filled ? '部分成交' : statusLabel[o.status] ?? o.status }}</td></tr>
                <tr v-if="summary && !summary.orders.length"><td colspan="8" class="none">当日没有委托</td></tr>
              </tbody>
            </table>
            <template v-else>
              <table class="grid-table">
                <thead><tr><th class="l">日期</th><th class="l">时间</th><th class="l">代码</th><th class="l">名称</th><th class="l">方向</th><th>股数</th><th>成交价</th><th>费用</th></tr></thead>
                <tbody>
                  <tr v-for="(f, i) in history?.items ?? []" :key="i"><td class="mono l">{{ f.day }}</td><td class="mono l dim">{{ f.time ?? '—' }}</td><td class="mono">{{ f.symbol }}</td><td class="l">{{ f.name }}</td><td class="l" :class="f.side === 'buy' ? 'up' : 'down'">{{ sideLabel[f.side] }}</td><td>{{ count(f.qty) }}</td><td>{{ money(f.price, 3) }}</td><td>{{ money(f.fee) }}</td></tr>
                  <tr v-if="history && !history.total"><td colspan="8" class="none">还没有成交</td></tr>
                </tbody>
              </table>
              <button v-if="history && history.items.length < history.total" class="button more" @click="loadHistory()">更多</button>
            </template>
          </div>
        </section>
      </div>

      <div class="side">
        <section class="panel">
          <div class="panel-header">{{ latestTrades }}<span v-if="summary" class="push muted">{{ summary.fills_day }}</span></div>
          <div v-if="summary" class="side-body kv">
            <span>买入 / 卖出笔数</span><b>{{ summary.trading.buys }} / {{ summary.trading.sells }}</b>
            <span>买入 / 卖出（万元）</span><b>{{ money(summary.trading.buy_amount / 1e4, 1) }} / {{ money(summary.trading.sell_amount / 1e4, 1) }}</b>
            <span>换手</span><b>{{ percent(summary.trading.turnover) }}</b>
            <span>费用</span><b>{{ money(summary.trading.fees) }} 元</b>
            <span>未成交</span><b>{{ summary.trading.unfilled }}</b>
          </div>
        </section>
        <section class="panel fills-panel">
          <div class="panel-header">成交</div>
          <div class="scroll">
            <table class="grid-table">
              <thead><tr><th class="l">代码</th><th class="l">名称</th><th class="l">方向</th><th>股数</th><th>成交价</th><th>费用</th></tr></thead>
              <tbody>
                <tr v-for="(f, i) in summary?.fills ?? []" :key="i"><td class="mono">{{ f.symbol }}</td><td class="l">{{ f.name }}</td><td class="l" :class="f.side === 'buy' ? 'up' : 'down'">{{ sideLabel[f.side] }}</td><td>{{ count(f.qty) }}</td><td>{{ money(f.price, 3) }}</td><td>{{ money(f.fee) }}</td></tr>
                <tr v-if="summary && !summary.fills.length"><td colspan="6" class="none">还没有成交</td></tr>
              </tbody>
            </table>
          </div>
        </section>
        <section class="panel industry-panel">
          <div class="panel-header">行业分布</div>
          <div class="side-body kv scroll">
            <template v-for="[name, weight] in summary?.industries ?? []" :key="name"><span>{{ name }}</span><b>{{ percent(weight, 1) }}</b></template>
          </div>
        </section>
      </div>
      <div v-if="error" class="toast-line" role="alert">{{ error }}</div>
    </template>
    <template v-else>
      <!-- 账户列表还没到：按真实三栏摆骨架，数据到了整块换成内容。 -->
      <section class="panel account-panel" aria-busy="true"><div class="panel-header">账户</div><div class="side-body"><UiSkeleton :lines="3" /></div></section>
      <div class="center" aria-busy="true">
        <section class="panel"><div class="panel-header">权益</div><div class="chart-body"><div class="hero-skeleton"><UiSkeleton shape="block" height="40px" /></div><div class="chart"><UiSkeleton shape="block" /></div></div></section>
        <section class="panel table-panel"><div class="panel-header">持仓</div><div class="scroll skeleton-rows"><UiSkeleton :lines="6" /></div></section>
      </div>
      <div class="side" aria-busy="true">
        <section class="panel"><div class="panel-header">当日交易</div><div class="side-body skeleton-rows"><UiSkeleton :lines="4" /></div></section>
        <section class="panel fills-panel"><div class="panel-header">成交</div><div class="side-body skeleton-rows"><UiSkeleton :lines="5" /></div></section>
      </div>
    </template>
    <PaperNewAccount v-if="creating" @created="created" @close="creating = false" />
  </main>
</template>

<style scoped>
.paper-page{height:100%;min-height:0;padding:8px;display:grid;grid-template-columns:248px minmax(0,1fr) 420px;gap:8px;font-size:12.5px;line-height:1.4}
.center{display:grid;grid-template-rows:auto minmax(0,1fr);gap:8px;min-width:0;min-height:0}
.side{display:flex;flex-direction:column;gap:8px;min-height:0;min-width:0}
.side>.panel{flex:none}.side>.fills-panel{flex:1 1 0;min-height:140px}.side>.industry-panel{flex:0 1 auto;min-height:96px;max-height:42%}
.panel-header{font-weight:600}
.account{display:block;width:100%;text-align:left;padding:10px;border:0;border-bottom:1px solid var(--grid);background:var(--panel);font-size:12.5px}
.account.on{background:var(--selected)}.account b{display:block;font-weight:500}.account em{font-style:normal;color:var(--ink3);font-size:11.5px}
.side-body{padding:2px 10px 8px}.account-panel .side-body{padding-top:8px;overflow:auto}
h5{font-size:11.5px;font-weight:600;color:var(--ink3);margin:12px 0 5px}h5:first-child{margin-top:2px}
.kv{display:grid;grid-template-columns:1fr auto}
.kv>span{color:var(--ink3);height:24px;display:flex;align-items:center;border-bottom:1px solid var(--grid);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.kv>b{font-weight:500;height:24px;display:flex;align-items:center;justify-content:flex-end;gap:5px;border-bottom:1px solid var(--grid);font-family:var(--mono);font-size:12px;white-space:nowrap;padding-left:10px}
.kv>b.plain,.kv>b.state{font-family:inherit;font-size:12.5px}.state{color:var(--ink3)}.state.running{color:var(--green)}.warn{color:var(--warn)}
.create-row{padding:10px}.create{width:100%}.empty .create{width:auto}
.note{color:var(--ink3);font-size:11.5px;line-height:1.6;margin:8px 0 0}
.legend{display:flex;align-items:center;gap:6px;font-weight:400;color:var(--ink2)}.legend i{width:12px;height:2.5px;display:block}.legend i.ink{background:var(--ink)}.legend i.accent{background:var(--accent)}
.chart-body{padding:8px 10px}
.hero{display:flex;align-items:baseline;gap:26px;padding:4px 2px 8px;min-height:40px;white-space:nowrap;overflow:hidden}
.hero span{color:var(--ink3)}.hero b{font-family:var(--mono);font-weight:500;margin-left:6px}
.hero .equity{font-size:24px;font-weight:500}
.hero-skeleton{padding:4px 2px 8px}.skeleton-rows{padding:12px 10px}.skeleton-row td{padding-top:8px;padding-bottom:8px}
.chart{height:218px;position:relative}.chart svg{display:block;font-size:10.5px;fill:var(--ink3);touch-action:none}
.chart .grid{stroke:var(--grid)}.chart .zero{stroke:var(--line)}.chart .cursor{stroke:var(--line);stroke-dasharray:3 3}
.chart .own{fill:none;stroke:var(--ink);stroke-width:1.7;stroke-linejoin:round}.chart .bench{fill:none;stroke:var(--accent);stroke-width:1.2;stroke-linejoin:round}
.chart .readout{fill:var(--ink2);font-size:11px}
.chart-empty{height:100%;display:grid;place-items:center;color:var(--ink3)}
.table-panel{min-height:0}.scroll{flex:1;min-height:0;overflow:auto}
.table-panel .scroll{padding:2px 10px 8px}.fills-panel .scroll{padding:2px 10px 6px}
.grid-table{width:100%;border-collapse:collapse}
.grid-table th{font-weight:500;color:var(--ink3);text-align:right;height:24px;border-bottom:1px solid var(--line);font-size:11.5px;white-space:nowrap;position:sticky;top:0;background:var(--panel);padding-left:10px}
.grid-table td{height:21px;border-bottom:1px solid var(--grid);white-space:nowrap;text-align:right;font-family:var(--mono);font-size:12px;padding-left:10px}
.grid-table th:first-child,.grid-table td:first-child{padding-left:0}
.grid-table .l{text-align:left;font-family:inherit;font-size:12.5px}.grid-table td.mono{text-align:left}.grid-table td.mono.l{font-family:var(--mono);font-size:12px}
.grid-table tbody tr:last-child td{border-bottom:0}
.dim{color:var(--ink3)}.grid-table td.l.dim{color:var(--ink2)}
.up{color:var(--red)}.down{color:var(--green)}
.none{text-align:center!important;color:var(--ink3);font-family:inherit!important;padding:18px 0!important}
.more{margin:8px auto 0;display:flex}
.industry-panel .kv{overflow:auto;align-content:start}
.empty{grid-column:1/-1;align-self:center;justify-self:center;padding:28px 32px;gap:10px;align-items:flex-start;max-width:760px}
.empty b{font-size:14px;font-weight:600}.empty span{color:var(--ink2);line-height:1.7}
.empty code{display:block;margin-top:6px;font-family:var(--mono);font-size:12px;color:var(--ink);background:var(--chrome);border:1px solid var(--grid);border-radius:3px;padding:6px 9px;white-space:nowrap;overflow-x:auto}
.error-text{color:var(--red)}
.toast-line{position:fixed;right:18px;bottom:38px;z-index:50;padding:10px 14px;border:1px solid var(--line);border-radius:5px;background:var(--panel);box-shadow:0 6px 25px var(--overlay-shadow);color:var(--ink2)}
/* 1440 以下收窄两侧栏；中栏再窄时先收起可卖、成本两列。 */
@media (max-width:1400px){.paper-page{grid-template-columns:232px minmax(0,1fr) 360px}.hero{gap:18px}.hero .equity{font-size:22px}}
@media (max-width:1300px){.grid-table .opt{display:none}.hero{gap:14px}}
</style>
