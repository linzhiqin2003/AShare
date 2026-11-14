<script setup lang="ts">
import { computed, shallowRef, watch } from 'vue'
import { ChevronLeft, ChevronRight } from 'lucide-vue-next'
import { friendly } from '../api'
import * as client from './api'
import { int, money, num, pct, tone } from './format'
import type { BacktestDocument, Page, Part, RebalanceRow, TradeRow } from './types'

const props = defineProps<{ document: BacktestDocument | null; backtest: string | null; part: Part }>()
type Tab = 'yearly' | 'rebalances' | 'trades' | 'industry' | 'drawdowns'
const TABS: { key: Tab; label: string }[] = [
  { key: 'yearly', label: '分年与归因' }, { key: 'rebalances', label: '调仓记录' }, { key: 'trades', label: '成交' },
  { key: 'industry', label: '行业' }, { key: 'drawdowns', label: '回撤区间' },
]
const LIMIT = 50
const tab = shallowRef<Tab>('yearly'), offset = shallowRef(0), error = shallowRef('')
const page = shallowRef<Page<RebalanceRow> | Page<TradeRow> | null>(null)
let request = 0
async function load() {
  page.value = null; error.value = ''
  if (!props.backtest || !props.document || (tab.value !== 'rebalances' && tab.value !== 'trades')) return
  const id = ++request
  try {
    const result = tab.value === 'rebalances'
      ? await client.rebalances(props.backtest, props.part, offset.value, LIMIT)
      : await client.trades(props.backtest, props.part, offset.value, LIMIT)
    if (id === request) page.value = result
  } catch (e) { if (id === request) error.value = friendly(e).message }
}
watch(() => [props.backtest, props.part, props.document] as const, () => { offset.value = 0; void load() })
watch([tab, offset], load)
function select(key: Tab) { if (tab.value !== key) { tab.value = key; offset.value = 0 } }
const rebalanceRows = computed(() => tab.value === 'rebalances' ? (page.value?.items ?? []) as RebalanceRow[] : [])
const tradeRows = computed(() => tab.value === 'trades' ? (page.value?.items ?? []) as TradeRow[] : [])
const pageText = computed(() => page.value && page.value.total ? `${offset.value + 1}–${Math.min(offset.value + LIMIT, page.value.total)} / ${int(page.value.total)}` : '')
const abs = (x: number | null) => (x === null ? null : Math.abs(x))
</script>

<template>
  <section class="panel tables">
    <div class="panel-header" role="tablist" aria-label="回测明细">
      <button v-for="t in TABS" :key="t.key" class="panel-tab" role="tab" :class="{ active: tab === t.key }" :aria-selected="tab === t.key" @click="select(t.key)">{{ t.label }}</button>
      <span v-if="pageText" class="push pager">
        <span class="muted mono">{{ pageText }}</span>
        <button class="icon-button" aria-label="上一页" :disabled="offset === 0" @click="offset -= LIMIT"><ChevronLeft :size="14" /></button>
        <button class="icon-button" aria-label="下一页" :disabled="!page || offset + LIMIT >= page.total" @click="offset += LIMIT"><ChevronRight :size="14" /></button>
      </span>
    </div>
    <div class="body">
      <div v-if="error" class="muted note">{{ error }}</div>
      <table v-else-if="document && tab === 'yearly'" class="tb">
        <thead><tr><th class="l">年份</th><th>策略</th><th v-if="document.benchmark">基准</th><th v-if="document.benchmark">超额</th><th class="opt">波动</th><th class="opt">回撤</th><th>Sharpe</th><th>价格贡献</th><th>分红（税后）</th><th>佣金与税</th><th>滑点</th></tr></thead>
        <tbody><tr v-for="y in document.yearly" :key="y.year">
          <td class="n l">{{ y.year }}</td>
          <td class="n" :class="tone(y.strategy)">{{ pct(y.strategy, 2, true) }}</td>
          <td v-if="document.benchmark" class="n">{{ pct(y.benchmark, 2, true) }}</td>
          <td v-if="document.benchmark" class="n" :class="tone(y.excess)">{{ pct(y.excess, 2, true) }}</td>
          <td class="n opt">{{ pct(y.volatility) }}</td><td class="n opt">{{ pct(abs(y.drawdown)) }}</td><td class="n">{{ num(y.sharpe) }}</td>
          <td class="n">{{ pct(y.price, 2, true) }}</td><td class="n">{{ pct(y.dividend, 2, true) }}</td><td class="n">{{ pct(y.fees, 2, true) }}</td><td class="n">{{ pct(y.slippage, 2, true) }}</td>
        </tr></tbody>
      </table>
      <table v-else-if="document && tab === 'rebalances' && page" class="tb">
        <thead><tr><th class="l">日期</th><th>目标只数</th><th>买入</th><th>卖出</th><th>新进</th><th>调出</th><th>换手（双边）</th><th>佣金与税费</th><th>滑点</th><th>未成交</th></tr></thead>
        <tbody><tr v-for="r in rebalanceRows" :key="r.date">
          <td class="n l">{{ r.date }}</td><td class="n">{{ r.targets }}</td><td class="n">{{ r.buys }}</td><td class="n">{{ r.sells }}</td><td class="n">{{ r.new_names }}</td><td class="n">{{ r.dropped_names }}</td>
          <td class="n">{{ pct(r.turnover, 1) }}</td><td class="n">{{ money(r.fees) }}</td><td class="n">{{ money(r.slippage) }}</td><td class="n">{{ r.unfilled || '' }}</td>
        </tr></tbody>
      </table>
      <table v-else-if="document && tab === 'trades' && page" class="tb">
        <thead><tr><th class="l">日期</th><th class="l">代码</th><th class="l">名称</th><th class="l">方向</th><th>数量</th><th>价格</th><th>金额</th><th>佣金与税费</th><th>滑点</th></tr></thead>
        <tbody><tr v-for="(r, i) in tradeRows" :key="`${r.date}${r.symbol}${i}`">
          <td class="n l">{{ r.date }}</td><td class="n l">{{ r.symbol }}</td><td class="l">{{ r.name }}</td>
          <td class="l" :class="r.side === 'buy' ? 'up' : 'down'">{{ r.side === 'buy' ? '买入' : '卖出' }}</td>
          <td class="n">{{ int(r.qty) }}</td><td class="n">{{ money(r.price) }}</td><td class="n">{{ money(r.amount) }}</td><td class="n">{{ money(r.fees) }}</td><td class="n">{{ money(r.slippage) }}</td>
        </tr></tbody>
      </table>
      <table v-else-if="document && tab === 'industry'" class="tb">
        <thead><tr><th class="l">行业</th><th>期末权重</th><th>调仓日平均</th><th>调仓日最高</th></tr></thead>
        <tbody><tr v-for="r in document.industry" :key="r.name"><td class="l">{{ r.name }}</td><td class="n">{{ pct(r.latest) }}</td><td class="n">{{ pct(r.mean) }}</td><td class="n">{{ pct(r.max) }}</td></tr></tbody>
      </table>
      <table v-else-if="document && tab === 'drawdowns'" class="tb">
        <thead><tr><th class="l">开始</th><th class="l">谷底</th><th class="l">修复</th><th>深度</th><th>下跌天数</th><th>持续天数</th></tr></thead>
        <tbody><tr v-for="e in document.drawdowns" :key="e.peak"><td class="n l">{{ e.peak }}</td><td class="n l">{{ e.trough }}</td><td class="n l">{{ e.recovery ?? '未修复' }}</td><td class="n">{{ pct(abs(e.depth)) }}</td><td class="n">{{ e.days_to_trough }}</td><td class="n">{{ e.duration_days }}</td></tr></tbody>
      </table>
    </div>
  </section>
</template>

<style scoped>
.tables{min-height:0;container-type:inline-size}
.body{flex:1;min-height:0;overflow:auto;padding:2px 10px 6px}
.pager{display:flex;align-items:center;gap:2px;font-weight:400;font-size:11.5px}
.pager .muted{margin-right:6px}
.tb{width:100%;border-collapse:collapse}
.tb th{font-weight:500;color:var(--ink3);text-align:right;height:26px;border-bottom:1px solid var(--line);font-size:11.5px;white-space:nowrap;position:sticky;top:0;background:var(--panel);padding-left:10px}
.tb th:first-child{padding-left:0}
.tb td{height:26px;border-bottom:1px solid var(--grid);white-space:nowrap;padding-left:10px}
.tb td:first-child{padding-left:0}
.tb tbody tr:last-child td{border-bottom:0}
.tb .n{text-align:right;font-family:var(--mono);font-size:12px}
.tb .l,.tb th.l{text-align:left}
.up{color:var(--red)}.down{color:var(--green)}
.note{padding:10px 0}
/* 窄屏时年度表先收起波动与回撤两列（年度风险看图即可），保证收益与归因列完整。 */
@container (max-width: 640px){.opt{display:none}}
</style>
