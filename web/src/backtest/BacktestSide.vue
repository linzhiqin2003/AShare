<script setup lang="ts">
import { computed } from 'vue'
import { int, num, pct } from './format'
import type { BacktestDocument } from './types'

const props = defineProps<{ document: BacktestDocument | null }>()
const s = computed(() => props.document?.summary ?? null)
const abs = (x: number | null | undefined) => (typeof x === 'number' ? Math.abs(x) : x)
const rows = computed(() => {
  const v = s.value
  if (!v) return []
  const bench = !!props.document?.benchmark
  return [
    ['年化收益', pct(v.annual_return), bench ? pct(v.benchmark_annual_return) : ''],
    ['年化波动', pct(v.annual_volatility), bench ? pct(v.benchmark_annual_volatility) : ''],
    ['Sharpe', num(v.sharpe), bench ? num(v.benchmark_sharpe) : ''],
    ['最大回撤', pct(abs(v.max_drawdown)), bench ? pct(abs(v.benchmark_max_drawdown)) : ''],
    ['期末净值', num(v.final_nav, 3), bench && v.benchmark_cumulative_return !== null ? num(1 + v.benchmark_cumulative_return, 3) : ''],
  ]
})
const more = computed(() => {
  const v = s.value
  if (!v) return []
  const bench = props.document?.benchmark
  return [
    ...(bench ? [
      ['超额年化', pct(v.excess_annual_return)],
      ['跟踪误差', pct(v.tracking_error)],
      ['信息比率', num(v.information_ratio)],
      ['Beta', num(v.beta)],
    ] : []),
    ['Calmar', num(v.calmar)],
    ['月胜率', pct(v.monthly_win_rate, 1)],
    ['年换手（单边）', pct(v.annual_turnover_one_sided, 0)],
    ['佣金与税费 ÷ 初始资金', pct(v.total_fees_to_initial)],
    ['滑点 ÷ 初始资金', pct(v.total_slippage_to_initial)],
  ]
})
const range = computed(() => props.document ? `${props.document.start} 至 ${props.document.end}` : '')
</script>

<template>
  <div class="side">
    <section class="panel">
      <div class="panel-header">统计<span class="push muted plain">{{ range }}</span></div>
      <div class="body">
        <table v-if="document" class="tb">
          <thead><tr><th /><th>策略</th><th v-if="document?.benchmark">基准</th></tr></thead>
          <tbody><tr v-for="r in rows" :key="r[0]"><td>{{ r[0] }}</td><td class="n">{{ r[1] }}</td><td v-if="document?.benchmark" class="n">{{ r[2] }}</td></tr></tbody>
        </table>
        <dl v-if="document" class="kv"><template v-for="r in more" :key="r[0]"><dt>{{ r[0] }}</dt><dd>{{ r[1] }}</dd></template></dl>
      </div>
    </section>
    <section class="panel">
      <div class="panel-header">成交约束</div>
      <div class="body">
        <dl v-if="document" class="kv">
          <dt>调仓次数</dt><dd>{{ int(document.constraints.rebalances) }}</dd>
          <dt>当日未成交的委托</dt><dd>{{ int(document.constraints.unfilled) }}</dd>
          <dt>其中涨停买不进</dt><dd>{{ int(document.constraints.limit_up) }}</dd>
          <dt>跌停卖不出</dt><dd>{{ int(document.constraints.limit_down) }}</dd>
          <dt>停牌</dt><dd>{{ int(document.constraints.suspended) }}</dd>
        </dl>
      </div>
    </section>
    <section class="panel holdings">
      <div class="panel-header">最新持仓<span v-if="document" class="push muted plain">{{ document.holdings.length }} 只</span></div>
      <div class="body scroll">
        <table v-if="document" class="tb">
          <thead><tr><th class="l">代码</th><th class="l">名称</th><th class="l">行业</th><th>权重</th></tr></thead>
          <tbody><tr v-for="h in document.holdings" :key="h.symbol"><td class="mono code">{{ h.symbol }}</td><td class="l">{{ h.name ?? '—' }}</td><td class="l dim">{{ h.industry ?? '—' }}</td><td class="n">{{ pct(h.weight) }}</td></tr></tbody>
        </table>
      </div>
    </section>
  </div>
</template>

<style scoped>
.side{display:grid;grid-template-rows:auto auto minmax(0,1fr);gap:8px;min-height:0;min-width:0}
.plain{font-weight:400}
.body{padding:2px 10px 8px}
.scroll{flex:1;min-height:0;overflow:auto}
.tb{width:100%;border-collapse:collapse}
.tb th{font-weight:500;color:var(--ink3);text-align:right;height:24px;border-bottom:1px solid var(--line);font-size:11.5px;white-space:nowrap;position:sticky;top:0;background:var(--panel)}
.tb td{height:24px;border-bottom:1px solid var(--grid);white-space:nowrap}
.tb tbody tr:last-child td{border-bottom:0}
.tb .n{text-align:right;font-family:var(--mono);font-size:12px}
.tb .l,.tb th.l{text-align:left}
.tb .code{font-size:12px}
.tb .dim{color:var(--ink2)}
.kv{display:grid;grid-template-columns:1fr auto;margin:6px 0 0}
.kv dt{color:var(--ink3);height:26px;display:flex;align-items:center;border-bottom:1px solid var(--grid)}
.kv dd{margin:0;font-weight:500;height:26px;display:flex;align-items:center;justify-content:flex-end;border-bottom:1px solid var(--grid);font-family:var(--mono);font-size:12px}
.kv dt:nth-last-child(2),.kv dd:last-child{border-bottom:0}
</style>
