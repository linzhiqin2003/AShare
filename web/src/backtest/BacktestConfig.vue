<script setup lang="ts">
import { computed } from 'vue'
import BenchmarkSelect from '../components/BenchmarkSelect.vue'
import { UiNumberInput, UiSelect } from '../components/ui'
import type { BacktestOptions, BacktestSettings } from './types'

const props = defineProps<{ modelValue: BacktestSettings; options: BacktestOptions; disabled?: boolean }>()
const emit = defineEmits<{ 'update:modelValue': [value: BacktestSettings] }>()

const LABELS: Record<string, Record<string, string>> = {
  universe: { csi300: '沪深 300 时点成分', csi500: '中证 500 时点成分', csi1000: '中证 1000 时点成分', TOP800: '沪深 300 + 中证 500', TOP1800: '沪深 300 + 中证 500 + 中证 1000', ALL: '全部上市股票' },
  rebalance: { day: '每个交易日', week: '每周最后一个交易日', month: '每月最后一个交易日', quarter: '每季最后一个交易日', year: '每年最后一个交易日' },
  trade_at: { open: '次日开盘价', close: '次日收盘价' },
  dividend_tax: { '0.2': '20%', '0.1': '10%', '0': '不扣' },
  carry_blocked: { true: '不成交，逐日重试', false: '不成交，等下次调仓' },
  lots: { board: '按板块规则', none: '不取整' },
}
type Choice = string | number | boolean
const choices = (key: string) => (props.options.choices[key] ?? []).map((value: Choice) => ({ value, label: LABELS[key]?.[String(value)] ?? String(value) }))
const SELECTS = [
  ['universe', '股票池'], ['rebalance', '调仓'], ['trade_at', '成交'],
  ['dividend_tax', '红利税'], ['carry_blocked', '涨跌停与停牌'], ['lots', '整手'],
] as const
const menus = computed(() => Object.fromEntries(SELECTS.map(([key]) => [key, choices(key)])) as Record<(typeof SELECTS)[number][0], { value: Choice; label: string }[]>)

function set<K extends keyof BacktestSettings>(key: K, value: BacktestSettings[K]) {
  if (props.disabled || value === props.modelValue[key]) return
  emit('update:modelValue', { ...props.modelValue, [key]: value })
}
/** 数字设置：显示单位 × scale = 存储单位的倒数；按白名单范围收紧，整数项取整。 */
function limit(key: keyof BacktestSettings, scale = 1) {
  const [lo, hi] = props.options.limits[key as string] ?? [0, Infinity]
  return { min: Number((lo * scale).toPrecision(10)), max: Number((hi * scale).toPrecision(10)) }
}
function number(key: keyof BacktestSettings, value: number | null, scale = 1, integer = false) {
  const { min, max } = limit(key, scale)
  if (value === null || !Number.isFinite(value)) return
  let v = Math.min(max, Math.max(min, integer ? Math.round(value) : value))
  v = Number((v / scale).toPrecision(10))
  set(key, v as never)
}
const shown = (key: keyof BacktestSettings, scale = 1) => Number(((props.modelValue[key] as number) * scale).toPrecision(10))
</script>

<template>
  <div class="config">
    <h5>组合</h5>
    <div class="rows">
      <span>股票池</span>
      <b><UiSelect :disabled="disabled" :model-value="modelValue.universe" :options="menus.universe" aria-label="股票池" @update:model-value="set('universe', $event as string)" /></b>
      <span>调仓</span>
      <b><UiSelect :disabled="disabled" :model-value="modelValue.rebalance" :options="menus.rebalance" aria-label="调仓" @update:model-value="set('rebalance', $event as string)" /></b>
      <span>构建</span>
      <b>信号最高的前 N 只，等权</b>
      <span>持仓</span>
      <b><UiNumberInput :disabled="disabled" :model-value="modelValue.top_n" v-bind="limit('top_n')" :step="1" unit="只" aria-label="持仓只数" @update:model-value="number('top_n', $event, 1, true)" /></b>
      <span>成交</span>
      <b><UiSelect :disabled="disabled" :model-value="modelValue.trade_at" :options="menus.trade_at" aria-label="成交价" @update:model-value="set('trade_at', $event as string)" /></b>
      <span>初始资金</span>
      <b><UiNumberInput :disabled="disabled" :model-value="modelValue.capital" v-bind="limit('capital')" :step="100" unit="万元" aria-label="初始资金" @update:model-value="number('capital', $event)" /></b>
      <span>基准</span>
      <b class="benchmark"><BenchmarkSelect :model-value="modelValue.benchmark" @update:model-value="set('benchmark', $event)" /></b>
    </div>
    <h5>成本与规则</h5>
    <div class="rows">
      <span>佣金</span>
      <b class="pair"><UiNumberInput :disabled="disabled" :model-value="shown('commission', 100)" v-bind="limit('commission', 100)" :step="0.005" unit="%" aria-label="佣金率" @update:model-value="number('commission', $event, 100)" /><em>最低</em><UiNumberInput :disabled="disabled" :model-value="modelValue.commission_min" v-bind="limit('commission_min')" :step="1" unit="元" aria-label="最低佣金" @update:model-value="number('commission_min', $event)" /></b>
      <span>印花税与过户费</span>
      <b>按历年规定</b>
      <span>滑点</span>
      <b><UiNumberInput :disabled="disabled" :model-value="modelValue.slippage_bps" v-bind="limit('slippage_bps')" :step="1" unit="bps" aria-label="滑点" @update:model-value="number('slippage_bps', $event)" /></b>
      <span>红利税</span>
      <b><UiSelect :disabled="disabled" :model-value="modelValue.dividend_tax" :options="menus.dividend_tax" aria-label="红利税" @update:model-value="set('dividend_tax', $event as number)" /></b>
      <span>涨跌停与停牌</span>
      <b><UiSelect :disabled="disabled" :model-value="modelValue.carry_blocked" :options="menus.carry_blocked" aria-label="涨跌停与停牌" @update:model-value="set('carry_blocked', $event as boolean)" /></b>
      <span>整手</span>
      <b><UiSelect :disabled="disabled" :model-value="modelValue.lots" :options="menus.lots" aria-label="整手" @update:model-value="set('lots', $event as string)" /></b>
    </div>
  </div>
</template>

<style scoped>
.config{min-width:0}
h5{font-size:11.5px;font-weight:600;color:var(--ink3);margin:14px 0 5px}
.rows{display:grid;grid-template-columns:auto 1fr}
.rows>span{color:var(--ink3);height:27px;display:flex;align-items:center;border-bottom:1px solid var(--grid);padding-right:14px;white-space:nowrap}
.rows>b{font-weight:500;height:27px;display:flex;align-items:center;justify-content:flex-end;border-bottom:1px solid var(--grid);min-width:0;white-space:nowrap}
.pair{gap:4px}.pair em{font-style:normal;margin:0 3px 0 12px}
.benchmark :deep(button){font-weight:500}
</style>
