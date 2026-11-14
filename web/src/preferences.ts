import type { Scalar } from './types'

export type BacktestDefaults = Record<string, Scalar>
export interface PreferencePayload { backtest: BacktestDefaults; builtin: BacktestDefaults; keys: string[] }
export interface Option { value: Scalar; label: string }
export interface BacktestField { key: string; label: string; kind: 'select' | 'number'; options?: Option[]; min?: number; max?: number; unit?: string; invalid?: string }

// 与服务端 config.CHOICES 一致；服务端再校验一次。中性化不在默认设定里（新建 tab 固定以两组扫描开始）。
export const BACKTEST_FIELDS: BacktestField[] = [
  { key: 'universe', label: '股票池', kind: 'select', options: [{ value: 'TOP800', label: 'TOP800' }, { value: 'TOP1800', label: 'TOP1800' }, { value: 'ALL', label: '全市场' }] },
  { key: 'benchmark', label: '基准', kind: 'select' },
  { key: 'rebalance', label: '调仓周期', kind: 'number', min: 1, max: 2520, unit: '天', invalid: '调仓周期取 1–2520 个交易日' },
  { key: 'decay', label: '衰减', kind: 'number', min: 0, max: 2520, unit: '天', invalid: '衰减取 0–2520 天' },
  { key: 'trade_limitation', label: '涨跌停与停牌', kind: 'select', options: [{ value: true, label: '限制' }, { value: false, label: '不限制' }] },
  { key: 'trade_at', label: '成交', kind: 'select', options: [{ value: 'open', label: 'T+1 开盘' }, { value: 'close', label: 'T+1 收盘' }] },
  { key: 'n_groups', label: '分组', kind: 'select', options: [{ value: 5, label: '5' }, { value: 10, label: '10' }] },
  { key: 'weights_portfolio', label: '组合', kind: 'select', options: [{ value: false, label: '分组多空' }, { value: true, label: '另算权重多空' }] },
  { key: 'sub_universes', label: '子池检验', kind: 'select', options: [{ value: true, label: '开' }, { value: false, label: '关' }] },
  { key: 'by', label: '分段', kind: 'select', options: [{ value: 'year', label: '按年' }, { value: 'quarter', label: '按季' }] },
]

/** 控件给出的值是否合法（服务端会再校验一次）；不合法返回 undefined。 */
export function validate(field: BacktestField, value: Scalar): Scalar | undefined {
  if (field.kind === 'number') return typeof value === 'number' && Number.isInteger(value) && value >= (field.min ?? -Infinity) && value <= (field.max ?? Infinity) ? value : undefined
  if (!field.options) return typeof value === 'string' && value ? value : undefined
  return field.options.some(o => o.value === value) ? value : undefined
}
