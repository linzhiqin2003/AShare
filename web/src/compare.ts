// 因子对比页的纯函数：指标列、每列最优、同名区分、设定摘要、导出和波动对齐。
import { configParts } from './chart'
import type { Line, Packed, Settings } from './types'
import { VALIDATION_START } from './validation'

export interface CompareYear { period: string; sharpe: number | null; return: number | null; maxdrawdown: number | null }
export interface CompareStats {
  sharpe: number | null; return: number | null; turnover: number | null; maxdrawdown: number | null; margin: number | null
  fitness: number | null; t: number | null; ic: number | null; coverage: number | null; holding_days: number | null
  years: CompareYear[]; exposure: Record<string, number | null>; subs: Record<string, { sharpe: number | null }>
}
export interface CompareItem {
  id: string; name: string; source: string; settings: Settings; status: string; latest: string | null
  is: CompareStats | null; validation?: CompareStats | null; correlation: { name: string; value: number } | null
}
export interface CompareData { items: CompareItem[]; dates: string[]; series: Record<string, Packed>; correlation: (number | null)[][] }

/** better：1 越大越好，-1 越小越好，-2 绝对值越小越好，0 不比较。 */
export interface MetricColumn {
  key: string; group: string; label: string; value: (item: CompareItem) => number | null
  percent: boolean; digits: number; better: number; heat?: number; year?: boolean
}
interface MetricOption { key: string; label: string; validation?: boolean }

const isValue = (key: keyof CompareStats) => (item: CompareItem) => (item.is?.[key] as number | null | undefined) ?? null
const exposures: [string, string][] = [['size', '市值'], ['reversal_20', '20 日反转'], ['reversal_60', '60 日反转'], ['momentum', '动量']]

// 表格列顺序固定，和设计稿一致；选择只决定显示哪些。
const SIMPLE: Record<string, Omit<MetricColumn, 'key'>> = {
  sharpe: { group: 'Sharpe', label: '样本内', value: isValue('sharpe'), percent: false, digits: 2, better: 1 },
  validation_sharpe: { group: 'Sharpe', label: '验证', value: item => item.validation?.sharpe ?? null, percent: false, digits: 2, better: 1 },
  return: { group: '样本内 · 多空', label: '年化收益', value: isValue('return'), percent: true, digits: 1, better: 1 },
  maxdrawdown: { group: '样本内 · 多空', label: '最大回撤', value: isValue('maxdrawdown'), percent: true, digits: 1, better: -2 },
  turnover: { group: '样本内 · 多空', label: '换手', value: isValue('turnover'), percent: true, digits: 1, better: -1 },
  margin: { group: '样本内 · 多空', label: 'Margin', value: isValue('margin'), percent: false, digits: 1, better: 1 },
  fitness: { group: '样本内 · 多空', label: 'Fitness', value: isValue('fitness'), percent: false, digits: 2, better: 1 },
  holding_days: { group: '样本内 · 多空', label: '持仓天数', value: isValue('holding_days'), percent: false, digits: 1, better: 0 },
  ic: { group: '样本内 · 多空', label: 'Rank IC', value: isValue('ic'), percent: true, digits: 1, better: 1 },
  t: { group: '样本内 · 多空', label: 't 值', value: isValue('t'), percent: false, digits: 2, better: 1 },
  coverage: { group: '样本内 · 多空', label: '覆盖率', value: isValue('coverage'), percent: true, digits: 1, better: 1 },
  ...Object.fromEntries(exposures.map(([key, label]) => [`exposure_${key}`, { group: '风格暴露', label, value: (item: CompareItem) => item.is?.exposure[key] ?? null, percent: false, digits: 3, better: 0 }])),
  sub_top800: { group: '子池与相关', label: 'TOP800 Sharpe', value: item => item.is?.subs.TOP800?.sharpe ?? null, percent: false, digits: 2, better: 1 },
  correlation: { group: '子池与相关', label: '库内最大相关', value: item => item.correlation ? Math.abs(item.correlation.value) : null, percent: false, digits: 2, better: -1 },
}
const YEARLY: Record<string, { group: string; field: keyof CompareYear; percent: boolean; digits: number; heat?: number }> = {
  years_sharpe: { group: '分年 Sharpe', field: 'sharpe', percent: false, digits: 1, heat: 3 },
  years_return: { group: '分年收益', field: 'return', percent: true, digits: 0, heat: .3 },
  years_drawdown: { group: '分年回撤', field: 'maxdrawdown', percent: true, digits: 0 },
}
const ORDER = [...Object.keys(SIMPLE), ...Object.keys(YEARLY)]

export const METRIC_GROUPS: { label: string; items: MetricOption[] }[] = [
  { label: '收益与风险', items: [{ key: 'sharpe', label: 'Sharpe' }, { key: 'return', label: '年化收益' }, { key: 'maxdrawdown', label: '最大回撤' }] },
  { label: '交易', items: [{ key: 'turnover', label: '换手' }, { key: 'margin', label: 'Margin' }, { key: 'fitness', label: 'Fitness' }, { key: 'holding_days', label: '持仓天数' }] },
  { label: '预测力', items: [{ key: 'ic', label: 'Rank IC' }, { key: 't', label: 't 值' }, { key: 'coverage', label: '覆盖率' }] },
  { label: '分段', items: [{ key: 'validation_sharpe', label: '验证 Sharpe', validation: true }, { key: 'years_sharpe', label: '分年 Sharpe' }, { key: 'years_return', label: '分年收益' }, { key: 'years_drawdown', label: '分年回撤' }] },
  { label: '风格暴露', items: exposures.map(([key, label]) => ({ key: `exposure_${key}`, label })) },
  { label: '子池与相关', items: [{ key: 'sub_top800', label: 'TOP800 Sharpe' }, { key: 'correlation', label: '库内最大相关' }] },
]
export const DEFAULT_METRICS = ['sharpe', 'validation_sharpe', 'return', 'maxdrawdown', 'turnover', 'margin', 'fitness', 'ic', 't', 'years_sharpe']

/** 验证期开关关闭时：不出「验证」列，分年列只到样本内末。 */
export function metricColumns(selected: string[], items: CompareItem[], showValidation: boolean): MetricColumn[] {
  const columns: MetricColumn[] = []
  for (const key of ORDER) {
    if (!selected.includes(key) || (key === 'validation_sharpe' && !showValidation)) continue
    if (SIMPLE[key]) { columns.push({ key, ...SIMPLE[key] }); continue }
    const spec = YEARLY[key]
    const years = [...new Set(items.flatMap(item => [...(item.is?.years ?? []), ...(showValidation ? item.validation?.years ?? [] : [])]).map(y => y.period))]
      .filter(period => showValidation || period < VALIDATION_START).sort()
    for (const period of years) {
      const find = (item: CompareItem) => [...(item.is?.years ?? []), ...(showValidation ? item.validation?.years ?? [] : [])].find(y => y.period === period)
      columns.push({ key: `${key}:${period}`, group: spec.group, label: period.slice(2, 4), year: true, percent: spec.percent, digits: spec.digits, heat: spec.heat, better: 0, value: item => (find(item)?.[spec.field] as number | null | undefined) ?? null })
    }
  }
  return columns
}

/** 每列最优的因子 id；少于两个有效值不标。 */
export function bestValues(columns: MetricColumn[], items: CompareItem[]): Map<string, string> {
  const result = new Map<string, string>()
  for (const column of columns) {
    if (!column.better) continue
    const scored = items.map(item => ({ id: item.id, value: column.value(item) })).filter((v): v is { id: string; value: number } => v.value != null && Number.isFinite(v.value))
    if (scored.length < 2) continue
    const score = (v: number) => column.better === -2 ? -Math.abs(v) : column.better * v
    result.set(column.key, scored.reduce((a, b) => score(b.value) > score(a.value) ? b : a).id)
  }
  return result
}

const SETTING_NAMES: Record<string, string> = { universe: '股票池', group_neutralize: '中性化', decay: '衰减', rebalance: '调仓', trade_at: '成交', n_groups: '分组', weights_portfolio: '组合', trade_limitation: '涨跌停' }
// 缺省值与显式默认值视为相同，不算「真有不同」。
const settingKey = (settings: Settings, key: string) => configParts(settings)[key]
function differingKeys(settings: Settings[]) { return Object.keys(SETTING_NAMES).filter(key => new Set(settings.map(s => settingKey(s, key))).size > 1) }

/** 只在设定真有不同时返回一句话，一致时返回空串（2026-11-14 用户要求不放多余说明）。 */
export function settingsSummary(settings: Settings[]): string {
  const differ = differingKeys(settings)
  return differ.length ? `设定不同：${differ.map(k => SETTING_NAMES[k]).join('、')}` : ''
}

const TOKEN = /[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|\S/g
/**
 * 同名因子附上它们之间真正不同的项，例如 `low_vol · 20 · 行业`：
 * 表达式按记号逐个比较，取不同位置上的取值；配置只列这几个同名因子之间不同的项。名字唯一的不变。
 */
export function distinctLabels(rows: { id: string; name: string; source: string; settings: Settings }[]): Map<string, string> {
  const labels = new Map(rows.map(r => [r.id, r.name]))
  const groups = new Map<string, typeof rows>()
  for (const row of rows) groups.set(row.name, [...(groups.get(row.name) ?? []), row])
  for (const group of groups.values()) {
    if (new Set(group.map(r => r.id)).size < 2) continue
    const tokens = group.map(r => r.source.replace(/#.*$/gm, '').match(TOKEN) ?? [])
    const aligned = tokens.every(t => t.length === tokens[0].length)
    const positions = aligned ? tokens[0].map((_, i) => i).filter(i => new Set(tokens.map(t => t[i])).size > 1) : []
    const keys = differingKeys(group.map(r => r.settings))
    group.forEach((row, index) => {
      const parts = [...positions.map(i => tokens[index][i]), ...keys.map(key => configParts(row.settings)[key])]
      if (!parts.length && !aligned) parts.push(`表达式 ${index + 1}`)
      if (parts.length) labels.set(row.id, `${row.name} · ${parts.join(' · ')}`)
    })
  }
  return labels
}

export function toCsv(columns: MetricColumn[], items: CompareItem[], names?: Map<string, string>): string {
  const quote = (text: string) => /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text
  const header = ['因子', ...columns.map(c => `${c.group} ${c.year ? c.key.split(':')[1] : c.label}`)]
  const rows = items.map(item => [names?.get(item.id) ?? item.name, ...columns.map(c => { const v = c.value(item); return v == null || !Number.isFinite(v) ? '' : String(Number(v.toPrecision(6))) })])
  return [header, ...rows].map(row => row.map(quote).join(',')).join('\n') + '\n'
}

/** 按日收益标准差把各条线缩放到同一波动（取各线波动的均值），只改变量级不改变形状。 */
export function volatilityAligned(lines: Line[]): Line[] {
  const std = (values: number[]) => {
    const finite = values.filter(Number.isFinite)
    if (finite.length < 2) return NaN
    const mean = finite.reduce((a, b) => a + b, 0) / finite.length
    return Math.sqrt(finite.reduce((a, b) => a + (b - mean) ** 2, 0) / (finite.length - 1))
  }
  const stds = lines.map(l => std(l.values))
  const valid = stds.filter(s => Number.isFinite(s) && s > 0)
  if (!valid.length) return lines
  const target = valid.reduce((a, b) => a + b, 0) / valid.length
  return lines.map((line, i) => Number.isFinite(stds[i]) && stds[i] > 0 ? { ...line, values: line.values.map(v => v * target / stds[i]) } : line)
}
