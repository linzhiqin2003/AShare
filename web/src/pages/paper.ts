// 模拟盘页的数据类型与纯计算（格式化、区间截取、曲线归一），不碰 DOM，便于单测。
export interface PaperAccountItem { id: string; since: string | null; running: boolean }
export interface PaperPosition { symbol: string; name: string; industry: string | null; total: number; sellable: number; cost: number; mark: number; value: number; pnl: number; weight: number | null }
export interface PaperOrder { id: number; symbol: string; name: string; side: 'buy' | 'sell'; kind: string; qty: number; filled: number; price: number; status: string; time: string | null }
export interface PaperFill { day: string | null; time: string | null; symbol: string; name: string; side: 'buy' | 'sell'; qty: number; price: number; fee: number }
export interface PaperTrading { buys: number; sells: number; buy_amount: number; sell_amount: number; turnover: number | null; fees: number; open_orders: number; unfilled: number; rejected: [string, number][] }
export interface PaperSummary {
  id: string; day: string | null; phase: string; settled: boolean; running: boolean; quote_time: string | null; since: string | null
  equity: number; cash: number; frozen: number; market_value: number; invested: number
  day_pnl: number; day_return: number | null; month_return: number | null; year_return: number | null; nav: number | null
  positions: PaperPosition[]; industries: [string, number][]
  concentration: { max_weight: number | null; max_symbol: string | null; max_industry: string | null; max_industry_weight: number | null; cash_weight: number | null }
  trading: PaperTrading; risk: { open_orders: number; rejected: [string, number][] }; orders: PaperOrder[]; fills_day: string | null; fills: PaperFill[]
}
export interface PaperCurve { dates: string[]; nav: (number | null)[]; benchmark: { symbol: string; name: string } | null; closes: (number | null)[] | null; today_close?: number | null }
export type Range = '1m' | '6m' | '1y' | 'all'

const integer = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 })
export function money(value: number | null | undefined, digits = 2) {
  if (value == null || !Number.isFinite(value)) return '—'
  return new Intl.NumberFormat('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(value)
}
export function count(value: number | null | undefined) { return value == null || !Number.isFinite(value) ? '—' : integer.format(value) }
export function percent(value: number | null | undefined, digits = 2, signed = false) {
  if (value == null || !Number.isFinite(value)) return '—'
  const text = (value * 100).toFixed(digits)
  return `${signed && value > 0 ? '+' : ''}${text === `-${(0).toFixed(digits)}` ? (0).toFixed(digits) : text}%`
}
export function signed(value: number | null | undefined, digits = 0) {
  if (value == null || !Number.isFinite(value)) return '—'
  return `${value > 0 ? '+' : ''}${money(value, digits)}`
}
// A 股习惯：涨红跌绿。
export function tone(value: number | null | undefined) { return value == null || !Number.isFinite(value) || value === 0 ? '' : value > 0 ? 'up' : 'down' }
export const statusLabel: Record<string, string> = { open: '未成交', filled: '已成交', cancelled: '已撤', expired: '已过期' }

export interface Point { date: string; nav: number | null; close: number | null }
// 已结算日 + 当天盘中点（当天尚未结算时才追加）。
export function points(curve: PaperCurve | null, summary: PaperSummary | null): Point[] {
  const result: Point[] = (curve?.dates ?? []).map((date, i) => ({ date, nav: curve!.nav[i], close: curve!.closes?.[i] ?? null }))
  if (summary?.day && summary.nav != null && !result.some(p => p.date === summary.day) && (!result.length || summary.day > result[result.length - 1].date))
    result.push({ date: summary.day, nav: summary.nav, close: curve?.today_close ?? null })
  return result
}
function shift(date: string, months: number) {
  const [y, m, d] = date.split('-').map(Number)
  const target = new Date(Date.UTC(y, m - 1 - months, 1))
  const last = new Date(Date.UTC(target.getUTCFullYear(), target.getUTCMonth() + 1, 0)).getUTCDate()
  target.setUTCDate(Math.min(d, last))
  return target.toISOString().slice(0, 10)
}
// 区间起点取「最后一天往前 N 个月」之前的最后一个点，使区间收益以那天收盘为基。
export function slice(all: Point[], range: Range): Point[] {
  if (range === 'all' || all.length < 2) return all
  const cutoff = shift(all[all.length - 1].date, range === '1m' ? 1 : range === '6m' ? 6 : 12)
  let start = 0
  for (let i = 0; i < all.length; i++) if (all[i].date <= cutoff) start = i
  return all.slice(start)
}
// 两条线都以区间第一个有效值为 0%。
export function relative(values: (number | null)[]): number[] {
  const base = values.find(v => v != null && Number.isFinite(v) && v > 0)
  return values.map(v => (v != null && base ? v / base - 1 : NaN))
}
// 每个自然月第一个点标一次月份；点太密时隔月标。
export function monthTicks(dates: string[], maxLabels = 8) {
  const ticks: { index: number; label: string }[] = []
  dates.forEach((d, i) => { if (i === 0 || d.slice(0, 7) !== dates[i - 1].slice(0, 7)) ticks.push({ index: i, label: d.slice(0, 7) }) })
  if (dates.length && ticks.length <= 2 && dates.length > 1) return [{ index: 0, label: dates[0] }, { index: dates.length - 1, label: dates[dates.length - 1] }]
  const step = Math.ceil(ticks.length / maxLabels)
  return ticks.filter((_, i) => i % step === 0)
}
// 有运行器时 10 秒一轮；盘中没在跑 30 秒；收盘或已结算 60 秒。
export function pollDelay(summary: PaperSummary | null) {
  if (!summary) return 30_000
  if (summary.running) return 10_000
  return summary.settled || summary.phase === '已收盘' ? 60_000 : 30_000
}
