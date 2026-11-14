// 模拟盘页的数据类型与纯计算（格式化、区间截取、曲线归一），不碰 DOM，便于单测。
export interface PaperStrategy { name: string; universe: string; rebalance: string; top_n: number; trade_at: string; initial_cash: number; enabled: boolean; origin: string | null }
export interface PaperPlan { last_signal_day: string | null; last_execution_day: string | null; next_signal_day: string | null; pending_execution_day: string | null }
export interface PaperAccountItem { id: string; since: string | null; running: boolean; started?: boolean; strategy?: PaperStrategy | null; plan?: PaperPlan | null }
export interface PaperBacktest { id: string; name: string; universe: string | null; rebalance: string | null; top_n: number | null; capital: number | null }
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

// 账户与策略绑定（2026-11-14）：账户列表的副标题、从回测新建账户时的默认名称。
export const rebalanceLabel: Record<string, string> = { day: '每日调仓', week: '每周调仓', month: '月度调仓', quarter: '季度调仓', year: '年度调仓' }
export const universeLabel: Record<string, string> = { csi300: '沪深300', csi500: '中证500', csi1000: '中证1000', TOP800: 'TOP800', TOP1800: 'TOP1800', ALL: '全市场' }
export function accountLine(item: PaperAccountItem) {
  const s = item.strategy
  const parts = [s && !s.enabled ? '已停用' : item.running ? '运行中' : s ? '纸面跟踪' : '实时模拟']
  if (s) parts.push(rebalanceLabel[s.rebalance] ?? s.rebalance)
  if (item.since) parts.push(`自 ${item.since}`)
  else if (s && item.started === false) parts.push('等待首个交易日')
  return parts.join(' · ')
}
export function backtestLabel(b: PaperBacktest) {
  const parts = [b.name]
  if (b.universe) parts.push(universeLabel[b.universe] ?? b.universe)
  if (b.rebalance) parts.push(rebalanceLabel[b.rebalance] ?? b.rebalance)
  if (b.top_n) parts.push(`前 ${b.top_n} 只`)
  return parts.join(' · ')
}
// 与后端账户名规则一致：首字符为字母、数字、汉字或下划线，其余可含点和短横线，最长 64。
const NAME = /^[\p{L}\p{N}_][\p{L}\p{N}_.-]{0,63}$/u
export function validName(name: string) { return NAME.test(name) }
export function suggestName(text: string) {
  const cleaned = text.trim().replace(/[^\p{L}\p{N}_.-]+/gu, '-').replace(/^[.-]+/, '').replace(/-+$/, '')
  return [...cleaned].slice(0, 64).join('')
}
