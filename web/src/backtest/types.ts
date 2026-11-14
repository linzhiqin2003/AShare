export type Part = 'is' | 'validation'
export type BacktestStatus = 'queued' | 'running' | 'complete' | 'failed' | 'cancelled' | 'evicted'

export interface BacktestSettings {
  universe: string
  rebalance: string
  top_n: number
  trade_at: string
  capital: number
  commission: number
  commission_min: number
  slippage_bps: number
  dividend_tax: number
  carry_blocked: boolean
  lots: string
  benchmark?: string
}

export interface BacktestOptions {
  defaults: BacktestSettings
  choices: Record<string, (string | number | boolean)[]>
  limits: Record<string, [number, number]>
  windows: Record<Part, [string, string]>
}

export interface BacktestState {
  id: string
  name: string
  status: BacktestStatus
  progress: number | null
  queue_position: number | null
  elapsed: number | null
  settings: BacktestSettings
}

export interface YearRow {
  year: number; partial: boolean; strategy: number | null; benchmark: number | null; excess: number | null
  volatility: number | null; drawdown: number | null; sharpe: number | null
  price: number | null; dividend: number | null; fees: number | null; slippage: number | null
}
export interface Episode { peak: string; trough: string; recovery: string | null; depth: number; days_to_trough: number; duration_days: number; recovered: boolean }
export interface IndustryRow { name: string; latest: number | null; mean: number | null; max: number | null }
export interface Holding { symbol: string; name: string | null; industry: string | null; weight: number }

export interface BacktestSummary {
  annual_return: number | null; annual_volatility: number | null; sharpe: number | null; max_drawdown: number | null; final_nav: number | null
  benchmark_annual_return: number | null; benchmark_annual_volatility: number | null; benchmark_sharpe: number | null
  benchmark_max_drawdown: number | null; benchmark_cumulative_return: number | null
  excess_annual_return: number | null; tracking_error: number | null; information_ratio: number | null; beta: number | null
  calmar: number | null; monthly_win_rate: number | null; annual_turnover_one_sided: number | null
  total_fees_to_initial: number | null; total_slippage_to_initial: number | null; rebalances: number
}

export interface BacktestDocument {
  start: string
  end: string
  benchmark: { symbol: string; name: string } | null
  summary: BacktestSummary
  constraints: { rebalances: number; unfilled: number; limit_up: number; limit_down: number; suspended: number }
  series: { dates: string[]; nav: (number | null)[]; benchmark: (number | null)[] | null; drawdown: (number | null)[] }
  yearly: YearRow[]
  drawdowns: Episode[]
  industry: IndustryRow[]
  industry_sampling: string | null
  holdings: Holding[]
  rebalance_count: number
  trade_count: number
}

export interface RebalanceRow { date: string; targets: number; buys: number; sells: number; new_names: number; dropped_names: number; turnover: number | null; fees: number | null; slippage: number | null; unfilled: number }
export interface TradeRow { date: string; symbol: string; name: string; side: 'buy' | 'sell'; qty: number; price: number; amount: number; fees: number; slippage: number }
export interface Page<T> { items: T[]; total: number }

export interface BacktestTab {
  id: string
  name: string
  source: string
  settings: BacktestSettings
  backtest: string | null
  state: BacktestState | null
  entry?: string
}
