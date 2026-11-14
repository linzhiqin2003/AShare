export interface IndexQuote { symbol: string; name: string; close: number; change: number | null }
export interface IndustryMove { name: string; change: number | null; count: number }
export interface MarketSummary { date: string | null; indexes: IndexQuote[]; breadth: { up: number; down: number; flat: number } | null; industries: IndustryMove[] }

/** 带符号的百分比，默认两位小数；空值显示破折号。 */
export function signedPercent(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  const text = (value * 100).toFixed(digits)
  return (value > 0 ? '+' : '') + (text === '-0.00' ? '0.00' : text) + '%'
}

/**
 * 行业热力格的颜色：涨用 --rise、跌用 --fall，按幅度相对 scale 的比例与面板底色混合。
 * 用 color-mix 跟随主题和「涨跌颜色」设置，不写死色值。
 */
export function heatColor(change: number, scale: number): string {
  const share = Math.round(Math.min(1, Math.abs(change) / (scale || 1)) * 85 + 15)
  return `color-mix(in srgb, var(${change >= 0 ? '--rise' : '--fall'}) ${share}%, var(--panel))`
}
