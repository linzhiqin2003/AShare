// 回测页的数字格式：百分比、带符号百分比、比值与金额。缺失一律显示「—」。
const DASH = '—'
const ok = (x: number | null | undefined): x is number => typeof x === 'number' && Number.isFinite(x)

export function pct(x: number | null | undefined, digits = 2, sign = false) {
  if (!ok(x)) return DASH
  const text = (x * 100).toFixed(digits)
  return `${sign && x > 0 ? '+' : ''}${text === `-${(0).toFixed(digits)}` ? (0).toFixed(digits) : text}%`
}
export function num(x: number | null | undefined, digits = 2) {
  return ok(x) ? x.toFixed(digits) : DASH
}
export function int(x: number | null | undefined) {
  return ok(x) ? Math.round(x).toLocaleString('en-US') : DASH
}
export function money(x: number | null | undefined, digits = 2) {
  return ok(x) ? x.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits }) : DASH
}
/** A 股习惯：涨红跌绿；零与缺失不着色。 */
export function tone(x: number | null | undefined) {
  return !ok(x) || x === 0 ? '' : x > 0 ? 'up' : 'down'
}
