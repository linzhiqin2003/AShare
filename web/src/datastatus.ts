// 数据页：接口类型与纯函数（格式化、迷你图、坐标刻度），便于单测。
export interface PanelInfo { version: string; start: string | null; end: string | null; days: number; symbols: number | null; fields: number; size: number; built_at: string | null }
export interface VerifyCheck { name: string; checked: number | null; failed: number | null; pass_rate: number | null; note: string; verdict?: string | null }
export interface VerifyInfo { generated_at: string | null; current: boolean; verdict: string | null; checks: VerifyCheck[] }
export interface RawTable { name: string; group: string; partitions: number; rows: number; failed: number; first: string | null; last: string | null; fetched: string | null }
export interface SyncRecord { date: string; group: string; first: string; last: string; datasets: string[]; partitions: number; rows: number; calls: number }
export interface IndexInfo { symbol: string; name: string; start: string; end: string }
export interface Dataset { key: string; name: string; panel: PanelInfo | null; verify: VerifyInfo | null; tables: RawTable[]; history: SyncRecord[]; indexes?: IndexInfo[] }
export interface Overview { datasets: Dataset[] }
export interface FieldRow { name: string; label: string; category: string; dtype: string; unit: string; start: string | null; latest: number | null; spark: (number | null)[]; references: number }
export interface FieldList { state: 'ready' | 'pending' | 'failed' | 'missing'; version: string | null; fields: FieldRow[] }
export interface Distribution { low: number; high: number; counts: number[]; quantiles: (number | null)[]; n: number }
export interface FieldDetail { name: string; label: string; category: string; dtype: string; unit: string; start: string | null; latest: number | null; overall: number | null; latest_count: number; monthly: [string, number | null][]; distribution: Distribution | null; true_share: number | null; factors: { name: string; sharpe: number | null }[] }

export function percent(value: number | null | undefined, digits = 1): string {
  return value === null || value === undefined || !Number.isFinite(value) ? '—' : `${(value * 100).toFixed(digits)}%`
}

export function bytes(value: number): string {
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let size = value, unit = 0
  while (size >= 1024 && unit < units.length - 1) { size /= 1024; unit++ }
  return `${unit === 0 ? size : size.toFixed(1)} ${units[unit]}`
}

export function integer(value: number | null | undefined): string {
  return value === null || value === undefined ? '—' : value.toLocaleString('en-US')
}

/** 分区名：8 位日期、6 位年月转成可读形式，其余（状态、交易所、指数代码）原样。 */
export function partition(value: string | null): string {
  if (!value) return '—'
  if (/^\d{8}$/.test(value)) return `${value.slice(0, 4)}-${value.slice(4, 6)}-${value.slice(6)}`
  if (/^\d{6}$/.test(value)) return `${value.slice(0, 4)}-${value.slice(4)}`
  return value
}

/** ISO 时间取「日期 时:分」，无效时给破折号。 */
export function stamp(value: string | null | undefined, withDate = true): string {
  if (!value) return '—'
  const match = /^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})/.exec(value)
  if (!match) return value
  return withDate ? `${match[1]} ${match[2]}` : match[2]
}

/** 字段单位决定数值显示：比例按百分比，金额股数用万/亿。 */
export function formatValue(value: number | null | undefined, unit: string): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  if (unit === '比例') return `${trim((value * 100).toPrecision(3))}%`
  const size = Math.abs(value)
  if (size >= 1e8) return `${trim((value / 1e8).toFixed(2))}亿`
  if (size >= 1e4) return `${trim((value / 1e4).toFixed(2))}万`
  if (size >= 100) return trim(value.toFixed(1))
  return trim(value.toPrecision(3))
}
function trim(text: string) { return text.includes('.') && !text.includes('e') ? text.replace(/\.?0+$/, '') : text }

/** 1、2、2.5、5 × 10ⁿ 的整齐刻度，落在 [low, high] 内，约 count 个。 */
export function ticks(low: number, high: number, count = 5): number[] {
  if (!(high > low)) return Number.isFinite(low) ? [low] : []
  const raw = (high - low) / Math.max(1, count - 1)
  const power = 10 ** Math.floor(Math.log10(raw))
  const step = [1, 2, 2.5, 5, 10].map(m => m * power).find(s => s >= raw) ?? 10 * power
  const output: number[] = []
  for (let value = Math.ceil(low / step - 1e-9) * step; value <= high + step * 1e-9; value += step) output.push(Number(value.toPrecision(12)))
  return output
}

/** 覆盖率折线的 SVG 路径：0–1 映射到 [height - pad, pad]，空值断开。 */
export function linePath(values: (number | null)[], width: number, height: number, pad = 2, left = 0): string {
  if (!values.length) return ''
  const span = Math.max(1, values.length - 1)
  let path = '', open = false
  values.forEach((value, index) => {
    if (value === null || !Number.isFinite(value)) { open = false; return }
    const x = left + (index / span) * (width - left), y = height - pad - Math.min(1, Math.max(0, value)) * (height - 2 * pad)
    path += `${open ? 'L' : 'M'}${x.toFixed(1)} ${y.toFixed(1)} `
    open = true
  })
  return path.trim()
}

/** 详情图横轴的年份刻度：按跨度取 1/2/3/5 年一档。 */
export function yearTicks(months: string[], wanted = 7): { index: number; label: string }[] {
  if (!months.length) return []
  const first = Number(months[0].slice(0, 4)), last = Number(months[months.length - 1].slice(0, 4))
  const step = [1, 2, 3, 5, 10].find(s => (last - first) / s <= wanted) ?? 10
  const output: { index: number; label: string }[] = []
  months.forEach((month, index) => {
    const year = Number(month.slice(0, 4))
    if (month.slice(4) === '01' && (year - first) % step === 0) output.push({ index, label: String(year) })
  })
  return output
}

export function matches(field: FieldRow, query: string): boolean {
  const needle = query.trim().toLowerCase()
  return !needle || [field.name, field.label, field.category].some(v => v.toLowerCase().includes(needle))
}
