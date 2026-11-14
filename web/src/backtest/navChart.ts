// 净值图的纯计算：时间窗、窗内重新起点、窗内回撤、纵轴刻度和年份刻度。组件只负责画。
export type Range = '1y' | '3y' | '5y' | 'all'
export const RANGE_YEARS: Record<Exclude<Range, 'all'>, number> = { '1y': 1, '3y': 3, '5y': 5 }

/** 时间窗起点：末日往前 N 个自然年之后的第一个交易日；数据不够长时从头开始。 */
export function rangeStart(dates: string[], range: Range) {
  if (range === 'all' || !dates.length) return 0
  const last = dates[dates.length - 1]
  const from = `${Number(last.slice(0, 4)) - RANGE_YEARS[range]}${last.slice(4)}`
  const index = dates.findIndex(d => d >= from)
  return Math.max(0, index)
}

/** 数据跨度够不够显示这个时间窗（窗口比跨度还长时按钮不出现）。 */
export function rangeAvailable(dates: string[], range: Range) {
  if (range === 'all') return true
  return rangeStart(dates, range) > 0
}

/** 窗内以起点为 1 重新起算；起点缺值时取窗内第一个有效值。 */
export function rebase(values: (number | null)[], start: number): (number | null)[] {
  let base: number | null = null
  for (let i = start; i < values.length; i++) { const v = values[i]; if (v !== null && Number.isFinite(v) && v > 0) { base = v; break } }
  return values.slice(start).map(v => (base === null || v === null || !Number.isFinite(v)) ? null : v / base)
}

/** 窗内回撤 `nav / 历史最高 − 1`，与报告的回撤公式相同。 */
export function drawdown(nav: (number | null)[]): (number | null)[] {
  let peak = -Infinity
  return nav.map(v => {
    if (v === null || !Number.isFinite(v)) return null
    peak = Math.max(peak, v)
    return v / peak - 1
  })
}

/** 净值纵轴：1/2/2.5/5 × 10ⁿ 的整齐刻度，包住数据但不强行包含 0。 */
export function navTicks(values: (number | null)[][], count = 4) {
  let lo = Infinity, hi = -Infinity
  for (const series of values) for (const v of series) if (v !== null && Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v) }
  if (!Number.isFinite(lo)) { lo = 0.9; hi = 1.1 }
  if (hi - lo < 1e-9) { lo -= 0.05; hi += 0.05 }
  const target = (hi - lo) / count, magnitude = 10 ** Math.floor(Math.log10(target))
  const step = ([1, 2, 2.5, 5, 10].find(f => f * magnitude >= target * (1 - 1e-9)) ?? 10) * magnitude
  const first = Math.ceil(lo / step - 1e-9) * step
  const ticks: number[] = []
  for (let t = first; t <= hi + step * 1e-9; t += step) ticks.push(Number(t.toPrecision(12)))
  return { min: lo, max: hi, ticks, step }
}

/** 刻度标签的小数位：由步长决定，全轴一致（0.25 步长要两位，不能把 1.25 写成 1.3）。 */
export function tickDigits(step: number) {
  const text = Number(step.toPrecision(12)).toString()
  return Math.max(1, text.includes('.') ? text.split('.')[1].length : 0)
}

/** 回撤纵轴：0 到最深回撤，按 5%/10%/20% 等整数百分比分格。 */
export function drawdownTicks(values: (number | null)[]) {
  let lo = 0
  for (const v of values) if (v !== null && Number.isFinite(v)) lo = Math.min(lo, v)
  const depth = Math.max(-lo, 0.01)
  const step = [0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5].find(s => depth / s <= 2.5) ?? 0.5
  const ticks: number[] = []
  for (let t = 0; t >= -depth - 1e-12; t -= step) ticks.push(Number(t.toFixed(4)))
  return { min: -depth, ticks }
}

/** 每年第一个交易日的位置，用于横轴年份；只保留彼此间隔够宽的。 */
export function yearTicks(dates: string[], minimumGap = 0) {
  const ticks: { index: number; label: string }[] = []
  for (let i = 0; i < dates.length; i++) {
    const year = dates[i].slice(0, 4)
    if (i === 0 || year !== dates[i - 1].slice(0, 4)) {
      if (!ticks.length || i - ticks[ticks.length - 1].index >= minimumGap) ticks.push({ index: i, label: year })
    }
  }
  return ticks
}

/** 月份刻度（一年以内的窗口用）。 */
export function monthTicks(dates: string[]) {
  const ticks: { index: number; label: string }[] = []
  for (let i = 1; i < dates.length; i++) if (dates[i].slice(5, 7) !== dates[i - 1].slice(5, 7)) ticks.push({ index: i, label: `${Number(dates[i].slice(5, 7))}月` })
  return ticks
}

/** 折线路径；缺值断开。 */
export function linePath(values: (number | null)[], x: (i: number) => number, y: (v: number) => number) {
  let path = '', open = false
  values.forEach((v, i) => {
    if (v === null || !Number.isFinite(v)) { open = false; return }
    path += `${open ? 'L' : 'M'}${x(i).toFixed(1)} ${y(v).toFixed(1)}`
    open = true
  })
  return path
}

/** 回撤面积：从 0 线到回撤值的闭合区域。 */
export function areaPath(values: (number | null)[], x: (i: number) => number, y: (v: number) => number) {
  let path = '', run: number[] = []
  const flush = () => {
    if (run.length) {
      path += `M${x(run[0]).toFixed(1)} ${y(0).toFixed(1)}` + run.map(i => `L${x(i).toFixed(1)} ${y(values[i] as number).toFixed(1)}`).join('') + `L${x(run[run.length - 1]).toFixed(1)} ${y(0).toFixed(1)}Z`
      run = []
    }
  }
  values.forEach((v, i) => { if (v === null || !Number.isFinite(v)) flush(); else run.push(i) })
  flush()
  return path
}
