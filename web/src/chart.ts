import type { BenchmarkData, ChartData, Line, Packed, PlotPart, ReturnSelection, Run, View } from './types'
export const colors = ['#14171A', '#1F4FBF', '#A14A14', '#7C43B0', '#1E7B57', '#A86A12', '#487E98', '#A04668']
export function unpack(p: Packed): number[] {
  const bytes = Uint8Array.from(atob(p.data), c => c.charCodeAt(0))
  const data = new DataView(bytes.buffer)
  return Array.from({ length: bytes.length / 4 }, (_, i) => data.getFloat32(i * 4, true))
}
export function curves(values: number[], compound: boolean, start = 0, end = values.length - 1, benchmark?: number[]) {
  let value = compound ? 1 : 0
  let wealth = 1, baseline = 1
  let peak = value
  const cumulative: number[] = [], drawdown: number[] = []
  for (let i = start; i <= Math.min(end, values.length - 1); i++) {
    const raw = values[i]
    const ret = Number.isFinite(raw) ? raw : 0
    // 缺失基准日按零收益处理；复利比较净值之比，不能复利逐日收益差。
    const base = benchmark && Number.isFinite(benchmark[i]) ? benchmark[i] : 0
    if (compound) {
      wealth *= 1 + ret; baseline *= 1 + base
      value = baseline > 0 ? wealth / baseline : NaN
    } else value += ret - base
    if (!Number.isFinite(value)) { cumulative.push(NaN); drawdown.push(NaN); continue }
    peak = Math.max(peak, value)
    cumulative.push(compound ? value - 1 : value)
    drawdown.push(compound ? (peak ? value / peak - 1 : 0) : value - peak)
  }
  return { cumulative, drawdown }
}
export function chartData(runs: Run[], parts: PlotPart[][], view: View): ChartData {
  const dates = [...new Set(parts.flatMap(p => p.filter(v => Object.keys(v.series).length).flatMap(v => v.dates)))].sort()
  const positions = new Map(dates.map((d, i) => [d, i]))
  const lines: Line[] = []
  runs.forEach((run, ri) => {
    const collection = new Map<string, number[]>()
    for (const part of parts[ri] ?? []) {
      for (const [name, packed] of Object.entries(part.series)) {
        const rows = packed.shape.length === 2 ? packed.shape[0] : 1
        const data = unpack(packed)
        for (let row = 0; row < rows; row++) {
          const kind = name === 'pnl.long_short' ? 'long_short' : name === 'pnl.long' ? 'long' : name === 'pnl.weights' ? 'weights' : undefined
          const label = view === 'returns' ? variantLabel(run.label) + (kind === 'weights' ? ' · 权重' : kind === 'long' ? ' · 第 1 组' : '') : view === 'groups' ? `第 ${row + 1} 组` : view === 'ic' ? 'Rank IC' : name.replace('sub.', '')
          const key = `${name}:${row}`
          if (!collection.has(key)) collection.set(key, Array(dates.length).fill(NaN))
          const values = collection.get(key)!
          part.dates.forEach((date, i) => { values[positions.get(date)!] = data[row * part.dates.length + i] })
          if (!lines.some(l => l.id === `${run.id}:${key}`)) lines.push({ id: `${run.id}:${key}`, name: label, kind, color: view === 'groups' ? `hsl(220, 65%, ${32 + row * 5}%)` : colors[(view === 'returns' ? run.ordinal : lines.length) % colors.length], values })
        }
      }
    }
  })
  return { dates, lines }
}
export function relativeLines(data: ChartData, selection: ReturnSelection): ChartData {
  return { ...data, lines: data.lines.filter(l => l.benchmark || (l.kind !== 'weights' && (selection === 'both' || l.kind === selection))).map(l => l.kind === 'long_short' ? { ...l, name: `${l.name} · 多空` } : l) }
}
export function fmt(value: number | null | undefined, percent = false, digits = 2) {
  return value == null || !Number.isFinite(value) ? '—' : `${(percent ? value * 100 : value).toFixed(digits)}${percent ? '%' : ''}`
}
export function configLabel(settings: Record<string, unknown>) {
  const neutral: Record<string, string> = { none: '无', industry: '行业', cap: '市值', 'cap+industry': '市值 + 行业' }
  return `${settings.universe} · ${neutral[String(settings.group_neutralize ?? 'none')]} · 衰减 ${settings.decay} · 调仓 ${settings.rebalance ?? 1} 天`
}
export function variantLabel(label: string) {
  return label.replace('cap+industry', '市值 + 行业').replace('industry', '行业').replace('cap', '市值')
}

export function withBenchmark(data: ChartData, benchmark: BenchmarkData | null): ChartData {
  if (!benchmark?.symbol || !data.dates.length) return data
  const values = new Map<string,number>()
  for (const part of benchmark.parts) { const packed = part.series.benchmark; if (!packed) continue; const numbers = unpack(packed); part.dates.forEach((date,i) => values.set(date,numbers[i])) }
  return { ...data, lines: [...data.lines, { id: 'benchmark', name: benchmark.name, color: '#8A9199', values: data.dates.map(d => values.get(d) ?? NaN), benchmark: true }] }
}

export function niceAxis(values: number[][], count = 4, integerPercent = false, tightLower = false): { min: number; max: number; ticks: number[] } {
  let lo = 0, hi = 0
  for (const series of values) for (const value of series) if (Number.isFinite(value)) { lo = Math.min(lo,value); hi = Math.max(hi,value) }
  const span = hi - lo || .1
  const scale = integerPercent ? 100 : 1
  const target = Math.max(integerPercent ? 1 : 0, span / count * scale), magnitude = 10 ** Math.floor(Math.log10(target))
  const step = ([1,2,2.5,5,10].find(v => v * magnitude >= target - Math.abs(target) * 1e-10 && (!integerPercent || Number.isInteger(v * magnitude))) ?? 10) * magnitude / scale
  // 整数边界允许浮点误差；主图下沿留少量呼吸空间，不扩成一整档空刻度。
  const firstTick = (tightLower ? Math.ceil(lo / step - 1e-10) : Math.floor(lo / step + 1e-10)) * step
  const min = tightLower && lo < 0 ? lo - span * .025 : firstTick
  const max = Math.ceil((hi === lo ? hi + step : hi) / step - 1e-10) * step
  return { min, max, ticks: Array.from({ length: Math.round((max-firstTick)/step)+1 },(_,i) => Number((firstTick+i*step).toPrecision(12))) }
}
