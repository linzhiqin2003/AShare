import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { bestValues, distinctLabels, metricColumns, settingsSummary, toCsv, volatilityAligned, DEFAULT_METRICS, type CompareData, type CompareItem, type CompareStats } from './compare'
import { setShowValidation } from './validation'
import FactorCompare from './components/FactorCompare.vue'
import type { Entry, Packed } from './types'

const pack = (values: number[]): Packed => ({ shape: [values.length], data: btoa(String.fromCharCode(...new Uint8Array(new Float32Array(values).buffer))) })
const year = (period: string, sharpe: number) => ({ period, sharpe, return: sharpe / 10, maxdrawdown: .05 })
const stats = (sharpe: number, years: ReturnType<typeof year>[], extra: Partial<CompareStats> = {}): CompareStats => ({ sharpe, return: .1, turnover: .2, maxdrawdown: .12, margin: 10, fitness: 1, t: 3, ic: .03, coverage: .9, holding_days: 5, years, exposure: { size: .1 }, subs: { TOP800: { sharpe: .5 } }, ...extra })
const settings = { universe: 'TOP1800', group_neutralize: 'industry', decay: 0, rebalance: 1, trade_at: 'open' }
const item = (id: string, sharpe: number, validation?: number, extra: Partial<CompareStats> = {}): CompareItem => ({ id, name: `f_${id}`, source: 'rank(close)', settings, status: 'pass', latest: id, is: stats(sharpe, [year('2023', sharpe), year('2024', -sharpe)], extra), ...(validation === undefined ? {} : { validation: stats(validation, [year('2025', validation)]) }), correlation: { name: 'other', value: -.4 } })

afterEach(() => { setShowValidation(false); localStorage.clear(); vi.useRealTimers() })

describe('对比页纯函数（2026-11-14 因子对比页）', () => {
  it('验证期关闭时没有验证列，分年列只到样本内末', () => {
    const items = [item('a', 1.5, -0.8), item('b', .9, 1.2)]
    const hidden = metricColumns(DEFAULT_METRICS, items, false)
    expect(hidden.map(c => c.key)).not.toContain('validation_sharpe')
    expect(hidden.filter(c => c.year).map(c => c.label)).toEqual(['23', '24'])
    const shown = metricColumns(DEFAULT_METRICS, items, true)
    expect(shown.map(c => c.key)).toContain('validation_sharpe')
    expect(shown.filter(c => c.year).map(c => c.label)).toEqual(['23', '24', '25'])
    // 列顺序固定，与勾选顺序无关。
    expect(metricColumns(['t', 'sharpe'], items, false).map(c => c.key)).toEqual(['sharpe', 't'])
  })
  it('每列最优：越大越好、越小越好、回撤按绝对值；只有一个有效值不标', () => {
    const items = [item('a', 1.5, undefined, { turnover: .5, maxdrawdown: -.3 }), item('b', .9, undefined, { turnover: .1, maxdrawdown: -.1 }), item('c', 1.1, undefined, { margin: null })]
    const columns = metricColumns(['sharpe', 'turnover', 'maxdrawdown', 'holding_days'], items, false)
    const best = bestValues(columns, items)
    expect(best.get('sharpe')).toBe('a')
    expect(best.get('turnover')).toBe('b')
    expect(best.get('maxdrawdown')).toBe('b')
    expect(best.has('holding_days')).toBe(false)
    expect(bestValues(metricColumns(['sharpe'], [items[0]], false), [items[0]]).size).toBe(0)
  })
  it('同名因子附上它们之间真正不同的项：表达式取值和配置差异；名字唯一的不变（2026-11-14 用户要求）', () => {
    const row = (id: string, name: string, source: string, extra: Record<string, unknown> = {}) => ({ id, name, source, settings: { ...settings, ...extra } })
    const labels = distinctLabels([
      row('a', 'low_vol', 'vol = ts_std(returns, 20)\nlow_vol = -vol'),
      row('b', 'low_vol', 'vol = ts_std(returns, 60)\nlow_vol = -vol'),
      row('c', 'low_vol', 'vol = ts_std(returns, 20)\nlow_vol = -vol', { group_neutralize: 'cap+industry' }),
      row('d', 'reversal', '-ts_delta(close, 20)'),
      row('e', 'size', 'rank(cap)', { rebalance: 5 }),
    ])
    expect(labels.get('a')).toBe('low_vol · 20 · 行业')
    expect(labels.get('b')).toBe('low_vol · 60 · 行业')
    expect(labels.get('c')).toBe('low_vol · 20 · 市值 + 行业')
    expect(labels.get('d')).toBe('reversal')
    expect(labels.get('e')).toBe('size')
    // 只有配置不同；缺省调仓与显式 1 天视为相同。
    const configOnly = distinctLabels([row('x', 'f', 'rank(close)', { universe: 'TOP800' }), row('y', 'f', 'rank(close)', { universe: 'ALL', rebalance: null })])
    expect([configOnly.get('x'), configOnly.get('y')]).toEqual(['f · TOP800', 'f · 全市场'])
    // 表达式结构不同、配置相同时按序号区分，不会两个看起来一样。
    const shapes = distinctLabels([row('p', 'g', 'rank(close)'), row('q', 'g', 'rank(close / open)')])
    expect(new Set([shapes.get('p'), shapes.get('q')]).size).toBe(2)
  })
  it('设定只在不同时给出一句话', () => {
    // 2026-11-14 用户要求：只在真有不同时显示，一致时不放说明。
    expect(settingsSummary([settings, settings])).toBe('')
    expect(settingsSummary([settings, { ...settings, rebalance: null }])).toBe('')
    expect(settingsSummary([settings, { ...settings, universe: 'TOP800', decay: 3 }])).toBe('设定不同：股票池、衰减')
    expect(settingsSummary([])).toBe('')
  })
  it('按波动对齐只改量级：缩放后各线日波动相同，符号与零值不变', () => {
    const lines = volatilityAligned([{ id: 'a', name: 'a', color: '', values: [.01, -.01, .02, NaN] }, { id: 'b', name: 'b', color: '', values: [.03, -.03, .06, 0] }])
    const std = (v: number[]) => { const f = v.filter(Number.isFinite); const m = f.reduce((a, b) => a + b, 0) / f.length; return Math.sqrt(f.reduce((a, b) => a + (b - m) ** 2, 0) / (f.length - 1)) }
    expect(std(lines[0].values)).toBeCloseTo(std(lines[1].values), 12)
    expect(lines[0].values.map(Math.sign).slice(0, 3)).toEqual([1, -1, 1])
    expect(Number.isNaN(lines[0].values[3])).toBe(true)
  })
  it('导出的是表上看到的列，关闭验证期时不含验证数值', () => {
    const items = [item('a', 1.5, -0.8), item('b', .9, 1.2)]
    const csv = toCsv(metricColumns(DEFAULT_METRICS, items, false), items)
    expect(csv.split('\n')[0]).toContain('Sharpe 样本内')
    expect(csv).not.toContain('验证')
    expect(csv).not.toContain('-0.8')
    expect(csv.split('\n')[1].startsWith('f_a,1.5,')).toBe(true)
  })
})

const entries = ['a', 'b'].map(id => ({ id, name: `f_${id}`, source: 'rank(close)', settings, sharpe: 1, latest: id }) as unknown as Entry)
function response(validation: boolean): CompareData {
  const dates = ['20241230', '20241231', ...(validation ? ['20250102'] : [])]
  return { items: [item('a', 1.5, validation ? -0.77 : undefined), item('b', .9, validation ? 1.23 : undefined)].map(i => validation ? i : (({ validation: _, ...rest }) => rest)(i)), dates, series: { a: pack(dates.map(() => .01)), b: pack(dates.map(() => -.01)) }, correlation: [[1, .91], [.91, 1]] }
}
async function mountCompare(transform: (data: CompareData) => CompareData = d => d) {
  vi.useFakeTimers()
  const requests: string[] = []
  vi.stubGlobal('fetch', vi.fn(async (input: string) => {
    requests.push(input)
    if (input.startsWith('/api/compare')) return { ok: true, json: async () => transform(response(input.includes('validation=true'))) }
    return { ok: true, json: async () => ({ items: entries, total: 2, counts: {}, groups: [], offset: 0, limit: 50 }) }
  }))
  const wrapper = mount(FactorCompare, { props: { selected: entries }, global: { stubs: { ResultChart: { props: ['data'], template: '<div class="chart-stub" :data-lines="data.lines.map(l => l.name).join(\'|\')">{{ data.dates.join(",") }}</div>' } } } })
  await vi.advanceTimersByTimeAsync(200); await flushPromises()
  return { wrapper, requests }
}

describe('因子对比页（2026-11-14，照 p3 设计稿）', () => {
  it('默认不请求也不显示验证期，图只到样本内末；打开开关重新请求并恢复', async () => {
    const { wrapper, requests } = await mountCompare()
    expect(requests.find(r => r.startsWith('/api/compare'))).toContain('validation=false')
    expect(wrapper.get('.chart-stub').text()).toBe('20241230,20241231')
    const headers = wrapper.findAll('.metrics-table th').map(t => t.text())
    expect(headers).not.toContain('验证')
    expect(headers).not.toContain('25')
    expect(wrapper.text()).not.toContain('-0.77')
    // 2026-11-14 用户要求：设定一致时不显示说明，相关矩阵下不放「两两相关」那句话。
    expect(wrapper.text()).not.toContain('设定')
    expect(wrapper.text()).not.toContain('两两相关')
    setShowValidation(true)
    await vi.advanceTimersByTimeAsync(10); await flushPromises()
    expect(requests.filter(r => r.startsWith('/api/compare')).at(-1)).toContain('validation=true')
    expect(wrapper.get('.chart-stub').text()).toContain('20250102')
    expect(wrapper.findAll('.metrics-table th').map(t => t.text())).toContain('验证')
    expect(wrapper.text()).toContain('-0.77')
    wrapper.unmount()
  })
  it('选择指标：验证 Sharpe 在开关关闭时不可选；存为视图后保留', async () => {
    const { wrapper } = await mountCompare()
    await wrapper.get('.chooser-toggle').trigger('click')
    const validation = wrapper.findAll('.chooser label').find(l => l.text() === '验证 Sharpe')!.get('[role=checkbox]')
    expect(validation.attributes('disabled')).toBeDefined()
    await wrapper.findAll('.chooser label').find(l => l.text() === 'Margin')!.get('[role=checkbox]').trigger('click')
    expect(wrapper.findAll('.metrics-table th').map(t => t.text())).not.toContain('Margin')
    await wrapper.findAll('.chooser button').find(b => b.text() === '存为视图')!.trigger('click')
    expect(JSON.parse(localStorage.getItem('ashare.compare.metrics')!)).not.toContain('margin')
    expect(wrapper.find('.chooser').exists()).toBe(false)
    wrapper.unmount()
  })
  it('左侧勾选与取消交给因子库的同一份选择', async () => {
    const { wrapper } = await mountCompare()
    await wrapper.get('[role=checkbox][aria-label="对比 f_b"]').trigger('click')
    expect(wrapper.emitted('toggle')![0][0]).toMatchObject({ id: 'b' })
    await wrapper.findAll('button').find(b => b.text() === '因子库')!.trigger('click')
    expect(wrapper.emitted('close')).toHaveLength(1)
    wrapper.unmount()
  })
  it('同名因子在图例、左侧列表、矩阵和指标表里都附上区分它们的配置（2026-11-14 用户要求）', async () => {
    const same = (data: CompareData) => ({ ...data, items: data.items.map((item, i) => ({ ...item, name: 'low_vol', source: `-ts_std(returns, ${i ? 60 : 20})` })) })
    const { wrapper } = await mountCompare(same)
    expect(wrapper.get('.chart-stub').attributes('data-lines')).toBe('low_vol · 20|low_vol · 60')
    expect(wrapper.findAll('.matrix tbody th').map(t => t.text())).toEqual(['low_vol · 20', 'low_vol · 60'])
    expect(wrapper.findAll('.factor-cell').map(t => t.text())).toEqual(['low_vol · 20', 'low_vol · 60'])
    wrapper.unmount()
  })
})
