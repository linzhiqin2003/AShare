import { mount } from '@vue/test-utils'
import { defineComponent, nextTick } from 'vue'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { setShowValidation, showValidation, VALIDATION_KEY, visibleParts, visiblePeriods } from './validation'
import { useResearch, type ResearchTab } from './useResearch'
import ResearchResults from './components/ResearchResults.vue'
import CheckTable from './components/CheckTable.vue'
import TabSettings from './components/TabSettings.vue'
import type { Packed, Run, Summary } from './types'

const pack = (values: number[]): Packed => ({ shape: [values.length], data: btoa(String.fromCharCode(...new Uint8Array(new Float32Array(values).buffer))) })
const parts = [
  { part: 'is', dates: ['20240102', '20240103'], series: { 'pnl.long_short': pack([.1, .2]) } },
  { part: 'validation', dates: ['20250102', '20250103'], series: { 'pnl.long_short': pack([.3, .4]) } },
]
const period = (name: string, sharpe: number) => ({ period: name, sharpe, return: .1, turnover: .2, maxdrawdown: .05, margin: 3 })
const stats = (sharpe: number, periods: ReturnType<typeof period>[]) => ({ ...period('', sharpe), n: 10, fitness: 1, t: 2, ic: .02, coverage: .9, holding_days: 3, exposure: {}, periods, years: periods, subs: {} })
const summary: Summary = { is: stats(1.11, [period('2024', 1.11)]), validation: stats(-0.77, [period('2025', -0.77)]), trial_count: 1, correlations: [], checks: [{ key: 'validation_ratio', label: '验证 Sharpe ÷ 样本内', value: -0.69, threshold: '≥ 0.50', status: 'fail' }, { key: 'sharpe', label: 'Sharpe（样本内）', value: 1.11, threshold: '≥ 1.00', status: 'pass' }] }
const run = { id: 'validation-run', ordinal: 0, label: '20 · industry', status: 'complete', settings: { by: 'year' }, summary, elapsed: 1 } as Run
const tab = (): ResearchTab => ({ id: 'tab', name: 't', source: 'rank(close)', settings: {}, count: 1, error: null, running: false, progress: 100, job: { id: 'job', status: 'complete', total: 1, elapsed: 1, runs: [run] }, selected: run.id, visible: [run.id], chart: { dates: [], lines: [] }, view: 'returns', compound: false, relative: false, returnSelection: 'both', autoName: false, drawdown: null, benchmark: null })
afterEach(() => { setShowValidation(false); localStorage.clear() })

describe('验证期开关（2026-11-14 用户要求默认隐藏验证期）', () => {
  it('默认关闭；只认 ashare.showValidation=1；读写失败不报错', () => {
    expect(showValidation.value).toBe(false)
    setShowValidation(true)
    expect(localStorage.getItem(VALIDATION_KEY)).toBe('1')
    setShowValidation(false)
    expect(localStorage.getItem(VALIDATION_KEY)).toBe('0')
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('denied') })
    expect(() => setShowValidation(true)).not.toThrow()
    expect(showValidation.value).toBe(true)
    vi.restoreAllMocks()
  })
  it('关闭时只保留样本内部分和样本内年份', () => {
    expect(visibleParts(parts, false).map(p => p.part)).toEqual(['is'])
    expect(visibleParts(parts, true).map(p => p.part)).toEqual(['is', 'validation'])
    expect(visiblePeriods([period('2024', 1), period('2025', 2), period('2025Q1', 3)], false).map(p => p.period)).toEqual(['2024'])
  })
  it('研究台曲线关时只画到样本内末，打开开关自动重画', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, json: async () => parts })))
    let research!: ReturnType<typeof useResearch>
    const wrapper = mount(defineComponent({ setup() { research = useResearch(); return () => null } }))
    const t = tab()
    for (const view of ['returns', 'groups', 'ic'] as const) {
      t.view = view
      await research.loadChart(t)
      expect(t.chart.dates.every(d => d < '20250101')).toBe(true)
    }
    t.view = 'returns'
    setShowValidation(true)
    await research.loadChart(t)
    expect(t.chart.dates).toContain('20250103')
    wrapper.unmount()
  })
  it('参数组没有验证列、分年没有验证行，检验保留状态不显示数值；打开后恢复', async () => {
    const wrapper = mount(ResearchResults, { props: { tab: tab() }, global: { stubs: { ResultChart: true, HoldingsTable: true } } })
    expect(wrapper.findAll('.parameters th').map(t => t.text())).not.toContain('验证')
    expect(wrapper.text()).not.toContain('-0.77')
    expect(wrapper.text()).not.toContain('-0.69')
    expect(wrapper.text()).toContain('验证 Sharpe ÷ 样本内')
    expect(wrapper.text()).toContain('1 项未通过')
    await wrapper.findAll('.panel-tab').find(b => b.text() === '分年')!.trigger('click')
    expect(wrapper.text()).not.toContain('2025')
    expect(wrapper.text()).toContain('2024')
    setShowValidation(true); await nextTick()
    expect(wrapper.text()).toContain('2025 验证')
    expect(wrapper.text()).toContain('-0.77')
    await wrapper.findAll('.panel-tab').find(b => b.text() === '参数组')!.trigger('click')
    expect(wrapper.findAll('.parameters th').map(t => t.text())).toContain('验证')
    expect(wrapper.text()).toContain('-0.69')
    wrapper.unmount()
  })
  it('检验表只隐藏依赖验证期的那一条数值', async () => {
    const wrapper = mount(CheckTable, { props: { checks: summary.checks } })
    const rows = wrapper.findAll('tbody tr')
    expect(rows[0].findAll('td')[1].text()).toBe('')
    expect(rows[0].text()).toContain('未通过')
    expect(rows[1].findAll('td')[1].text()).toBe('1.11')
    setShowValidation(true); await nextTick()
    expect(wrapper.findAll('tbody tr')[0].findAll('td')[1].text()).toBe('-0.69')
    wrapper.unmount()
  })
  it('入口只在 tab 设置的「验证 2025 年」一行', async () => {
    const settings = { universe: 'TOP800', group_neutralize: [null], n_groups: 10, trade_at: 'open', by: 'year', weights_portfolio: false, sub_universes: true, trade_limitation: true, decay: 0 }
    const wrapper = mount(TabSettings, { props: { modelValue: settings, name: 'a', defaults: settings }, global: { stubs: { BenchmarkSelect: true } } })
    const toggle = wrapper.get('[role="switch"][aria-label="显示验证期"]')
    expect(toggle.element.closest('.setting-row')!.textContent).toContain('验证 2025 年')
    expect(toggle.attributes('aria-checked')).toBe('false')
    await toggle.trigger('click')
    expect(showValidation.value).toBe(true)
    expect(localStorage.getItem(VALIDATION_KEY)).toBe('1')
    expect(toggle.attributes('aria-checked')).toBe('true')
    // 开关是全局显示偏好，不改 tab 的回测设置。
    expect(wrapper.emitted('update:modelValue')).toBeUndefined()
    wrapper.unmount()
  })
})

describe('参数组的扣费年化列（2026-11-14 用户要求）', () => {
  it('按年化 − 日均换手 × 240 × 10 bp 显示在年化后面', () => {
    const wrapper = mount(ResearchResults, { props: { tab: tab() }, global: { stubs: { ResultChart: true, HoldingsTable: true } } })
    const heads = wrapper.findAll('.parameters th').map(t => t.text())
    expect(heads.indexOf('扣费年化')).toBe(heads.indexOf('年化') + 1)
    // 年化 10%，日均换手 20%：扣 0.2 × 240 × 0.001 = 4.8 个百分点。
    expect(wrapper.find('.parameters td.col-net').text()).toBe('5.2%')
    wrapper.unmount()
  })
})
