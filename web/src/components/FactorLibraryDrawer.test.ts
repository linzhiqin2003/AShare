import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'
import FactorLibrary from './FactorLibrary.vue'
import { setShowValidation } from '../validation'
import type { Entry, Summary } from '../types'

const stats = (sharpe: number) => ({ sharpe, return: .1, turnover: .2, maxdrawdown: .1, margin: 10, holding_days: 5, fitness: 1, t: 3, ic: .03, coverage: .9, n: 100, period: '', exposure: {}, periods: [], years: [], subs: {} })
const summary = (validation: number): Summary => ({ is: stats(1.23), validation: stats(validation), trial_count: 1, correlations: [], checks: [{ key: 'validation_ratio', label: '验证 Sharpe ÷ 样本内', value: validation / 1.23, threshold: '≥ 0.50', status: 'fail' }] })
const entry = (id: string, validation: number): Entry => ({ id, name: `factor_${id}`, source: 'rank(close)', settings: { universe: 'TOP800', group_neutralize: 'industry', decay: 0, rebalance: 1 }, expression_id: id, status: 'pass', sharpe: 1.23, validation, turnover: .2, fitness: 1, annual_return: .1, drawdown: .1, coverage: .9, t: 3, margin: 10, latest: null })
const entries = [entry('a', -0.77), entry('b', 0.66), entry('c', 0.55)]

function mockFetch() {
  const requests: string[] = []
  vi.stubGlobal('fetch', vi.fn(async (input: string) => {
    requests.push(input)
    const detail = input.match(/^\/api\/library\/(\w)$/)
    if (detail) return { ok: true, json: async () => ({ ...entries.find(e => e.id === detail[1])!, summary: summary(entries.find(e => e.id === detail[1])!.validation!) }) }
    if (input.startsWith('/api/library/correlation')) return { ok: true, json: async () => ({ ids: [], matrix: [] }) }
    if (input.startsWith('/api/compare')) return { ok: true, json: async () => ({ items: [], dates: [], series: {}, correlation: [] }) }
    return { ok: true, json: async () => ({ items: entries, total: entries.length, counts: { pass: 3 }, groups: [], offset: 0, limit: 50 }) }
  }))
  return requests
}
async function mountLibrary() {
  vi.useFakeTimers()
  const requests = mockFetch()
  const wrapper = mount(FactorLibrary, { props: { completions: [] }, attachTo: document.body, global: { stubs: { ResultChart: true, ExpressionEditor: true, LibraryAnalytics: true, FactorCompare: { props: ['selected'], template: '<div class="compare-stub">{{ selected.map(e => e.id).join(",") }}</div>' } } } })
  await vi.advanceTimersByTimeAsync(180); await flushPromises()
  return { wrapper, requests }
}
afterEach(() => { vi.useRealTimers(); setShowValidation(false); localStorage.clear() })

describe('因子详情从右侧滑出（2026-11-14 用户要求，原先用 SplitPane 挤压列表）', () => {
  it('浮在列表上方，不挤压列表；点别的因子直接换内容；Esc 关闭并把焦点还给原来的行', async () => {
    const { wrapper } = await mountLibrary()
    const rows = () => wrapper.findAll('tr.entry-row')
    await rows()[0].trigger('click'); await flushPromises()
    const drawer = wrapper.get('aside.detail-drawer')
    // 不再出现列表与详情之间的分隔条：列表宽度不变。
    expect(wrapper.find('[aria-label="调整因子列表与详情宽度"]').exists()).toBe(false)
    expect(drawer.element.parentElement?.classList.contains('split-child')).toBe(false)
    expect(document.activeElement).toBe(drawer.element)
    expect(drawer.text()).toContain('factor_a')
    await rows()[1].trigger('click'); await flushPromises()
    // 同一个抽屉元素换内容，没有关闭再打开。
    expect(wrapper.get('aside.detail-drawer').element).toBe(drawer.element)
    expect(drawer.text()).toContain('factor_b')
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }))
    await flushPromises()
    expect(wrapper.find('aside.detail-drawer').exists()).toBe(false)
    expect(document.activeElement).toBe(rows()[1].element)
    wrapper.unmount()
  })
  it('点 × 关闭；勾选框点击不打开详情', async () => {
    const { wrapper } = await mountLibrary()
    await wrapper.get('input[aria-label="选择 factor_a"]').trigger('click')
    expect(wrapper.find('aside.detail-drawer').exists()).toBe(false)
    await wrapper.findAll('tr.entry-row')[2].trigger('click'); await flushPromises()
    await wrapper.get('button[aria-label="关闭详情"]').trigger('click')
    expect(wrapper.find('aside.detail-drawer').exists()).toBe(false)
    expect(document.activeElement).toBe(wrapper.findAll('tr.entry-row')[2].element)
    wrapper.unmount()
  })
})

describe('验证期默认隐藏（2026-11-14 用户要求，避免看着验证期调参）', () => {
  it('关闭时：列表没有验证列且不能勾选、不能按它排序，详情只有样本内一列，检验不显示数值', async () => {
    const { wrapper } = await mountLibrary()
    expect(wrapper.findAll('th').map(t => t.text())).not.toContain('验证')
    expect(wrapper.text()).not.toContain('-0.77')
    await wrapper.findAll('button').find(b => b.text() === '视图')!.trigger('click')
    const columns = wrapper.findAll('.group-menu-anchor')[1]
    await columns.trigger('mouseenter')
    const box = columns.findAll('label').find(l => l.text() === '验证')!.get('input')
    expect(box.attributes('disabled')).toBeDefined()
    expect((box.element as HTMLInputElement).checked).toBe(false)
    await wrapper.findAll('tr.entry-row')[0].trigger('click'); await flushPromises()
    const drawer = wrapper.get('aside.detail-drawer')
    expect(drawer.findAll('.detail-metrics th').map(t => t.text())).toEqual(['', '样本内'])
    expect(drawer.text()).not.toContain('-0.77')
    expect(drawer.text()).not.toContain('-0.63')
    expect(drawer.text()).toContain('验证 Sharpe ÷ 样本内')
    expect(drawer.text()).toContain('未通过')
    wrapper.unmount()
  })
  it('打开时恢复：可显示验证列并排序，详情有验证列和检验数值', async () => {
    setShowValidation(true)
    const { wrapper, requests } = await mountLibrary()
    expect(wrapper.findAll('th').map(t => t.text())).not.toContain('验证')
    await wrapper.findAll('button').find(b => b.text() === '视图')!.trigger('click')
    const columns = wrapper.findAll('.group-menu-anchor')[1]
    await columns.trigger('mouseenter')
    await columns.findAll('label').find(l => l.text() === '验证')!.get('input').setValue(true)
    expect(wrapper.findAll('th').map(t => t.text())).toContain('验证')
    expect(wrapper.text()).toContain('-0.77')
    await wrapper.findAll('th button').find(b => b.text() === '验证')!.trigger('click')
    await vi.advanceTimersByTimeAsync(180); await flushPromises()
    expect(requests.filter(r => r.startsWith('/api/library?')).at(-1)).toContain('sort=validation')
    // 关掉开关：按验证排序退回默认，列随之消失。
    setShowValidation(false)
    await vi.advanceTimersByTimeAsync(180); await flushPromises()
    expect(requests.filter(r => r.startsWith('/api/library?')).at(-1)).toContain('sort=updated')
    expect(wrapper.findAll('th').map(t => t.text())).not.toContain('验证')
    setShowValidation(true); await flushPromises()
    await wrapper.findAll('tr.entry-row')[0].trigger('click'); await flushPromises()
    const drawer = wrapper.get('aside.detail-drawer')
    expect(drawer.findAll('.detail-metrics th').map(t => t.text())).toEqual(['', '样本内', '验证'])
    expect(drawer.text()).toContain('-0.77')
    expect(drawer.text()).toContain('-0.63')
    wrapper.unmount()
  })
})

describe('勾选后进入对比（2026-11-14 用户要求，照 p5 设计稿）', () => {
  it('选两个以上才能对比，进入后把所选因子交给对比页，返回仍保留勾选', async () => {
    const { wrapper } = await mountLibrary()
    const compare = () => wrapper.findAll('button').find(b => b.text() === '对比')!
    expect(compare().attributes('disabled')).toBeDefined()
    await wrapper.get('input[aria-label="选择 factor_a"]').setValue(true)
    expect(wrapper.text()).toContain('已选 1 个')
    expect(compare().attributes('disabled')).toBeDefined()
    await wrapper.get('input[aria-label="选择 factor_c"]').setValue(true)
    expect(wrapper.text()).toContain('已选 2 个')
    await compare().trigger('click')
    expect(wrapper.get('.compare-stub').text()).toBe('a,c')
    wrapper.unmount()
  })
})
