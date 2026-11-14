// 因子库「送去回测」（2026-11-14 新增）：只选中一个因子时可用，跳到 #/backtest?entry=<id>。
import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'
import FactorLibrary from './FactorLibrary.vue'

afterEach(() => { vi.useRealTimers(); vi.restoreAllMocks(); location.hash = '' })
const entry = (id: string) => ({ id, name: id, source: 'close', settings: { universe: 'TOP800' }, expression_id: id, status: 'pass', sharpe: 1, validation: null, turnover: 0.1, fitness: 1, annual_return: 0.1, drawdown: -0.1, coverage: 1, t: 2, margin: null, latest: null })

describe('送去回测', () => {
  it('选一个时可用并跳到回测页；选两个时不可用', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, json: async () => ({ items: [entry('e1'), entry('e2')], total: 2, counts: { pass: 2 }, groups: [], offset: 0, limit: 50 }) })))
    const wrapper = mount(FactorLibrary, { props: { completions: [] }, global: { stubs: { ResultChart: true, ExpressionEditor: true, LibraryAnalytics: true } } })
    await vi.advanceTimersByTimeAsync(180); await flushPromises()
    const send = () => wrapper.findAll('button').find(b => b.text() === '送去回测')!
    expect(send().attributes('disabled')).toBeDefined()
    await wrapper.get('[role=checkbox][aria-label="选择 e1"]').trigger('click')
    expect(send().attributes('disabled')).toBeUndefined()
    await send().trigger('click')
    expect(location.hash).toBe('#/backtest?entry=e1')
    await wrapper.get('[role=checkbox][aria-label="选择 e2"]').trigger('click')
    expect(send().attributes('disabled')).toBeDefined()
    wrapper.unmount()
  })
})
