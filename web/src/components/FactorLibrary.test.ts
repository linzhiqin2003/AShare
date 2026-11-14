import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'
import FactorLibrary from './FactorLibrary.vue'

afterEach(() => { vi.useRealTimers(); vi.restoreAllMocks() })
describe('因子库服务端列表', () => {
  it('筛选后全部计数和分组数采用同一次响应，排序来自表头', async () => {
    vi.useFakeTimers()
    const requests: string[] = []
    vi.stubGlobal('fetch', vi.fn(async (input: string) => {
      requests.push(input)
      const filtered = input.includes('universe=ALL')
      const total = filtered ? 7 : 1003
      return { ok: true, json: async () => ({ items: [], total, counts: { pass: total }, groups: [], offset: 0, limit: 50 }) }
    }))
    const wrapper = mount(FactorLibrary, { props: { completions: [] }, global: { stubs: { ResultChart: true, ExpressionEditor: true } } })
    await vi.advanceTimersByTimeAsync(180); await flushPromises()
    // 2026-11-14 用户要求去掉工具栏的「全部 N」，总数只在分页处出现一次。
    expect(wrapper.text()).not.toContain('全部')
    expect(wrapper.get('.pagination').text()).toContain('/ 1003')
    await wrapper.findAll('button').find(b => b.text() === '筛选')!.trigger('click')
    await wrapper.get('[role=checkbox][aria-label="股票池 TOP1800"]').trigger('click')
    await wrapper.get('[role=checkbox][aria-label="股票池 TOP800"]').trigger('click')
    await vi.advanceTimersByTimeAsync(180); await flushPromises()
    expect(wrapper.get('.pagination').text()).toContain('/ 7')
    expect(wrapper.text()).not.toContain('1003')
    await wrapper.findAll('th button').find(b => b.text() === 'Sharpe')!.trigger('click')
    await vi.advanceTimersByTimeAsync(180); await flushPromises()
    expect(requests.at(-1)).toContain('sort=sharpe')
    expect(requests.at(-1)).toContain('universe=ALL')
    expect(wrapper.findAll('select').every(s => !s.text().includes('排序'))).toBe(true)
    wrapper.unmount()
  })
  it('鼠标移入再点击不会关闭分组和显示子菜单，选项仍可操作', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, json: async () => ({ items: [], total: 0, counts: {}, groups: [], offset: 0, limit: 50 }) })))
    const wrapper = mount(FactorLibrary, { props: { completions: [] }, global: { stubs: { ResultChart: true, ExpressionEditor: true } } })
    await vi.advanceTimersByTimeAsync(180)
    await wrapper.findAll('button').find(b => b.text() === '视图')!.trigger('click')
    await wrapper.find('.group-menu-anchor').trigger('mouseenter')
    await wrapper.find('.group-menu-anchor button').trigger('click')
    expect(wrapper.find('.submenu').text()).toContain('不分组')
    await wrapper.findAll('.submenu button').find(b => b.text() === '表达式')!.trigger('click')
    expect(wrapper.find('.view-menu').exists()).toBe(false)
    await vi.advanceTimersByTimeAsync(180); await flushPromises()
    expect(vi.mocked(fetch).mock.calls.at(-1)?.[0]).toContain('group=expression')
    await wrapper.findAll('button').find(b => b.text() === '视图')!.trigger('click')
    const columns = wrapper.findAll('.group-menu-anchor')[1]
    await columns.trigger('mouseenter'); await columns.find('button').trigger('click')
    expect(columns.find('.column-menu').exists()).toBe(true)
    expect(wrapper.findAll('.submenu')).toHaveLength(1)
    wrapper.unmount()
  })
  it('取消全部后重新选择两项，不会误选第三项', async () => {
    vi.useFakeTimers()
    const requests: string[] = []
    vi.stubGlobal('fetch',vi.fn(async (input: string) => { requests.push(input); return {ok:true,json:async () => ({items:[],total:0,counts:{},groups:[],offset:0,limit:50})} }))
    const wrapper = mount(FactorLibrary,{props:{completions:[]},global:{stubs:{ResultChart:true,ExpressionEditor:true}}})
    await vi.advanceTimersByTimeAsync(180)
    await wrapper.findAll('button').find(b => b.text() === '筛选')!.trigger('click')
    for (const label of ['通过','未通过','待检验']) await wrapper.get(`[role=checkbox][aria-label="状态 ${label}"]`).trigger('click')
    await wrapper.get('[role=checkbox][aria-label="状态 通过"]').trigger('click')
    await wrapper.get('[role=checkbox][aria-label="状态 未通过"]').trigger('click')
    expect(wrapper.get('[role=checkbox][aria-label="状态 待检验"]').attributes('aria-checked')).toBe('false')
    await vi.advanceTimersByTimeAsync(180); await flushPromises()
    expect(requests.at(-1)).toContain('status=pass%2Cfail')
    wrapper.unmount()
  })

})
