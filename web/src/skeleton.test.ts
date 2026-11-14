// 2026-11-14 用户：各页首屏用骨架屏 + 流光占位，替代空白面板或转圈。
// 只在「第一次还没有数据」时出现；数据到了骨架消失；之后的刷新保留旧内容，不闪回骨架。
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import UiSkeleton from './components/ui/UiSkeleton.vue'
import DataPage from './pages/DataPage.vue'
import SettingsPage from './pages/SettingsPage.vue'
import PaperPage from './pages/PaperPage.vue'
import BacktestPage from './pages/BacktestPage.vue'
import FactorLibrary from './components/FactorLibrary.vue'
import App from './App.vue'

type Table = Record<string, (init?: RequestInit) => unknown>
/** 假 fetch：请求先挂起，release() 后按表应答；hold() 让之后的请求重新挂起（模拟慢刷新）。 */
function gated(table: Table) {
  let open = false
  const waiting: (() => void)[] = []
  const calls: string[] = []
  vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
    const path = url.replace(/^\/api/, '').split('?')[0]
    calls.push(`${init?.method ?? 'GET'} ${path}`)
    if (!open) await new Promise<void>(resolve => waiting.push(resolve))
    const key = Object.keys(table).find(k => k.endsWith('*') ? path.startsWith(k.slice(0, -1)) : path === k)
    return new Response(JSON.stringify(key ? table[key](init) : { message: '没有' }), { status: key ? 200 : 404 })
  }))
  return {
    calls,
    async release() { open = true; waiting.splice(0).forEach(resolve => resolve()); await flushPromises(); await flushPromises() },
    hold() { open = false },
  }
}
const skeletons = (wrapper: { findAll: (s: string) => unknown[] }) => wrapper.findAll('.ui-skeleton').length
// unstubAllGlobals 会连带撤掉 test-setup 里的 ResizeObserver 替身，每个用例前补上。
beforeEach(() => vi.stubGlobal('ResizeObserver', class { observe() {} unobserve() {} disconnect() {} }))
afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); localStorage.clear() })

describe('UiSkeleton', () => {
  it('块与多行两种形状，对读屏隐藏，多行宽度错落', () => {
    const block = mount(UiSkeleton, { props: { shape: 'block', height: '40px' } })
    expect(block.get('.ui-skeleton.block').attributes('style')).toContain('height: 40px')
    expect(block.attributes('aria-hidden')).toBe('true')
    const lines = mount(UiSkeleton, { props: { lines: 4 } })
    const widths = lines.findAll('.ui-skeleton.line').map(l => l.attributes('style'))
    expect(widths).toHaveLength(4)
    expect(new Set(widths).size).toBeGreaterThan(1)
  })
  it('流光只动 transform、颜色走主题变量，减少动态效果时停掉', () => {
    const source = readFileSync(join(__dirname, 'components/ui/UiSkeleton.vue'), 'utf8')
    expect(source).toMatch(/background:var\(--skeleton\)/)
    expect(source).toMatch(/var\(--skeleton-shine\)/)
    expect(source).toMatch(/@keyframes ui-skeleton-shimmer\{to\{transform:translateX\(100%\)\}\}/)
    expect(source).toMatch(/prefers-reduced-motion:reduce\)\{\.ui-skeleton::after\{animation:none/)
    // 主内容不能靠 opacity:0 等 JS 加类才出现（首屏白屏的老坑），骨架是额外的一层。
    expect(source).not.toMatch(/opacity:0/)
  })
})

const field = (name: string) => ({ name, label: name, category: '行情', dtype: 'float64', unit: '元', start: '2007-01-04', latest: 0.9, spark: [0.9, 1], references: 0 })
const overview = { datasets: [{ key: 'stock', name: '股票日频面板', panel: { version: 'abcdef1234567890', start: '2007-01-04', end: '2026-11-12', days: 4825, symbols: 5923, fields: 48, size: 6.78e9, built_at: null }, verify: null, tables: [], history: [] }] }
const detail = (name: string) => ({ ...field(name), overall: 0.8, latest_count: 5000, monthly: [['200701', 0.5]], distribution: null, true_share: null, factors: [] })

describe('数据页', () => {
  it('概览、字段表、字段详情首次加载显示骨架，统计未就绪时保持，就绪后消失', async () => {
    vi.useFakeTimers()
    let ready = false
    const gate = gated({ '/data/overview': () => overview, '/data/fields': () => ({ state: ready ? 'ready' : 'pending', version: 'v', fields: ready ? [field('close')] : [] }), '/data/fields/*': () => detail('close') })
    const wrapper = mount(DataPage, { props: { completions: [] } })
    expect(wrapper.find('.datasets .ui-skeleton').exists()).toBe(true)
    expect(wrapper.find('.fields-table .ui-skeleton').exists()).toBe(true)
    expect(wrapper.find('.detail .ui-skeleton').exists()).toBe(true)
    await gate.release()
    expect(wrapper.find('.datasets .ui-skeleton').exists()).toBe(false)
    expect(wrapper.find('.fields-table .ui-skeleton').exists()).toBe(true)
    ready = true
    await vi.advanceTimersByTimeAsync(1600); await flushPromises()
    expect(skeletons(wrapper)).toBe(0)
    expect(wrapper.text()).toContain('close')
    wrapper.unmount()
  })
  it('接口失败时不再转骨架，只给友好提示', async () => {
    const gate = gated({})
    const wrapper = mount(DataPage, { props: { completions: [] } })
    await gate.release()
    expect(skeletons(wrapper)).toBe(0)
    expect(wrapper.find('[role=alert]').exists()).toBe(true)
    wrapper.unmount()
  })
})

describe('设置页', () => {
  it('默认设定先显示带名称的骨架行；保存时保留当前值，不闪回骨架', async () => {
    const gate = gated({ '/preferences': () => ({ backtest: { universe: 'TOP1800', rebalance: 1 }, builtin: {}, keys: [] }), '/benchmarks': () => ({ items: [] }) })
    const wrapper = mount(SettingsPage)
    expect(wrapper.find('#settings-backtest .ui-skeleton').exists()).toBe(true)
    expect(wrapper.get('#settings-backtest').text()).toContain('股票池')
    await gate.release()
    expect(skeletons(wrapper)).toBe(0)
    gate.hold()
    await wrapper.findAll('button').find(b => b.text() === '恢复内置默认')!.trigger('click')
    await flushPromises()
    expect(gate.calls.at(-1)).toBe('DELETE /preferences')
    expect(skeletons(wrapper)).toBe(0)
    await gate.release()
    wrapper.unmount()
  })
})

const libraryPage = (items: { id: string; name: string }[]) => ({ items: items.map(i => ({ ...i, source: 'x', settings: {}, status: 'pass', sharpe: 1, turnover: null, latest: null })), total: items.length, counts: { pass: items.length }, groups: [], offset: 0, limit: 50, facets: {} })

describe('因子库', () => {
  it('第一次列表到之前列表和分析区是骨架；改搜索重新请求时保留旧行', async () => {
    vi.useFakeTimers()
    const gate = gated({ '/library': () => libraryPage([{ id: 'e1', name: 'low_vol' }]) })
    const wrapper = mount(FactorLibrary, { props: { completions: [] }, global: { stubs: { ResultChart: true, ExpressionEditor: true } } })
    await vi.advanceTimersByTimeAsync(180)
    expect(wrapper.find('.library-skeleton .ui-skeleton').exists()).toBe(true)
    expect(wrapper.find('.analytics-skeleton').exists()).toBe(true)
    expect(wrapper.text()).not.toContain('没有符合条件的因子')
    await gate.release()
    expect(skeletons(wrapper)).toBe(0)
    expect(wrapper.text()).toContain('low_vol')
    gate.hold()
    await wrapper.get('input[aria-label="搜索因子"]').setValue('low')
    await vi.advanceTimersByTimeAsync(400); await flushPromises()
    expect(gate.calls.filter(c => c.startsWith('GET /library')).length).toBeGreaterThan(1)
    expect(skeletons(wrapper)).toBe(0)
    expect(wrapper.text()).toContain('low_vol')
    await gate.release()
    wrapper.unmount()
  })
})

describe('模拟盘', () => {
  it('账户列表到之前摆三栏骨架，到了以后换成内容', async () => {
    const gate = gated({ '/paper/accounts': () => ({ items: [] }) })
    const wrapper = mount(PaperPage)
    expect(wrapper.findAll('[aria-busy=true]').length).toBeGreaterThan(0)
    expect(skeletons(wrapper)).toBeGreaterThan(0)
    await gate.release()
    expect(skeletons(wrapper)).toBe(0)
    expect(wrapper.text()).toContain('还没有模拟账户')
    wrapper.unmount()
  })
})

const backtestOptions = { defaults: { universe: 'csi300', rebalance: 'month', top_n: 30, trade_at: 'open', capital: 1000, commission: 0.00025, commission_min: 5, slippage_bps: 5, dividend_tax: 0.2, carry_blocked: true, lots: 'board', benchmark: '000001.SH' }, choices: { universe: ['csi300'], trade_at: ['open'] }, limits: {}, windows: {} }

describe('回测页', () => {
  it('选项到之前左侧配置、净值图、右侧统计都是骨架，到了以后换成真实组件', async () => {
    const gate = gated({ '/backtests/options': () => backtestOptions, '/completions': () => [] })
    const stubs = { ExpressionEditor: true, BacktestConfig: true, BacktestSide: true, BacktestTables: true, NavChart: true }
    const wrapper = mount(BacktestPage, { global: { stubs } })
    await flushPromises()
    expect(wrapper.find('.bt-skeleton .ui-skeleton').exists()).toBe(true)
    expect(wrapper.find('.chart-skeleton .ui-skeleton').exists()).toBe(true)
    expect(wrapper.find('.side-skeleton .ui-skeleton').exists()).toBe(true)
    await gate.release()
    expect(skeletons(wrapper)).toBe(0)
    expect(wrapper.findComponent({ name: 'BacktestSide' }).exists()).toBe(true)
    wrapper.unmount()
  })
  it('选项读取失败时不再转骨架', async () => {
    const gate = gated({})
    const wrapper = mount(BacktestPage, { global: { stubs: { ExpressionEditor: true, BacktestConfig: true, BacktestSide: true, BacktestTables: true, NavChart: true } } })
    await gate.release()
    expect(skeletons(wrapper)).toBe(0)
    wrapper.unmount()
  })
})

describe('研究台', () => {
  it('标签和结果第一次恢复之前，编辑器和结果区是骨架', async () => {
    const gate = gated({ '/settings': () => ({ defaults: {} }), '/completions': () => [], '/tabs': init => init?.method === 'POST' ? { id: 't1' } : [], '/meta': () => ({ data_date: null, running: 0, max_workers: 4 }) })
    const stubs = { ExpressionEditor: true, ResearchResults: true, StatusBar: true, TabSettings: true }
    const wrapper = mount(App, { global: { stubs } })
    await flushPromises()
    expect(wrapper.find('.editor-skeleton .ui-skeleton').exists()).toBe(true)
    expect(wrapper.find('.results-skeleton .ui-skeleton').exists()).toBe(true)
    await gate.release()
    expect(wrapper.find('.results-skeleton').exists()).toBe(false)
    expect(wrapper.find('.editor-skeleton').exists()).toBe(false)
    expect(wrapper.findComponent({ name: 'ResearchResults' }).exists()).toBe(true)
    wrapper.unmount()
  })
})
