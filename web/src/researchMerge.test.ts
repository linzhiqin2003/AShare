// 研究台与回测合并成一个「研究」页（2026-11-14，方向 A：视图切换）。
// 旧链接跳转、旧 localStorage 迁移、两个视图共用一份源码和成交设置、运行按钮只跑当前视图。
import { flushPromises, mount } from '@vue/test-utils'
import { defineComponent, nextTick } from 'vue'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { pageFromHash, PAGES } from './router'
import { useResearch } from './useResearch'

const A = 'a'.repeat(24), B = 'b'.repeat(24), C = 'c'.repeat(24)
const BT_DEFAULTS = { universe: 'csi300', rebalance: 'month', top_n: 30, trade_at: 'open', capital: 1000, commission: 0.00025, commission_min: 5, slippage_bps: 5, dividend_tax: 0.2, carry_blocked: true, lots: 'board', benchmark: '000001.SH' }
const RESEARCH_DEFAULTS = { universe: 'TOP1800', trade_at: 'open', trade_limitation: true, n_groups: 10, group_neutralize: null, decay: 0, rebalance: 1, by: 'year', weights_portfolio: false, sub_universes: true }
const btState = (id: string, status: string) => ({ id, name: 'vol', status, progress: null, queue_position: null, elapsed: null, settings: BT_DEFAULTS })

interface Call { method: string; path: string; body: unknown }
let calls: Call[] = []
let ids = 0
/** 假后端：按路径应答；记录每次请求。 */
function server(extra: Record<string, (body: unknown, path: string) => unknown> = {}) {
  calls = []; ids = 0
  const states = new Map([[A, btState(A, 'complete')], [B, btState(B, 'complete')]])
  vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
    const full = url.replace(/^\/api/, ''), path = full.split('?')[0], method = init?.method ?? 'GET'
    const body = init?.body ? JSON.parse(String(init.body)) : undefined
    calls.push({ method, path: full, body })
    const reply = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status })
    const custom = Object.keys(extra).find(k => path === k)
    if (custom) return reply(extra[custom](body, full))
    if (path === '/settings') return reply({ defaults: RESEARCH_DEFAULTS })
    if (path === '/completions') return reply([])
    if (path === '/meta') return reply({ data_date: null, running: 0, max_workers: 4 })
    if (path === '/tabs' && method === 'POST') return reply({ id: (body as { id: string }).id })
    if (path === '/tabs') return reply([])
    if (/^\/tabs\/\w+\/name$/.test(path)) return reply({})
    if (/^\/tabs\/\w+\/runs$/.test(path)) return reply({ id: `job${++ids}` })
    if (path.startsWith('/jobs/') && path.endsWith('/events')) return reply({})
    if (path === '/validate') return reply({ count: (body as { settings: Record<string, unknown> }).settings.n_groups ? 4 : 2, name: 'vol' })
    if (path === '/backtests/options') return reply({ defaults: BT_DEFAULTS, choices: { universe: ['csi300', 'TOP800', 'TOP1800'], trade_at: ['open', 'close'] }, limits: {}, windows: {} })
    if (path === '/backtests/states') {
      const asked = new URLSearchParams(full.split('?')[1]).get('ids')!.split(',')
      return reply({ items: asked.filter(id => states.has(id)).map(id => states.get(id)) })
    }
    if (path === '/backtests' && method === 'POST') return reply({ ...btState(C, 'running'), variants: [{ ...btState(C, 'running'), label: '' }] })
    if (path === '/library/e1') return reply({ id: 'e1', name: 'low_vol', source: '-ts_std(returns, 60)', settings: { universe: 'TOP800', trade_at: 'close', group_neutralize: 'industry', benchmark: '000300.SH' } })
    return reply({ message: '没有' }, 404)
  }))
}
class EventSourceStub { static CLOSED = 2; readyState = 0; onmessage = null; onerror = null; close() {} }

function harness() {
  let research!: ReturnType<typeof useResearch>
  const wrapper = mount(defineComponent({ setup() { research = useResearch(); return () => null } }))
  return { research: () => research, wrapper }
}

beforeEach(() => { localStorage.clear(); vi.stubGlobal('EventSource', EventSourceStub); vi.stubGlobal('ResizeObserver', class { observe() {} unobserve() {} disconnect() {} }) })
afterEach(() => { vi.unstubAllGlobals(); localStorage.clear(); history.replaceState(null, '', '#/') })

describe('顶栏与旧链接', () => {
  it('顶栏只剩研究、因子库、数据、模拟盘；#/backtest 落到研究页', () => {
    expect(PAGES.map(p => p.label)).toEqual(['研究', '因子库', '数据', '模拟盘'])
    expect(pageFromHash('#/backtest')).toBe('research')
    expect(pageFromHash('#/backtest?entry=e1')).toBe('research')
  })
})

describe('旧 localStorage 迁移', () => {
  it('旧研究草稿是因子检验视图；旧回测 tab 变成策略回测视图的研究 tab，空白 tab 不迁', async () => {
    server()
    localStorage.setItem('ashare.drafts', JSON.stringify([{ id: '1'.repeat(32), name: 'mom', source: 'ts_mean(returns, 20)', settings: { ...RESEARCH_DEFAULTS, universe: 'ALL' }, autoName: false }]))
    localStorage.setItem('ashare.active', '1'.repeat(32))
    localStorage.setItem('ashare.backtest.tabs', JSON.stringify([
      { id: 'old1', name: '低波', source: '-ts_std(returns, 20)', settings: { ...BT_DEFAULTS, top_n: 10, trade_at: 'close', benchmark: '000300.SH' }, backtest: A },
      { id: 'old2', name: '', source: '', settings: {}, groups: [], selected: null },
      { id: 'old3', name: '', source: 'rank(close)', settings: {}, groups: [{ id: A, label: '20' }, { id: B, label: '60' }], selected: B, entry: 'e9' },
    ]))
    const { research, wrapper } = harness()
    await research().init(); await flushPromises()
    const tabs = research().tabs as unknown as Record<string, any>[]
    expect(tabs.map(t => [t.name, t.mode])).toEqual([['mom', 'factor'], ['低波', 'backtest'], ['vol', 'backtest']])
    expect(tabs[0].settings.universe).toBe('ALL')
    const old = tabs[1]
    expect(old.source).toBe('-ts_std(returns, 20)')
    expect(old.settings.trade_at).toBe('close')
    expect(old.settings.benchmark).toBe('000300.SH')
    expect(old.backtest.settings.top_n).toBe(10)
    expect(old.backtest.groups.map((g: any) => [g.id, g.label, g.state?.status])).toEqual([[A, '', 'complete']])
    expect(tabs[2].backtest.groups.map((g: any) => g.label)).toEqual(['20', '60'])
    expect(tabs[2].backtest.selected).toBe(B)
    expect(tabs[2].backtest.entry).toBe('e9')
    expect(tabs.every(t => /^[a-f0-9]{32}$/.test(t.id))).toBe(true)
    expect(localStorage.getItem('ashare.backtest.tabs')).toBeNull()
    const drafts = JSON.parse(localStorage.getItem('ashare.drafts')!)
    expect(drafts.map((d: any) => d.mode)).toEqual(['factor', 'backtest', 'backtest'])
    expect(drafts[1].backtest.groups).toEqual([{ id: A, label: '' }])
    // 成交价和基准只存一份（在研究设置里），回测设置里不再重复。
    expect(drafts[1].backtest.settings).not.toHaveProperty('trade_at')
    expect(drafts[1].backtest.settings).not.toHaveProperty('benchmark')
    expect(drafts[1].settings.trade_at).toBe('close')
    // 再加载一次不会重复迁移。
    wrapper.unmount()
    const again = harness()
    await again.research().init(); await flushPromises()
    expect(again.research().tabs.map(t => t.name)).toEqual(['mom', '低波', 'vol'])
    again.wrapper.unmount()
  })
})

describe('旧链接与送去回测', () => {
  it('#/backtest 切到策略回测视图；带 entry 新开 tab，同一条目再进入切回原 tab', async () => {
    server()
    const { research, wrapper } = harness()
    const r = research() as any
    await r.init(); await flushPromises()
    expect(r.current.value.mode).toBe('factor')
    expect(await r.openRoute('#/backtest')).toBe(true)
    expect(r.current.value.mode).toBe('backtest')
    expect(r.tabs).toHaveLength(1)
    await r.openRoute('#/backtest?entry=e1'); await flushPromises()
    expect(r.tabs).toHaveLength(2)
    const tab = r.current.value
    expect(tab).toMatchObject({ name: 'low_vol', source: '-ts_std(returns, 60)', mode: 'backtest' })
    expect(tab.backtest.entry).toBe('e1')
    // 原来的带入行为：回测认得的股票池、成交时点和基准，其余回测默认值。
    expect(tab.backtest.settings).toMatchObject({ universe: 'TOP800', top_n: 30 })
    expect(tab.settings).toMatchObject({ trade_at: 'close', benchmark: '000300.SH', universe: 'TOP800', group_neutralize: 'industry' })
    r.activeId.value = r.tabs[0].id
    await r.openRoute('#/backtest?entry=e1')
    expect(r.tabs).toHaveLength(2)
    expect(r.current.value.backtest.entry).toBe('e1')
    expect(calls.filter(c => c.path === '/library/e1')).toHaveLength(1)
    expect(await r.openRoute('#/research')).toBe(false)
    wrapper.unmount()
  })
})

describe('研究页', () => {
  async function page() {
    server()
    const { default: ResearchPage } = await import('./pages/ResearchPage.vue')
    const { research, wrapper: host } = harness()
    await research().init(); await flushPromises()
    const stubs = { ExpressionEditor: true, ResearchResults: true, BacktestResults: true, BenchmarkSelect: true }
    const wrapper = mount(ResearchPage, { props: { research: research() }, global: { stubs }, attachTo: document.body })
    await flushPromises()
    return { r: research() as any, wrapper, host }
  }
  const run = (w: ReturnType<typeof mount>) => w.findAll('button').find(b => b.text().startsWith('运行'))!
  const view = async (w: ReturnType<typeof mount>, label: string) => { await w.findAll('[role=tab]').find(b => b.text() === label)!.trigger('click'); await flushPromises() }

  it('切换视图保留两边结果，设置换成对应视图的那一套', async () => {
    const { r, wrapper, host } = await page()
    const tab = r.current.value
    tab.job = { id: 'job0', status: 'complete', total: 1, elapsed: 1, runs: [] }
    tab.backtest.groups = [{ id: A, label: '', state: btState(A, 'complete') }]; tab.backtest.selected = A
    expect(wrapper.findComponent({ name: 'ResearchResults' }).exists()).toBe(true)
    expect(wrapper.text()).toContain('分组检验')
    await view(wrapper, '策略回测')
    expect(tab.mode).toBe('backtest')
    expect(wrapper.findComponent({ name: 'ResearchResults' }).exists()).toBe(false)
    expect(wrapper.findComponent({ name: 'BacktestResults' }).exists()).toBe(true)
    expect(wrapper.text()).toContain('成本与规则')
    expect(wrapper.text()).not.toContain('分组检验')
    await view(wrapper, '因子检验')
    expect(tab.job.id).toBe('job0')
    expect(tab.backtest.groups.map((g: any) => g.id)).toEqual([A])
    expect(wrapper.findComponent({ name: 'ResearchResults' }).exists()).toBe(true)
    wrapper.unmount(); host.unmount()
  })

  it('成交价和基准只出现一次，两个视图的运行都用它；运行按钮只跑当前视图', async () => {
    const { r, wrapper, host } = await page()
    const { pick } = await import('./components/ui/testing')
    for (const mode of ['因子检验', '策略回测']) {
      await view(wrapper, mode)
      expect(wrapper.findAll('[aria-label="成交时点"]')).toHaveLength(1)
      expect(wrapper.findAll('[aria-label="成交价"]')).toHaveLength(0)
      expect(wrapper.findAllComponents({ name: 'BenchmarkSelect' })).toHaveLength(1)
    }
    await view(wrapper, '因子检验')
    await pick(wrapper, '成交时点', 'T+1 收盘'); await flushPromises()
    expect(run(wrapper).text()).toContain('运行 4 组')
    await run(wrapper).trigger('click'); await flushPromises()
    const factorRuns = calls.filter(c => /^\/tabs\/\w+\/runs$/.test(c.path))
    expect(factorRuns).toHaveLength(1)
    expect((factorRuns[0].body as any).settings.trade_at).toBe('close')
    expect(calls.filter(c => c.path === '/backtests' && c.method === 'POST')).toHaveLength(0)
    await view(wrapper, '策略回测')
    expect(run(wrapper).text()).toContain('运行 2 组')
    await run(wrapper).trigger('click'); await flushPromises()
    const submitted = calls.filter(c => c.path === '/backtests' && c.method === 'POST')
    expect(submitted).toHaveLength(1)
    expect((submitted[0].body as any).settings).toMatchObject({ trade_at: 'close', top_n: 30 })
    expect((submitted[0].body as any).source).toBe(r.current.value.source)
    expect(calls.filter(c => /^\/tabs\/\w+\/runs$/.test(c.path))).toHaveLength(1)
    await nextTick()
    wrapper.unmount(); host.unmount()
  })
})
