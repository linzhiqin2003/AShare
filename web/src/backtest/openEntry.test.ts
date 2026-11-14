// 因子库「送去回测」的接收方（2026-11-14 新增）：同一条目只开一个 tab。
// 2026-11-14 研究台与回测合并后接收方是研究页：新开一个 tab、停在策略回测视图；研究 tab 总有表达式，
// 不再有「空白 tab 直接复用」，所以第一次带入就是新开（原断言里的 tab 数各加一）。
import { flushPromises, mount } from '@vue/test-utils'
import { defineComponent } from 'vue'
import { describe, expect, it, vi } from 'vitest'

const calls = vi.hoisted(() => [] as string[])
vi.mock('../api', () => ({
  api: vi.fn(async (path: string) => {
    calls.push(path)
    if (path === '/backtests/options') return { defaults: { universe: 'csi300', rebalance: 'month', top_n: 30, trade_at: 'open', capital: 1000, commission: 0.00025, commission_min: 5, slippage_bps: 5, dividend_tax: 0.2, carry_blocked: true, lots: 'board', benchmark: '000001.SH' }, choices: { universe: ['csi300', 'TOP800'], trade_at: ['open', 'close'] }, limits: {}, windows: {} }
    if (path === '/completions') return []
    if (path === '/settings') return { defaults: { universe: 'TOP1800', trade_at: 'open' } }
    if (path === '/tabs') return []
    if (path === '/validate') return { count: 1, name: 'low_vol' }
    if (path === '/meta') return { data_date: null, running: 0, max_workers: 4 }
    if (path === '/library/e1') return { id: 'e1', name: 'low_vol', source: '-ts_std(returns, 60)', settings: { universe: 'TOP800', trade_at: 'close', group_neutralize: 'industry' } }
    throw new Error(path)
  }),
  post: vi.fn(async (path: string, body?: { id: string }) => {
    if (path === '/tabs') return { id: body!.id }
    throw new Error(path)
  }),
  series: vi.fn(),
  friendly: () => ({ message: '读取失败' }),
  RequestError: class extends Error {},
}))

describe('送去回测', () => {
  it('带入表达式、名称和可识别设置；重复进入切回同一 tab', async () => {
    localStorage.clear()
    const { useResearch } = await import('../useResearch')
    let b!: ReturnType<typeof useResearch>
    const host = mount(defineComponent({ setup() { b = useResearch(); return () => null } }))
    await b.init()
    expect(b.tabs).toHaveLength(1)
    const tab = await b.openEntry('e1')
    expect(b.tabs).toHaveLength(2)
    expect(tab).toMatchObject({ name: 'low_vol', source: '-ts_std(returns, 60)', mode: 'backtest' })
    expect(tab!.backtest.entry).toBe('e1')
    expect(tab!.backtest.settings).toMatchObject({ universe: 'TOP800', trade_at: 'close', top_n: 30 })
    expect(tab!.settings.trade_at).toBe('close')
    await b.addTab()
    await b.openEntry('e1')
    expect(b.tabs).toHaveLength(3)
    expect(b.current.value?.backtest.entry).toBe('e1')
    expect(calls.filter(c => c === '/library/e1')).toHaveLength(1)
    await flushPromises()
    expect(JSON.parse(localStorage.getItem('ashare.drafts')!)[1].backtest.entry).toBe('e1')
    host.unmount()
  })
})
