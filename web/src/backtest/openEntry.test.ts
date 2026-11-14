// 因子库「送去回测」的接收方（2026-11-14 新增）：同一条目只开一个 tab，空白 tab 直接复用。
import { flushPromises } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'

const calls = vi.hoisted(() => [] as string[])
vi.mock('../api', () => ({
  api: vi.fn(async (path: string) => {
    calls.push(path)
    if (path === '/backtests/options') return { defaults: { universe: 'csi300', rebalance: 'month', top_n: 30, trade_at: 'open', capital: 1000, commission: 0.00025, commission_min: 5, slippage_bps: 5, dividend_tax: 0.2, carry_blocked: true, lots: 'board', benchmark: '000001.SH' }, choices: { universe: ['csi300', 'TOP800'], trade_at: ['open', 'close'] }, limits: {}, windows: {} }
    if (path === '/completions') return []
    if (path === '/library/e1') return { id: 'e1', name: 'low_vol', source: '-ts_std(returns, 60)', settings: { universe: 'TOP800', trade_at: 'close', group_neutralize: 'industry' } }
    throw new Error(path)
  }),
  post: vi.fn(),
  friendly: () => ({ message: '读取失败' }),
  RequestError: class extends Error {},
}))

describe('送去回测', () => {
  it('带入表达式、名称和可识别设置；重复进入切回同一 tab', async () => {
    localStorage.clear()
    const { useBacktests } = await import('./useBacktests')
    const b = useBacktests()
    await b.init()
    expect(b.tabs.value).toHaveLength(1)
    const tab = await b.openEntry('e1')
    expect(b.tabs.value).toHaveLength(1)
    expect(tab).toMatchObject({ name: 'low_vol', source: '-ts_std(returns, 60)', entry: 'e1' })
    expect(tab!.settings).toMatchObject({ universe: 'TOP800', trade_at: 'close', top_n: 30 })
    b.addTab()
    await b.openEntry('e1')
    expect(b.tabs.value).toHaveLength(2)
    expect(b.current.value?.entry).toBe('e1')
    expect(calls.filter(c => c === '/library/e1')).toHaveLength(1)
    await flushPromises()
    expect(JSON.parse(localStorage.getItem('ashare.backtest.tabs')!)[0].entry).toBe('e1')
    b.dispose()
  })
})
