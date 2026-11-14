// 回测页一次跑候选列表展开的多组（2026-11-14 新增）：旧 tab 格式兼容、整批进度、切换与整批取消。
import { flushPromises } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { batchProgress, restoreTab, selectedGroup } from './useBacktests'
import type { BacktestGroup, BacktestSettings, BacktestState } from './types'

const defaults: BacktestSettings = { universe: 'csi300', rebalance: 'month', top_n: 30, trade_at: 'open', capital: 1000, commission: 0.00025, commission_min: 5, slippage_bps: 5, dividend_tax: 0.2, carry_blocked: true, lots: 'board', benchmark: '000001.SH' }
const A = 'a'.repeat(24), B = 'b'.repeat(24), C = 'c'.repeat(24)
const state = (id: string, status: BacktestState['status'], progress: number | null = null): BacktestState =>
  ({ id, name: 'vol', status, progress, queue_position: null, elapsed: null, settings: defaults })
const group = (id: string, label: string, s: BacktestState | null): BacktestGroup => ({ id, label, state: s })

const calls = vi.hoisted(() => [] as { path: string; body?: unknown }[])
const server = vi.hoisted(() => ({ states: new Map<string, object>() }))
vi.mock('../api', () => ({
  api: vi.fn(async (path: string) => {
    calls.push({ path })
    if (path === '/backtests/options') return { defaults: { universe: 'csi300', rebalance: 'month', top_n: 30, trade_at: 'open', capital: 1000, commission: 0.00025, commission_min: 5, slippage_bps: 5, dividend_tax: 0.2, carry_blocked: true, lots: 'board' }, choices: {}, limits: {}, windows: {} }
    if (path === '/completions') return []
    if (path.startsWith('/backtests/states?')) {
      const ids = new URLSearchParams(path.split('?')[1]).get('ids')!.split(',')
      return { items: ids.filter(id => server.states.has(id)).map(id => server.states.get(id)) }
    }
    throw new Error(path)
  }),
  post: vi.fn(async (path: string, body?: unknown) => {
    calls.push({ path, body })
    if (path === '/backtests') return { ...server.states.get(A), variants: [{ ...server.states.get(A), label: '20' }, { ...server.states.get(B), label: '60' }] }
    if (path === '/backtests/cancel') return { cancelled: (body as { ids: string[] }).ids }
    throw new Error(path)
  }),
  friendly: () => ({ message: '读取失败' }),
  RequestError: class extends Error {},
}))

afterEach(() => { localStorage.clear(); calls.length = 0; server.states.clear() })

describe('旧 tab 格式', () => {
  it('只有一个 backtest 身份的旧 tab 读成一组，组名为空', () => {
    const tab = restoreTab({ id: 't1', name: '低波', source: '-ts_std(returns, 20)', settings: { top_n: 10 }, backtest: A }, defaults)!
    expect(tab.groups).toEqual([{ id: A, label: '', state: null }])
    expect(tab.selected).toBe(A)
    expect(tab.settings).toEqual({ ...defaults, top_n: 10 })
    expect(selectedGroup(tab)?.id).toBe(A)
  })
  it('没有运行过的旧 tab 没有组；新格式保留组名和选中组，非法项忽略', () => {
    expect(restoreTab({ id: 't1', name: '', source: '', settings: {}, backtest: null }, defaults)!.groups).toEqual([])
    const tab = restoreTab({ id: 't2', groups: [{ id: A, label: '20' }, { id: B, label: '60' }, { label: 'x' }], selected: B }, defaults)!
    expect(tab.groups.map(g => [g.id, g.label])).toEqual([[A, '20'], [B, '60']])
    expect(tab.selected).toBe(B)
    expect(restoreTab({ id: 't3', groups: [{ id: A, label: '20' }], selected: 'gone' }, defaults)!.selected).toBe(A)
    expect(restoreTab({ name: 'x' } as Record<string, unknown>, defaults)).toBeNull()
  })
})

describe('整批进度', () => {
  it('只有一组时与原来相同', () => {
    expect(batchProgress([group(A, '', state(A, 'running', 0.45))])).toBe('45%')
    expect(batchProgress([group(A, '', state(A, 'queued'))])).toBe('排队')
    expect(batchProgress([group(A, '', state(A, 'running'))])).toBe('')
    expect(batchProgress([group(A, '', state(A, 'complete'))])).toBe('')
  })
  it('多组取平均，已结束（含失败）的组按 100% 计', () => {
    const groups = [group(A, '20', state(A, 'complete')), group(B, '60', state(B, 'running', 0.5)), group(C, '120', state(C, 'queued'))]
    expect(batchProgress(groups)).toBe('50%')
    groups[0].state = state(A, 'failed')
    expect(batchProgress(groups)).toBe('50%')
    expect(batchProgress(groups.map(g => ({ ...g, state: state(g.id, 'queued') })))).toBe('排队')
  })
})

describe('多组运行', () => {
  it('提交后每组一个身份；切换选中组；整批取消一次请求；存回新格式', async () => {
    server.states.set(A, state(A, 'running', 0.2)); server.states.set(B, state(B, 'queued'))
    const { useBacktests } = await import('./useBacktests')
    const b = useBacktests()
    await b.init()
    const tab = b.current.value!
    tab.source = '-ts_std(returns, [20, 60])'
    await b.run(tab)
    expect(tab.groups.map(g => [g.id, g.label, g.state?.status])).toEqual([[A, '20', 'running'], [B, '60', 'queued']])
    expect(tab.selected).toBe(A)
    expect(tab.name).toBe('vol')
    expect(b.active(tab)).toBe(true)
    b.select(tab, B)
    expect(selectedGroup(tab)?.label).toBe('60')
    b.select(tab, 'nope')
    expect(tab.selected).toBe(B)
    server.states.set(A, state(A, 'cancelled')); server.states.set(B, state(B, 'cancelled'))
    await b.cancel(tab)
    expect(calls.filter(c => c.path === '/backtests/cancel')).toEqual([{ path: '/backtests/cancel', body: { ids: [A, B] } }])
    expect(tab.groups.map(g => g.state?.status)).toEqual(['cancelled', 'cancelled'])
    expect(b.active(tab)).toBe(false)
    await flushPromises()
    const saved = JSON.parse(localStorage.getItem('ashare.backtest.tabs')!)[0]
    expect(saved.groups).toEqual([{ id: A, label: '20' }, { id: B, label: '60' }])
    expect(saved.selected).toBe(B)
    expect(saved).not.toHaveProperty('backtest')
    b.dispose()
  })
  it('读回旧格式的 tab：一次请求取状态，服务端已不认识的组去掉', async () => {
    server.states.set(A, state(A, 'complete'))
    localStorage.setItem('ashare.backtest.tabs', JSON.stringify([
      { id: 't1', name: '低波', source: '-ts_std(returns, 20)', settings: {}, backtest: A },
      { id: 't2', name: '已清', source: 'close', settings: {}, backtest: C },
    ]))
    const { useBacktests } = await import('./useBacktests')
    const b = useBacktests()
    await b.init()
    expect(b.tabs.value[0].groups.map(g => [g.id, g.state?.status])).toEqual([[A, 'complete']])
    expect(b.tabs.value[1].groups).toEqual([])
    expect(b.tabs.value[1].selected).toBeNull()
    expect(calls.filter(c => c.path.startsWith('/backtests/states?'))).toHaveLength(2)
    b.dispose()
  })
})
