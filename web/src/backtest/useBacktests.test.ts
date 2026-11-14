// 回测页的路由约定、验证期开关和因子库带入（2026-11-14 新增）。
import { afterEach, describe, expect, it } from 'vitest'
import { pct, tone } from './format'
import { routeEntry, settingsFromEntry } from './useBacktests'
import type { BacktestSettings } from './types'

const defaults: BacktestSettings = { universe: 'csi300', rebalance: 'month', top_n: 30, trade_at: 'open', capital: 1000, commission: 0.00025, commission_min: 5, slippage_bps: 5, dividend_tax: 0.2, carry_blocked: true, lots: 'board', benchmark: '000001.SH' }
const choices = { universe: ['csi300', 'TOP800', 'TOP1800', 'ALL'], trade_at: ['open', 'close'] }

afterEach(() => localStorage.clear())

describe('从因子库送去回测', () => {
  it('#/backtest?entry=<id> 取出条目，非法值忽略', () => {
    expect(routeEntry('#/backtest?entry=abc123')).toBe('abc123')
    expect(routeEntry('#/backtest')).toBeNull()
    expect(routeEntry('#/backtest?entry=../x')).toBeNull()
  })
  it('只带入回测页认得的股票池、成交时点和基准，其余用回测默认值', () => {
    const out = settingsFromEntry({ universe: 'TOP1800', trade_at: 'close', benchmark: '000300.SH', group_neutralize: 'industry', rebalance: 5 }, defaults, choices)
    expect(out).toEqual({ ...defaults, universe: 'TOP1800', trade_at: 'close', benchmark: '000300.SH' })
    expect(settingsFromEntry({ universe: 'TEST' }, defaults, choices).universe).toBe('csi300')
  })
})

describe('数字格式', () => {
  it('带符号百分比、负零与涨红跌绿', () => {
    expect(pct(0.3116, 2, true)).toBe('+31.16%')
    expect(pct(-0.00001)).toBe('0.00%')
    expect(pct(null)).toBe('—')
    expect([tone(0.1), tone(-0.1), tone(0), tone(null)]).toEqual(['up', 'down', '', ''])
  })
})
