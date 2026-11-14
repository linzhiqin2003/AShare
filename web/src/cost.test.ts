import { describe, expect, it } from 'vitest'
import { COST_BPS, netReturn } from './cost'

describe('扣费年化估算（2026-11-14 用户：研究台加一列扣成本后的估算）', () => {
  it('按单边 10 bp × 买卖合计换手扣减', () => {
    expect(COST_BPS).toBe(10)
    // 日均换手 19.2%，一年 240 天，单边 10 bp：扣 4.608 个百分点。
    expect(netReturn(0.065, 0.192)).toBeCloseTo(0.065 - 0.04608, 12)
    expect(netReturn(0.0757, 0.0842, 20)).toBeCloseTo(0.0757 - 0.0842 * 240 * 0.002, 12)
  })
  it('与逐日扣费后再年化一致', () => {
    const pnl = [0.002, -0.001, 0.0035, 0.0004], turn = [0.3, 0.1, 0.25, 0.05]
    const mean = (x: number[]) => x.reduce((a, b) => a + b, 0) / x.length
    const daily = pnl.map((p, i) => p - turn[i] * COST_BPS / 10000)
    expect(netReturn(mean(pnl) * 240, mean(turn))).toBeCloseTo(mean(daily) * 240, 12)
  })
  it('缺数据时不估算', () => {
    expect(netReturn(null, 0.1)).toBeNull()
    expect(netReturn(0.1, undefined)).toBeNull()
    expect(netReturn(Number.NaN, 0.1)).toBeNull()
  })
})
