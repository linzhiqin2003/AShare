// 回测页净值图的纯计算（2026-11-14 新增）。
import { describe, expect, it } from 'vitest'
import { areaPath, drawdown, drawdownTicks, linePath, navTicks, rangeAvailable, rangeStart, rebase, tickDigits, yearTicks } from './navChart'

const dates = ['2016-01-04', '2016-06-01', '2017-01-03', '2019-01-02', '2021-03-01', '2024-12-31']

describe('时间窗与重新起点', () => {
  it('从末日往前 N 年的第一个交易日开始；跨度不够的时间窗不提供', () => {
    expect(rangeStart(dates, 'all')).toBe(0)
    expect(dates[rangeStart(dates, '1y')]).toBe('2024-12-31')
    expect(dates[rangeStart(dates, '5y')]).toBe('2021-03-01')
    expect(rangeAvailable(['2025-01-02', '2025-12-31'], '1y')).toBe(false)
    expect(rangeAvailable(dates, '3y')).toBe(true)
  })
  it('窗内以第一个有效值为 1，缺值保留为缺口', () => {
    expect(rebase([2, null, 3, 4], 1)).toEqual([null, 1, 4 / 3])
    expect(rebase([null, null], 0)).toEqual([null, null])
  })
  it('回撤与报告同一公式：nav / 历史最高 − 1', () => {
    expect(drawdown([1, 1.2, 0.9, null, 1.3])).toEqual([0, 0, 0.9 / 1.2 - 1, null, 0])
  })
})

describe('坐标轴', () => {
  it('净值刻度包住数据但不强行包含 0', () => {
    const axis = navTicks([[0.95, 1.4, 2.1]], 5)
    expect(axis.ticks[0]).toBeGreaterThan(0.9)
    expect(axis.ticks.at(-1)).toBeLessThanOrEqual(2.1)
    expect(axis.step).toBe(0.25)
  })
  // 来源：2026-11-14 回测页 1280 截图，0.25 步长的刻度被写成 1.3、1.8。
  it('刻度小数位按步长统一，1.25 不会被四舍五入成 1.3', () => {
    const axis = navTicks([[0.9, 2.0]], 5)
    expect(axis.ticks.map(t => t.toFixed(tickDigits(axis.step)))).toContain('1.25')
    expect(tickDigits(0.5)).toBe(1)
    expect(tickDigits(0.025)).toBe(3)
    expect(tickDigits(1)).toBe(1)
  })
  it('回撤轴从 0 到最深回撤，按整数百分比分格', () => {
    expect(drawdownTicks([0, -0.34, -0.1]).ticks).toEqual([0, -0.2])
    expect(drawdownTicks([0, -0.07]).ticks).toEqual([0, -0.05])
    expect(drawdownTicks([0, 0]).min).toBe(-0.01)
  })
  it('年份刻度取每年第一个交易日，过密时跳过', () => {
    expect(yearTicks(dates).map(t => t.label)).toEqual(['2016', '2017', '2019', '2021', '2024'])
    expect(yearTicks(dates, 2).map(t => t.index)).toEqual([0, 2, 4])
  })
  it('折线与面积在缺值处断开', () => {
    const x = (i: number) => i, y = (v: number) => v
    expect(linePath([1, null, 2, 3], x, y)).toBe('M0.0 1.0M2.0 2.0L3.0 3.0')
    expect(areaPath([-1, null, -2], x, y)).toBe('M0.0 0.0L0.0 -1.0L0.0 0.0ZM2.0 0.0L2.0 -2.0L2.0 0.0Z')
  })
})
