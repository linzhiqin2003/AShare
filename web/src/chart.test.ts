import { describe, expect, it } from 'vitest'
import { chartData, curves, unpack, niceAxis, withBenchmark } from './chart'
import type { PlotPart, Run } from './types'

describe('窗口重算与列式传输', () => {
  it('拖动起点后从零重算累加和回撤', () => {
    expect(curves([0.5, -0.2, 0.1, -0.3], false, 2, 3).cumulative).toEqual([0.1, -0.19999999999999998])
    expect(curves([0.5, -0.2, 0.1, -0.3], false, 2, 3).drawdown[1]).toBeCloseTo(-0.3)
  })
  it('复利回撤以窗口内初始净值一为基准', () => {
    const v = curves([0.5, -0.2, 0.1], true, 1, 2)
    expect(v.cumulative[0]).toBeCloseTo(-0.2)
    expect(v.cumulative[1]).toBeCloseTo(-0.12)
    expect(v.drawdown[1]).toBeCloseTo(-0.12)
  })
  it('缺失值不让曲线和回撤断成 NaN', () => {
    expect(curves([NaN, 0.1, NaN, -0.2], false).cumulative).toEqual([0, 0.1, 0.1, -0.1])
  })
  it('按日期对齐样本内和验证，支持多个变体的不同日期', () => {
    function part(dates: string[], values: number[]): PlotPart {
      const bytes = new Uint8Array(new Float32Array(values).buffer)
      return { part: 'is', dates, series: { 'pnl.long_short': { shape: [values.length], data: btoa(String.fromCharCode(...bytes)) } } }
    }
    const packed = part(['20240101', '20240102'], [0.1, -0.1])
    expect(unpack(packed.series['pnl.long_short'])[0]).toBeCloseTo(0.1)
    const chart = chartData([{ id: 'a', ordinal: 0, label: 'a' }, { id: 'b', ordinal: 1, label: 'b' }] as Run[], [[packed], [part(['20240102', '20250101'], [0.2, 0.3])]], 'returns')
    expect(chart.dates).toEqual(['20240101', '20240102', '20250101'])
    expect(chart.lines[1].values[0]).toBeNaN()
    expect(chart.lines[1].values[1]).toBeCloseTo(0.2)
  })
})


describe('整刻度与基准', () => {
  it('按数据跨度选择整齐刻度而不是端点百分比', () => {
    expect(niceAxis([[.83,.37,.15,-.08]])).toEqual({min:-.25,max:1,ticks:[-.25,0,.25,.5,.75,1]})
    expect(niceAxis([[.083]],4,true).ticks).toEqual([0,.05,.1])
    expect(niceAxis([[.001]],4,true).ticks).toEqual([0,.01])
    expect(niceAxis([[-.14,-.03,0]],2).ticks).toEqual([-.2,-.1,0])
  })
  it('主图下沿贴近最低数据，整档刻度不额外延伸', () => {
    const axis = niceAxis([[-.43,.33]],4,true,true)
    expect(axis.ticks).toEqual([-.4,-.2,0,.2,.4])
    expect(axis.min).toBeLessThan(-.43)
    expect(axis.min).toBeGreaterThan(-.46)
    expect(niceAxis([[-.1-.2,.3]],4,true).ticks).toEqual([-.4,-.2,0,.2,.4])
    expect(niceAxis([[-.1-.2,0]],3,true).ticks).toEqual([-.3,-.2,-.1,0])
  })
  it('基准按日期对齐，和因子采用同一个窗口重算函数', () => {
    const bytes = new Uint8Array(new Float32Array([.01,.02]).buffer)
    const benchmark = {symbol:'synthetic',name:'合成指数',parts:[{part:'is' as const,dates:['20240102','20240103'],series:{benchmark:{shape:[2],data:btoa(String.fromCharCode(...bytes))}}}],relatives:{},requested:'synthetic',fallback:false}
    const chart = withBenchmark({dates:['20240101','20240102','20240103'],lines:[]},benchmark)
    expect(chart.lines[0].benchmark).toBe(true)
    expect(chart.lines[0].values[0]).toBeNaN()
    expect(curves(chart.lines[0].values,false,2,2).cumulative[0]).toBeCloseTo(.02)
    expect(curves(chart.lines[0].values,true,1,2).cumulative[1]).toBeCloseTo(.0302)
  })
})
