// 时间窗刷子（2026-11-14 用户：策略回测的净值图也要像研究结果图那样拖底部缩略图改窗口）：
// 两种图共用 TimeBrush；拖动后净值图的主图从窗口起点画起、以起点为 1 归一，回撤在窗内重算。
import { flushPromises, mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import TimeBrush from './TimeBrush.vue'
import ResultChart from './ResultChart.vue'
import NavChart from '../backtest/NavChart.vue'
import { moveWindow } from '../chart'
import type { BacktestDocument } from '../backtest/types'

const dates = Array.from({ length: 10 }, (_, i) => `2016-01-${String(i + 4).padStart(2, '0')}`)
const nav = [1, 1.1, 1.2, 1.5, 1.2, 1.8, 1.9, 2.1, 2.0, 2.4]
const doc = { start: dates[0], end: dates[9], benchmark: null, series: { dates, nav, benchmark: null, drawdown: [] } } as unknown as BacktestDocument

async function drag(target: { trigger: (e: string, o: object) => Promise<void> }, from: number, to: number) {
  await target.trigger('pointerdown', { clientX: from, pointerId: 1 })
  window.dispatchEvent(new MouseEvent('pointermove', { clientX: to }))
  window.dispatchEvent(new MouseEvent('pointerup', { clientX: to }))
  await flushPromises()
}

describe('时间窗刷子', () => {
  it('拉左端、拉右端、整体平移都不出界，窗口至少 3 个点', () => {
    expect(moveWindow('left', { start: 0, end: 9 }, 3, 10)).toEqual({ start: 3, end: 9 })
    expect(moveWindow('left', { start: 0, end: 9 }, 20, 10)).toEqual({ start: 7, end: 9 })
    expect(moveWindow('right', { start: 2, end: 9 }, -20, 10)).toEqual({ start: 2, end: 4 })
    expect(moveWindow('body', { start: 2, end: 5 }, 10, 10)).toEqual({ start: 6, end: 9 })
    expect(moveWindow('body', { start: 2, end: 5 }, -10, 10)).toEqual({ start: 0, end: 3 })
  })

  it('研究结果图与净值图用的是同一个刷子组件，净值图不再有 1/3/5 年按钮', () => {
    const research = mount(ResultChart, { props: { data: { dates: dates.map(d => d.replaceAll('-', '')), lines: [{ id: 'a', name: 'a', color: 'black', values: nav.map(v => v / 100) }] } } })
    const backtest = mount(NavChart, { props: { document: doc, part: 'is', showValidation: false } })
    expect(research.findComponent(TimeBrush).exists()).toBe(true)
    expect(backtest.findComponent(TimeBrush).exists()).toBe(true)
    expect(backtest.find('[aria-label="时间窗"]').exists()).toBe(false)
    expect(backtest.text()).not.toMatch(/[135] 年|全部/)
    expect(backtest.text()).toContain('对数')
    research.unmount(); backtest.unmount()
  })

  it('净值图拖动刷子左端后，主图从窗口起点画起并以起点为 1，回撤在窗内重算', async () => {
    const wrapper = mount(NavChart, { props: { document: doc, part: 'is', showValidation: false }, attachTo: document.body })
    const brush = wrapper.getComponent(TimeBrush)
    expect([brush.props('start'), brush.props('end')]).toEqual([0, 9])
    // 默认画布 720 宽，绘图区 664：每个点 664/9 像素，拖 3 个点。
    const step = 664 / 9
    await drag(wrapper.findAll('.brush-handle')[0], 46, 46 + 3 * step)
    expect([brush.props('start'), brush.props('end')]).toEqual([3, 9])
    const main = wrapper.get('svg[aria-label="策略与基准净值"]')
    await main.trigger('pointermove', { clientX: 46 })
    expect(wrapper.get('.tip b').text()).toBe(dates[3])
    expect(wrapper.get('.tip').text()).toContain('1.000')
    // 窗内第 2 个点 1.2 / 1.5 − 1 = −20%：回撤从窗口起点重算，不带窗口外的历史高点。
    await main.trigger('pointermove', { clientX: 46 + 664 / 6 })
    expect(wrapper.get('.tip b').text()).toBe(dates[4])
    expect(wrapper.get('.tip').text()).toContain('0.800')
    expect(wrapper.get('.tip').text()).toContain('-20.00%')
    // 主图横轴只剩窗口里的 7 个点：最后一个点在绘图区右端。
    await main.trigger('pointermove', { clientX: 46 + 664 })
    expect(wrapper.get('.tip b').text()).toBe(dates[9])
    expect(wrapper.get('.tip').text()).toContain(`${(2.4 / 1.5).toFixed(3)}`)
    // 整体平移：窗口长度不变。
    await drag(wrapper.get('.brush-body'), 300, 300 - 2 * step)
    expect([brush.props('start'), brush.props('end')]).toEqual([1, 7])
    wrapper.unmount()
  })

  it('换了文档（日期轴不同）窗口回到全部；键盘左右键每次 10 个点', async () => {
    const wrapper = mount(NavChart, { props: { document: doc, part: 'is', showValidation: false } })
    await drag(wrapper.findAll('.brush-handle')[0], 46, 46 + 5 * 664 / 9)
    expect(wrapper.getComponent(TimeBrush).props('start')).toBe(5)
    const longer = { ...doc, series: { ...doc.series, dates: [...dates, '2016-01-14'], nav: [...nav, 2.5] } } as BacktestDocument
    await wrapper.setProps({ document: longer })
    expect([wrapper.getComponent(TimeBrush).props('start'), wrapper.getComponent(TimeBrush).props('end')]).toEqual([0, 10])
    await wrapper.findAll('.brush-handle')[1].trigger('keydown', { key: 'ArrowLeft' })
    expect(wrapper.getComponent(TimeBrush).props('end')).toBe(2)
    wrapper.unmount()
  })
})
