// 模拟盘页（2026-11-14 新增）：纯计算与页面行为，接口用合成数据替身。
import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { monthTicks, percent, points, pollDelay, relative, signed, slice, tone } from './paper'
import type { PaperCurve, PaperSummary } from './paper'
import PaperPage from './PaperPage.vue'

const api = vi.hoisted(() => vi.fn())
vi.mock('../api', () => ({ api, friendly: () => ({ message: '账户暂时读取不到，请稍后再试' }) }))

function summary(extra: Partial<PaperSummary> = {}): PaperSummary {
  return {
    id: '演示', day: '2026-11-04', phase: '连续竞价', settled: false, running: true, quote_time: '2026-11-04 14:59:00', since: '2026-10-30',
    equity: 1_002_000, cash: 500_000, frozen: 0, market_value: 502_000, invested: 1_000_000,
    day_pnl: 2000, day_return: 0.002, month_return: -0.01, year_return: 0.002, nav: 1.002,
    positions: [{ symbol: '600000.SH', name: '合成一', industry: '银行', total: 1000, sellable: 900, cost: 10, mark: 10.5, value: 10_500, pnl: 500, weight: 0.0105 }],
    industries: [['银行', 0.0105]],
    concentration: { max_weight: 0.0105, max_symbol: '合成一', max_industry: '银行', max_industry_weight: 0.0105, cash_weight: 0.499 },
    trading: { buys: 1, sells: 0, buy_amount: 1000, sell_amount: 0, turnover: 0.0005, fees: 5, open_orders: 1, unfilled: 1, rejected: [] },
    risk: { open_orders: 1, rejected: [['资金不足', 2]] },
    orders: [{ id: 1, symbol: '600000.SH', name: '合成一', side: 'buy', kind: 'limit', qty: 200, filled: 100, price: 10.5, status: 'open', time: '09:31:00' }],
    fills_day: '2026-11-04',
    fills: [{ day: '2026-11-04', time: '09:32:00', symbol: '600000.SH', name: '合成一', side: 'buy', qty: 100, price: 10.5, fee: 5 }],
    ...extra,
  }
}
const curve: PaperCurve = { dates: ['2026-10-30', '2026-11-02', '2026-11-03'], nav: [0.99, 1.0, 1.01], benchmark: { symbol: '000300.SH', name: '沪深300' }, closes: [4000, 4040, null], today_close: 4100 }

describe('纯计算', () => {
  it('格式与涨跌色：涨红跌绿，负零不显示负号', () => {
    expect(percent(0.0123, 2, true)).toBe('+1.23%')
    expect(percent(-0.00001, 2)).toBe('0.00%')
    expect(percent(null)).toBe('—')
    expect(signed(-1234.5, 1)).toBe('-1,234.5')
    expect([tone(1), tone(-1), tone(0)]).toEqual(['up', 'down', ''])
  })
  it('盘中当天点只在未结算时追加，且基准用当天收盘', () => {
    expect(points(curve, summary()).at(-1)).toEqual({ date: '2026-11-04', nav: 1.002, close: 4100 })
    expect(points(curve, summary({ day: '2026-11-03' }))).toHaveLength(3)
    expect(points(null, summary())).toEqual([{ date: '2026-11-04', nav: 1.002, close: null }])
  })
  it('区间起点取 N 个月前最后一个点；相对收益以首个有效值为 0', () => {
    const all = ['2026-04-30', '2026-05-29', '2026-06-01', '2026-11-04'].map((date, i) => ({ date, nav: 1 + i / 10, close: null }))
    expect(slice(all, '6m').map(p => p.date)).toEqual(['2026-04-30', '2026-05-29', '2026-06-01', '2026-11-04'])
    expect(slice(all, '1m').map(p => p.date)).toEqual(['2026-06-01', '2026-11-04'])
    expect(relative([null, 2, 3])).toEqual([NaN, 0, 0.5])
    expect(relative([null, null]).every(Number.isNaN)).toBe(true)
  })
  it('月份刻度每月一个，过密时隔月；轮询间隔随运行状态', () => {
    const dates = ['2026-05-29', '2026-06-01', '2026-06-02', '2026-07-01']
    expect(monthTicks(dates).map(t => t.label)).toEqual(['2026-05', '2026-06', '2026-07'])
    expect(pollDelay(summary())).toBe(10_000)
    expect(pollDelay(summary({ running: false }))).toBe(30_000)
    expect(pollDelay(summary({ running: false, settled: true }))).toBe(60_000)
  })
})

describe('页面', () => {
  beforeEach(() => { api.mockReset(); vi.useFakeTimers() })
  afterEach(() => { vi.useRealTimers(); localStorage.clear() })

  it('没有账户时给空状态和一句命令行用法', async () => {
    api.mockResolvedValue({ items: [] })
    const wrapper = mount(PaperPage)
    await flushPromises()
    expect(wrapper.text()).toContain('还没有模拟账户')
    expect(wrapper.find('code').text()).toContain('python -m ashare.paper start --directory')
    expect(api).toHaveBeenCalledTimes(1)
    wrapper.unmount()
  })

  it('展示账户、持仓、委托、风控与成交；按运行状态轮询，页面隐藏时暂停', async () => {
    api.mockImplementation(async (url: string) => url === '/paper/accounts' ? { items: [{ id: '演示', since: '2026-10-30', running: true }] } : url.endsWith('/curve') ? curve : summary())
    const wrapper = mount(PaperPage)
    await flushPromises()
    expect(api).toHaveBeenCalledWith(`/paper/accounts/${encodeURIComponent('演示')}`)
    expect(wrapper.find('.equity').text()).toBe('¥ 1,002,000')
    expect(wrapper.find('.hero .up').text()).toBe('+2,000（+0.20%）')
    expect(wrapper.text()).toContain('拒单 · 资金不足')
    const rows = wrapper.findAll('.table-panel tbody tr')
    expect(rows).toHaveLength(1)
    expect(rows[0].text()).toContain('合成一')
    expect(wrapper.find('svg[aria-label="权益曲线"]').exists()).toBe(true)
    await wrapper.findAll('.panel-tab')[1].trigger('click')
    expect(wrapper.find('.table-panel tbody').text()).toContain('部分成交')
    const calls = api.mock.calls.length
    await vi.advanceTimersByTimeAsync(10_000)
    expect(api.mock.calls.length).toBe(calls + 1) // 只重取摘要，曲线在日期未变时不重取
    Object.defineProperty(document, 'hidden', { configurable: true, value: true })
    document.dispatchEvent(new Event('visibilitychange'))
    await vi.advanceTimersByTimeAsync(60_000)
    expect(api.mock.calls.length).toBe(calls + 1)
    Object.defineProperty(document, 'hidden', { configurable: true, value: false })
    document.dispatchEvent(new Event('visibilitychange'))
    await flushPromises()
    expect(api.mock.calls.length).toBe(calls + 2)
    wrapper.unmount()
    await vi.advanceTimersByTimeAsync(60_000)
    expect(api.mock.calls.length).toBe(calls + 2)
  })

  it('读取失败只显示一句友好提示', async () => {
    api.mockImplementation(async (url: string) => { if (url === '/paper/accounts') return { items: [{ id: 'a', since: null, running: false }] }; throw new Error('boom /secret/path') })
    const wrapper = mount(PaperPage)
    await flushPromises()
    expect(wrapper.find('[role=alert]').text()).toBe('账户暂时读取不到，请稍后再试')
    expect(wrapper.text()).not.toContain('secret')
    wrapper.unmount()
  })

  it('历史成交按页加载更多', async () => {
    const fill = (i: number) => ({ day: '2026-11-03', time: null, symbol: `S${i}`, name: '', side: 'sell', qty: 100, price: 1, fee: 5 })
    api.mockImplementation(async (url: string) => {
      if (url === '/paper/accounts') return { items: [{ id: 'a', since: null, running: false }] }
      if (url.endsWith('/curve')) return curve
      if (url.includes('/fills?offset=0')) return { total: 3, items: [fill(0), fill(1)] }
      if (url.includes('/fills?offset=2')) return { total: 3, items: [fill(2)] }
      return summary({ running: false })
    })
    const wrapper = mount(PaperPage)
    await flushPromises()
    await wrapper.findAll('.panel-tab')[2].trigger('click')
    await flushPromises()
    expect(wrapper.findAll('.table-panel tbody tr')).toHaveLength(2)
    await wrapper.get('.more').trigger('click')
    await flushPromises()
    expect(wrapper.findAll('.table-panel tbody tr')).toHaveLength(3)
    expect(wrapper.find('.more').exists()).toBe(false)
    wrapper.unmount()
  })
})
