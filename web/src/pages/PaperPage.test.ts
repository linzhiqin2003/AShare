// 模拟盘页（2026-11-14 新增）：纯计算与页面行为，接口用合成数据替身。
import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { accountLine, backtestLabel, monthTicks, percent, points, pollDelay, relative, signed, slice, suggestName, tone, validName } from './paper'
import { pick, selectText } from '../components/ui/testing'
import type { PaperCurve, PaperSummary } from './paper'
import PaperPage from './PaperPage.vue'

const api = vi.hoisted(() => vi.fn())
vi.mock('../api', () => ({
  api,
  post: (path: string, body: unknown) => api(path, { method: 'POST', body: JSON.stringify(body) }),
  friendly: (e: unknown) => ({ message: e instanceof Error && e.message === 'conflict' ? '已有同名账户' : '账户暂时读取不到，请稍后再试' }),
}))

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
    expect(wrapper.find('code').text()).toContain('python -m ashare.paper account create')
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

// 2026-11-14 模拟盘每日运行：账户与策略绑定、从回测新建账户。
describe('账户与策略', () => {
  beforeEach(() => { api.mockReset(); vi.useFakeTimers() })
  afterEach(() => { vi.useRealTimers(); localStorage.clear(); document.body.innerHTML = '' })
  const strategy = { name: '低波动 30', universe: 'csi300', rebalance: 'month', top_n: 30, trade_at: 'open', initial_cash: 1e7, enabled: true, origin: 'backtest' }
  const plan = { last_signal_day: '2026-10-30', last_execution_day: '2026-11-02', next_signal_day: '2026-11-30', pending_execution_day: null }

  it('副标题、默认账户名与名称规则', () => {
    expect(accountLine({ id: 'a', since: '2015-01-05', running: false, started: true, strategy, plan })).toBe('纸面跟踪 · 月度调仓 · 自 2015-01-05')
    expect(accountLine({ id: 'a', since: null, running: false, started: false, strategy, plan: null })).toBe('纸面跟踪 · 月度调仓 · 等待首个交易日')
    expect(accountLine({ id: 'a', since: null, running: false, started: true, strategy: { ...strategy, enabled: false } })).toBe('已停用 · 月度调仓')
    expect(accountLine({ id: 'a', since: '2026-10-30', running: true })).toBe('运行中 · 自 2026-10-30')
    expect(suggestName(' 低波动 30 / 月度 ')).toBe('低波动-30-月度')
    expect(suggestName('...x')).toBe('x')
    expect([validName('低波动-30'), validName('.ops'), validName('a/b'), validName('x'.repeat(65))]).toEqual([true, false, false, false])
    expect(backtestLabel({ id: 'k', name: '回测', universe: 'csi300', rebalance: 'week', top_n: 20, capital: 1000 })).toBe('回测 · 沪深300 · 每周调仓 · 前 20 只')
  })

  it('还没开始运行的账户只显示策略和计划，不请求摘要', async () => {
    api.mockImplementation(async (url: string) => url === '/paper/accounts' ? { items: [{ id: 'lowvol', since: null, running: false, started: false, strategy, plan: { ...plan, last_execution_day: null } }] } : summary())
    const wrapper = mount(PaperPage)
    await flushPromises()
    expect(api.mock.calls.map(c => c[0])).toEqual(['/paper/accounts', '/paper/accounts'])
    expect(wrapper.text()).toContain('下一个交易日开盘后开始运行')
    expect(wrapper.text()).toContain('下次信号日2026-11-30')
    expect(wrapper.find('.account b').text()).toBe('低波动 30')
    await vi.advanceTimersByTimeAsync(60_000)
    expect(api.mock.calls.every(c => c[0] === '/paper/accounts')).toBe(true)
    wrapper.unmount()
  })

  it('从回测新建账户：自绘下拉、默认名称和资金、建好后选中新账户', async () => {
    const items: unknown[] = [{ id: 'old', since: '2026-10-30', running: false, started: true, strategy: null, plan: null }]
    const backtests = [
      { id: 'a'.repeat(24), name: '低波动 30', universe: 'csi300', rebalance: 'month', top_n: 30, capital: 2000 },
      { id: 'b'.repeat(24), name: '动量 50', universe: 'ALL', rebalance: 'week', top_n: 50, capital: 500 },
    ]
    api.mockImplementation(async (url: string, init?: RequestInit) => {
      if (url === '/paper/accounts' && init?.method === 'POST') {
        const body = JSON.parse(String(init.body))
        if (body.name === 'taken') throw new Error('conflict')
        const created = { id: body.name, since: null, running: false, started: false, strategy: { ...strategy, name: '动量 50' }, plan: null }
        items.unshift(created) // 服务端之后的列表里有它
        return created
      }
      if (url === '/paper/accounts') return { items }
      if (url === '/paper/backtests') return { items: backtests }
      if (url.endsWith('/curve')) return curve
      return summary()
    })
    const wrapper = mount(PaperPage, { attachTo: document.body })
    await flushPromises()
    await wrapper.get('.create-row .create').trigger('click')
    await flushPromises()
    const dialog = wrapper.get('[role=dialog]')
    expect(dialog.find('select').exists()).toBe(false)
    expect(selectText(wrapper, '回测')).toBe('低波动 30 · 沪深300 · 月度调仓 · 前 30 只')
    expect((dialog.get('input[aria-label="账户名称"]').element as HTMLInputElement).value).toBe('低波动-30')
    await pick(wrapper, '回测', '动量 50 · 全市场 · 每周调仓 · 前 50 只')
    expect((dialog.get('input[aria-label="账户名称"]').element as HTMLInputElement).value).toBe('动量-50')
    await dialog.get('input[aria-label="账户名称"]').setValue('taken')
    await wrapper.get('.actions .primary').trigger('click')
    await flushPromises()
    expect(wrapper.get('[role=dialog] [role=alert]').text()).toBe('已有同名账户')
    await wrapper.get('input[aria-label="账户名称"]').setValue('../bad')
    expect(wrapper.get('.actions .primary').attributes('disabled')).toBeDefined()
    await wrapper.get('input[aria-label="账户名称"]').setValue('momentum')
    await wrapper.get('.actions .primary').trigger('click')
    await flushPromises()
    const posts = api.mock.calls.filter(c => c[1]?.method === 'POST').map(c => JSON.parse(String(c[1].body)))
    expect(posts.at(-1)).toEqual({ name: 'momentum', backtest: 'b'.repeat(24), capital: 500 })
    expect(wrapper.find('[role=dialog]').exists()).toBe(false)
    expect(wrapper.findAll('.account').map(b => b.find('b').text())).toEqual(['动量 50', 'old'])
    expect(wrapper.find('.account.on b').text()).toBe('动量 50')
    wrapper.unmount()
  })

  it('没有已完成的回测时说清楚，Esc 关闭', async () => {
    api.mockImplementation(async (url: string) => url === '/paper/backtests' ? { items: [] } : { items: [] })
    const wrapper = mount(PaperPage, { attachTo: document.body })
    await flushPromises()
    await wrapper.get('.empty .create').trigger('click')
    await flushPromises()
    expect(wrapper.get('[role=dialog]').text()).toContain('还没有已完成的回测')
    expect(wrapper.get('.actions .primary').attributes('disabled')).toBeDefined()
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }))
    await flushPromises()
    expect(wrapper.find('[role=dialog]').exists()).toBe(false)
    wrapper.unmount()
  })
})
