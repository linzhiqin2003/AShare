// 2026-11-14 数据页、设置页与状态栏组件：接口全部用假 fetch。
import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'
import DataPage from './DataPage.vue'
import SettingsPage from './SettingsPage.vue'
import StatusBar from '../components/StatusBar.vue'
import UiSelect from '../components/ui/UiSelect.vue'
import UiNumberInput from '../components/ui/UiNumberInput.vue'

const field = (name: string, label: string, category = '行情') => ({ name, label, category, dtype: 'float64', unit: '元', start: '2007-01-04', latest: 0.9, spark: [0.9, 1], references: name === 'close' ? 2 : 0 })
const overview = { datasets: [
  { key: 'stock', name: '股票日频面板', panel: { version: 'abcdef1234567890', start: '2007-01-04', end: '2026-11-12', days: 4825, symbols: 5923, fields: 48, size: 6.78e9, built_at: null }, verify: { generated_at: null, current: true, verdict: null, checks: [{ name: 'ohlc_high', checked: 10, failed: 0, pass_rate: 1, note: '' }, { name: 'price_limits', checked: 10, failed: 2, pass_rate: 0.8, note: '涨跌停' }] }, tables: [{ name: 'daily', group: 'stock', partitions: 3, rows: 30, failed: 0, first: '20070104', last: '20261112', fetched: null }], history: [] },
  { key: 'index', name: '指数行情', panel: null, verify: null, tables: [], history: [], indexes: [{ symbol: '000300.SH', name: '沪深300', start: '2007-01-04', end: '2026-11-12' }] },
] }
function detail(name: string) { return { ...field(name, name), overall: 0.8, latest_count: 5000, monthly: [['200701', 0.5], ['200801', 0.9]], distribution: null, true_share: null, factors: [{ name: 'momentum', sharpe: 1.2 }] } }

function routes(table: Record<string, (init?: RequestInit) => unknown>) {
  const calls: { path: string; init?: RequestInit }[] = []
  vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
    const path = url.replace(/^\/api/, '')
    calls.push({ path, init })
    const key = Object.keys(table).find(k => path === k || path.startsWith(k + '?') || (k.endsWith('*') && path.startsWith(k.slice(0, -1))))
    if (!key) return new Response(JSON.stringify({ message: '没有' }), { status: 404 })
    return new Response(JSON.stringify(table[key](init)), { status: 200 })
  }))
  return calls
}
afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); localStorage.clear() })

describe('数据页', () => {
  it('统计未就绪时轮询，就绪后显示字段、选中行加载详情', async () => {
    vi.useFakeTimers()
    let ready = false
    const calls = routes({
      '/data/overview': () => overview,
      '/data/fields': () => ready ? { state: 'ready', version: 'v', fields: [field('open', '开盘价'), field('close', '收盘价'), field('pe', '市盈率', '估值')] } : { state: 'pending', version: 'v', fields: [] },
      '/data/fields/*': () => detail(calls[calls.length - 1].path.split('/').pop()!),
    })
    const wrapper = mount(DataPage, { props: { completions: [] } })
    await flushPromises()
    expect(wrapper.text()).toContain('正在统计各字段覆盖')
    expect(wrapper.text()).toContain('5923 只 × 4825 个交易日，48 个字段')
    expect(wrapper.text()).toContain('1 项有差异')
    ready = true
    await vi.advanceTimersByTimeAsync(1600); await flushPromises()
    const rows = wrapper.findAll('.fields-table tbody tr')
    expect(rows.map(r => r.find('td').text())).toEqual(['open', 'close', 'pe'])
    // 上次选中的字段默认是 close。
    expect(wrapper.find('tr.selected td').text()).toBe('close')
    expect(calls.some(c => c.path === '/data/fields/close')).toBe(true)
    await rows[2].trigger('click'); await flushPromises()
    expect(calls.at(-1)!.path).toBe('/data/fields/pe')
    expect(wrapper.find('.detail .panel-header').text()).toContain('pe')
    expect(localStorage.getItem('ashare.data.field')).toBe('pe')
    await wrapper.find('.find input').setValue('估值'); await flushPromises()
    expect(wrapper.findAll('.fields-table tbody tr')).toHaveLength(1)
    wrapper.unmount()
    const before = calls.length
    await vi.advanceTimersByTimeAsync(5000)
    expect(calls.length).toBe(before)
  })
  it('切换数据集换成该数据集的页签，指数行情没有字段详情', async () => {
    routes({ '/data/overview': () => overview, '/data/fields': () => ({ state: 'ready', version: 'v', fields: [field('close', '收盘价')] }), '/data/fields/*': () => detail('close') })
    const wrapper = mount(DataPage, { props: { completions: [{ label: 'ts_mean', kind: 'operator', signature: 'ts_mean(x, window)', detail: '滚动均值', hint: '' }] } })
    await flushPromises()
    expect(wrapper.findAll('.tabs-header .panel-tab').map(t => t.text())).toEqual(['字段', '原始表', '更新记录', '校验', '算子'])
    await wrapper.findAll('.tabs-header .panel-tab')[4].trigger('click')
    expect(wrapper.text()).toContain('滚动均值')
    await wrapper.findAll('.dataset')[1].trigger('click'); await flushPromises()
    expect(wrapper.findAll('.tabs-header .panel-tab').map(t => t.text())).toEqual(['指数', '原始表', '更新记录'])
    expect(wrapper.find('.detail').exists()).toBe(false)
    expect(wrapper.text()).toContain('沪深300')
    expect(localStorage.getItem('ashare.data.dataset')).toBe('index')
  })
  it('接口失败只给友好提示', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ message: '操作失败，请稍后重试' }), { status: 500 })))
    const wrapper = mount(DataPage, { props: { completions: [] } })
    await flushPromises()
    expect(wrapper.find('[role=alert]').text()).toBe('操作失败，请稍后重试')
  })
})

describe('状态栏', () => {
  it('只显示当天有数据的项，涨跌用变量上色', async () => {
    routes({ '/market': () => ({ date: '2026-11-12', indexes: [{ symbol: '000001.SH', name: '上证指数', close: 3655.8295, change: -0.0067 }], breadth: { up: 875, down: 4635, flat: 63 }, industries: [{ name: '银行', change: 0.0068, count: 42 }, { name: '传媒', change: -0.0319, count: 100 }] }) })
    const wrapper = mount(StatusBar, { props: { elapsed: 14.62 } })
    await flushPromises()
    const text = wrapper.text()
    expect(text).toContain('已收盘 11-12'); expect(text).toContain('3,655.83'); expect(text).toContain('-0.67%')
    expect(text).toContain('银行+0.68%'); expect(text).toContain('传媒-3.19%'); expect(text).toContain('上次运行 14.6 s')
    expect(wrapper.find('.ix b.fall').text()).toBe('-0.67%')
    expect(wrapper.findAll('.cells i')).toHaveLength(2)
    wrapper.unmount()
  })
  it('没有行情时不显示行情，只留运行耗时', async () => {
    routes({ '/market': () => ({ date: null, indexes: [], breadth: null, industries: [] }) })
    const wrapper = mount(StatusBar, { props: { elapsed: null } })
    await flushPromises()
    expect(wrapper.text()).toBe('')
  })
})

describe('设置页', () => {
  it('主题点选立即生效；默认设定按原始类型保存，失败回滚并提示', async () => {
    let fail = false
    const backtest = { universe: 'TOP1800', benchmark: '000001.SH', rebalance: 1, decay: 0, trade_limitation: true, trade_at: 'open', n_groups: 10, weights_portfolio: false, sub_universes: true, by: 'year' }
    const calls = routes({
      '/preferences': init => {
        if (init?.method === 'PUT') { if (fail) throw new Error('x'); return { backtest: JSON.parse(String(init.body)), builtin: {}, keys: [] } }
        return { backtest, builtin: backtest, keys: [] }
      },
      '/benchmarks': () => ({ items: [{ symbol: '000001.SH', name: '上证指数' }, { symbol: '000300.SH', name: '沪深300' }], total: 2, selected: {}, default: {} }),
    })
    const wrapper = mount(SettingsPage)
    await flushPromises()
    await wrapper.findAll('.theme-card')[1].trigger('click')
    expect(document.documentElement.dataset.theme).toBe('dark')
    // 共享控件只按约定的签名交互：aria-label 找控件，update:modelValue 改值。
    const control = (label: string) => wrapper.findAllComponents(UiSelect).concat(wrapper.findAllComponents(UiNumberInput) as never).find(c => c.attributes('aria-label') === label || c.props('ariaLabel' as never) === label)!
    control('涨跌停与停牌').vm.$emit('update:modelValue', false); await flushPromises()
    const body = JSON.parse(String(calls.filter(c => c.init?.method === 'PUT').at(-1)!.init!.body))
    expect(body.trade_limitation).toBe(false)
    expect(Object.keys(body).sort()).toEqual(Object.keys(backtest).sort())
    control('基准').vm.$emit('update:modelValue', '000300.SH'); await flushPromises()
    expect(control('基准').props('modelValue')).toBe('000300.SH')
    control('调仓周期').vm.$emit('update:modelValue', 0); await flushPromises()
    expect(wrapper.find('[role=alert]').text()).toBe('调仓周期取 1–2520 个交易日')
    fail = true
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ message: '请求参数不受支持' }), { status: 422 })))
    control('股票池').vm.$emit('update:modelValue', 'TOP800'); await flushPromises()
    expect(wrapper.find('[role=alert]').text()).toBe('请求参数不受支持')
    expect(control('股票池').props('modelValue')).toBe('TOP1800')
    control('涨跌颜色').vm.$emit('update:modelValue', 'green'); await flushPromises()
    expect(document.documentElement.dataset.rise).toBe('green')
    document.documentElement.removeAttribute('data-theme')
  })
})
