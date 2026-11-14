// 2026-11-14 外壳：路由、外观偏好、数据页与状态栏的纯函数。
import { afterEach, describe, expect, it, vi } from 'vitest'
import { pageFromHash, PAGES } from './router'
import { applyAppearance, isDark, rise, setRise, setTheme, theme, themeShortcut } from './appearance'
import { bytes, formatValue, linePath, matches, partition, percent, stamp, ticks, yearTicks } from './datastatus'
import { heatColor, signedPercent } from './market'
import { BACKTEST_FIELDS, validate } from './preferences'

afterEach(() => { localStorage.clear(); document.documentElement.removeAttribute('data-theme'); vi.unstubAllGlobals() })

describe('hash 路由', () => {
  it('五个顶栏页面与设置页可直达，未知地址回到研究台', () => {
    expect(PAGES.map(p => p.label)).toEqual(['研究台', '因子库', '数据', '回测', '模拟盘'])
    for (const page of ['library', 'data', 'backtest', 'paper', 'settings']) expect(pageFromHash(`#/${page}`)).toBe(page)
    expect(pageFromHash('#/data/close')).toBe('data')
    expect(pageFromHash('')).toBe('research')
    expect(pageFromHash('#/nope')).toBe('research')
    expect(pageFromHash('#/__proto__')).toBe('research')
  })
})

describe('外观偏好', () => {
  it('浅色/深色写属性并记住，跟随系统时去掉属性', () => {
    setTheme('dark')
    expect(document.documentElement.dataset.theme).toBe('dark')
    expect(localStorage.getItem('ashare.theme')).toBe('dark')
    setTheme('system')
    expect(document.documentElement.hasAttribute('data-theme')).toBe(false)
    setTheme('light')
    expect(document.documentElement.dataset.theme).toBe('light')
  })
  it('⌘⇧L 在浅色与深色之间切换；跟随系统时切到与当前显示相反的一种', () => {
    setTheme('light')
    const event = new KeyboardEvent('keydown', { key: 'L', metaKey: true, shiftKey: true, cancelable: true })
    themeShortcut(event)
    expect(theme.value).toBe('dark'); expect(event.defaultPrevented).toBe(true)
    themeShortcut(new KeyboardEvent('keydown', { key: 'l', ctrlKey: true, shiftKey: true }))
    expect(theme.value).toBe('light')
    themeShortcut(new KeyboardEvent('keydown', { key: 'l', metaKey: true }))
    expect(theme.value).toBe('light')
    vi.stubGlobal('matchMedia', () => ({ matches: true }))
    setTheme('system'); expect(isDark()).toBe(true)
    themeShortcut(new KeyboardEvent('keydown', { key: 'l', metaKey: true, shiftKey: true }))
    expect(theme.value).toBe('light')
  })
  it('涨跌颜色写到根元素上', () => {
    setRise('green'); expect(document.documentElement.dataset.rise).toBe('green')
    setRise('red'); applyAppearance(); expect(document.documentElement.dataset.rise).toBe('red'); expect(rise.value).toBe('red')
  })
})

describe('数据页格式化', () => {
  it('百分比、字节、分区与时间', () => {
    expect(percent(0.6694)).toBe('66.9%'); expect(percent(null)).toBe('—'); expect(percent(1, 2)).toBe('100.00%')
    expect(bytes(6780122776)).toBe('6.3 GB'); expect(bytes(512)).toBe('512 B')
    expect(partition('20261112')).toBe('2026-11-12'); expect(partition('202611')).toBe('2026-11'); expect(partition('SHFE')).toBe('SHFE'); expect(partition(null)).toBe('—')
    expect(stamp('2026-11-13T01:28:54+08:00')).toBe('2026-11-13 01:28'); expect(stamp('2026-11-13T01:28:54+08:00', false)).toBe('01:28'); expect(stamp(null)).toBe('—')
  })
  it('按单位显示数值：比例用百分比，大数用万/亿', () => {
    expect(formatValue(0.0109, '比例')).toBe('1.09%')
    expect(formatValue(2.5e9, '元')).toBe('25亿')
    expect(formatValue(35000, '股')).toBe('3.5万')
    expect(formatValue(13.2, '元')).toBe('13.2')
    expect(formatValue(null, '元')).toBe('—')
  })
  it('整齐刻度落在区间内', () => {
    expect(ticks(0.0007, 0.071, 5)).toEqual([0.02, 0.04, 0.06])
    expect(ticks(2, 200, 5)).toEqual([50, 100, 150, 200])
    expect(ticks(3, 3)).toEqual([3])
    for (const value of ticks(-1.3, 7.9)) { expect(value).toBeGreaterThanOrEqual(-1.3); expect(value).toBeLessThanOrEqual(7.9) }
  })
  it('迷你图遇到空值断开，数值夹在 0–1', () => {
    expect(linePath([], 100, 10)).toBe('')
    expect(linePath([1, null, 0], 100, 10, 0)).toBe('M0.0 0.0 M100.0 10.0')
    expect(linePath([2, -1], 10, 10, 0)).toBe('M0.0 0.0 L10.0 10.0')
  })
  it('年份刻度按跨度取步长，且只标一月', () => {
    const months = Array.from({ length: 240 }, (_, i) => `${2007 + Math.floor(i / 12)}${String(i % 12 + 1).padStart(2, '0')}`)
    const labels = yearTicks(months).map(t => t.label)
    expect(labels[0]).toBe('2007'); expect(labels.length).toBeLessThanOrEqual(8)
    expect(yearTicks(['200703', '200704'])).toEqual([])
  })
  it('字段搜索看名称、中文名与类别', () => {
    const field = { name: 'pe_ttm', label: '滚动市盈率', category: '估值', dtype: 'float32', unit: '倍', start: null, latest: null, spark: [], references: 0 }
    expect(matches(field, 'PE')).toBe(true); expect(matches(field, '市盈')).toBe(true); expect(matches(field, '估值')).toBe(true); expect(matches(field, 'close')).toBe(false); expect(matches(field, ' ')).toBe(true)
  })
})

describe('状态栏与默认设定', () => {
  it('涨跌幅带符号，负零显示为 0', () => {
    expect(signedPercent(0.02)).toBe('+2.00%'); expect(signedPercent(-0.0346)).toBe('-3.46%'); expect(signedPercent(-0.00001)).toBe('0.00%'); expect(signedPercent(null)).toBe('—')
  })
  it('热力格颜色跟随涨跌变量，幅度越大越深', () => {
    expect(heatColor(0.02, 0.02)).toBe('color-mix(in srgb, var(--rise) 100%, var(--panel))')
    expect(heatColor(-0.001, 0.02)).toContain('var(--fall)')
    expect(heatColor(0, 0)).toBe('color-mix(in srgb, var(--rise) 15%, var(--panel))')
  })
  it('默认设定只接受选项里的值和范围内的整数', () => {
    const field = (key: string) => BACKTEST_FIELDS.find(f => f.key === key)!
    expect(validate(field('trade_limitation'), false)).toBe(false)
    expect(validate(field('trade_limitation'), 'false')).toBeUndefined()
    expect(validate(field('n_groups'), 5)).toBe(5)
    expect(validate(field('universe'), 'TOP9')).toBeUndefined()
    expect(validate(field('rebalance'), 20)).toBe(20)
    expect(validate(field('rebalance'), 0)).toBeUndefined()
    expect(validate(field('rebalance'), 2.5)).toBeUndefined()
    expect(validate(field('decay'), null)).toBeUndefined()
    expect(validate(field('benchmark'), '000300.SH')).toBe('000300.SH')
    expect(validate(field('benchmark'), '')).toBeUndefined()
  })
})
