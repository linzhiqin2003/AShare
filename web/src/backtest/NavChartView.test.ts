// 验证期窗口切换只在全局开关打开时出现（2026-11-14 新增）。
import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import NavChart from './NavChart.vue'
import type { BacktestDocument } from './types'

const doc = {
  start: '2016-01-04', end: '2016-01-07', benchmark: { symbol: '000300.SH', name: '沪深300' },
  series: { dates: ['2016-01-04', '2016-01-05', '2016-01-06', '2016-01-07'], nav: [1, 1.02, 0.99, 1.05], benchmark: [1, 0.98, 0.97, 1], drawdown: [0, 0, -0.03, 0] },
} as unknown as BacktestDocument

describe('净值图', () => {
  it('默认没有验证期切换；打开开关后出现并发出窗口', async () => {
    const hidden = mount(NavChart, { props: { document: doc, part: 'is', showValidation: false } })
    expect(hidden.text()).not.toContain('验证')
    expect(hidden.find('path.nav').attributes('d')).toMatch(/^M/)
    expect(hidden.text()).toContain('沪深300')
    const shown = mount(NavChart, { props: { document: doc, part: 'is', showValidation: true } })
    await shown.get('button[aria-pressed="false"]').trigger('click')
    expect(shown.emitted('part')?.[0]).toEqual(['validation'])
  })
  it('数据不足一年时不提供 1/3/5 年时间窗', () => {
    const wrapper = mount(NavChart, { props: { document: doc, part: 'is', showValidation: false } })
    expect(wrapper.text()).not.toContain('1 年')
  })
})
