// 净值图的组切换行（2026-11-14 新增）：只有一组时不出现；多组时叠画其余组、点选切换、失败组标出。
import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import NavChart from './NavChart.vue'
import type { BacktestDocument } from './types'

const dates = ['2016-01-04', '2016-01-05', '2016-01-06', '2016-01-07']
const doc = (nav: number[]) => ({
  start: dates[0], end: dates[3], benchmark: null,
  series: { dates, nav, benchmark: null, drawdown: [0, 0, 0, 0] },
}) as unknown as BacktestDocument
const first = doc([1, 1.02, 0.99, 1.05]), second = doc([1, 0.9, 0.95, 1.4])

describe('净值图组切换', () => {
  it('只有一组时没有切换行，图例照旧', () => {
    const wrapper = mount(NavChart, { props: { document: first, part: 'is', showValidation: false, groups: [{ id: 'a', label: '', status: 'complete', document: first }], selected: 'a' } })
    expect(wrapper.find('.groups').exists()).toBe(false)
    expect(wrapper.find('path.other').exists()).toBe(false)
    expect(wrapper.text()).toContain('策略')
    const bare = mount(NavChart, { props: { document: first, part: 'is', showValidation: false } })
    expect(bare.find('.groups').exists()).toBe(false)
  })
  it('多组：其余组叠画、纵轴包住全部曲线，点组名切换，失败组标出', async () => {
    const groups = [
      { id: 'a', label: '20', status: 'complete' as const, document: first },
      { id: 'b', label: '60', status: 'complete' as const, document: second },
      { id: 'c', label: '120', status: 'failed' as const, document: null },
      { id: 'd', label: '250', status: 'running' as const, document: null },
    ]
    const wrapper = mount(NavChart, { props: { document: first, part: 'is', showValidation: false, groups, selected: 'a' } })
    const buttons = wrapper.findAll('.groups button')
    expect(buttons.map(b => b.text())).toEqual(['20', '60', '120', '250'])
    expect(buttons[0].attributes('aria-selected')).toBe('true')
    expect(buttons[0].classes()).toContain('active')
    expect(buttons[2].classes()).toContain('failed')
    expect(wrapper.findAll('path.other')).toHaveLength(1)
    // 第二组最高 1.4：纵轴刻度要包住它，而不只是选中组的 1.05。
    const ticks = wrapper.findAll('svg.chart')[0].findAll('g text').map(t => Number(t.text()))
    expect(Math.max(...ticks)).toBeGreaterThanOrEqual(1.4)
    expect(wrapper.text()).not.toContain('策略')
    await buttons[1].trigger('click')
    expect(wrapper.emitted('select')?.[0]).toEqual(['b'])
  })
  it('选中组失败、没有文档时切换行仍在，可以切走', async () => {
    const groups = [
      { id: 'a', label: '20', status: 'failed' as const, document: null },
      { id: 'b', label: '60', status: 'complete' as const, document: second },
    ]
    const wrapper = mount(NavChart, { props: { document: null, part: 'is', showValidation: false, groups, selected: 'a' } })
    expect(wrapper.findAll('.groups button')).toHaveLength(2)
    await wrapper.findAll('.groups button')[1].trigger('click')
    expect(wrapper.emitted('select')?.[0]).toEqual(['b'])
  })
})
