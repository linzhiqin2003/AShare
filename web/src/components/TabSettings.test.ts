import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import TabSettings from './TabSettings.vue'

describe('tab 设置隔离', () => {
  it('只发出当前 tab 的副本，不修改另一个 tab 或默认值', async () => {
    const settings = { universe: 'TOP800', group_neutralize: ['industry'], n_groups: 10, trade_at:'open', by:'year', weights_portfolio:false, sub_universes:true, trade_limit:true, decay:0 }
    const defaults = { ...settings }
    const wrapper = mount(TabSettings, { props: { modelValue: settings, name: 'alpha', defaults }, global: { stubs: { BenchmarkSelect:true } } })
    await wrapper.findAll('[aria-label="股票池"] button').find(b => b.text() === '全市场')!.trigger('click')
    const emitted = wrapper.emitted('update:modelValue')![0][0]
    expect(emitted).toEqual({ ...settings, universe: 'ALL' })
    expect(settings.universe).toBe('TOP800')
    expect(defaults.universe).toBe('TOP800')
    expect(wrapper.text()).not.toContain('正交向量')
    expect(wrapper.text()).not.toContain('延迟')
    document.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape'}))
    expect(wrapper.emitted('close')).toHaveLength(1)
    wrapper.unmount()
  })

  it('没开权重组合时衰减置灰（2026-11-14：衰减只作用于权重组合，界面看不出来）', async () => {
    const base = { universe: 'TOP800', group_neutralize: [null], n_groups: 10, trade_at:'open', by:'year', sub_universes:true, trade_limitation:true, decay:5 }
    const controls = (w: ReturnType<typeof mount>) => ['减少衰减','衰减天数','增加衰减'].map(label => w.find(`[aria-label="${label}"]`).attributes('disabled') !== undefined)
    const cases: [boolean | boolean[], boolean][] = [[false, true], [true, false], [[true, false], false]]
    for (const [weights, disabled] of cases) {
      const settings = { ...base, weights_portfolio: weights }
      const wrapper = mount(TabSettings, { props: { modelValue: settings, name: 'alpha', defaults: settings }, global: { stubs: { BenchmarkSelect:true } } })
      expect(controls(wrapper)).toEqual([disabled, disabled, disabled])
      await wrapper.find('[aria-label="增加衰减"]').trigger('click')
      expect(wrapper.emitted('update:modelValue')?.length ?? 0).toBe(disabled ? 0 : 1)
      wrapper.unmount()
    }
  })
})
