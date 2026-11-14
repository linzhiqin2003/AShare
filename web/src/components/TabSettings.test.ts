import { mount } from '@vue/test-utils'
import { nextTick } from 'vue'
import { describe, expect, it } from 'vitest'
import TabSettings from './TabSettings.vue'
import SharedSettings from './SharedSettings.vue'
import { options, pick, selectText } from './ui/testing'

describe('tab 设置隔离', () => {
  it('只发出当前 tab 的副本，不修改另一个 tab 或默认值', async () => {
    const settings = { universe: 'TOP800', group_neutralize: ['industry'], n_groups: 10, trade_at:'open', by:'year', weights_portfolio:false, sub_universes:true, trade_limit:true, decay:0 }
    const defaults = { ...settings }
    const wrapper = mount(TabSettings, { props: { modelValue: settings, defaults }, global: { stubs: { BenchmarkSelect:true } } })
    await pick(wrapper, '股票池', '全市场')
    const emitted = wrapper.emitted('update:modelValue')![0][0]
    expect(emitted).toEqual({ ...settings, universe: 'ALL' })
    expect(settings.universe).toBe('TOP800')
    expect(defaults.universe).toBe('TOP800')
    expect(wrapper.text()).not.toContain('正交向量')
    expect(wrapper.text()).not.toContain('延迟')
    // 2026-11-14 合并成研究页后设置摊在编辑器下方，不再是弹出层：没有对话框，也没有「关闭」可发。
    expect(wrapper.find('[role=dialog]').exists()).toBe(false)
    document.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape'}))
    expect(wrapper.emitted('close')).toBeUndefined()
    wrapper.unmount()
  })

  it('没开权重组合时衰减置灰（2026-11-14：衰减只作用于权重组合，界面看不出来）', async () => {
    const base = { universe: 'TOP800', group_neutralize: [null], n_groups: 10, trade_at:'open', by:'year', sub_universes:true, trade_limitation:true, decay:5 }
    const controls = (w: ReturnType<typeof mount>) => ['减少衰减天数','衰减天数','增加衰减天数'].map(label => w.find(`[aria-label="${label}"]`).attributes('disabled') !== undefined)
    const cases: [boolean | boolean[], boolean][] = [[false, true], [true, false], [[true, false], false]]
    for (const [weights, disabled] of cases) {
      const settings = { ...base, weights_portfolio: weights }
      const wrapper = mount(TabSettings, { props: { modelValue: settings, defaults: settings }, global: { stubs: { BenchmarkSelect:true } } })
      expect(controls(wrapper)).toEqual([disabled, disabled, disabled])
      await wrapper.find('[aria-label="增加衰减天数"]').trigger('click')
      expect(wrapper.emitted('update:modelValue')?.length ?? 0).toBe(disabled ? 0 : 1)
      wrapper.unmount()
    }
  })

  // 2026-11-14 用户：「这种都直接下拉选框吧，好看点」——设置里的分段按钮全部换成下拉，中性化是多选下拉。
  it('设置项都是下拉：单选选一项发出单个值，中性化多选切换勾选并保持打开', async () => {
    const settings = { universe: 'TOP800', group_neutralize: ['industry'], n_groups: 10, trade_at: 'open', by: 'year', weights_portfolio: false, sub_universes: true, trade_limitation: true, decay: 0 }
    const wrapper = mount(TabSettings, { props: { modelValue: settings, defaults: settings }, global: { stubs: { BenchmarkSelect: true } }, attachTo: document.body })
    expect(wrapper.find('.segmented').exists()).toBe(false)
    // 成交时点挪到共用的「样本与成交」（SharedSettings），见下面的用例。
    expect(['股票池', '中性化', '分组数', '组合', '分段'].map(label => selectText(wrapper, label))).toEqual(['TOP800', '行业', '10', '分组多空', '按年'])
    expect(wrapper.find('[aria-label="成交时点"]').exists()).toBe(false)
    await pick(wrapper, '分组数', '5')
    expect(wrapper.emitted('update:modelValue')!.at(-1)![0]).toMatchObject({ n_groups: 5 })
    await wrapper.get('button[aria-label="中性化"]').trigger('click'); await nextTick()
    options().find(o => o.textContent?.trim() === '市值')!.click(); await nextTick()
    expect(wrapper.emitted('update:modelValue')!.at(-1)![0]).toMatchObject({ group_neutralize: ['industry', 'cap'] })
    expect(document.querySelector('[role=listbox]')).not.toBeNull()
    wrapper.unmount()
  })
  it('值本来是数组（扫参或旧预设）时单选下拉显示拼接文字，选一项变回单个值', async () => {
    const settings = { universe: ['TOP800', 'ALL'], group_neutralize: [null, 'cap+industry'], n_groups: 10, trade_at: ['open', 'close'], by: 'year', weights_portfolio: [false, true], sub_universes: true, trade_limitation: true, decay: 0 }
    const wrapper = mount(TabSettings, { props: { modelValue: settings, defaults: settings }, global: { stubs: { BenchmarkSelect: true } }, attachTo: document.body })
    expect(selectText(wrapper, '股票池')).toBe('TOP800、全市场')
    expect(selectText(wrapper, '组合')).toBe('分组多空、另算权重多空')
    expect(selectText(wrapper, '中性化')).toBe('无、市值+行业')
    await pick(wrapper, '股票池', 'TOP800')
    expect(wrapper.emitted('update:modelValue')!.at(-1)![0]).toMatchObject({ universe: 'TOP800' })
    wrapper.unmount()
  })

  it('预设和恢复默认只换因子检验自己的设置，共用的成交价和基准不变', async () => {
    const settings = { universe: 'ALL', trade_at: 'close', benchmark: '000300.SH', n_groups: 5 }
    const defaults = { universe: 'TOP1800', trade_at: 'open', benchmark: '000001.SH', n_groups: 10 }
    const wrapper = mount(TabSettings, { props: { modelValue: settings, defaults } })
    await wrapper.findAll('button').find(b => b.text() === '恢复默认')!.trigger('click')
    expect(wrapper.emitted('update:modelValue')!.at(-1)![0]).toEqual({ universe: 'TOP1800', trade_at: 'close', benchmark: '000300.SH', n_groups: 10 })
    wrapper.unmount()
  })
})

describe('样本与成交（两个视图共用）', () => {
  it('成交时点是下拉，选一项发出单个值；值是数组（扫参或旧预设）时显示拼接文字', async () => {
    const wrapper = mount(SharedSettings, { props: { modelValue: { trade_at: 'open', universe: 'TOP800' } }, global: { stubs: { BenchmarkSelect: true } }, attachTo: document.body })
    expect(wrapper.find('.segmented').exists()).toBe(false)
    expect(selectText(wrapper, '成交时点')).toBe('T+1 开盘')
    await pick(wrapper, '成交时点', 'T+1 收盘')
    expect(wrapper.emitted('update:modelValue')!.at(-1)![0]).toEqual({ trade_at: 'close', universe: 'TOP800' })
    await wrapper.setProps({ modelValue: { trade_at: ['open', 'close'] } })
    expect(selectText(wrapper, '成交时点')).toBe('T+1 开盘、T+1 收盘')
    expect(wrapper.text()).toContain('2016-01-01 至 2024-12-31')
    wrapper.unmount()
  })
})
