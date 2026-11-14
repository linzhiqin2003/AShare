import { mount } from '@vue/test-utils'
import { afterEach, describe, expect, it } from 'vitest'
import { nextTick } from 'vue'
import RebalanceControl from './RebalanceControl.vue'
import { options } from './ui/testing'

afterEach(() => { document.body.innerHTML = '' })

describe('调仓周期下拉（2026-11-14 用户：按钮加输入框的拼法难看）', () => {
  it('选一个发数字，多选发列表（扫参）', async () => {
    const wrapper = mount(RebalanceControl, { props: { modelValue: 1 }, attachTo: document.body })
    expect(wrapper.get('button[aria-label="调仓周期"]').text()).toBe('1 天')
    expect(wrapper.find('input').exists()).toBe(false)
    await wrapper.get('button[aria-label="调仓周期"]').trigger('click'); await nextTick()
    options().find(o => o.textContent?.trim() === '5 天')!.click(); await nextTick()
    expect(wrapper.emitted('update:modelValue')!.at(-1)).toEqual([[1, 5]])
    await wrapper.setProps({ modelValue: [1, 5] })
    options().find(o => o.textContent?.trim() === '1 天')!.click(); await nextTick()
    expect(wrapper.emitted('update:modelValue')!.at(-1)).toEqual([5])
    wrapper.unmount()
  })
  it('旧配置里不在预设中的值照样显示', () => {
    const wrapper = mount(RebalanceControl, { props: { modelValue: [7, 20] } })
    expect(wrapper.get('button[aria-label="调仓周期"]').text()).toBe('7 天、20 天')
    wrapper.unmount()
  })
})
