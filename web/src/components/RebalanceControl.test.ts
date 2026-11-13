import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import RebalanceControl from './RebalanceControl.vue'

describe('调仓周期输入', () => {
  it('预设与列表发出真实配置，错误输入也阻止后端继续采用旧值', async () => {
    const wrapper = mount(RebalanceControl,{props:{modelValue:1}})
    await wrapper.findAll('button').find(b => b.text() === '5')!.trigger('click')
    expect(wrapper.emitted('update:modelValue')!.at(-1)).toEqual([5])
    await wrapper.get('input').setValue('[1,5,20]')
    expect(wrapper.emitted('update:modelValue')!.at(-1)).toEqual([[1,5,20]])
    await wrapper.get('input').setValue('0')
    expect(wrapper.get('[role=alert]').text()).toContain('1–2520')
    expect(wrapper.emitted('update:modelValue')!.at(-1)).toEqual(['0'])
    wrapper.unmount()
  })
})
