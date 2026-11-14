import { mount } from '@vue/test-utils'
import { defineComponent, h, nextTick, shallowRef, withDirectives } from 'vue'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { UiCheckbox, UiNumberInput, UiSelect, UiSwitch, vTip, TIP_DELAY } from './index'
import { place } from './position'
import { options } from './testing'

afterEach(() => { vi.useRealTimers(); document.body.innerHTML = '' })
const key = (target: Element, name: string) => target.dispatchEvent(new KeyboardEvent('keydown', { key: name, bubbles: true }))
const choices = [{ value: 'sum', label: '累加' }, { value: 'compound', label: '复利' }, { value: 'relative', label: '相对基准', disabled: true }, { value: 'cash', label: 'Cash' }, { value: 'carry', label: 'Carry' }]

function mountSelect(value = 'sum') {
  const model = shallowRef<string>(value)
  const wrapper = mount(defineComponent({ setup: () => () => h(UiSelect, { modelValue: model.value, options: choices, 'aria-label': '累计方式', 'onUpdate:modelValue': (v: string | number | boolean | null) => model.value = String(v) }) }), { attachTo: document.body })
  return { wrapper, model }
}

describe('UiSelect（2026-11-14 用户要求：研究台「累加/复利」弹出的是系统原生菜单）', () => {
  it('不是原生 select；点开是挂在 body 上的 listbox，选中项打勾，点选发出新值并收起', async () => {
    const { wrapper, model } = mountSelect()
    expect(wrapper.find('select').exists()).toBe(false)
    const trigger = wrapper.get('button[aria-label="累计方式"]')
    expect(trigger.attributes('aria-haspopup')).toBe('listbox')
    expect(trigger.text()).toBe('累加')
    await trigger.trigger('click'); await nextTick()
    const list = document.querySelector('[role=listbox]')!
    expect(list.parentElement).toBe(document.body)
    expect(trigger.attributes('aria-expanded')).toBe('true')
    expect(options()[0].getAttribute('aria-selected')).toBe('true')
    expect(options()[0].querySelector('svg')).not.toBeNull()
    expect(options()[2].getAttribute('aria-disabled')).toBe('true')
    options()[2].click(); await nextTick()
    expect(model.value).toBe('sum')
    options()[1].click(); await nextTick()
    expect(model.value).toBe('compound')
    expect(document.querySelector('[role=listbox]')).toBeNull()
    expect(document.activeElement).toBe(trigger.element)
    wrapper.unmount()
  })
  it('键盘：下键打开，上下跳过禁用项，Home/End，首字母跳转，回车选中，Esc 关闭且不冒泡', async () => {
    const { wrapper, model } = mountSelect()
    const trigger = wrapper.get('button[aria-label="累计方式"]').element
    const outer = vi.fn(); const escapes = (e: KeyboardEvent) => { if (e.key === 'Escape') outer() }; document.addEventListener('keydown', escapes)
    key(trigger, 'ArrowDown'); await nextTick(); await nextTick()
    const list = document.querySelector<HTMLElement>('[role=listbox]')!
    expect(document.activeElement).toBe(list)
    const active = () => list.getAttribute('aria-activedescendant')!.split('-').at(-1)
    expect(active()).toBe('0')
    key(list, 'ArrowDown'); await nextTick(); expect(active()).toBe('1')
    key(list, 'ArrowDown'); await nextTick(); expect(active()).toBe('3')
    key(list, 'Home'); await nextTick(); expect(active()).toBe('0')
    key(list, 'End'); await nextTick(); expect(active()).toBe('4')
    key(list, 'c'); await nextTick(); expect(active()).toBe('3')
    key(list, 'Enter'); await nextTick()
    expect(model.value).toBe('cash')
    key(trigger, 'Enter'); await nextTick(); await nextTick()
    const reopened = document.querySelector<HTMLElement>('[role=listbox]')!
    key(reopened, 'Escape'); await nextTick()
    expect(document.querySelector('[role=listbox]')).toBeNull()
    expect(outer).not.toHaveBeenCalled()
    expect(model.value).toBe('cash')
    document.removeEventListener('keydown', escapes)
    wrapper.unmount()
  })
  it('点外面关闭，不改值', async () => {
    const { wrapper, model } = mountSelect()
    await wrapper.get('button').trigger('click'); await nextTick()
    document.body.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true }))
    await nextTick()
    expect(document.querySelector('[role=listbox]')).toBeNull()
    expect(model.value).toBe('sum')
    wrapper.unmount()
  })
})

describe('浮层贴边翻转', () => {
  it('下方放不下且上方更宽时翻到上方，水平方向收回视口', () => {
    vi.stubGlobal('innerHeight', 800); vi.stubGlobal('innerWidth', 1000)
    const rect = (left: number, top: number) => ({ left, top, bottom: top + 24, right: left + 80, width: 80, height: 24, x: left, y: top, toJSON: () => ({}) }) as DOMRect
    expect(place(rect(10, 100), { width: 120, height: 200 })).toEqual({ left: 10, top: 128, side: 'below' })
    expect(place(rect(950, 700), { width: 120, height: 200 })).toEqual({ left: 872, top: 496, side: 'above' })
    vi.unstubAllGlobals()
  })
})

describe('UiCheckbox / UiSwitch / UiNumberInput', () => {
  it('勾选框：role=checkbox、空格切换、半选态、系列颜色、禁用不发出', async () => {
    const wrapper = mount(UiCheckbox, { props: { modelValue: false, color: '#1F4FBF', 'aria-label': '选择 a' }, attachTo: document.body })
    const box = wrapper.get('[role=checkbox]')
    expect(wrapper.find('input').exists()).toBe(false)
    expect(box.attributes('aria-checked')).toBe('false')
    await box.trigger('click')
    expect(wrapper.emitted('update:modelValue')![0]).toEqual([true])
    await wrapper.setProps({ modelValue: true })
    expect(box.attributes('style')).toContain('#1F4FBF')
    expect(box.find('svg').exists()).toBe(true)
    await wrapper.setProps({ indeterminate: true })
    expect(box.attributes('aria-checked')).toBe('mixed')
    await box.trigger('click')
    expect(wrapper.emitted('update:modelValue')!.at(-1)).toEqual([true])
    await wrapper.setProps({ disabled: true, indeterminate: false })
    await box.trigger('click')
    expect(wrapper.emitted('update:modelValue')).toHaveLength(2)
    wrapper.unmount()
  })
  it('放在 label 里点文字也切换', async () => {
    const value = shallowRef(false)
    const wrapper = mount(defineComponent({ setup: () => () => h('label', [h(UiCheckbox, { modelValue: value.value, 'onUpdate:modelValue': (v: boolean) => value.value = v }), '换手']) }), { attachTo: document.body })
    wrapper.get('label').element.click(); await nextTick()
    expect(value.value).toBe(true)
    wrapper.unmount()
  })
  it('开关：role=switch，点击取反', async () => {
    const wrapper = mount(UiSwitch, { props: { modelValue: true, 'aria-label': '子池检验' } })
    expect(wrapper.get('[role=switch]').attributes('aria-checked')).toBe('true')
    await wrapper.get('[role=switch]').trigger('click')
    expect(wrapper.emitted('update:modelValue')![0]).toEqual([false])
    wrapper.unmount()
  })
  it('数字输入：没有原生 spinner；−/+ 按步长并夹在范围内；上下键；清空发出 null；越界在失焦时收回', async () => {
    const wrapper = mount(UiNumberInput, { props: { modelValue: 0.05, min: 0, max: 0.1, step: 0.01, unit: '天', 'aria-label': '换手', 'onUpdate:modelValue': (v: number | null) => wrapper.setProps({ modelValue: v }) } })
    const input = wrapper.get('input')
    expect(input.attributes('type')).toBe('text')
    expect(wrapper.find('input[type=number]').exists()).toBe(false)
    expect(wrapper.text()).toContain('天')
    await wrapper.get('button[aria-label="增加换手"]').trigger('click')
    expect(wrapper.emitted('update:modelValue')!.at(-1)).toEqual([0.06])
    await wrapper.get('button[aria-label="增加换手"]').trigger('click'); await wrapper.get('button[aria-label="增加换手"]').trigger('click'); await wrapper.get('button[aria-label="增加换手"]').trigger('click'); await wrapper.get('button[aria-label="增加换手"]').trigger('click')
    expect(wrapper.props('modelValue')).toBe(0.1)
    expect(wrapper.get('button[aria-label="增加换手"]').attributes('disabled')).toBeDefined()
    await input.trigger('keydown', { key: 'ArrowDown' })
    expect(wrapper.emitted('update:modelValue')!.at(-1)).toEqual([0.09])
    await input.setValue('')
    expect(wrapper.emitted('update:modelValue')!.at(-1)).toEqual([null])
    await input.setValue('5'); await input.trigger('blur')
    expect(wrapper.emitted('update:modelValue')!.at(-1)).toEqual([0.1])
    expect((input.element as HTMLInputElement).value).toBe('0.1')
    await input.setValue('abc'); await input.trigger('blur')
    expect((input.element as HTMLInputElement).value).toBe('0.1')
    wrapper.unmount()
  })
})

describe('v-tip（2026-11-14 用户要求替换原生 title）', () => {
  it('悬停 400 ms 后出现、挂在 body 上并关联 aria-describedby，移开即消失；空文字不显示', async () => {
    vi.useFakeTimers()
    const text = shallowRef('调仓周期（交易日）')
    const host = mount(defineComponent({ setup: () => () => withDirectives(h('span', { class: 'target' }, ['调仓']), [[vTip, text.value]]) }), { attachTo: document.body })
    const target = host.get('.target')
    expect(target.attributes('title')).toBeUndefined()
    await target.trigger('mouseenter')
    vi.advanceTimersByTime(TIP_DELAY - 1)
    expect(document.getElementById('ui-tip')?.style.display ?? 'none').toBe('none')
    vi.advanceTimersByTime(1)
    const tip = document.getElementById('ui-tip')!
    expect(tip.parentElement).toBe(document.body)
    expect(tip.style.display).toBe('block')
    expect(tip.textContent).toBe('调仓周期（交易日）')
    expect(target.attributes('aria-describedby')).toBe('ui-tip')
    await target.trigger('mouseleave')
    expect(tip.style.display).toBe('none')
    text.value = ''; await nextTick()
    await target.trigger('mouseenter'); vi.advanceTimersByTime(TIP_DELAY)
    expect(tip.style.display).toBe('none')
    host.unmount()
  })
})
