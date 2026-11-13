import { mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'
import SplitPane from './SplitPane.vue'

afterEach(() => { localStorage.clear(); vi.restoreAllMocks() })
describe('分栏比例与边界', () => {
  it('拖动按像素约束、保存比例、双击复位并适应缩小后的窗口', async () => {
    let resize!: ResizeObserverCallback
    vi.stubGlobal('ResizeObserver', class { constructor(callback: ResizeObserverCallback) { resize = callback } observe() {} disconnect() {} })
    const wrapper = mount(SplitPane, { props: {storageKey:'test',label:'调整分栏',initial:.4,minFirst:300,maxFirst:600,minSecond:400} })
    const size = async (width: number) => { resize([{contentRect:{width,height:500}}] as ResizeObserverEntry[],{} as ResizeObserver); await wrapper.vm.$nextTick() }
    await size(1008)
    expect(wrapper.attributes('style')).toContain('400px')
    const separator = wrapper.get('[role=separator]')
    Object.assign(separator.element,{setPointerCapture:vi.fn()})
    vi.stubGlobal('requestAnimationFrame',vi.fn(() => 1))
    vi.stubGlobal('cancelAnimationFrame',vi.fn())
    await separator.trigger('pointerdown',{clientX:400,pointerId:1})
    await separator.trigger('pointermove',{clientX:900,pointerId:1})
    await separator.trigger('pointerup')
    expect(wrapper.attributes('style')).toContain('600px')
    expect(localStorage.getItem('ashare.split.test')).toBe('0.6')
    await separator.trigger('dblclick')
    expect(wrapper.attributes('style')).toContain('400px')
    expect(localStorage.getItem('ashare.split.test')).toBe('0.4')
    await size(508)
    const first = parseFloat(wrapper.element.style.gridTemplateColumns)
    expect(first).toBeCloseTo(300 / 700 * 500)
    expect(first).toBeLessThan(500)
    wrapper.unmount()
  })
})
