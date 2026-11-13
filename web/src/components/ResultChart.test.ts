import { mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'
import ResultChart from './ResultChart.vue'
import type { ChartData } from '../types'

const data: ChartData = { dates:['20240101','20240102','20250101'], lines:[
  {id:'a',name:'多空',color:'#123456',values:[.1,-.05,.03],kind:'long_short'},
  {id:'b',name:'第 1 组',color:'#123456',values:[.2,-.1,.02],kind:'long'},
  {id:'benchmark',name:'合成基准',color:'#888888',values:[.04,-.02,.01],benchmark:true},
] }
afterEach(() => vi.restoreAllMocks())

describe('收益图实际绘制', () => {
  it('抽屉隐藏回撤刻度，收益标签都是不重复的整数百分比', async () => {
    const wrapper = mount(ResultChart,{props:{data,compact:true}})
    expect(wrapper.findAll('.draw-tick')).toHaveLength(0)
    const labels = wrapper.findAll('.main-tick text').map(t => t.text())
    expect(new Set(labels).size).toBe(labels.length)
    expect(labels.length).toBeLessThanOrEqual(4)
    expect(labels.every(t => /^-?\d+%$/.test(t))).toBe(true)
    wrapper.unmount()
  })
  it('相对模式隐藏基准线，悬停显示超额收益与超额回撤', async () => {
    vi.spyOn(window,'requestAnimationFrame').mockImplementation(callback => { callback(0); return 1 })
    const wrapper = mount(ResultChart,{props:{data,relative:true,selectedId:'a'}})
    expect(wrapper.findAll('.return-line')).toHaveLength(2)
    expect(wrapper.find('[data-kind="benchmark"]').exists()).toBe(false)
    expect(wrapper.get('[data-kind="long"]').attributes('stroke-dasharray')).toBeTruthy()
    const svg = wrapper.get('svg')
    vi.spyOn(svg.element,'getBoundingClientRect').mockReturnValue({x:0,y:0,left:0,top:0,right:1000,bottom:450,width:1000,height:450,toJSON:() => ({})})
    await svg.trigger('pointermove',{clientX:48,clientY:40})
    expect(wrapper.get('.hover-tip').text()).toContain('2024-01-01 · 超额')
    expect(wrapper.get('.hover-tip').text()).toContain('6.00%')
    expect(wrapper.get('.hover-tip').text()).toContain('16.00%')
    expect(wrapper.get('.hover-tip').text()).toContain('超额回撤 · 多空')
    await wrapper.setProps({relative:false})
    expect(wrapper.find('[data-kind="benchmark"]').exists()).toBe(true)
    wrapper.unmount()
  })
  it('基准尚未加载时不把绝对收益冒充超额', () => {
    const wrapper = mount(ResultChart,{props:{data:{...data,lines:data.lines.slice(0,2)},relative:true}})
    expect(wrapper.find('svg').exists()).toBe(false)
    expect(wrapper.text()).toContain('基准数据暂不可用')
    wrapper.unmount()
  })
  it('八个变体的多空和多头标签都保持在图内，互不重叠', () => {
    const many = {dates:['20240102'],lines:Array.from({length:16},(_,i) => ({id:String(i),name:String(i),color:'black',values:[i ? -.4 - i * .001 : .4]}))}
    const wrapper = mount(ResultChart,{props:{data:many}})
    const positions = wrapper.findAll('.line-label').map(t => Number(t.attributes('y'))).sort((a,b) => a-b)
    expect(positions).toHaveLength(16)
    expect(positions[0]).toBeGreaterThanOrEqual(18)
    const drawTicks = wrapper.findAll('.draw-tick text').map(t => Number(t.attributes('y')))
    expect(positions.at(-1)).toBeLessThan(Math.min(...drawTicks))
    for (let i=1;i<positions.length;i++) expect(positions[i]-positions[i-1]).toBeGreaterThanOrEqual(12.99)
    wrapper.unmount()
  })
})
