import { mount, flushPromises } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'
import HoldingsTable from './HoldingsTable.vue'
const api = vi.hoisted(() => vi.fn())
vi.mock('../api',() => ({api,friendly:() => ({message:'读取失败'})}))
describe('持仓日期与两侧查询', () => {
  it('默认最后一天，日期联动、比较和搜索只使用该日数据', async () => {
    api.mockImplementation(async (url:string) => url.includes('?') ? {date:'20250103',signal_date:'20250102',snapshot:'20250103',previous:'20250102',items:[{symbol:'S1',name:'合成一',industry:'合成行业',factor:1,weight:.5,side:'long',status:'new'},{symbol:'S2',name:'合成二',industry:'合成行业',factor:-1,weight:-.5,side:'short',status:'exit'}]} : {dates:['20250102','20250103'],date:'20250103'})
    const wrapper = mount(HoldingsTable,{props:{run:'a'}})
    await flushPromises()
    expect((wrapper.get('[aria-label="持仓日期"]').element as HTMLSelectElement).value).toBe('20250103')
    expect(wrapper.get('tbody').text()).toContain('合成一')
    await wrapper.setProps({date:'20250102'}); await flushPromises()
    expect(api).toHaveBeenLastCalledWith('/runs/a/holdings?date=20250102&compare=false')
    await wrapper.get('[role=switch]').trigger('click'); await flushPromises()
    expect(api).toHaveBeenLastCalledWith('/runs/a/holdings?date=20250102&compare=true')
    await wrapper.get('[aria-label="持仓侧"]').setValue('short')
    expect(wrapper.get('tbody').text()).toContain('调出')
    await wrapper.get('[aria-label="搜索持仓"]').setValue('不存在')
    expect(wrapper.findAll('tbody tr')).toHaveLength(0)
    wrapper.unmount()
  })
})
