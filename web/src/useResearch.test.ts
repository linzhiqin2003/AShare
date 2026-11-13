import { mount } from '@vue/test-utils'
import { defineComponent } from 'vue'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { useResearch, type ResearchTab } from './useResearch'
import type { Packed, Run } from './types'

const pack = (values: number[]): Packed => ({shape:[values.length],data:btoa(String.fromCharCode(...new Uint8Array(new Float32Array(values).buffer)))})
afterEach(() => vi.unstubAllGlobals())

describe('相对基准序列按需加载', () => {
  it('第 1 组只下载一条序列，切回普通模式复用缓存且不提交回测', async () => {
    vi.stubGlobal('fetch',vi.fn(async (url: string) => ({ok:true,json:async () => [
      {part:'is',dates:['20240102','20240103'],series:url.endsWith('view=long') ? {'pnl.long':pack([.2,.3])} : {'pnl.long_short':pack([.1,.15]),'pnl.weights':pack([.3,.4])}},
    ]})))
    let research!: ReturnType<typeof useResearch>
    const wrapper = mount(defineComponent({setup() { research = useResearch(); return () => null }}))
    const run = {id:'relative-fetch-test',ordinal:0,label:'20 · industry',status:'complete',settings:{},summary:null,elapsed:1} as Run
    const tab: ResearchTab = {
      id:'tab',name:'test',source:'rank(close)',settings:{benchmark:'000001.SH'},count:1,error:null,
      running:false,progress:100,job:{id:'job',status:'complete',total:1,elapsed:1,runs:[run]},selected:run.id,visible:[run.id],
      chart:{dates:[],lines:[]},view:'returns',compound:false,relative:false,returnSelection:'both',autoName:false,drawdown:null,
      benchmark:{symbol:'000001.SH',name:'上证指数',requested:'000001.SH',fallback:false,relatives:{},parts:[{part:'is',dates:['20240102','20240103'],series:{benchmark:pack([.01,.02])}}]},
    }
    await research.loadChart(tab)
    expect(tab.chart.lines.map(l => l.kind)).toEqual(['long_short','weights',undefined])
    expect(fetch).toHaveBeenCalledTimes(1)
    tab.relative = true
    await research.loadChart(tab)
    expect(tab.chart.lines.map(l => l.kind)).toEqual(['long_short','long',undefined])
    expect(tab.drawdown?.kind).toBe('long_short')
    expect(fetch).toHaveBeenCalledTimes(2)
    expect(vi.mocked(fetch).mock.calls[1][0]).toBe(`/api/runs/${run.id}/series?view=long`)
    tab.returnSelection = 'long'
    await research.loadChart(tab)
    expect(tab.chart.lines.map(l => l.kind)).toEqual(['long',undefined])
    expect(tab.drawdown?.kind).toBe('long')
    tab.relative = false
    await research.loadChart(tab)
    expect(tab.chart.lines.map(l => l.kind)).toEqual(['long_short','weights',undefined])
    expect(fetch).toHaveBeenCalledTimes(2)
    expect(tab.settings).toEqual({benchmark:'000001.SH'})
    expect(tab.job?.runs[0]).toBe(run)
    wrapper.unmount()
  })
})
