import { computed, onUnmounted, reactive, shallowRef, watch } from 'vue'
import { api, friendly, post, series } from './api'
import { chartData, relativeLines, withBenchmark } from './chart'
import { showValidation, visibleParts } from './validation'
import type { ApiError, BenchmarkData, ChartData, Completion, Job, Line, ReturnSelection, Settings, View } from './types'

export interface ResearchTab {
  id: string; name: string; source: string; settings: Settings; count: number; error: ApiError | null;
  running: boolean; progress: number; job: Job | null; selected: string; visible: string[];
  chart: ChartData; view: View; compound: boolean; relative: boolean; returnSelection: ReturnSelection; autoName: boolean; drawdown: Line | null; benchmark: BenchmarkData | null
}
const emptyChart = (): ChartData => ({ dates: [], lines: [] })
export function useResearch() {
  const tabs = reactive<ResearchTab[]>([])
  const activeId = shallowRef('')
  const current = computed(() => tabs.find(t => t.id === activeId.value))
  const completions = shallowRef<Completion[]>([])
  const defaults = shallowRef<Settings>({})
  const metadata = shallowRef({ data_date: null as string | null, running: 0, max_workers: 4 })
  const message = shallowRef('')
  const elapsed = shallowRef<number | null>(null)
  const events = new Map<string, EventSource>()
  const validations = new Map<string, AbortController>()
  const chartVersions = new Map<string, number>()
  let ready = false
  const benchmarkVersions = new Map<string, number>()
  let metadataTimer: ReturnType<typeof setInterval> | undefined

  async function refreshMeta() { try { metadata.value = await api('/meta') } catch { /* 保留已有状态，操作时报告连接错误。 */ } }
  async function addTab(draft?: Partial<ResearchTab>) {
    try {
      const { id } = await post<{ id: string }>('/tabs', { id: draft?.id ?? crypto.randomUUID().replaceAll('-', '') })
      const tab: ResearchTab = { id, name: draft?.name ?? 'low_vol', source: draft?.source ?? '# 已实现波动率取负：低波动做多\nvol = ts_std(returns, [20, 60])\nlow_vol = -vol', settings: draft?.settings ?? { ...defaults.value, group_neutralize: ['industry', 'cap+industry'] }, count: 4, error: null, running: false, progress: 0, job: null, selected: '', visible: [], chart: emptyChart(), view: 'returns', compound: false, relative: false, returnSelection: 'both', autoName: draft?.autoName ?? (!draft?.name || /^alpha_\d+$/.test(draft.name)), benchmark: null, drawdown: null }
      tabs.push(tab); activeId.value = id
      await validate(tabs[tabs.length - 1])
      return tabs[tabs.length - 1]
    } catch (error) { message.value = friendly(error).message }
  }
  async function renameTab(tab: ResearchTab, name: string) {
    const previous = tab.name
    tab.name = name; tab.autoName = false
    try { await post(`/tabs/${tab.id}/name`,{name}) }
    catch (error) { tab.name = previous; message.value = friendly(error).message }
  }
  async function closeTab(tab: ResearchTab) {
    try {
      await api(`/tabs/${tab.id}`, { method: 'DELETE' })
      events.get(tab.id)?.close(); events.delete(tab.id)
      validations.get(tab.id)?.abort()
      tabs.splice(tabs.indexOf(tab), 1)
      localStorage.removeItem(`ashare.job.${tab.id}`)
      if (activeId.value === tab.id) activeId.value = tabs[0]?.id ?? ''
    } catch (error) { message.value = friendly(error).message }
  }
  async function validate(tab: ResearchTab) {
    validations.get(tab.id)?.abort()
    const controller = new AbortController(); validations.set(tab.id, controller)
    try {
      const result = await api<{ count: number; name: string }>('/validate', { method: 'POST', signal: controller.signal, body: JSON.stringify({ source: tab.source, settings: tab.settings }) })
      if (!controller.signal.aborted) { tab.count = result.count; tab.error = null; if (tab.autoName) tab.name = result.name }
    } catch (error) {
      if (!controller.signal.aborted) { tab.error = friendly(error); tab.count = 0 }
    }
  }
  async function loadChart(tab: ResearchTab) {
    const version = (chartVersions.get(tab.id) ?? 0) + 1; chartVersions.set(tab.id, version)
    const view = tab.view, relative = tab.relative && view === 'returns', selection = tab.returnSelection
    const runs = tab.job?.runs.filter(r => r.status === 'complete' && (view === 'returns' ? tab.visible.includes(r.id) : r.id === tab.selected)) ?? []
    // 验证期开关关闭时只取样本内部分，图表只画到样本内末。
    const show = showValidation.value
    async function partsFor(id: string) {
      if (!relative) return visibleParts(await series(id, view), show)
      const parts = await Promise.all([
        ...(selection !== 'long' ? [series(id, 'returns')] : []),
        ...(selection !== 'long_short' ? [series(id, 'long')] : []),
      ])
      return visibleParts(parts.flat(), show)
    }
    try {
      const parts = await Promise.all(runs.map(r => partsFor(r.id)))
      const selected = tab.job?.runs.find(r => r.id === tab.selected && r.status === 'complete')
      const selectedParts = selected && view === 'returns' ? await partsFor(selected.id) : null
      if (chartVersions.get(tab.id) !== version) return
      const drawKind = relative && selection === 'long' ? 'long' : 'long_short'
      tab.drawdown = selected && selectedParts ? chartData([selected],[selectedParts],'returns').lines.find(l => l.kind === drawKind) ?? null : null
      const data = withBenchmark(chartData(runs, parts, view), view === 'returns' ? tab.benchmark : null)
      tab.chart = relative ? relativeLines(data, selection) : data
    } catch (error) { tab.error = friendly(error) }
  }
  async function loadJob(tab: ResearchTab, id: string) {
    const job = await api<Job>(`/jobs/${id}`)
    const previous = new Set(tab.job?.id === id ? tab.job.runs.filter(r => r.status === 'complete').map(r => r.id) : [])
    const completed = job.runs.filter(r => r.status === 'complete')
    if (job.id !== tab.job?.id) tab.visible = []
    tab.job = job
    for (const run of completed) if (!previous.has(run.id)) tab.visible.push(run.id)
    if (!completed.some(r => r.id === tab.selected)) tab.selected = completed[0]?.id ?? ''
    await loadBenchmark(tab)
    if (job.status !== 'running') {
      tab.running = false; elapsed.value = job.elapsed
      if (job.status === 'failed') tab.error = { message: '运行失败，请稍后重试' }
      if (job.status === 'interrupted') tab.error = { message: '服务已重启，请重新运行' }
      events.get(tab.id)?.close(); events.delete(tab.id)
    }
  }
  function subscribe(tab: ResearchTab, id: string) {
    events.get(tab.id)?.close()
    const total = tab.job?.id === id ? tab.job.runs.length : tab.count
    const source = new EventSource(`/api/jobs/${id}/events`); events.set(tab.id, source)
    const progress = new Map<number, number>()
    let updates = Promise.resolve()
    source.onmessage = event => {
      const data = JSON.parse(event.data)
      if (data.kind === 'progress') progress.set(data.ordinal, ({ evaluate: .03, prepared: .15, is: .2, validation: .7 } as Record<string, number>)[data.phase] ?? 0)
      if (data.kind === 'variant') progress.set(data.ordinal, 1)
      tab.progress = Math.round([...progress.values()].reduce((a, b) => a + b, 0) / Math.max(1, total) * 100)
      if (['variant', 'complete', 'cancelled', 'failed', 'interrupted'].includes(data.kind)) {
        updates = updates.then(() => loadJob(tab, id)).catch(error => { tab.error = friendly(error) })
        void refreshMeta()
      }
    }
    source.onerror = () => { if (source.readyState === EventSource.CLOSED) void loadJob(tab, id).catch(error => { tab.error = friendly(error) }) }
  }
  async function run(tab: ResearchTab) {
    if (tab.running) return
    await validate(tab)
    if (tab.error) return
    try {
      const { id } = await post<{ id: string }>(`/tabs/${tab.id}/runs`, { source: tab.source, settings: tab.settings, name: tab.name })
      tab.running = true; tab.progress = 0; tab.error = null
      // 已有曲线保留到第一组新结果到达。
      localStorage.setItem(`ashare.job.${tab.id}`, id)
      subscribe(tab, id); void refreshMeta()
    } catch (error) { tab.error = friendly(error) }
  }
  async function cancel(tab: ResearchTab) {
    const id = localStorage.getItem(`ashare.job.${tab.id}`)
    if (id) { try { await post(`/jobs/${id}/cancel`) } catch (error) { tab.error = friendly(error) } }
  }
  async function loadBenchmark(tab: ResearchTab) {
    if (!tab.job) return loadChart(tab)
    const version = (benchmarkVersions.get(tab.id) ?? 0) + 1; benchmarkVersions.set(tab.id, version)
    const job = tab.job.id
    if (tab.benchmark && tab.benchmark.requested !== String(tab.settings.benchmark ?? '')) {
      tab.benchmark = null
      tab.chart = { ...tab.chart, lines: tab.chart.lines.filter(line => !line.benchmark) }
    }
    try {
      const result = await api<BenchmarkData>(`/jobs/${job}/benchmark?${new URLSearchParams({ symbol: String(tab.settings.benchmark ?? '') })}`)
      if (benchmarkVersions.get(tab.id) !== version || tab.job.id !== job) return
      tab.benchmark = result
      await loadChart(tab)
    } catch (error) {
      if (benchmarkVersions.get(tab.id) !== version || tab.job.id !== job) return
      tab.benchmark = null; tab.error = friendly(error); await loadChart(tab)
    }
  }
  async function init() {
    try {
      const [settings, suggestions, existing] = await Promise.all([api<{ defaults: Settings }>('/settings'), api<Completion[]>('/completions'), api<{ id: string; job: string | null }[]>('/tabs')])
      defaults.value = settings.defaults; completions.value = suggestions
      let drafts: Partial<ResearchTab>[] = []
      try { const value = JSON.parse(localStorage.getItem('ashare.drafts') ?? '[]'); if (Array.isArray(value)) drafts = value } catch { /* 损坏的草稿不阻塞新建。 */ }
      for (const draft of drafts) {
        const tab = await addTab(draft)
        const job = existing.find(t => t.id === draft.id)?.job ?? (draft.id ? localStorage.getItem(`ashare.job.${draft.id}`) : null)
        if (tab && job) {
          try { await loadJob(tab, job); if (tab.job?.status === 'running') { tab.running = true; subscribe(tab, job) } } catch { /* 结果库更换后草稿仍保留，旧任务无需阻塞界面。 */ }
        }
      }
      if (!tabs.length) await addTab()
      const active = localStorage.getItem('ashare.active')
      if (tabs.some(t => t.id === active)) activeId.value = active!
      ready = true; persist()
      await refreshMeta(); metadataTimer = setInterval(refreshMeta, 5000)
    } catch (error) { message.value = friendly(error).message }
  }
  function persist() {
    if (!ready) return
    localStorage.setItem('ashare.drafts', JSON.stringify(tabs.map(t => ({ id:t.id, name:t.name, source:t.source, settings:t.settings, autoName:t.autoName }))))
    localStorage.setItem('ashare.active',activeId.value)
  }
  watch(() => tabs.map(t => ({ id: t.id, name: t.name, source: t.source, settings: t.settings, autoName: t.autoName })), persist, { deep: true })
  watch(activeId,persist)
  watch(showValidation, () => { for (const tab of tabs) void loadChart(tab) })
  onUnmounted(() => { clearInterval(metadataTimer); events.forEach(e => e.close()); validations.forEach(v => v.abort()) })
  return { tabs, activeId, current, completions, defaults, metadata, message, elapsed, init, addTab, closeTab, renameTab, validate, loadChart, loadBenchmark, run, cancel }
}
