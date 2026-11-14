import { computed, ref, shallowRef, watch } from 'vue'
import { api, friendly, RequestError } from '../api'
import type { ApiError, Completion, Entry } from '../types'
import * as client from './api'
import type { BacktestGroup, BacktestOptions, BacktestSettings, BacktestState, BacktestTab } from './types'

export const TABS_KEY = 'ashare.backtest.tabs'
export const ACTIVE_KEY = 'ashare.backtest.active'
function read(key: string) { try { return localStorage.getItem(key) } catch { return null } }
function write(key: string, value: string) { try { localStorage.setItem(key, value) } catch { /* 隐私模式等：只是不记住 */ } }

/** `#/backtest?entry=<因子库条目 id>`：从因子库「送去回测」进入。 */
export function routeEntry(hash: string) {
  const query = hash.split('?')[1]
  const entry = query ? new URLSearchParams(query).get('entry') : null
  return entry && /^[\w-]{1,80}$/.test(entry) ? entry : null
}

/** 因子库条目的设置里，回测页认得的只有股票池、成交时点和基准；其余用回测默认值。 */
export function settingsFromEntry(entry: Record<string, unknown>, defaults: BacktestSettings, choices: BacktestOptions['choices']): BacktestSettings {
  const out = { ...defaults }
  if (typeof entry.universe === 'string' && choices.universe?.includes(entry.universe)) out.universe = entry.universe
  if (typeof entry.trade_at === 'string' && choices.trade_at?.includes(entry.trade_at)) out.trade_at = entry.trade_at
  if (typeof entry.benchmark === 'string') out.benchmark = entry.benchmark
  return out
}

const RUNNING = ['queued', 'running']
const running = (state: BacktestState | null) => !!state && RUNNING.includes(state.status)

/**
 * localStorage 里的一个 tab → 页面状态。第一版只存一个 `backtest` 身份（一次只跑一个表达式），
 * 读的时候当成只有一组；组名为空，界面与单组时一样。
 */
export function restoreTab(saved: Record<string, unknown>, defaults: BacktestSettings): BacktestTab | null {
  if (!saved || typeof saved.id !== 'string') return null
  let groups: BacktestGroup[] = []
  if (Array.isArray(saved.groups)) {
    groups = saved.groups
      .filter((g): g is { id: string; label?: unknown } => !!g && typeof g.id === 'string')
      .map(g => ({ id: g.id, label: typeof g.label === 'string' ? g.label : '', state: null }))
  } else if (typeof saved.backtest === 'string') {
    groups = [{ id: saved.backtest, label: '', state: null }]
  }
  const selected = groups.some(g => g.id === saved.selected) ? saved.selected as string : groups[0]?.id ?? null
  return {
    id: saved.id,
    name: typeof saved.name === 'string' ? saved.name : '',
    source: typeof saved.source === 'string' ? saved.source : '',
    settings: { ...defaults, ...(saved.settings as Partial<BacktestSettings> | undefined) },
    groups, selected,
    ...(typeof saved.entry === 'string' ? { entry: saved.entry } : {}),
  }
}

export function selectedGroup(tab: BacktestTab | null) {
  return tab?.groups.find(g => g.id === tab.selected) ?? tab?.groups[0] ?? null
}

/** tab 标签上的进度：只有一组时与原来相同；多组时是整批的平均进度，已结束的组按 100% 计。 */
export function batchProgress(groups: BacktestGroup[]) {
  if (!groups.some(g => running(g.state))) return ''
  if (groups.length === 1) {
    const s = groups[0].state!
    return s.status === 'running' && s.progress !== null ? `${Math.round(s.progress * 100)}%` : s.status === 'queued' ? '排队' : ''
  }
  if (groups.every(g => g.state?.status === 'queued')) return '排队'
  const done = groups.reduce((sum, g) => sum + (!g.state || g.state.status === 'queued' ? 0 : g.state.status === 'running' ? g.state.progress ?? 0 : 1), 0)
  return `${Math.round(done / groups.length * 100)}%`
}

const uid = () => (globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`).replace(/-/g, '').slice(0, 16)

export function useBacktests() {
  const tabs = ref<BacktestTab[]>([])
  const activeId = shallowRef('')
  const options = shallowRef<BacktestOptions | null>(null)
  const completions = shallowRef<Completion[]>([])
  const message = shallowRef('')
  const errors = ref<Record<string, ApiError | null>>({})
  const current = computed(() => tabs.value.find(t => t.id === activeId.value) ?? null)
  let timer: ReturnType<typeof setInterval> | undefined

  function blank(settings?: Partial<BacktestSettings>): BacktestTab {
    return { id: uid(), name: '', source: '', settings: { ...options.value!.defaults, ...settings }, groups: [], selected: null }
  }
  function persist() {
    write(TABS_KEY, JSON.stringify(tabs.value.map(({ id, name, source, settings, groups, selected, entry }) => ({
      id, name, source, settings, groups: groups.map(({ id, label }) => ({ id, label })), selected, entry,
    }))))
    write(ACTIVE_KEY, activeId.value)
  }
  const active = (tab: BacktestTab) => tab.groups.some(g => running(g.state))
  /** 一次请求刷新一个 tab 全部组的状态；服务端已不认识的组从 tab 里去掉。 */
  async function refresh(tab: BacktestTab) {
    const asked = tab.groups
    if (!asked.length) return
    const { items } = await client.states(asked.map(g => g.id))
    if (tab.groups !== asked) return  // 等待期间重新运行过：这批回答已经过时
    const found = new Map(items.map(item => [item.id, item]))
    tab.groups = asked.filter(g => found.has(g.id)).map(g => ({ ...g, state: found.get(g.id)! }))
    if (!tab.groups.some(g => g.id === tab.selected)) tab.selected = tab.groups[0]?.id ?? null
  }
  function poll() {
    clearInterval(timer)
    if (!tabs.value.some(active)) return
    timer = setInterval(async () => {
      const busy = tabs.value.filter(active)
      if (!busy.length) return clearInterval(timer)
      await Promise.all(busy.map(tab => refresh(tab).catch(() => { /* 下一次再问 */ })))
    }, 600)
  }

  async function init() {
    try {
      const [loaded, words] = await Promise.all([client.options(), api<Completion[]>('/completions').catch(() => [])])
      options.value = loaded; completions.value = words
    } catch (e) { message.value = friendly(e).message; return }
    let saved: unknown = []
    try { saved = JSON.parse(read(TABS_KEY) ?? '[]') } catch { saved = [] }
    tabs.value = (Array.isArray(saved) ? saved : []).map(t => restoreTab(t, options.value!.defaults)).filter((t): t is BacktestTab => !!t)
    if (!tabs.value.length) tabs.value = [blank()]
    activeId.value = tabs.value.some(t => t.id === read(ACTIVE_KEY)) ? read(ACTIVE_KEY)! : tabs.value[0].id
    await Promise.all(tabs.value.map(tab => refresh(tab).catch(() => { tab.groups = []; tab.selected = null })))
    poll()
  }

  function addTab(seed?: Partial<BacktestTab>) {
    const tab = { ...blank(), ...seed }
    tabs.value.push(tab); activeId.value = tab.id
    return tab
  }
  function closeTab(tab: BacktestTab) {
    const index = tabs.value.findIndex(t => t.id === tab.id)
    tabs.value.splice(index, 1)
    if (!tabs.value.length) tabs.value.push(blank())
    if (activeId.value === tab.id) activeId.value = tabs.value[Math.max(0, index - 1)].id
  }
  function select(tab: BacktestTab, group: string) {
    if (tab.groups.some(g => g.id === group)) tab.selected = group
  }

  async function run(tab: BacktestTab) {
    if (!tab.source.trim() || active(tab)) return
    errors.value[tab.id] = null
    try {
      const submitted = await client.submit({ source: tab.source, name: tab.name, settings: tab.settings })
      // 重跑时留在同名的那一组上（改了设置再跑，多半还想看刚才那组）。
      const label = selectedGroup(tab)?.label
      tab.groups = submitted.variants.map(({ label, ...state }) => ({ id: state.id, label, state }))
      tab.selected = (tab.groups.find(g => g.label === label) ?? tab.groups[0]).id
      if (!tab.name) tab.name = submitted.name
      poll()
    } catch (e) {
      const detail = friendly(e)
      if (e instanceof RequestError && detail.line) errors.value[tab.id] = detail
      else message.value = detail.message
    }
  }
  async function cancel(tab: BacktestTab) {
    const ids = tab.groups.filter(g => running(g.state)).map(g => g.id)
    if (!ids.length) return
    try { await client.cancel(ids); await refresh(tab) } catch (e) { message.value = friendly(e).message }
  }

  /** 从因子库带入：同一条目已经开过就切过去，否则新开一个 tab。 */
  async function openEntry(entry: string) {
    const existing = tabs.value.find(t => t.entry === entry)
    if (existing) { activeId.value = existing.id; return existing }
    try {
      const detail = await api<Entry>(`/library/${entry}`)
      const settings = settingsFromEntry(detail.settings as Record<string, unknown>, options.value!.defaults, options.value!.choices)
      const empty = current.value && !current.value.source.trim() && !current.value.groups.length ? current.value : null
      if (empty) { Object.assign(empty, { name: detail.name, source: detail.source, settings, entry }); return empty }
      return addTab({ name: detail.name, source: detail.source, settings, entry })
    } catch (e) { message.value = friendly(e).message; return null }
  }

  watch([tabs, activeId], persist, { deep: true })
  const dispose = () => clearInterval(timer)
  return { tabs, activeId, current, options, completions, message, errors, init, addTab, closeTab, select, run, cancel, openEntry, dispose, active }
}
