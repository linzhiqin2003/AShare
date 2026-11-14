import { computed, ref, shallowRef, watch } from 'vue'
import { api, friendly, RequestError } from '../api'
import type { ApiError, Completion, Entry } from '../types'
import * as client from './api'
import type { BacktestOptions, BacktestSettings, BacktestTab } from './types'

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
    return { id: uid(), name: '', source: '', settings: { ...options.value!.defaults, ...settings }, backtest: null, state: null }
  }
  function persist() {
    write(TABS_KEY, JSON.stringify(tabs.value.map(({ id, name, source, settings, backtest, entry }) => ({ id, name, source, settings, backtest, entry }))))
    write(ACTIVE_KEY, activeId.value)
  }
  const active = (tab: BacktestTab) => tab.state?.status === 'queued' || tab.state?.status === 'running'
  function poll() {
    clearInterval(timer)
    if (!tabs.value.some(active)) return
    timer = setInterval(async () => {
      const running = tabs.value.filter(active)
      if (!running.length) return clearInterval(timer)
      await Promise.all(running.map(async tab => {
        try { tab.state = await client.state(tab.backtest!) } catch { /* 下一次再问 */ }
      }))
    }, 600)
  }

  async function init() {
    try {
      const [loaded, words] = await Promise.all([client.options(), api<Completion[]>('/completions').catch(() => [])])
      options.value = loaded; completions.value = words
    } catch (e) { message.value = friendly(e).message; return }
    let saved: BacktestTab[] = []
    try { saved = JSON.parse(read(TABS_KEY) ?? '[]') } catch { saved = [] }
    tabs.value = (Array.isArray(saved) ? saved : []).filter(t => t && typeof t.id === 'string').map(t => ({
      ...t, settings: { ...options.value!.defaults, ...t.settings }, state: null,
    }))
    if (!tabs.value.length) tabs.value = [blank()]
    activeId.value = tabs.value.some(t => t.id === read(ACTIVE_KEY)) ? read(ACTIVE_KEY)! : tabs.value[0].id
    await Promise.all(tabs.value.filter(t => t.backtest).map(async tab => {
      try { tab.state = await client.state(tab.backtest!) } catch { tab.backtest = null }
    }))
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

  async function run(tab: BacktestTab) {
    if (!tab.source.trim() || active(tab)) return
    errors.value[tab.id] = null
    try {
      tab.state = await client.submit({ source: tab.source, name: tab.name, settings: tab.settings })
      tab.backtest = tab.state.id
      if (!tab.name) tab.name = tab.state.name
      poll()
    } catch (e) {
      const detail = friendly(e)
      if (e instanceof RequestError && detail.line) errors.value[tab.id] = detail
      else message.value = detail.message
    }
  }
  async function cancel(tab: BacktestTab) {
    if (!tab.backtest) return
    try { await client.cancel(tab.backtest); tab.state = await client.state(tab.backtest) } catch (e) { message.value = friendly(e).message }
  }

  /** 从因子库带入：同一条目已经开过就切过去，否则新开一个 tab。 */
  async function openEntry(entry: string) {
    const existing = tabs.value.find(t => t.entry === entry)
    if (existing) { activeId.value = existing.id; return existing }
    try {
      const detail = await api<Entry>(`/library/${entry}`)
      const settings = settingsFromEntry(detail.settings as Record<string, unknown>, options.value!.defaults, options.value!.choices)
      const empty = current.value && !current.value.source.trim() && !current.value.backtest ? current.value : null
      if (empty) { Object.assign(empty, { name: detail.name, source: detail.source, settings, entry }); return empty }
      return addTab({ name: detail.name, source: detail.source, settings, entry })
    } catch (e) { message.value = friendly(e).message; return null }
  }

  watch([tabs, activeId], persist, { deep: true })
  const dispose = () => clearInterval(timer)
  return { tabs, activeId, current, options, completions, message, errors, init, addTab, closeTab, run, cancel, openEntry, dispose, active }
}
