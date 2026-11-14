import { shallowRef } from 'vue'
import { friendly, RequestError } from '../api'
import type { Scalar, Settings } from '../types'
import * as client from './api'
import type { BacktestGroup, BacktestOptions, BacktestRun, BacktestSettings, BacktestState, BacktestTab } from './types'

/** 合并前回测页自己的 localStorage；研究页启动时读一次、迁移成研究 tab 后删掉。 */
export const TABS_KEY = 'ashare.backtest.tabs'
export const ACTIVE_KEY = 'ashare.backtest.active'

/** 与后端 `backtest/settings.py` 的 DEFAULTS 相同；只在回测选项还没读到时兜底。 */
export const FALLBACK_SETTINGS: BacktestSettings = { universe: 'csi300', rebalance: 'month', top_n: 30, trade_at: 'open', capital: 1000, commission: 0.00025, commission_min: 5, slippage_bps: 5, dividend_tax: 0.2, carry_blocked: true, lots: 'board' }

/**
 * 两个视图共用、只在「样本与成交」里出现一次的设置：成交价（次日开盘或收盘，两边同一含义）和基准
 * （同一份主流指数目录）。存在研究设置里；回测设置里的同名键不再保存，提交时用共用值覆盖。
 * 涨跌停与停牌两边含义不同（研究：是否计入涨跌停与停牌；回测：总是计入，选的是挡住的委托怎么处理），不合并。
 */
export const SHARED_KEYS = ['trade_at', 'benchmark'] as const

/** `#/backtest?entry=<因子库条目 id>`：从因子库「送去回测」进入。 */
export function routeEntry(hash: string) {
  const query = hash.split('?')[1]
  const entry = query ? new URLSearchParams(query).get('entry') : null
  return entry && /^[\w-]{1,80}$/.test(entry) ? entry : null
}

/** 因子库条目的设置里，回测认得的只有股票池、成交时点和基准；其余用回测默认值。 */
export function settingsFromEntry(entry: Record<string, unknown>, defaults: BacktestSettings, choices: BacktestOptions['choices']): BacktestSettings {
  const out = { ...defaults }
  if (typeof entry.universe === 'string' && choices.universe?.includes(entry.universe)) out.universe = entry.universe
  if (typeof entry.trade_at === 'string' && choices.trade_at?.includes(entry.trade_at)) out.trade_at = entry.trade_at
  if (typeof entry.benchmark === 'string') out.benchmark = entry.benchmark
  return out
}

const first = (value: Scalar | Scalar[] | undefined) => (Array.isArray(value) ? value[0] : value)
/** 提交回测用的设置：回测自己的组合与成本规则 + 研究设置里的共用成交价和基准（扫参时取第一项）。 */
export function sharedSettings(research: Settings, own: BacktestSettings): BacktestSettings {
  const trade = first(research.trade_at), bench = first(research.benchmark)
  return { ...own, ...(typeof trade === 'string' ? { trade_at: trade } : {}), ...(typeof bench === 'string' && bench ? { benchmark: bench } : {}) }
}

const RUNNING = ['queued', 'running']
const running = (state: BacktestState | null) => !!state && RUNNING.includes(state.status)

function restoreGroups(saved: Record<string, unknown>) {
  let groups: BacktestGroup[] = []
  if (Array.isArray(saved.groups)) {
    groups = saved.groups
      .filter((g): g is { id: string; label?: unknown } => !!g && typeof g.id === 'string')
      .map(g => ({ id: g.id, label: typeof g.label === 'string' ? g.label : '', state: null }))
  } else if (typeof saved.backtest === 'string') {
    groups = [{ id: saved.backtest, label: '', state: null }]
  }
  const selected = groups.some(g => g.id === saved.selected) ? saved.selected as string : groups[0]?.id ?? null
  return { groups, selected }
}

/**
 * 合并前回测页 localStorage 里的一个 tab → 页面状态。第一版只存一个 `backtest` 身份（一次只跑一个表达式），
 * 读的时候当成只有一组；组名为空，界面与单组时一样。
 */
export function restoreTab(saved: Record<string, unknown>, defaults: BacktestSettings): BacktestTab | null {
  if (!saved || typeof saved.id !== 'string') return null
  return {
    id: saved.id,
    name: typeof saved.name === 'string' ? saved.name : '',
    source: typeof saved.source === 'string' ? saved.source : '',
    settings: { ...defaults, ...(saved.settings as Partial<BacktestSettings> | undefined) },
    ...restoreGroups(saved),
    ...(typeof saved.entry === 'string' ? { entry: saved.entry } : {}),
  }
}

/** 研究草稿里的 `backtest` 部分 → 回测状态；缺了或格式不对就是一个没运行过的回测。 */
export function restoreRun(saved: unknown, defaults: BacktestSettings): BacktestRun {
  const value = saved && typeof saved === 'object' ? saved as Record<string, unknown> : {}
  const settings = value.settings && typeof value.settings === 'object' ? value.settings as Partial<BacktestSettings> : {}
  return {
    settings: { ...defaults, ...settings },
    ...restoreGroups(value),
    ...(typeof value.entry === 'string' ? { entry: value.entry } : {}),
    count: typeof value.count === 'number' ? value.count : 1,
    error: null,
  }
}

/** 存进研究草稿的回测部分：只存组的身份和组名（状态每次向服务端问），不存共用设置。 */
export function storedRun(run: BacktestRun) {
  const settings: Partial<BacktestSettings> = { ...run.settings }
  for (const key of SHARED_KEYS) delete settings[key]
  return { settings, groups: run.groups.map(({ id, label }) => ({ id, label })), selected: run.selected, ...(run.entry ? { entry: run.entry } : {}) }
}

/** 一个旧回测 tab 迁移成研究 tab 要用的字段；没写过表达式也没运行过的空白 tab 不迁。 */
export interface MigratedTab {
  old: string
  name?: string
  autoName: boolean
  source: string
  settings: Settings
  mode: 'backtest'
  backtest: BacktestRun
}
/** 读合并前回测页的 tab 列表：要迁的 tab 和全部原始项（没迁成功的由调用方写回）。 */
export function migrateBacktestTabs(raw: string | null, defaults: BacktestSettings, researchDefaults: Settings): { tabs: MigratedTab[]; items: unknown[] } {
  let items: unknown[] = []
  try { const value = JSON.parse(raw ?? '[]'); if (Array.isArray(value)) items = value } catch { /* 损坏的旧数据不阻塞研究页 */ }
  const tabs: MigratedTab[] = []
  for (const item of items) {
    const tab = restoreTab(item as Record<string, unknown>, defaults)
    if (!tab || (!tab.source.trim() && !tab.groups.length)) continue
    tabs.push({
      old: tab.id, ...(tab.name ? { name: tab.name } : {}), autoName: !tab.name, source: tab.source, mode: 'backtest',
      settings: { ...researchDefaults, trade_at: tab.settings.trade_at, ...(tab.settings.benchmark ? { benchmark: tab.settings.benchmark } : {}) },
      backtest: { settings: tab.settings, groups: tab.groups, selected: tab.selected, ...(tab.entry ? { entry: tab.entry } : {}), count: Math.max(1, tab.groups.length), error: null },
    })
  }
  return { tabs, items }
}

export function selectedGroup(tab: { groups: BacktestGroup[]; selected: string | null } | null) {
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

/**
 * 研究页里策略回测的运行状态：选项、提交、整批取消、状态轮询。tab 列表归研究页，
 * `all()` 返回全部 tab 的回测部分，轮询只问还在跑的。
 */
export function useBacktestRuns(all: () => BacktestRun[], report: (message: string) => void) {
  const options = shallowRef<BacktestOptions | null>(null)
  let timer: ReturnType<typeof setInterval> | undefined

  const active = (run: BacktestRun) => run.groups.some(g => running(g.state))
  const defaults = () => options.value?.defaults ?? FALLBACK_SETTINGS
  async function load() {
    try { options.value = await client.options() } catch (e) { report(friendly(e).message) }
    return options.value
  }
  /** 一次请求刷新一个 tab 全部组的状态；服务端已不认识的组从 tab 里去掉。 */
  async function refresh(run: BacktestRun) {
    const asked = run.groups
    if (!asked.length) return
    const { items } = await client.states(asked.map(g => g.id))
    if (run.groups !== asked) return  // 等待期间重新运行过：这批回答已经过时
    const found = new Map(items.map(item => [item.id, item]))
    run.groups = asked.filter(g => found.has(g.id)).map(g => ({ ...g, state: found.get(g.id)! }))
    if (!run.groups.some(g => g.id === run.selected)) run.selected = run.groups[0]?.id ?? null
  }
  async function restore(runs: BacktestRun[]) {
    await Promise.all(runs.map(run => refresh(run).catch(() => { run.groups = []; run.selected = null })))
    poll()
  }
  function poll() {
    clearInterval(timer)
    if (!all().some(active)) return
    timer = setInterval(async () => {
      const busy = all().filter(active)
      if (!busy.length) return clearInterval(timer)
      await Promise.all(busy.map(run => refresh(run).catch(() => { /* 下一次再问 */ })))
    }, 600)
  }
  function select(run: BacktestRun, group: string) {
    if (run.groups.some(g => g.id === group)) run.selected = group
  }
  /** 提交一批；返回服务端起的名字（tab 没名字时用）。表达式错误（带行号）记在 run.error，其余报给页面。 */
  async function submit(run: BacktestRun, body: { source: string; name: string; settings: BacktestSettings }) {
    if (!body.source.trim() || active(run)) return null
    run.error = null
    try {
      const submitted = await client.submit(body)
      // 重跑时留在同名的那一组上（改了设置再跑，多半还想看刚才那组）。
      const label = selectedGroup(run)?.label
      run.groups = submitted.variants.map(({ label, ...state }) => ({ id: state.id, label, state }))
      run.selected = (run.groups.find(g => g.label === label) ?? run.groups[0]).id
      poll()
      return submitted.name
    } catch (e) {
      const detail = friendly(e)
      if (e instanceof RequestError && detail.line) run.error = detail
      else report(detail.message)
      return null
    }
  }
  async function cancel(run: BacktestRun) {
    const ids = run.groups.filter(g => running(g.state)).map(g => g.id)
    if (!ids.length) return
    try { await client.cancel(ids); await refresh(run) } catch (e) { report(friendly(e).message) }
  }
  const dispose = () => clearInterval(timer)
  return { options, defaults, active, load, refresh, restore, poll, select, submit, cancel, dispose }
}
