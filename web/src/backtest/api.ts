import { api, post } from '../api'
import type { BacktestDocument, BacktestOptions, BacktestSettings, BacktestState, BacktestSubmitted, Page, Part, RebalanceRow, TradeRow } from './types'

/** 回测页的接口；结果不可变，按 id + 窗口缓存，切 tab 不重复下载。 */
const documents = new Map<string, Promise<BacktestDocument>>()

export const options = () => api<BacktestOptions>('/backtests/options')
export const submit = (body: { source: string; name: string; settings: BacktestSettings }) => post<BacktestSubmitted>('/backtests', body)
export const state = (id: string) => api<BacktestState>(`/backtests/${id}`)
/** 一次问一个 tab 全部组的状态；找不到的组不在结果里。 */
export const states = (ids: string[]) => api<{ items: BacktestState[] }>(`/backtests/states?${new URLSearchParams({ ids: ids.join(',') })}`)
/** 整批取消：服务端先撤下排队的组，再结束在跑的那组。 */
export const cancel = (ids: string[]) => post<{ cancelled: string[] }>('/backtests/cancel', { ids })

export function result(id: string, part: Part): Promise<BacktestDocument> {
  const key = `${id}:${part}`
  if (!documents.has(key)) {
    if (documents.size >= 32) documents.delete(documents.keys().next().value!)
    documents.set(key, api<BacktestDocument>(`/backtests/${id}/result?part=${part}`).catch(e => { documents.delete(key); throw e }))
  }
  return documents.get(key)!
}
export const rebalances = (id: string, part: Part, offset: number, limit = 50) =>
  api<Page<RebalanceRow>>(`/backtests/${id}/rebalances?${new URLSearchParams({ part, offset: String(offset), limit: String(limit) })}`)
export const trades = (id: string, part: Part, offset: number, limit = 50) =>
  api<Page<TradeRow>>(`/backtests/${id}/trades?${new URLSearchParams({ part, offset: String(offset), limit: String(limit) })}`)
