import type { ApiError, PlotPart, View } from './types'

export class RequestError extends Error {
  detail: ApiError
  constructor(detail: ApiError) { super(detail.message); this.detail = detail }
}
export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try { response = await fetch(`/api${path}`, { ...init, headers: { 'Content-Type': 'application/json', ...init?.headers } }) }
  catch (e) { if (e instanceof DOMException && e.name === 'AbortError') throw e; throw new RequestError({ message: '连接失败，请检查本地服务' }) }
  // 经 VPN 访问时登录过期或口令更换：回到登录页。
  if (response.status === 401) { location.assign('/login'); throw new RequestError({ message: '请先登录' }) }
  const data = await response.json()
  if (!response.ok) throw new RequestError(data)
  return data as T
}
export const post = <T>(path: string, body?: unknown) => api<T>(path, { method: 'POST', body: JSON.stringify(body ?? {}) })
const seriesCache = new Map<string, Promise<PlotPart[]>>()
export function series(run: string, view: View | 'long'): Promise<PlotPart[]> {
  const key = `${run}:${view}`
  if (!seriesCache.has(key)) {
    if (seriesCache.size >= 128) seriesCache.delete(seriesCache.keys().next().value!)
    seriesCache.set(key, api<PlotPart[]>(`/runs/${run}/series?view=${view}`).catch(e => { seriesCache.delete(key); throw e }))
  }
  return seriesCache.get(key)!
}
export function friendly(error: unknown): ApiError {
  return error instanceof RequestError ? error.detail : { message: '操作失败，请稍后重试' }
}
