import { afterEach, describe, expect, it, vi } from 'vitest'
import { api, RequestError } from './api'

describe('登录过期', () => {
  afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks() })
  it('接口返回 401 时回到登录页（2026-11-14：经 VPN 访问加了口令登录）', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ message: '请先登录' }), { status: 401 })))
    const assign = vi.fn()
    vi.stubGlobal('location', { ...window.location, assign })
    await expect(api('/meta')).rejects.toBeInstanceOf(RequestError)
    expect(assign).toHaveBeenCalledWith('/login')
  })
  it('其他错误照常抛出，不跳转', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ message: '操作失败' }), { status: 500 })))
    const assign = vi.fn()
    vi.stubGlobal('location', { ...window.location, assign })
    await expect(api('/meta')).rejects.toMatchObject({ detail: { message: '操作失败' } })
    expect(assign).not.toHaveBeenCalled()
  })
})
