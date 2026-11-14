// @vitest-environment node
import { describe, expect, it } from 'vitest'
import config from '../vite.config'

describe('开发代理', () => {
  it('保留 Host（2026-11-14：字符串简写改写 Host，make web-dev 下所有写请求 403）', () => {
    const proxy = (config as { server?: { proxy?: Record<string, unknown> } }).server?.proxy?.['/api']
    expect(typeof proxy).toBe('object')
    expect((proxy as { changeOrigin?: boolean }).changeOrigin).toBe(false)
  })
})
