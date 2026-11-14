import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'

// 2026-11-14 用户：「不要这种蓝色框了」（点进数字框出现 2px 主色描边）。聚焦只加深边框或用中性色细线。
const files = (dir: string): string[] => readdirSync(dir).flatMap(name => {
  const path = join(dir, name)
  return statSync(path).isDirectory() ? files(path) : /\.(vue|css)$/.test(name) ? [path] : []
})
const focusRule = /[^{}]*:focus(?:-visible|-within)?[^{}]*\{[^}]*\}/g

describe('聚焦样式不用主色框', () => {
  it.each(files(__dirname).map(f => [f.slice(__dirname.length + 1), f]))('%s', (_, path) => {
    const rules = readFileSync(path, 'utf8').match(focusRule) ?? []
    const blue = rules.filter(r => /(outline|border-color|box-shadow)[^;}]*var\(--accent\)/.test(r))
    expect(blue).toEqual([])
  })
  it('文本输入框聚焦不画描边', () => {
    expect(readFileSync(join(__dirname, 'style.css'), 'utf8')).toContain('input:focus,input:focus-visible{outline:none;border-color:var(--ink3)}')
  })
})
