import { readdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'

// 2026-11-14 用户：「都还是系统原生的组件，在搞什么」（研究台「累加/复利」弹出系统菜单）。
// 页面组件只用 ui/ 里的自绘控件；文本输入框保留原生 input。
const folder = join(__dirname, '..')
const banned: [RegExp, string][] = [
  [/<select[\s>]/, '原生 select'],
  [/type="checkbox"/, '原生 checkbox'],
  [/type="number"/, '原生 number input'],
  [/class="switch"/, '旧 .switch 按钮'],
  [/\s:?title="/, '原生 title 提示'],
  [/<title>/, 'SVG 原生 title 提示'],
]

describe('页面组件不再使用系统原生控件', () => {
  it.each(readdirSync(folder).filter(name => name.endsWith('.vue')))('%s', name => {
    const source = readFileSync(join(folder, name), 'utf8')
    const found = banned.filter(([pattern]) => pattern.test(source)).map(([, label]) => label)
    expect(found).toEqual([])
  })
})
