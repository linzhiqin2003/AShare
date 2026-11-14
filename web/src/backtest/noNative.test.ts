// 回测页同样不用系统原生控件（2026-11-14 用户要求；共享扫描只覆盖 components/，这里补上回测组件和合并后的研究页）。
import { readdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'

const banned: [RegExp, string][] = [
  [/<select[\s>]/, '原生 select'],
  [/type="checkbox"/, '原生 checkbox'],
  [/type="radio"/, '原生 radio'],
  [/type="number"/, '原生 number input'],
  [/type="date"/, '原生 date input'],
  [/\s:?title="/, '原生 title 提示'],
  [/<title>/, 'SVG 原生 title 提示'],
]
const files = [
  ...readdirSync(__dirname).filter(name => name.endsWith('.vue')).map(name => join(__dirname, name)),
  join(__dirname, '../pages/ResearchPage.vue'),
]

describe('回测页不使用系统原生控件', () => {
  it.each(files)('%s', path => {
    const source = readFileSync(path, 'utf8')
    expect(banned.filter(([pattern]) => pattern.test(source)).map(([, label]) => label)).toEqual([])
  })
})
