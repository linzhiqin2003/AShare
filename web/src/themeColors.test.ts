import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join, relative } from 'node:path'
import { describe, expect, it } from 'vitest'

// 2026-11-14 暗色主题统一：组件里写死的颜色在深色下变成白底表头、白底行号栏、看不清的语法高亮。
// 颜色只在 style.css 里按亮暗两套定义；组件、TS 和其他样式表一律引用变量。这里扫 web/src 守住它。
const HEX = /(?<![\w&])#(?:[0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{3,4})(?![\w-])/g
const FUNCTION = /(?<![\w-])(?:rgba?|hsla?|hwb|lab|lch|oklab|oklch)\(/g
// 颜色名只在取颜色的属性里算（CSS 声明、SVG 属性、JS 对象键、canvas 赋值），类名、var(--red)、主题名不算。
const PROPERTY = /(?<![\w-])(?:color|background(?:-color)?|border(?:-(?:top|right|bottom|left))?(?:-color)?|outline(?:-color)?|fill|stroke|box-shadow|text-shadow|caret-color|accent-color|stop-color|flood-color|text-decoration(?:-color)?|column-rule(?:-color)?|fillStyle|strokeStyle|shadowColor)\s*[:=]\s*/g
const NAMED = /(?<![\w$.-])(?:white|black|red|green|blue|gr[ae]y|silver|orange|yellow|purple|pink|brown|navy|teal|maroon|olive|lime|aqua|cyan|magenta|fuchsia|gold|indigo|violet|beige|ivory|khaki|coral|salmon|tomato|crimson|orchid|plum|wheat|azure|lavender|snow|linen|whitesmoke|gainsboro|lightgr[ae]y|darkgr[ae]y|dimgr[ae]y)(?![\w-(])/g

function value(source: string, from: number) {
  const quote = source[from]
  if (quote === '"' || quote === "'" || quote === '`') {
    const end = source.indexOf(quote, from + 1)
    return source.slice(from + 1, end < 0 ? undefined : end)
  }
  return source.slice(from).match(/^[^;}\n<>]*/)![0]
}

/** 返回源码里写死的颜色（附行号）；空数组表示只用了变量和 transparent/currentColor/inherit。 */
export function findColorLiterals(source: string): string[] {
  const line = (index: number) => source.slice(0, index).split('\n').length
  const found: string[] = []
  for (const pattern of [HEX, FUNCTION]) for (const match of source.matchAll(pattern)) found.push(`${line(match.index!)}: ${match[0]}`)
  for (const match of source.matchAll(PROPERTY)) {
    const start = match.index! + match[0].length
    for (const name of value(source, start).matchAll(NAMED)) found.push(`${line(start)}: ${name[0]}`)
  }
  return found
}

const root = __dirname
function sources(folder: string): string[] {
  return readdirSync(folder).flatMap(name => {
    const path = join(folder, name)
    if (statSync(path).isDirectory()) return sources(path)
    if (!/\.(vue|ts|css)$/.test(name) || /\.test\.ts$/.test(name)) return []
    return relative(root, path) === 'style.css' ? [] : [relative(root, path)]
  })
}

describe('组件与 TS 不写死颜色', () => {
  it.each(sources(root))('%s', name => {
    expect(findColorLiterals(readFileSync(join(root, name), 'utf8'))).toEqual([])
  })
})

describe('颜色扫描本身', () => {
  // 守卫要抓得到这次清掉的每一类写法，也不能把类名、var(--red)、white-space、主题名当成颜色。
  it.each([
    ['background:white;', 'white'],
    ['.x{color:#A14A14}', '#A14A14'],
    ["{ background: '#fff', color: '#8A9199' }", '#fff'],
    ['box-shadow:0 12px 32px #00000024', '#00000024'],
    ['fill="rgba(31,79,191,.035)"', 'rgba('],
    ['color: `hsl(220, 65%, ${l}%)`', 'hsl('],
    ['stroke="white"', 'white'],
    [':fill="ok ? \'black\' : \'var(--ink)\'"', 'black'],
    ['ctx.fillStyle = "red"', 'red'],
    ['.year{border-left:1px solid white}', 'white'],
  ])('抓到 %s', (source, hit) => {
    expect(findColorLiterals(source).join(' ')).toContain(hit)
  })
  it.each([
    'white-space:nowrap',
    'color:var(--red);background:var(--green)',
    '<span class="green">',
    "setRise('green'); theme.value === 'dark'",
    'background:transparent;color:currentColor;fill:inherit',
    '<template #first>',
    "router.push('#/research')",
    'color-mix(in srgb, var(--rise) 30%, transparent)',
    'fill-opacity=".11" stroke-width="1"',
  ])('放过 %s', source => {
    expect(findColorLiterals(source)).toEqual([])
  })
})
