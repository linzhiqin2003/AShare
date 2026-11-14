import { shallowRef } from 'vue'

// 外观偏好只属于当前浏览器，存 localStorage；写在 <html> 的属性上，CSS 变量据此切换。
export type Theme = 'light' | 'dark' | 'system'
export type Rise = 'red' | 'green'
const THEME_KEY = 'ashare.theme', RISE_KEY = 'ashare.rise'

function read<T extends string>(key: string, allowed: readonly T[], fallback: T): T {
  try { const value = localStorage.getItem(key); return allowed.includes(value as T) ? value as T : fallback } catch { return fallback }
}
function write(key: string, value: string) { try { localStorage.setItem(key, value) } catch { /* 隐私模式下只在本次会话生效。 */ } }

export const theme = shallowRef<Theme>(read(THEME_KEY, ['light', 'dark', 'system'] as const, 'system'))
export const rise = shallowRef<Rise>(read(RISE_KEY, ['red', 'green'] as const, 'red'))

export function applyAppearance(root: HTMLElement = document.documentElement) {
  // 跟随系统时不写属性，由 prefers-color-scheme 决定。
  if (theme.value === 'system') root.removeAttribute('data-theme')
  else root.setAttribute('data-theme', theme.value)
  root.setAttribute('data-rise', rise.value)
}
export function setTheme(value: Theme) { theme.value = value; write(THEME_KEY, value); applyAppearance() }
export function setRise(value: Rise) { rise.value = value; write(RISE_KEY, value); applyAppearance() }

/** 当前实际显示的是否深色（跟随系统时看系统设置）。 */
export function isDark() {
  if (theme.value !== 'system') return theme.value === 'dark'
  return typeof matchMedia === 'function' && matchMedia('(prefers-color-scheme: dark)').matches
}
/** ⌘⇧L：在浅色与深色之间切换（跟随系统时切到与当前显示相反的一种）。 */
export function toggleTheme() { setTheme(isDark() ? 'light' : 'dark') }
export function themeShortcut(event: KeyboardEvent) {
  if ((event.metaKey || event.ctrlKey) && event.shiftKey && event.key.toLowerCase() === 'l') { event.preventDefault(); toggleTheme() }
}
