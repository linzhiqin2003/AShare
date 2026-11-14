// 自绘下拉的共用列表逻辑（2026-11-14 从 UiSelect 抽出，单选 UiSelect 与多选 UiMultiSelect 共用）：
// 打开收起、挂在 body 上的定位、点外部关闭、上下键与 Home/End、首字母跳转、回车空格选择、Esc 关闭。
// 选中后做什么由调用方的 choose 决定：单选选完收起，多选切换勾选后保持打开。
import { nextTick, onUnmounted, shallowRef } from 'vue'
import { place } from './position'
import type { UiOption } from './types'

export interface ListboxOptions<T> {
  options: () => UiOption<T>[]
  disabled: () => boolean | undefined
  /** 打开时高亮哪一项（通常是当前选中的第一项）；不可选时落到第一个可选项。 */
  initial: () => number
  choose: (index: number) => void
}

export function useListbox<T>(config: ListboxOptions<T>) {
  const trigger = shallowRef<HTMLButtonElement | null>(null), list = shallowRef<HTMLElement | null>(null)
  const open = shallowRef(false), active = shallowRef(-1)
  const style = shallowRef<Record<string, string>>({})
  let typed = '', typedTimer: ReturnType<typeof setTimeout> | undefined

  function position() {
    if (!trigger.value) return
    const rect = trigger.value.getBoundingClientRect()
    const width = Math.max(rect.width, list.value?.offsetWidth ?? 0)
    const height = list.value?.offsetHeight ?? Math.min(300, config.options().length * 28 + 10)
    const at = place(rect, { width, height })
    style.value = { left: `${at.left}px`, top: `${at.top}px`, minWidth: `${rect.width}px` }
  }
  function enabled(index: number) { const options = config.options(); return index >= 0 && index < options.length && !options[index].disabled }
  function step(from: number, direction: 1 | -1) {
    for (let i = from + direction; i >= 0 && i < config.options().length; i += direction) if (enabled(i)) return i
    return from
  }
  function scrollActive() { void nextTick(() => list.value?.querySelector<HTMLElement>(`[data-index="${active.value}"]`)?.scrollIntoView?.({ block: 'nearest' })) }
  async function show() {
    if (config.disabled() || open.value) return
    open.value = true
    const initial = config.initial()
    active.value = enabled(initial) ? initial : step(-1, 1)
    position()
    await nextTick(); position(); list.value?.focus(); scrollActive()
    document.addEventListener('pointerdown', outside, true)
    addEventListener('resize', position); addEventListener('scroll', reposition, true)
  }
  function hide(refocus = true) {
    if (!open.value) return
    open.value = false
    document.removeEventListener('pointerdown', outside, true)
    removeEventListener('resize', position); removeEventListener('scroll', reposition, true)
    if (refocus) trigger.value?.focus()
  }
  function toggle() { if (open.value) hide(); else void show() }
  function reposition(event: Event) { if (!list.value?.contains(event.target as Node)) position() }
  function outside(event: PointerEvent) { const target = event.target as Node; if (!list.value?.contains(target) && !trigger.value?.contains(target)) hide(false) }
  function pick(index: number) { if (enabled(index)) config.choose(index) }
  function typeahead(key: string) {
    const options = config.options()
    clearTimeout(typedTimer); typed += key.toLowerCase(); typedTimer = setTimeout(() => typed = '', 500)
    const order = [...options.keys()].map(i => (active.value + 1 + i) % options.length)
    const match = order.find(i => enabled(i) && options[i].label.toLowerCase().startsWith(typed)) ?? order.find(i => enabled(i) && options[i].label.toLowerCase().startsWith(key.toLowerCase()))
    if (match !== undefined) { active.value = match; scrollActive() }
  }
  function triggerKey(event: KeyboardEvent) {
    if (['ArrowDown', 'ArrowUp', 'Enter', ' '].includes(event.key)) { event.preventDefault(); void show() }
  }
  function listKey(event: KeyboardEvent) {
    const key = event.key
    if (key === 'Escape') { event.preventDefault(); event.stopPropagation(); hide() }
    else if (key === 'Tab') hide(false)
    else if (key === 'ArrowDown') { event.preventDefault(); active.value = step(active.value, 1); scrollActive() }
    else if (key === 'ArrowUp') { event.preventDefault(); active.value = step(active.value, -1); scrollActive() }
    else if (key === 'Home') { event.preventDefault(); active.value = step(-1, 1); scrollActive() }
    else if (key === 'End') { event.preventDefault(); active.value = step(config.options().length, -1); scrollActive() }
    else if (key === 'Enter' || key === ' ') { event.preventDefault(); pick(active.value) }
    else if (key.length === 1 && !event.metaKey && !event.ctrlKey && !event.altKey) { event.preventDefault(); typeahead(key) }
  }
  onUnmounted(() => { hide(false); clearTimeout(typedTimer) })
  return { trigger, list, open, active, style, enabled, show, hide, toggle, pick, triggerKey, listKey }
}

/** 值可能是单个也可能是数组（扫参或旧预设）：统一成数组。 */
export function asList<T>(value: T | readonly T[]): readonly T[] { return Array.isArray(value) ? value : [value as T] }
/** 触发器上的文字：选中项的名字用顿号拼起来，按选项顺序。 */
export function joinedLabel<T>(options: UiOption<T>[], value: T | readonly T[]) {
  const values = asList(value)
  return options.filter(o => values.includes(o.value)).map(o => o.label).join('、')
}
