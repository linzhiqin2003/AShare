<script setup lang="ts" generic="T extends string | number | boolean | null">
// 自绘下拉（2026-11-14 用户要求替换系统原生 select）。
// 用法：<UiSelect v-model="x" :options="[{ value, label, disabled? }]" aria-label="…" />
import { computed, nextTick, onUnmounted, shallowRef, useId } from 'vue'
import { Check, ChevronDown } from 'lucide-vue-next'
import { place } from './position'
import type { UiOption } from './types'

defineOptions({ inheritAttrs: false })
// 'aria-label' 声明成 prop：开着 strictTemplates 时调用方才能直接写 aria-label。
const props = defineProps<{ modelValue: T; options: UiOption<T>[]; placeholder?: string; disabled?: boolean; 'aria-label'?: string }>()
const emit = defineEmits<{ 'update:modelValue': [value: T] }>()
// Vue 把 'aria-label' 规范成 ariaLabel 存在 props 上。
const ariaLabel = computed(() => (props as unknown as { ariaLabel?: string }).ariaLabel)

const id = useId()
const trigger = shallowRef<HTMLButtonElement | null>(null), list = shallowRef<HTMLElement | null>(null)
const open = shallowRef(false), active = shallowRef(-1)
const style = shallowRef<Record<string, string>>({})
const selectedIndex = computed(() => props.options.findIndex(o => o.value === props.modelValue))
const current = computed(() => props.options[selectedIndex.value])
let typed = '', typedTimer: ReturnType<typeof setTimeout> | undefined

function position() {
  if (!trigger.value) return
  const rect = trigger.value.getBoundingClientRect()
  const width = Math.max(rect.width, list.value?.offsetWidth ?? 0)
  const height = list.value?.offsetHeight ?? Math.min(300, props.options.length * 28 + 10)
  const at = place(rect, { width, height })
  style.value = { left: `${at.left}px`, top: `${at.top}px`, minWidth: `${rect.width}px` }
}
function enabled(index: number) { return index >= 0 && index < props.options.length && !props.options[index].disabled }
function step(from: number, direction: 1 | -1) {
  for (let i = from + direction; i >= 0 && i < props.options.length; i += direction) if (enabled(i)) return i
  return from
}
function scrollActive() { void nextTick(() => list.value?.querySelector<HTMLElement>(`[data-index="${active.value}"]`)?.scrollIntoView?.({ block: 'nearest' })) }
async function show() {
  if (props.disabled || open.value) return
  open.value = true
  active.value = enabled(selectedIndex.value) ? selectedIndex.value : step(-1, 1)
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
function reposition(event: Event) { if (!list.value?.contains(event.target as Node)) position() }
function outside(event: PointerEvent) { const target = event.target as Node; if (!list.value?.contains(target) && !trigger.value?.contains(target)) hide(false) }
function choose(index: number) { if (!enabled(index)) return; if (props.options[index].value !== props.modelValue) emit('update:modelValue', props.options[index].value); hide() }
function typeahead(key: string) {
  clearTimeout(typedTimer); typed += key.toLowerCase(); typedTimer = setTimeout(() => typed = '', 500)
  const order = [...props.options.keys()].map(i => (active.value + 1 + i) % props.options.length)
  const match = order.find(i => enabled(i) && props.options[i].label.toLowerCase().startsWith(typed)) ?? order.find(i => enabled(i) && props.options[i].label.toLowerCase().startsWith(key.toLowerCase()))
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
  else if (key === 'End') { event.preventDefault(); active.value = step(props.options.length, -1); scrollActive() }
  else if (key === 'Enter' || key === ' ') { event.preventDefault(); choose(active.value) }
  else if (key.length === 1 && !event.metaKey && !event.ctrlKey && !event.altKey) { event.preventDefault(); typeahead(key) }
}
onUnmounted(() => { hide(false); clearTimeout(typedTimer) })
</script>
<template>
  <button ref="trigger" type="button" class="ui-select" v-bind="$attrs" :aria-label="ariaLabel" :disabled="disabled" aria-haspopup="listbox" :aria-expanded="open" :aria-controls="open ? `${id}-list` : undefined" @click="open ? hide() : show()" @keydown="triggerKey">
    <span class="ui-select-label" :class="{ placeholder: !current }">{{ current?.label ?? placeholder ?? '' }}</span><ChevronDown :size="12" class="ui-select-caret" />
  </button>
  <Teleport to="body">
    <ul v-if="open" :id="`${id}-list`" ref="list" class="ui-menu ui-select-menu" role="listbox" tabindex="-1" :aria-label="ariaLabel" :aria-activedescendant="active >= 0 ? `${id}-${active}` : undefined" :style="style" @keydown="listKey">
      <li v-for="(option, index) in options" :id="`${id}-${index}`" :key="String(option.value)" role="option" class="ui-menu-item" :class="{ active: index === active }" :data-index="index" :aria-selected="option.value === modelValue" :aria-disabled="option.disabled || undefined" @pointermove="enabled(index) && (active = index)" @click="choose(index)">
        <span>{{ option.label }}</span><Check v-if="option.value === modelValue" :size="12" class="ui-check" />
      </li>
    </ul>
  </Teleport>
</template>
<style>
.ui-select{display:inline-flex;align-items:center;gap:6px;height:24px;min-width:0;max-width:100%;padding:0 7px 0 8px;border:1px solid var(--line);border-radius:3px;background:var(--panel);color:var(--ink);font-size:11.5px;font-weight:400;white-space:nowrap;text-align:left}
.ui-select:hover:not(:disabled){border-color:var(--ink3)}
.ui-select[aria-expanded=true]{border-color:var(--accent)}
.ui-select-label{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis}
.ui-select-label.placeholder{color:var(--ink3)}
.ui-select-caret{color:var(--ink3);transition:transform .12s}
.ui-select[aria-expanded=true] .ui-select-caret{transform:rotate(180deg)}
.ui-menu{position:fixed;z-index:90;margin:0;padding:5px;list-style:none;background:var(--panel);border:1px solid var(--line);border-radius:6px;box-shadow:0 12px 32px #00000024;max-height:min(320px,calc(100vh - 16px));overflow:auto;outline:none;font-size:12px;color:var(--ink)}
.ui-menu.ui-menu:focus-visible{outline:none}
.ui-menu-item{display:flex;align-items:center;gap:8px;min-height:28px;padding:0 8px;border-radius:3px;cursor:pointer;white-space:nowrap}
.ui-menu-item>span{flex:1}
.ui-menu-item.active{background:var(--selected)}
.ui-menu-item[aria-disabled=true]{color:var(--ink3);cursor:default}
.ui-menu-item[aria-disabled=true].active{background:transparent}
.ui-check{color:var(--ink);flex:none}
@media (prefers-reduced-motion:reduce){.ui-select-caret{transition:none}}
</style>
