<script setup lang="ts">
import { computed, onMounted, onUnmounted, shallowRef } from 'vue'
const props = withDefaults(defineProps<{ storageKey: string; axis?: 'horizontal' | 'vertical'; initial?: number; minFirst?: number; maxFirst?: number; minSecond?: number; maxSecond?: number; active?: boolean; label: string }>(), { axis: 'horizontal', initial: .5, minFirst: 200, maxFirst: Infinity, minSecond: 200, maxSecond: Infinity, active: true })
const root = shallowRef<HTMLElement | null>(null), size = shallowRef(0), ratio = shallowRef(props.initial)
let observer: ResizeObserver | undefined, frame = 0
let drag: { start: number; first: number } | null = null
let pending = 0
try { const saved = Number(localStorage.getItem(`ashare.split.${props.storageKey}`)); if (saved > 0 && saved < 1) ratio.value = saved } catch { /* 无存储权限时保留默认比例。 */ }
const available = computed(() => Math.max(0, size.value - 8))
const bounds = computed(() => {
  const total = available.value
  const scale = Math.min(1, total / (props.minFirst + props.minSecond))
  return [Math.max(props.minFirst * scale, total - props.maxSecond), Math.min(props.maxFirst, total - props.minSecond * scale)]
})
const first = computed(() => Math.max(bounds.value[0], Math.min(bounds.value[1], available.value * ratio.value)))
const style = computed(() => props.active ? { [props.axis === 'horizontal' ? 'gridTemplateColumns' : 'gridTemplateRows']: `${first.value}px 8px minmax(0,1fr)` } : { gridTemplateColumns: 'minmax(0,1fr)' })
function save() { localStorage.setItem(`ashare.split.${props.storageKey}`, String(ratio.value)) }
function reset() { ratio.value = props.initial; save() }
function move(event: PointerEvent) {
  if (!drag) return
  pending = drag.first + (props.axis === 'horizontal' ? event.clientX : event.clientY) - drag.start
  if (!frame) frame = requestAnimationFrame(() => { ratio.value = Math.max(bounds.value[0], Math.min(bounds.value[1], pending)) / available.value; frame = 0 })
}
function end() { if (frame) { cancelAnimationFrame(frame); frame = 0; ratio.value = Math.max(bounds.value[0], Math.min(bounds.value[1], pending)) / available.value }; drag = null; save() }
function begin(event: PointerEvent) { drag = { start: props.axis === 'horizontal' ? event.clientX : event.clientY, first: first.value }; pending = first.value; (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId); event.preventDefault() }
function key(event: KeyboardEvent) { const direction = ['ArrowLeft', 'ArrowUp'].includes(event.key) ? -1 : ['ArrowRight', 'ArrowDown'].includes(event.key) ? 1 : 0; if (event.key === 'Home') reset(); if (direction) { event.preventDefault(); ratio.value = Math.max(bounds.value[0], Math.min(bounds.value[1], first.value + direction * 16)) / available.value; save() } }
onMounted(() => { observer = new ResizeObserver(entries => { const rect = entries[0].contentRect; const next = props.axis === 'horizontal' ? rect.width : rect.height; if (next > 0) size.value = next }); if (root.value) observer.observe(root.value) })
onUnmounted(() => { observer?.disconnect(); cancelAnimationFrame(frame) })
</script>
<template>
  <div ref="root" class="split-pane" :class="axis" :style="style" :data-split="storageKey" :data-initial="initial" :data-min-first="minFirst" :data-max-first="maxFirst" :data-min-second="minSecond" :data-max-second="maxSecond">
    <div class="split-child"><slot name="first" /></div>
    <div v-if="active" class="split-boundary" role="separator" tabindex="0" :aria-label="label" :aria-orientation="axis === 'horizontal' ? 'vertical' : 'horizontal'" :aria-valuenow="Math.round(ratio * 100)" @pointerdown="begin" @pointermove="move" @pointerup="end" @pointercancel="end" @dblclick="reset" @keydown="key"></div>
    <div v-if="active" class="split-child"><slot name="second" /></div>
  </div>
</template>
<style scoped>
.split-pane{display:grid;width:100%;height:100%;min-width:0;min-height:0}.split-child{min-width:0;min-height:0;position:relative}.split-child>:deep(*){height:100%;min-height:0}.split-boundary{position:relative;touch-action:none;z-index:4}.horizontal>.split-boundary{cursor:col-resize}.vertical>.split-boundary{cursor:row-resize}.split-boundary:after{content:'';position:absolute;background:transparent;border-radius:1px;transition:background .1s}.horizontal>.split-boundary:after{inset:0 3px}.vertical>.split-boundary:after{inset:3px 0}.split-boundary:hover:after,.split-boundary:focus-visible:after{background:var(--accent)}
</style>
