<script setup lang="ts">
// 数字输入：没有原生 spinner，两侧 −/+ 按钮（2026-11-14 用户要求替换系统原生控件）。
// 用法：<UiNumberInput v-model="n" :min :max :step unit="天" aria-label="…" />；清空时发出 null。
import { computed, shallowRef, watch } from 'vue'
import { Minus, Plus } from 'lucide-vue-next'
defineOptions({ inheritAttrs: false })
const props = withDefaults(defineProps<{ modelValue: number | null; min?: number; max?: number; step?: number; unit?: string; disabled?: boolean; 'aria-label'?: string }>(), { min: -Infinity, max: Infinity, step: 1 })
const emit = defineEmits<{ 'update:modelValue': [value: number | null] }>()
// Vue 把 'aria-label' 规范成 ariaLabel 存在 props 上。
const ariaLabel = computed(() => (props as unknown as { ariaLabel?: string }).ariaLabel)
const text = shallowRef('')
const decimals = computed(() => (String(props.step).split('.')[1] ?? '').length)
watch(() => props.modelValue, value => { if (parse(text.value) !== value) text.value = value == null ? '' : String(value) }, { immediate: true })
function parse(value: string): number | null { const trimmed = value.trim(); if (!trimmed) return null; const number = Number(trimmed); return Number.isFinite(number) ? number : NaN }
const clamp = (value: number) => Math.min(props.max, Math.max(props.min, Number(value.toFixed(decimals.value))))
function input(value: string) {
  text.value = value
  const number = parse(value)
  if (number === null) emit('update:modelValue', null)
  else if (!Number.isNaN(number) && number >= props.min && number <= props.max) emit('update:modelValue', number)
}
function commit() {
  const number = parse(text.value)
  const value = number === null ? null : Number.isNaN(number) ? props.modelValue : clamp(number)
  text.value = value == null ? '' : String(value)
  if (value !== props.modelValue) emit('update:modelValue', value)
}
function nudge(direction: 1 | -1) {
  if (props.disabled) return
  const base = props.modelValue ?? (Number.isFinite(props.min) ? props.min : 0)
  const value = clamp(base + direction * props.step)
  text.value = String(value)
  if (value !== props.modelValue) emit('update:modelValue', value)
}
const label = (prefix: string) => ariaLabel.value ? `${prefix}${ariaLabel.value}` : prefix
</script>
<template>
  <div class="ui-number" :class="[$attrs.class, { disabled }]">
    <button type="button" tabindex="-1" :aria-label="label('减少')" :disabled="disabled || (modelValue != null && modelValue <= min)" @click="nudge(-1)"><Minus :size="11" /></button>
    <input :value="text" type="text" inputmode="decimal" :aria-label="ariaLabel" :disabled="disabled" :aria-valuemin="Number.isFinite(min) ? min : undefined" :aria-valuemax="Number.isFinite(max) ? max : undefined" @input="input(($event.target as HTMLInputElement).value)" @blur="commit" @keydown.enter="commit" @keydown.up.prevent="nudge(1)" @keydown.down.prevent="nudge(-1)">
    <button type="button" tabindex="-1" :aria-label="label('增加')" :disabled="disabled || (modelValue != null && modelValue >= max)" @click="nudge(1)"><Plus :size="11" /></button>
    <span v-if="unit" class="ui-number-unit">{{ unit }}</span>
  </div>
</template>
<style>
.ui-number{display:inline-flex;align-items:center;height:23px;flex:none}
.ui-number input{width:48px;height:23px;min-height:23px;padding:0 2px;border:1px solid var(--line);border-left:0;border-right:0;border-radius:0;text-align:center;font-family:var(--mono);font-size:11px;background:var(--panel)}
.ui-number>button{height:23px;width:22px;display:grid;place-items:center;padding:0;border:1px solid var(--line);background:var(--panel);color:var(--ink3)}
.ui-number>button:first-child{border-radius:3px 0 0 3px}
.ui-number>button:nth-of-type(2){border-radius:0 3px 3px 0}
.ui-number>button:hover:not(:disabled){color:var(--ink)}
/* 2026-11-14 用户：聚焦时不要蓝色粗框；整组边框加深即可。 */
.ui-number:focus-within>input,.ui-number:focus-within>button{border-color:var(--ink3)}
.ui-number-unit{padding-left:7px;color:var(--ink3);font-size:11.5px}
.ui-number.disabled input,.ui-number.disabled .ui-number-unit{opacity:.4}
</style>
