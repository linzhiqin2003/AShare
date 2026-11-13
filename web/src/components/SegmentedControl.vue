<script setup lang="ts">
import type { Scalar } from '../types'
const props = defineProps<{ modelValue: Scalar | Scalar[]; options: { value: Scalar; label: string }[]; multiple?: boolean; label: string }>()
const emit = defineEmits<{ 'update:modelValue': [value: Scalar | Scalar[]] }>()
function selected(value: Scalar) { return Array.isArray(props.modelValue) ? props.modelValue.includes(value) : props.modelValue === value }
function choose(value: Scalar) { if (!props.multiple) return emit('update:modelValue', value); const values = Array.isArray(props.modelValue) ? props.modelValue : [props.modelValue]; const next = selected(value) ? values.filter(v => v !== value) : [...values,value]; if (next.length) emit('update:modelValue', next) }
</script>
<template><div class="segmented" role="group" :aria-label="label"><button v-for="option in options" :key="String(option.value)" :class="{ on: selected(option.value) }" :aria-pressed="selected(option.value)" @click="choose(option.value)">{{ option.label }}</button></div></template>
<style scoped>.segmented{display:flex;border:1px solid var(--line);border-radius:3px;overflow:hidden;flex:none}.segmented button{padding:2px 8px;min-height:21px;font-weight:500;color:var(--ink3);border:0;border-left:1px solid var(--line);background:white;white-space:nowrap;font-size:11.5px}.segmented button:first-child{border-left:0}.segmented button.on{background:var(--selected);color:var(--ink)}.segmented button:hover{color:var(--ink)}</style>
