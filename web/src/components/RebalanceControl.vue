<script setup lang="ts">
// 调仓周期（交易日数）。2026-11-14 用户：预设按钮加输入框的拼法难看，改成和其他设置一样的下拉。
// 多选就是扫参（展开成多组）；只选一个时发出数字，和以前的单值配置一致。
import { computed } from 'vue'
import type { Scalar } from '../types'
import { UiMultiSelect } from './ui'

const PRESETS = [1, 2, 3, 5, 10, 20, 40, 60, 120, 240]
const props = defineProps<{ modelValue?: Scalar | Scalar[] }>()
const emit = defineEmits<{ 'update:modelValue': [value: number | number[]] }>()
const valid = (v: unknown): v is number => typeof v === 'number' && Number.isInteger(v) && v >= 1 && v <= 2520
const current = computed(() => {
  const values = (Array.isArray(props.modelValue) ? props.modelValue : [props.modelValue ?? 1]).filter(valid)
  return values.length ? values : [1]
})
// 旧预设里的非常用值（如 7 天）也列出来，免得打开设置就看不到当前值。
const options = computed(() => [...new Set([...PRESETS, ...current.value])].sort((a, b) => a - b).map(n => ({ value: n, label: `${n} 天` })))
function update(values: number[]) { emit('update:modelValue', values.length === 1 ? values[0] : values) }
</script>
<template><UiMultiSelect :model-value="current" :options="options" aria-label="调仓周期" @update:model-value="update" /></template>
