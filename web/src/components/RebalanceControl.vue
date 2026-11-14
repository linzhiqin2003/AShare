<script setup lang="ts">
import { shallowRef, watch } from 'vue'
import type { Scalar } from '../types'
import { vTip } from './ui'
const props = defineProps<{ modelValue?: Scalar | Scalar[] }>()
const emit = defineEmits<{ 'update:modelValue': [value: number | number[] | string] }>()
const text = shallowRef('1'), error = shallowRef('')
watch(() => props.modelValue, value => { text.value = typeof value === 'string' ? value : JSON.stringify(value ?? 1); if (typeof value !== 'string') error.value = '' }, { immediate: true })
function change() {
  try {
    const value = JSON.parse(text.value), values = Array.isArray(value) ? value : [value]
    if (!values.length || values.length > 8 || !values.every(v => Number.isInteger(v) && v >= 1 && v <= 2520)) throw new Error()
    error.value = ''; emit('update:modelValue', value)
  } catch { error.value = '请输入 1–2520，或 [1,5,20]'; emit('update:modelValue',text.value) }
}
</script>
<template><div class="rebalance-control"><div class="rebalance-input"><button v-for="n in [1,5,10,20]" :key="n" :class="{active:modelValue === n}" @click="emit('update:modelValue',n)">{{ n }}</button><input v-model="text" aria-label="调仓周期" v-tip="'交易日数，支持 [1,5,20] 扫描'" @input="change" @blur="change"></div><span v-if="error" role="alert">{{ error }}</span></div></template>
<style scoped>
.rebalance-control{max-width:240px}.rebalance-input{display:flex;align-items:center;height:24px}.rebalance-input button{height:24px;padding:0 9px;border:1px solid var(--line);border-right:0;background:white;font-size:11px;color:var(--ink2)}.rebalance-input button:first-child{border-radius:3px 0 0 3px}.rebalance-input .active{background:var(--selected);color:var(--accent)}.rebalance-input input{width:92px;min-width:60px;min-height:24px;height:24px;border-radius:0 3px 3px 0;font-family:var(--mono);font-size:11px;padding:0 5px}.rebalance-control>span{font-size:10px;color:var(--red)}
</style>
