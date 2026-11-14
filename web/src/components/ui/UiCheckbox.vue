<script setup lang="ts">
import { computed } from 'vue'
// 自绘勾选框（2026-11-14 用户要求替换系统原生控件）。
// 用法：<UiCheckbox v-model="bool" :indeterminate? :color? aria-label="…" />；放在 <label> 里点文字也能切换。
const props = defineProps<{ modelValue: boolean; indeterminate?: boolean; color?: string; disabled?: boolean; 'aria-label'?: string }>()
const emit = defineEmits<{ 'update:modelValue': [value: boolean] }>()
// Vue 把 'aria-label' 规范成 ariaLabel 存在 props 上。
const ariaLabel = computed(() => (props as unknown as { ariaLabel?: string }).ariaLabel)
function toggle() { if (!props.disabled) emit('update:modelValue', props.indeterminate ? true : !props.modelValue) }
</script>
<template>
  <button type="button" role="checkbox" class="ui-checkbox" :aria-label="ariaLabel" :class="{ on: modelValue || indeterminate }" :aria-checked="indeterminate ? 'mixed' : modelValue" :disabled="disabled" :style="color ? { '--ui-check-color': color } : undefined" @click="toggle" @keydown.enter.prevent>
    <svg v-if="indeterminate" viewBox="0 0 12 12" aria-hidden="true"><path d="M3 6h6" /></svg>
    <svg v-else-if="modelValue" viewBox="0 0 12 12" aria-hidden="true"><path d="M2.6 6.2 5 8.5l4.4-5" /></svg>
  </button>
</template>
<style>
.ui-checkbox{--ui-check-color:var(--ink);flex:none;display:inline-grid;place-items:center;width:14px;height:14px;padding:0;border:1px solid var(--ink3);border-radius:3px;background:var(--panel);vertical-align:middle;transition:background .1s,border-color .1s}
.ui-checkbox:hover:not(:disabled){border-color:var(--ink2)}
.ui-checkbox.on{background:var(--ui-check-color);border-color:var(--ui-check-color)}
.ui-checkbox svg{width:10px;height:10px;fill:none;stroke:var(--panel);stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round}
.ui-checkbox:disabled{opacity:.4}
@media (prefers-reduced-motion:reduce){.ui-checkbox{transition:none}}
</style>
