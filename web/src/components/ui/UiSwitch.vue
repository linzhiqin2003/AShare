<script setup lang="ts">
import { computed } from 'vue'
// 开关（2026-11-14 统一原 .switch 按钮写法）。用法：<UiSwitch v-model="bool" aria-label="…" />
const props = defineProps<{ modelValue: boolean; disabled?: boolean; 'aria-label'?: string }>()
const emit = defineEmits<{ 'update:modelValue': [value: boolean] }>()
// Vue 把 'aria-label' 规范成 ariaLabel 存在 props 上。
const ariaLabel = computed(() => (props as unknown as { ariaLabel?: string }).ariaLabel)
</script>
<template>
  <button type="button" role="switch" class="ui-switch" :aria-label="ariaLabel" :aria-checked="modelValue" :disabled="disabled" @click="emit('update:modelValue', !props.modelValue)"><i /></button>
</template>
<style>
.ui-switch{flex:none;width:28px;height:16px;padding:2px;border:0;border-radius:10px;background:var(--line);transition:background .15s}
.ui-switch i{display:block;width:12px;height:12px;border-radius:50%;background:var(--panel);box-shadow:0 0 1px var(--overlay-shadow);transition:transform .15s}
.ui-switch[aria-checked=true]{background:var(--ink)}
.ui-switch[aria-checked=true] i{transform:translateX(12px)}
@media (prefers-reduced-motion:reduce){.ui-switch,.ui-switch i{transition:none}}
</style>
