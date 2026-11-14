<script setup lang="ts" generic="T extends string | number | boolean | null">
// 自绘下拉（2026-11-14 用户要求替换系统原生 select）。
// 用法：<UiSelect v-model="x" :options="[{ value, label, disabled? }]" aria-label="…" />
// 值可能本来就是数组（扫参或旧预设）：触发器显示拼接文字，所有选中项打勾，选一项后变回单个值。
// 列表逻辑在 listbox.ts，与多选 UiMultiSelect 共用。
import { computed, useId } from 'vue'
import { Check, ChevronDown } from 'lucide-vue-next'
import { asList, joinedLabel, useListbox } from './listbox'
import type { UiOption } from './types'

defineOptions({ inheritAttrs: false })
// 'aria-label' 声明成 prop：开着 strictTemplates 时调用方才能直接写 aria-label。
const props = defineProps<{ modelValue: T | readonly T[]; options: UiOption<T>[]; placeholder?: string; disabled?: boolean; 'aria-label'?: string }>()
const emit = defineEmits<{ 'update:modelValue': [value: T] }>()
// Vue 把 'aria-label' 规范成 ariaLabel 存在 props 上。
const ariaLabel = computed(() => (props as unknown as { ariaLabel?: string }).ariaLabel)

const id = useId()
const isSelected = (value: T) => asList(props.modelValue).includes(value)
const label = computed(() => joinedLabel(props.options, props.modelValue))
const { trigger, list, open, active, style, enabled, hide, toggle, pick, triggerKey, listKey } = useListbox({
  options: () => props.options,
  disabled: () => props.disabled,
  initial: () => props.options.findIndex(o => isSelected(o.value)),
  choose: index => { const value = props.options[index].value; if (Array.isArray(props.modelValue) || value !== props.modelValue) emit('update:modelValue', value); hide() },
})
</script>
<template>
  <button ref="trigger" type="button" class="ui-select" v-bind="$attrs" :aria-label="ariaLabel" :disabled="disabled" aria-haspopup="listbox" :aria-expanded="open" :aria-controls="open ? `${id}-list` : undefined" @click="toggle" @keydown="triggerKey">
    <span class="ui-select-label" :class="{ placeholder: !label }">{{ label || placeholder || '' }}</span><ChevronDown :size="12" class="ui-select-caret" />
  </button>
  <Teleport to="body">
    <ul v-if="open" :id="`${id}-list`" ref="list" class="ui-menu ui-select-menu" role="listbox" tabindex="-1" :aria-label="ariaLabel" :aria-activedescendant="active >= 0 ? `${id}-${active}` : undefined" :style="style" @keydown="listKey">
      <li v-for="(option, index) in options" :id="`${id}-${index}`" :key="String(option.value)" role="option" class="ui-menu-item" :class="{ active: index === active }" :data-index="index" :aria-selected="isSelected(option.value)" :aria-disabled="option.disabled || undefined" @pointermove="enabled(index) && (active = index)" @click="pick(index)">
        <span>{{ option.label }}</span><Check v-if="isSelected(option.value)" :size="12" class="ui-check" />
      </li>
    </ul>
  </Teleport>
</template>
<style>
.ui-select{display:inline-flex;align-items:center;gap:6px;height:24px;min-width:0;max-width:100%;padding:0 7px 0 8px;border:1px solid var(--line);border-radius:3px;background:var(--panel);color:var(--ink);font-size:11.5px;font-weight:400;white-space:nowrap;text-align:left}
.ui-select:hover:not(:disabled){border-color:var(--ink3)}
.ui-select[aria-expanded=true]{border-color:var(--ink3)}
.ui-select-label{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis}
.ui-select-label.placeholder{color:var(--ink3)}
.ui-select-caret{color:var(--ink3);transition:transform .12s}
.ui-select[aria-expanded=true] .ui-select-caret{transform:rotate(180deg)}
.ui-menu{position:fixed;z-index:90;margin:0;padding:5px;list-style:none;background:var(--panel);border:1px solid var(--line);border-radius:6px;box-shadow:0 12px 32px var(--overlay-shadow);max-height:min(320px,calc(100vh - 16px));overflow:auto;outline:none;font-size:12px;color:var(--ink)}
.ui-menu.ui-menu:focus-visible{outline:none}
.ui-menu-item{display:flex;align-items:center;gap:8px;min-height:28px;padding:0 8px;border-radius:3px;cursor:pointer;white-space:nowrap}
.ui-menu-item>span{flex:1}
.ui-menu-item.active{background:var(--selected)}
.ui-menu-item[aria-disabled=true]{color:var(--ink3);cursor:default}
.ui-menu-item[aria-disabled=true].active{background:transparent}
.ui-check{color:var(--ink);flex:none}
@media (prefers-reduced-motion:reduce){.ui-select-caret{transition:none}}
</style>
