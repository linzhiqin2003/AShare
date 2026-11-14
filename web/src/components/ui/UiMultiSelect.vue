<script setup lang="ts" generic="T extends string | number | boolean | null">
// 多选下拉（2026-11-14 研究台中性化：选多个就是扫参，展开成多组）。
// 用法：<UiMultiSelect :model-value="值或数组" :options="[{ value, label }]" aria-label="…" @update:model-value="…" />
// 点选项或回车、空格切换勾选，菜单保持打开；最后一个不能取消；触发器显示「行业、市值」这样的拼接文字；Esc 关闭。
// 列表逻辑在 listbox.ts，与单选 UiSelect 共用。
import { computed, useId } from 'vue'
import { Check, ChevronDown } from 'lucide-vue-next'
import { asList, joinedLabel, useListbox } from './listbox'
import type { UiOption } from './types'

defineOptions({ inheritAttrs: false })
const props = defineProps<{ modelValue: T | readonly T[]; options: UiOption<T>[]; placeholder?: string; disabled?: boolean; 'aria-label'?: string }>()
const emit = defineEmits<{ 'update:modelValue': [value: T[]] }>()
const ariaLabel = computed(() => (props as unknown as { ariaLabel?: string }).ariaLabel)

const id = useId()
const isSelected = (value: T) => asList(props.modelValue).includes(value)
const label = computed(() => joinedLabel(props.options, props.modelValue))
function toggleValue(value: T) {
  const values = asList(props.modelValue)
  if (!isSelected(value)) return emit('update:modelValue', props.options.map(o => o.value).filter(v => v === value || values.includes(v)))
  // 至少留一个：取消最后一个勾选不生效。
  if (values.length > 1) emit('update:modelValue', values.filter(v => v !== value))
}
const { trigger, list, open, active, style, enabled, toggle, pick, triggerKey, listKey } = useListbox({
  options: () => props.options,
  disabled: () => props.disabled,
  initial: () => props.options.findIndex(o => isSelected(o.value)),
  choose: index => toggleValue(props.options[index].value),
})
</script>
<template>
  <button ref="trigger" type="button" class="ui-select" v-bind="$attrs" :aria-label="ariaLabel" :disabled="disabled" aria-haspopup="listbox" :aria-expanded="open" :aria-controls="open ? `${id}-list` : undefined" @click="toggle" @keydown="triggerKey">
    <span class="ui-select-label" :class="{ placeholder: !label }">{{ label || placeholder || '' }}</span><ChevronDown :size="12" class="ui-select-caret" />
  </button>
  <Teleport to="body">
    <ul v-if="open" :id="`${id}-list`" ref="list" class="ui-menu ui-select-menu" role="listbox" aria-multiselectable="true" tabindex="-1" :aria-label="ariaLabel" :aria-activedescendant="active >= 0 ? `${id}-${active}` : undefined" :style="style" @keydown="listKey">
      <li v-for="(option, index) in options" :id="`${id}-${index}`" :key="String(option.value)" role="option" class="ui-menu-item" :class="{ active: index === active }" :data-index="index" :aria-selected="isSelected(option.value)" :aria-disabled="option.disabled || undefined" @pointermove="enabled(index) && (active = index)" @click="pick(index)">
        <i class="ui-multi-box" :class="{ on: isSelected(option.value) }"><Check v-if="isSelected(option.value)" :size="10" :stroke-width="3" /></i><span>{{ option.label }}</span>
      </li>
    </ul>
  </Teleport>
</template>
<style>
/* 多选用勾选框表示「可以选多个」，与单选的末尾对勾区分开。 */
.ui-multi-box{flex:none;width:13px;height:13px;border:1px solid var(--ink3);border-radius:2px;display:grid;place-items:center;color:var(--panel)}
.ui-multi-box.on{background:var(--ink);border-color:var(--ink)}
</style>
