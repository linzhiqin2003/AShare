<script setup lang="ts">
// 组件写法的提示：<UiTooltip text="…">内容</UiTooltip>，与 v-tip 指令同一实现。
import { vTip } from './tooltip'
defineProps<{ text?: string | null }>()
</script>
<template><span v-tip="text" class="ui-tooltip-anchor"><slot /></span></template>
