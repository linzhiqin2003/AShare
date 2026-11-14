<script setup lang="ts">
import { computed } from 'vue'
// 首屏加载占位（2026-11-14）：块（图表、卡片）或若干行（列表、表格）。只在「第一次还没有数据」时渲染，
// 数据一到由页面换成真实内容；主内容不靠透明度隐藏再显现，骨架是额外加上去的一层。
// 用法：<UiSkeleton shape="block" height="220px" />、<UiSkeleton :lines="6" />
const props = withDefaults(defineProps<{ shape?: 'block' | 'line'; lines?: number; width?: string; height?: string }>(), { shape: 'line', lines: 1 })
// 多行时宽度错落，像真实文字而不是一排等宽条。
const WIDTHS = ['100%', '82%', '94%', '68%', '88%', '76%']
const rows = computed(() => Array.from({ length: Math.max(1, props.lines) }, (_, i) => props.lines > 1 ? WIDTHS[i % WIDTHS.length] : '100%'))
</script>
<template>
  <span v-if="shape === 'block'" class="ui-skeleton block" :style="{ width, height }" aria-hidden="true" />
  <span v-else class="ui-skeleton-lines" :style="{ width }" aria-hidden="true">
    <span v-for="(rowWidth, index) in rows" :key="index" class="ui-skeleton line" :style="{ width: rowWidth, height }" />
  </span>
</template>
<style>
.ui-skeleton{position:relative;display:block;overflow:hidden;background:var(--skeleton);border-radius:3px}
.ui-skeleton.block{width:100%;height:100%;min-height:24px}
.ui-skeleton.line{height:8px;border-radius:2px}
.ui-skeleton-lines{display:flex;flex-direction:column;gap:14px;width:100%;flex:none;max-width:100%}
/* 流光：一条渐变高光从左扫到右，只动 transform。 */
.ui-skeleton::after{content:'';position:absolute;inset:0;background:linear-gradient(90deg,transparent,var(--skeleton-shine),transparent);transform:translateX(-100%);animation:ui-skeleton-shimmer 1.5s ease-in-out infinite}
@keyframes ui-skeleton-shimmer{to{transform:translateX(100%)}}
@media (prefers-reduced-motion:reduce){.ui-skeleton::after{animation:none;display:none}}
</style>
