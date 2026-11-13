<script setup lang="ts">
import { nextTick, shallowRef } from 'vue'
import { Check, LoaderCircle, Plus, X } from 'lucide-vue-next'
import type { ResearchTab } from '../useResearch'
defineProps<{ tabs: ResearchTab[]; active: string }>()
const emit = defineEmits<{ select: [id: string]; close: [tab: ResearchTab]; add: []; rename: [tab: ResearchTab, name: string] }>()
const editing = shallowRef(''), name = shallowRef(''), input = shallowRef<HTMLInputElement | null>(null)
async function edit(tab: ResearchTab) { editing.value = tab.id; name.value = tab.name; await nextTick(); input.value?.focus(); input.value?.select() }
function finish(tab: ResearchTab) { if (editing.value !== tab.id) return; const value = name.value.trim(); if (value) emit('rename',tab,value); editing.value = '' }
</script>
<template>
  <div class="panel-header editor-tabs" role="tablist" aria-label="研究标签">
    <div v-for="tab in tabs" :key="tab.id" class="editor-tab" :class="{ active:tab.id === active }">
      <input v-if="editing === tab.id" :ref="el => input = el as HTMLInputElement" v-model="name" class="tab-rename mono" aria-label="修改因子名" maxlength="80" @blur="finish(tab)" @keydown.enter="finish(tab)" @keydown.esc="editing = ''">
      <button v-else class="tab-name mono" role="tab" :aria-selected="tab.id === active" @click="emit('select',tab.id)" @dblclick="edit(tab)"><LoaderCircle v-if="tab.running" :size="12" class="spinning" /><Check v-else-if="tab.job?.status === 'complete'" :size="12" class="green" />{{ tab.name }}<span v-if="tab.running" class="muted">{{ tab.progress }}%</span></button>
      <button class="tab-close icon-button" :aria-label="`关闭 ${tab.name}`" @click="emit('close',tab)"><X :size="11" /></button>
    </div>
    <button class="icon-button" aria-label="新增研究 tab" @click="emit('add')"><Plus :size="15" /></button>
  </div>
</template>
<style scoped>.tab-rename{width:115px;min-width:50px;height:23px;min-height:23px;font-size:12px;padding:0 3px}.editor-tabs{scrollbar-width:none}.editor-tabs::-webkit-scrollbar{display:none}</style>
