<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, shallowRef, watch } from 'vue'
import { ChevronDown, Check, Search } from 'lucide-vue-next'
import { api, friendly } from '../api'
import type { IndexItem, IndexList } from '../types'
import { vTip } from './ui'
const props = defineProps<{ modelValue?: string }>()
const emit = defineEmits<{ 'update:modelValue': [value: string] }>()
const root = shallowRef<HTMLElement | null>(null), open = shallowRef(false), search = shallowRef(''), items = shallowRef<IndexItem[]>([])
const menuStyle = shallowRef({left:'0px',top:'0px'})
const searchInput = shallowRef<HTMLInputElement | null>(null)
async function toggle() { open.value = !open.value; if (open.value) { const rect = root.value!.getBoundingClientRect(); menuStyle.value = {left:`${Math.max(12,rect.right-300)}px`,top:`${Math.min(innerHeight-285,rect.bottom+4)}px`}; await nextTick(); searchInput.value?.focus() } }
const selected = shallowRef<IndexItem | null>(null), fallback = shallowRef(false), error = shallowRef('')
let controller: AbortController | undefined, timer: ReturnType<typeof setTimeout> | undefined
const label = computed(() => selected.value ? `${selected.value.name}${selected.value.name === selected.value.symbol ? '' : ` ${selected.value.symbol}`}` : '无本地指数')
async function load() {
  controller?.abort(); controller = new AbortController(); const active = controller
  try {
    const result = await api<IndexList>(`/benchmarks?${new URLSearchParams({ search: search.value, selected: props.modelValue ?? '' })}`, { signal: active.signal })
    if (active.signal.aborted) return
    items.value = result.items
    selected.value = result.selected.symbol ? result.selected as IndexItem : null
    fallback.value = result.default.fallback && result.selected.symbol === result.default.symbol
    error.value = ''
  } catch (e) { if (!active.signal.aborted) error.value = friendly(e).message }
}
function escapeMenu(event: KeyboardEvent) { if (open.value) { event.stopPropagation(); open.value = false } }
function choose(item: IndexItem) { selected.value = item; emit('update:modelValue', item.symbol); open.value = false }
function outside(event: PointerEvent) { if (!root.value?.contains(event.target as Node)) open.value = false }
watch(search, () => { clearTimeout(timer); timer = setTimeout(() => load(), 180) })
watch(() => props.modelValue, () => load())
onMounted(() => { void load(); document.addEventListener('pointerdown', outside) })
onUnmounted(() => { clearTimeout(timer); controller?.abort(); document.removeEventListener('pointerdown', outside) })
</script>
<template>
  <div ref="root" class="benchmark-select" @keydown.esc="escapeMenu">
    <button class="benchmark-button" aria-label="选择基准" :aria-expanded="open" @click="toggle"><span>{{ label }}</span><em v-if="fallback" v-tip="'本地没有上证指数，已选用列表第一项'">替代默认</em><ChevronDown :size="12" /></button>
    <div v-if="open" class="index-menu" :style="menuStyle">
      <label class="index-search"><Search :size="13" /><input ref="searchInput" v-model="search" placeholder="搜索指数名称或代码" aria-label="搜索基准指数"></label>
      <div class="index-options" role="listbox"><button v-for="item in items" :key="item.symbol" role="option" :aria-selected="item.symbol === modelValue" @click="choose(item)"><span>{{ item.name }}</span><span v-if="item.name !== item.symbol" class="mono muted">{{ item.symbol }}</span><Check v-if="item.symbol === modelValue" :size="12" /></button><div v-if="!items.length" class="index-empty">{{ error || '没有符合条件的指数' }}</div></div>
    </div>
  </div>
</template>
<style scoped>
.benchmark-select{position:relative;max-width:78%;min-width:0}.benchmark-button{display:flex;align-items:center;gap:6px;background:white;border:1px solid var(--line);border-radius:3px;min-height:24px;padding:2px 7px;width:100%;font-size:11.5px}.benchmark-button>span{overflow:hidden;white-space:nowrap;text-overflow:ellipsis}.benchmark-button em{font-style:normal;font-size:10px;color:var(--warn);white-space:nowrap}.index-menu{position:fixed;width:300px;max-width:calc(100vw - 48px);z-index:30;background:var(--panel);border:1px solid var(--line);border-radius:6px;box-shadow:0 12px 32px #00000024;padding:5px}.index-search{display:flex;gap:5px;align-items:center;padding:4px 7px;border-bottom:1px solid var(--grid)}.index-search input{min-width:0;width:100%;border:0;outline:0}.index-options{max-height:210px;overflow:auto}.index-options button{display:flex;align-items:center;gap:8px;width:100%;min-height:28px;padding:0 8px;border:0;border-radius:3px;background:var(--panel);text-align:left;font-size:12px}.index-options button:hover,.index-options button[aria-selected=true]{background:var(--selected)}.index-options button span:first-child{flex:1}.index-empty{padding:15px;color:var(--ink3)}
</style>
