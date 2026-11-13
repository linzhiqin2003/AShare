<script setup lang="ts">
import { onMounted, onUnmounted, shallowRef, watch } from 'vue'
import { Database, LoaderCircle, Plus, X, Play, Square, SlidersHorizontal, Clock, Check } from 'lucide-vue-next'
import { useResearch } from './useResearch'
import type { Completion, Settings, View } from './types'
import ExpressionEditor from './components/ExpressionEditor.vue'
import TabSettings from './components/TabSettings.vue'
import ResearchResults from './components/ResearchResults.vue'
import FactorLibrary from './components/FactorLibrary.vue'
import SplitPane from './components/SplitPane.vue'
import ResearchTabs from './components/ResearchTabs.vue'
const research = useResearch()
const { tabs, activeId, current, metadata, completions, defaults, elapsed, message } = research
const page = shallowRef(location.hash === '#/library' ? 'library' : 'research'), settingsOpen = shallowRef(false), signature = shallowRef<Completion | null>(null)
let debounce: ReturnType<typeof setTimeout> | undefined
function source(value: string) { if (current.value) { current.value.source = value; const tab = current.value; tab.error = null; clearTimeout(debounce); debounce = setTimeout(() => research.validate(tab), 600) } }
function settings(value: Settings) { if (current.value) { const benchmarkChanged = current.value.settings.benchmark !== value.benchmark; current.value.settings = value; void research.validate(current.value); if (benchmarkChanged) void research.loadBenchmark(current.value) } }
function select(id: string) { if (current.value) { current.value.selected = id; void research.loadChart(current.value) } }
function visible(id: string) { if (current.value) { current.value.visible = current.value.visible.includes(id) ? current.value.visible.filter(v => v !== id) : [...current.value.visible, id]; void research.loadChart(current.value) } }
function view(value: View) { if (current.value) { current.value.view = value; void research.loadChart(current.value) } }
watch(activeId, () => { settingsOpen.value = false; signature.value = null })
function navigate(value: string) { location.hash = '/' + value }
function route() { page.value = location.hash === '#/library' ? 'library' : 'research' }
let toastTimer: ReturnType<typeof setTimeout> | undefined
watch(message, value => { clearTimeout(toastTimer); if (value) toastTimer = setTimeout(() => message.value = '', 5000) })
onMounted(() => { window.addEventListener('hashchange',route); void research.init() })
onUnmounted(() => { clearTimeout(debounce); clearTimeout(toastTimer); window.removeEventListener('hashchange',route) })
</script>
<template>
  <div class="app-shell">
    <header class="topbar"><span class="brand">AShare</span><nav aria-label="主导航"><button :class="{ active: page === 'research' }" @click="navigate('research')">研究台</button><button :class="{ active: page === 'library' }" @click="navigate('library')">因子库</button></nav><div class="top-status"><span><LoaderCircle :size="13" :class="{ spinning: metadata.running > 0 }" />{{ metadata.running }} 个任务运行中</span><span><Database :size="14" />数据截至 {{ metadata.data_date ?? '—' }}</span></div></header>
    <div v-if="message" class="toast" role="alert">{{ message }}<button class="icon-button push" aria-label="关闭提示" @click="message = ''"><X :size="14" /></button></div>
    <div class="main-area">
      <main v-show="page === 'research'" class="research-page">
        <SplitPane storage-key="research-columns" label="调整编辑器宽度" :initial=".292" :min-first="310" :max-first="650" :min-second="700">
        <template #first><section class="panel editor-panel"><ResearchTabs :tabs="tabs" :active="activeId" @select="activeId = $event" @close="research.closeTab" @add="research.addTab()" @rename="research.renameTab" />
          <template v-if="current"><ExpressionEditor :key="current.id" :model-value="current.source" :completions="completions" :error="current.error" @update:model-value="source" @run="research.run(current)" @signature="signature = $event" /><div v-if="current.error" class="editor-error" role="alert">{{ current.error.message }}</div><div class="signature-line"><template v-if="signature"><b class="mono">{{ signature.signature }}</b><span>{{ signature.detail }}</span></template></div><div class="runbar"><button class="button settings-toggle" :class="{ on: settingsOpen }" @click="settingsOpen = !settingsOpen"><SlidersHorizontal :size="14" />设置</button><button v-if="current.running" class="button primary push" @click="research.cancel(current)"><Square :size="12" />取消</button><button v-else class="button primary push" :disabled="!current.count" @click="research.run(current)"><Play :size="13" />运行 {{ current.count || 0 }} 组<span class="run-key">⌘↵</span></button></div><TabSettings v-if="settingsOpen" :model-value="current.settings" :name="current.name" :defaults="defaults" @update:model-value="settings" @close="settingsOpen = false" /></template>
          <div v-else class="empty-editor"><button class="button" @click="research.addTab()"><Plus :size="14" />新建研究</button></div>
        </section></template>
        <template #second><ResearchResults v-if="current" :key="current.id" :tab="current" @select="select" @visible="visible" @view="view" @compound="current.compound = $event" @chart="research.loadChart(current)" />
        <section v-else class="panel"></section></template></SplitPane>
      </main>
      <FactorLibrary v-if="page === 'library'" :completions="completions" />
    </div>
    <footer class="bottom-bar"><span v-if="elapsed !== null"><Clock :size="12" />上次运行 {{ elapsed.toFixed(1) }} s</span></footer>
  </div>
</template>
