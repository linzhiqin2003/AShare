<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, shallowRef, watch } from 'vue'
import { Check, LoaderCircle, Maximize2, Minimize2, Play, Plus, Square, X } from 'lucide-vue-next'
import ExpressionEditor from '../components/ExpressionEditor.vue'
import BacktestConfig from '../backtest/BacktestConfig.vue'
import BacktestSide from '../backtest/BacktestSide.vue'
import BacktestTables from '../backtest/BacktestTables.vue'
import NavChart from '../backtest/NavChart.vue'
import { UiSkeleton } from '../components/ui'
import { result, state } from '../backtest/api'
import { friendly } from '../api'
import { routeEntry, useBacktests } from '../backtest/useBacktests'
import { showValidation as validation } from '../validation'
import type { BacktestDocument, BacktestTab, Part } from '../backtest/types'

const b = useBacktests()
const { tabs, activeId, current, options, completions, message, errors } = b
const part = shallowRef<Part>('is')
const doc = shallowRef<BacktestDocument | null>(null), loading = shallowRef(false), expanded = shallowRef(false)
const editing = shallowRef(''), draft = shallowRef('')
const effectivePart = computed<Part>(() => validation.value ? part.value : 'is')
const status = computed(() => current.value?.state ?? null)
const ENDED: Record<string, string> = { failed: '回测失败，请稍后重试', cancelled: '已取消', evicted: '结果已清理，重新运行即可' }
const running = computed(() => !!current.value && b.active(current.value))
// 首屏骨架：选项还没到（左侧配置、净值图、右侧统计），或已完成的回测第一次取结果；之后的刷新保留旧内容。
const booting = shallowRef(true)
const firstLoad = computed(() => !running.value && ((!options.value && booting.value) || (loading.value && !doc.value)))

let request = 0
async function load() {
  const tab = current.value, id = ++request
  if (!tab?.backtest || tab.state?.status !== 'complete') { if (!running.value) doc.value = null; return }
  loading.value = true
  try {
    const value = await result(tab.backtest, effectivePart.value)
    if (id === request) doc.value = value
  } catch (e) {
    if (id !== request) return
    // 结果可能刚被容量清理：刷新状态，改为提示重跑，而不是弹错误。
    try { tab.state = await state(tab.backtest) } catch { /* 保留原状态 */ }
    if (tab.state?.status === 'complete') message.value = friendly(e).message
    else doc.value = null
  }
  finally { if (id === request) loading.value = false }
}
watch(() => [current.value?.backtest, current.value?.state?.status, effectivePart.value] as const, load)

async function route() {
  const entry = routeEntry(location.hash)
  if (entry && options.value) {
    await b.openEntry(entry)
    // 带入一次即可；刷新页面不再重复新开 tab。
    history.replaceState(null, '', `${location.pathname}${location.search}#/backtest`)
  }
}
function key(event: KeyboardEvent) { if ((event.metaKey || event.ctrlKey) && event.key === 'Enter' && current.value) { event.preventDefault(); void b.run(current.value) } }
onMounted(async () => {
  window.addEventListener('hashchange', route)
  await b.init(); booting.value = false; await route()
})
onUnmounted(() => { b.dispose(); window.removeEventListener('hashchange', route) })

async function rename(tab: BacktestTab) { editing.value = tab.id; draft.value = tab.name; await nextTick(); (document.querySelector('.bt-rename') as HTMLInputElement | null)?.select() }
function finish(tab: BacktestTab) { if (editing.value === tab.id) { if (draft.value.trim()) tab.name = draft.value.trim(); editing.value = '' } }
const progress = (tab: BacktestTab) => tab.state?.status === 'running' && tab.state.progress !== null ? `${Math.round(tab.state.progress * 100)}%` : tab.state?.status === 'queued' ? '排队' : ''
function setSource(value: string) { if (current.value) { current.value.source = value; errors.value[current.value.id] = null } }
let toast: ReturnType<typeof setTimeout> | undefined
watch(message, value => { clearTimeout(toast); if (value) toast = setTimeout(() => message.value = '', 5000) })
</script>

<template>
  <main class="backtest-page" @keydown="key">
    <section class="panel left">
      <div class="panel-header bt-tabs" role="tablist" aria-label="回测标签">
        <div v-for="tab in tabs" :key="tab.id" class="bt-tab" :class="{ active: tab.id === activeId }">
          <input v-if="editing === tab.id" v-model="draft" class="bt-rename" aria-label="修改回测名" maxlength="80" @blur="finish(tab)" @keydown.enter="finish(tab)" @keydown.esc="editing = ''">
          <button v-else class="bt-name" role="tab" :aria-selected="tab.id === activeId" @click="activeId = tab.id" @dblclick="rename(tab)">
            <LoaderCircle v-if="b.active(tab)" :size="12" class="spinning accent" /><Check v-else-if="tab.state?.status === 'complete' && tab.id !== activeId" :size="12" class="muted" />
            {{ tab.name || '未命名' }}<em v-if="progress(tab)">{{ progress(tab) }}</em>
          </button>
          <button v-if="tabs.length > 1" class="tab-close icon-button" :aria-label="`关闭 ${tab.name || '未命名'}`" @click="b.closeTab(tab)"><X :size="11" /></button>
        </div>
        <button class="icon-button" aria-label="新建回测" :disabled="!options" @click="b.addTab()"><Plus :size="14" /></button>
      </div>
      <div v-if="current && options" class="left-body">
        <div class="signal-head"><h5>信号</h5><button class="icon-button" :aria-label="expanded ? '收起编辑器' : '展开编辑器'" @click="expanded = !expanded"><component :is="expanded ? Minimize2 : Maximize2" :size="13" /></button></div>
        <div class="editor" :class="{ expanded }"><ExpressionEditor :key="current.id" :model-value="current.source" :completions="completions" :error="errors[current.id]" @update:model-value="setSource" @run="b.run(current)" /></div>
        <div v-if="errors[current.id]" class="editor-error" role="alert">{{ errors[current.id]!.message }}</div>
        <BacktestConfig v-show="!expanded" :model-value="current.settings" :options="options" :disabled="running" @update:model-value="current.settings = $event" />
        <div class="actions">
          <button v-if="running" class="button primary grow" @click="b.cancel(current)"><Square :size="12" />取消</button>
          <button v-else class="button primary grow" :disabled="!current.source.trim()" @click="b.run(current)"><Play :size="13" />运行<span class="run-key">⌘↵</span></button>
        </div>
      </div>
      <div v-else-if="!options && booting" class="left-body bt-skeleton" aria-busy="true"><UiSkeleton shape="block" height="132px" /><UiSkeleton :lines="10" /></div>
    </section>
    <div class="middle">
      <div class="chart-wrap">
        <NavChart :document="doc" :part="effectivePart" :show-validation="validation" :loading="loading" @part="part = $event" />
        <div v-if="firstLoad" class="overlay chart-skeleton" aria-busy="true"><UiSkeleton shape="block" /></div>
        <div v-else-if="running && !doc" class="overlay muted"><LoaderCircle :size="14" class="spinning" />回测中 {{ current ? progress(current) : '' }}</div>
        <div v-else-if="status && status.status in ENDED && !doc" class="overlay muted">{{ ENDED[status.status] }}</div>
      </div>
      <BacktestTables :document="doc" :backtest="doc ? current?.backtest ?? null : null" :part="effectivePart" />
    </div>
    <BacktestSide v-if="!firstLoad" :document="doc" />
    <section v-else class="panel side-skeleton" aria-busy="true"><div class="panel-header">统计</div><div class="skeleton-body"><UiSkeleton :lines="10" /></div></section>
    <div v-if="message" class="toast" role="alert">{{ message }}<button class="icon-button push" aria-label="关闭提示" @click="message = ''"><X :size="14" /></button></div>
  </main>
</template>

<style scoped>
.backtest-page{height:100%;min-height:0;display:grid;grid-template-columns:clamp(320px,27.5vw,440px) minmax(0,1fr) clamp(290px,24.5vw,392px);gap:8px;padding:8px}
.left{overflow:visible}
.bt-tabs{gap:14px;overflow-x:auto;scrollbar-width:none;flex:none}
.bt-tab{height:100%;display:flex;align-items:center;gap:2px;border-bottom:2px solid transparent;flex:none;color:var(--ink3)}
.bt-tab.active{border-color:var(--accent);color:var(--ink)}
.bt-name{border:0;background:none;padding:0;display:flex;align-items:center;gap:6px;font-size:12px;font-weight:500;white-space:nowrap;color:inherit}
.bt-tab.active .bt-name{font-weight:600}
.bt-name em{font-style:normal;color:var(--ink3);font-size:11px;font-weight:400}
.bt-rename{width:110px;height:22px;min-height:22px;font-size:12px;padding:0 4px}
.tab-close{opacity:0}.bt-tab:hover .tab-close,.bt-tab:focus-within .tab-close{opacity:1}
.accent{color:var(--accent)}
.left-body{padding:8px 10px 10px;flex:1;min-height:0;display:flex;flex-direction:column;overflow:auto}
.signal-head{display:flex;align-items:center;justify-content:space-between}
.signal-head h5{font-size:11.5px;font-weight:600;color:var(--ink3);margin:2px 0 6px}
.editor{border:1px solid var(--line);border-radius:4px;height:132px;display:flex;flex:none;overflow:hidden}
.editor.expanded{flex:1;height:auto;min-height:200px}
.editor :deep(.cm-content){font-size:12.5px;line-height:21px!important;padding:6px 0!important}
.editor-error{color:var(--red);font-size:11px;padding:5px 2px 0}
.actions{display:flex;gap:8px;margin-top:14px}
.grow{flex:1}
.middle{display:grid;grid-template-rows:minmax(0,1fr) clamp(220px,36%,336px);gap:8px;min-height:0;min-width:0}
.chart-wrap{position:relative;min-height:0;display:flex}
.chart-wrap>:first-child{flex:1}
.overlay{position:absolute;inset:30px 0 0;display:flex;align-items:center;justify-content:center;gap:8px;font-size:12px;pointer-events:none}
.bt-skeleton{gap:18px}.chart-skeleton{align-items:stretch;padding:12px}.side-skeleton .skeleton-body{padding:12px 10px}
.toast{position:fixed;right:18px;bottom:38px;z-index:60;max-width:min(480px,calc(100vw - 36px));display:flex;align-items:center;gap:12px;padding:10px 14px;border:1px solid var(--line);border-radius:5px;background:var(--panel);box-shadow:0 6px 25px var(--overlay-shadow);color:var(--ink2);font-size:12px}
</style>
