<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, shallowRef, watch } from 'vue'
import { Check, LoaderCircle, Maximize2, Minimize2, Play, Plus, Square, X } from 'lucide-vue-next'
import ExpressionEditor from '../components/ExpressionEditor.vue'
import BacktestConfig from '../backtest/BacktestConfig.vue'
import BacktestSide from '../backtest/BacktestSide.vue'
import BacktestTables from '../backtest/BacktestTables.vue'
import NavChart from '../backtest/NavChart.vue'
import { result, state } from '../backtest/api'
import { friendly } from '../api'
import { batchProgress, routeEntry, selectedGroup, useBacktests } from '../backtest/useBacktests'
import { showValidation as validation } from '../validation'
import type { BacktestDocument, BacktestTab, Part } from '../backtest/types'

const b = useBacktests()
const { tabs, activeId, current, options, completions, message, errors } = b
const part = shallowRef<Part>('is')
const doc = shallowRef<BacktestDocument | null>(null), loading = shallowRef(false), expanded = shallowRef(false)
const editing = shallowRef(''), draft = shallowRef('')
const effectivePart = computed<Part>(() => validation.value ? part.value : 'is')
const group = computed(() => selectedGroup(current.value))
const status = computed(() => group.value?.state ?? null)
const ENDED: Record<string, string> = { failed: '回测失败，请稍后重试', cancelled: '已取消', evicted: '结果已清理，重新运行即可' }
const running = computed(() => !!current.value && b.active(current.value))
// 已完成各组同一窗口的文档，按「身份:窗口」存；净值图把它们叠在选中组下面。
const groupDocs = shallowRef<Record<string, BacktestDocument>>({})
const chartGroups = computed(() => (current.value?.groups ?? []).map(g => ({
  id: g.id, label: g.label, status: g.state?.status ?? null, document: groupDocs.value[`${g.id}:${effectivePart.value}`] ?? null,
})))

let request = 0, docGroup = ''
async function load() {
  const tab = current.value, id = ++request, selected = group.value
  void loadOthers(id)
  if (!tab || !selected || selected.state?.status !== 'complete') {
    // 重跑期间保留上一次的图；在同一批里切到还没算完或失败的组时不保留别组的图。
    if (!running.value || tab?.groups.some(g => g.id === docGroup)) doc.value = null
    return
  }
  loading.value = true
  try {
    const value = await result(selected.id, effectivePart.value)
    if (id === request) { doc.value = value; docGroup = selected.id }
  } catch (e) {
    if (id !== request) return
    // 结果可能刚被容量清理：刷新状态，改为提示重跑，而不是弹错误。
    try { selected.state = await state(selected.id) } catch { /* 保留原状态 */ }
    if (selected.state?.status === 'complete') message.value = friendly(e).message
    else doc.value = null
  }
  finally { if (id === request) loading.value = false }
}
/** 其余已完成组的文档（结果按身份缓存，切换选中组时不再下载）；读不到的组只是不画。 */
async function loadOthers(id: number) {
  const part = effectivePart.value
  const done = (current.value?.groups ?? []).filter(g => g.state?.status === 'complete')
  if (done.length < 2) return
  const loaded = await Promise.all(done.map(g => result(g.id, part).then(d => [`${g.id}:${part}`, d] as const, () => null)))
  if (id !== request) return
  groupDocs.value = Object.fromEntries(loaded.filter(x => x !== null))
}
watch(() => [current.value?.id, group.value?.id, current.value?.groups.map(g => g.state?.status).join(), effectivePart.value] as const, load)

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
  await b.init(); await route()
})
onUnmounted(() => { b.dispose(); window.removeEventListener('hashchange', route) })

async function rename(tab: BacktestTab) { editing.value = tab.id; draft.value = tab.name; await nextTick(); (document.querySelector('.bt-rename') as HTMLInputElement | null)?.select() }
function finish(tab: BacktestTab) { if (editing.value === tab.id) { if (draft.value.trim()) tab.name = draft.value.trim(); editing.value = '' } }
const progress = (tab: BacktestTab) => batchProgress(tab.groups)
const finished = (tab: BacktestTab) => !b.active(tab) && tab.groups.some(g => g.state?.status === 'complete')
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
            <LoaderCircle v-if="b.active(tab)" :size="12" class="spinning accent" /><Check v-else-if="finished(tab) && tab.id !== activeId" :size="12" class="muted" />
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
    </section>
    <div class="middle">
      <div class="chart-wrap">
        <NavChart :document="doc" :part="effectivePart" :show-validation="validation" :loading="loading" :groups="chartGroups" :selected="group?.id ?? null" @part="part = $event" @select="current && b.select(current, $event)" />
        <div v-if="running && !doc" class="overlay muted"><LoaderCircle :size="14" class="spinning" />回测中 {{ current ? progress(current) : '' }}</div>
        <div v-else-if="status && status.status in ENDED && !doc" class="overlay muted">{{ ENDED[status.status] }}</div>
      </div>
      <BacktestTables :document="doc" :backtest="doc ? group?.id ?? null : null" :part="effectivePart" />
    </div>
    <BacktestSide :document="doc" />
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
.toast{position:fixed;right:18px;bottom:38px;z-index:60;max-width:min(480px,calc(100vw - 36px));display:flex;align-items:center;gap:12px;padding:10px 14px;border:1px solid var(--line);border-radius:5px;background:var(--panel);box-shadow:0 6px 25px #0002;color:var(--ink2);font-size:12px}
</style>
