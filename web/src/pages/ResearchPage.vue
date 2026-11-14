<script setup lang="ts">
import { computed, onMounted, onUnmounted, shallowRef, watch } from 'vue'
import { ChartLine, CircleAlert, LoaderCircle, Play, Plus, Sigma, Square } from 'lucide-vue-next'
import type { Mode, useResearch } from '../useResearch'
import type { Completion, ReturnSelection, Settings, View } from '../types'
import { colors, variantLabel } from '../chart'
import { UiSkeleton } from '../components/ui'
import ExpressionEditor from '../components/ExpressionEditor.vue'
import SharedSettings from '../components/SharedSettings.vue'
import TabSettings from '../components/TabSettings.vue'
import ResearchResults from '../components/ResearchResults.vue'
import ResearchTabs from '../components/ResearchTabs.vue'
import SplitPane from '../components/SplitPane.vue'
import BacktestConfig from '../backtest/BacktestConfig.vue'
import BacktestResults from '../backtest/BacktestResults.vue'

// 研究页（2026-11-14 研究台与回测合并，方向 A：视图切换）。左栏一套 tab：编辑器在上，设置摊在下面；
// 结果区顶部切换「因子检验｜策略回测」和选中组。两个视图共用表达式与「样本与成交」，各自的结果分别保存在 tab 上。
const props = defineProps<{ research: ReturnType<typeof useResearch> }>()
const r = props.research
const { tabs, activeId, current, completions, defaults, loaded } = r
const options = r.backtests.options
const signature = shallowRef<Completion | null>(null)
let debounce: ReturnType<typeof setTimeout> | undefined
function source(value: string) { if (current.value) { current.value.source = value; const tab = current.value; tab.error = null; tab.backtest.error = null; clearTimeout(debounce); debounce = setTimeout(() => r.validate(tab), 600) } }
function settings(value: Settings) { if (current.value) { const benchmarkChanged = current.value.settings.benchmark !== value.benchmark; current.value.settings = value; void r.validate(current.value); if (benchmarkChanged) void r.loadBenchmark(current.value) } }
function select(id: string) { if (current.value) { current.value.selected = id; void r.loadChart(current.value) } }
function visible(id: string) { if (current.value) { current.value.visible = current.value.visible.includes(id) ? current.value.visible.filter(v => v !== id) : [...current.value.visible, id]; void r.loadChart(current.value) } }
function view(value: View) { if (current.value) { current.value.view = value; void r.loadChart(current.value) } }
function mode(value: string) { if (current.value) { current.value.compound = value.endsWith('compound'); current.value.relative = value.startsWith('relative-'); void r.loadChart(current.value) } }
function returns(value: ReturnSelection) { if (current.value) { current.value.returnSelection = value; void r.loadChart(current.value) } }
function setMode(value: Mode) { if (current.value) current.value.mode = value }
watch(activeId, () => { signature.value = null })

const MODES: { value: Mode; label: string; icon: typeof Sigma }[] = [{ value: 'factor', label: '因子检验', icon: Sigma }, { value: 'backtest', label: '策略回测', icon: ChartLine }]
const backtest = computed(() => current.value?.mode === 'backtest')
const busy = computed(() => !!current.value && r.busy(current.value))
const count = computed(() => (current.value ? (backtest.value ? current.value.backtest.count : current.value.count) : 0))
const runnable = computed(() => !!current.value && count.value > 0 && (!backtest.value || (!!options.value && !!current.value.source.trim())))
const editorError = computed(() => (current.value ? (backtest.value ? current.value.backtest.error : current.value.error) : null))
/** 结果区顶栏的组名：因子检验是本次运行的各组（与图上的线同色），策略回测是候选列表展开的各组。 */
const groups = computed(() => {
  const tab = current.value
  if (!tab) return []
  if (tab.mode === 'backtest') return tab.backtest.groups.map(g => ({ id: g.id, label: g.label, color: '', status: g.state?.status ?? null, active: g.id === tab.backtest.selected }))
  return (tab.job?.runs ?? []).map(run => ({ id: run.id, label: variantLabel(run.label).replace(/ · 调仓\d+天$/, ''), color: colors[run.ordinal % colors.length], status: run.status, active: run.id === tab.selected }))
})
function pickGroup(id: string) {
  const tab = current.value
  if (!tab) return
  if (tab.mode === 'backtest') r.backtests.select(tab.backtest, id)
  else select(id)
}
const running = (status: string | null) => status === 'queued' || status === 'running'

/** 合并前的 `#/backtest` 链接（含因子库「送去回测」）：研究页恢复完 tab 后处理，再把地址换成 `#/research`。 */
async function route() {
  if (!loaded.value) return
  if (await r.openRoute(location.hash)) history.replaceState(null, '', `${location.pathname}${location.search}#/research`)
}
watch(loaded, value => { if (value) void route() })
onMounted(() => window.addEventListener('hashchange', route))
onUnmounted(() => { clearTimeout(debounce); window.removeEventListener('hashchange', route) })
</script>

<template>
  <main class="research-page">
    <SplitPane storage-key="research-columns" label="调整编辑器宽度" :initial=".292" :min-first="310" :max-first="650" :min-second="700">
      <template #first>
        <section class="panel editor-panel">
          <ResearchTabs :tabs="tabs" :active="activeId" @select="activeId = $event" @close="r.closeTab" @add="r.addTab()" @rename="r.renameTab" />
          <template v-if="current">
            <div class="editor-box"><ExpressionEditor :key="current.id" :model-value="current.source" :completions="completions" :error="editorError" @update:model-value="source" @run="r.runCurrent(current)" @signature="signature = $event" /></div>
            <div v-if="editorError" class="editor-error" role="alert">{{ editorError.message }}</div>
            <div class="signature-line"><template v-if="signature"><b class="mono">{{ signature.signature }}</b><span>{{ signature.detail }}</span></template></div>
            <div class="settings-scroll">
              <SharedSettings :model-value="current.settings" @update:model-value="settings" />
              <TabSettings v-if="!backtest" :model-value="current.settings" :defaults="defaults" @update:model-value="settings" />
              <BacktestConfig v-else-if="options" :model-value="current.backtest.settings" :options="options" :disabled="busy" @update:model-value="current.backtest.settings = $event" />
            </div>
            <div class="runbar">
              <button v-if="busy" class="button primary grow" @click="r.cancelCurrent(current)"><Square :size="12" />取消</button>
              <button v-else class="button primary grow" :disabled="!runnable" @click="r.runCurrent(current)"><Play :size="13" />运行 {{ count || 0 }} 组<span class="run-key">⌘↵</span></button>
            </div>
          </template>
          <div v-else-if="!loaded" class="editor-skeleton" aria-busy="true"><UiSkeleton :lines="3" width="72%" /></div>
          <div v-else class="empty-editor"><button class="button" @click="r.addTab()"><Plus :size="14" />新建研究</button></div>
        </section>
      </template>
      <template #second>
        <div class="results-column">
          <div v-if="current && loaded" class="panel results-head">
            <div class="view-switch" role="tablist" aria-label="视图">
              <button v-for="m in MODES" :key="m.value" role="tab" :aria-selected="current.mode === m.value" :class="{ on: current.mode === m.value }" @click="setMode(m.value)"><component :is="m.icon" :size="14" />{{ m.label }}</button>
            </div>
            <div v-if="groups.length > 1" class="group-tabs" role="tablist" aria-label="组">
              <button v-for="g in groups" :key="g.id" class="group-tab mono" role="tab" :aria-selected="g.active" :class="{ active: g.active, failed: g.status === 'failed' }" @click="pickGroup(g.id)">
                <LoaderCircle v-if="running(g.status)" :size="11" class="spinning" /><CircleAlert v-else-if="g.status === 'failed'" :size="11" /><i v-else :style="g.color ? { background: g.color } : undefined" />{{ g.label }}
              </button>
            </div>
          </div>
          <div class="results-body">
            <template v-if="current && loaded">
              <BacktestResults v-if="backtest" :key="current.id" :run="current.backtest" :running="r.backtests.active(current.backtest)" @message="r.message.value = $event" />
              <ResearchResults v-else :key="current.id" :tab="current" @select="select" @visible="visible" @view="view" @mode="mode" @returns="returns" @chart="r.loadChart(current)" />
            </template>
            <section v-else-if="!loaded" class="panel results-skeleton" aria-busy="true"><UiSkeleton shape="block" height="58%" /><UiSkeleton :lines="5" /></section>
            <section v-else class="panel"></section>
          </div>
        </div>
      </template>
    </SplitPane>
  </main>
</template>

<style scoped>
.editor-panel{display:flex;flex-direction:column}
.editor-box{flex:1 1 150px;min-height:110px;display:flex;overflow:hidden}
.editor-box>:first-child{flex:1}
.settings-scroll{flex:0 1 auto;min-height:120px;overflow-y:auto;border-top:1px solid var(--line);padding-bottom:6px}
.runbar .grow{flex:1;height:30px}
.results-column{height:100%;min-height:0;display:flex;flex-direction:column;gap:8px}
.results-body{flex:1;min-height:0;display:flex;flex-direction:column}
.results-body>*{flex:1;min-height:0}
.results-head{flex:none;height:40px;flex-direction:row;align-items:center;gap:22px;padding:0 8px}
.view-switch{display:flex;padding:2px;border-radius:5px;background:var(--chrome);border:1px solid var(--line);flex:none}
.view-switch button{display:flex;align-items:center;gap:7px;height:26px;padding:0 14px;border:0;border-radius:4px;background:none;color:var(--ink3);font-size:12.5px;font-weight:500;white-space:nowrap}
.view-switch button:hover{color:var(--ink)}
.view-switch button.on{background:var(--panel);color:var(--ink);font-weight:600;box-shadow:0 0 0 1px var(--line)}
.group-tabs{display:flex;align-items:stretch;gap:22px;height:100%;min-width:0;overflow-x:auto;scrollbar-width:none;padding-left:22px;border-left:1px solid var(--line)}
.group-tab{display:flex;align-items:center;gap:7px;border:0;border-bottom:2px solid transparent;background:none;padding:0 2px;font-size:12.5px;color:var(--ink3);white-space:nowrap}
.group-tab:hover{color:var(--ink2)}
.group-tab i{width:13px;height:2.5px;display:block;background:var(--ink3);opacity:.6}
.group-tab.active{color:var(--ink);font-weight:600;border-bottom-color:var(--ink)}
.group-tab.active i{opacity:1}
.group-tab.active i:not([style]){background:var(--ink)}
.group-tab.failed{color:var(--red)}
</style>
