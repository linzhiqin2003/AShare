<script setup lang="ts">
import { computed, shallowRef } from 'vue'
import { colors, fmt, variantLabel } from '../chart'
import type { ResearchTab } from '../useResearch'
import ResultChart from './ResultChart.vue'
import CheckTable from './CheckTable.vue'
import SplitPane from './SplitPane.vue'
import type { ReturnSelection, View } from '../types'
const props = defineProps<{ tab: ResearchTab }>()
const emit = defineEmits<{ chart: []; select: [id: string]; visible: [id: string]; view: [view: View]; mode: [value: string]; returns: [value: ReturnSelection] }>()
const tableView = shallowRef('parameters')
const selected = computed(() => props.tab.job?.runs.find(r => r.id === props.tab.selected))
const checks = computed(() => selected.value?.summary?.checks ?? [])
const counts = computed(() => ['pass', 'fail', 'hint'].map(s => checks.value.filter(c => c.status === s).length))
const hasSubs = computed(() => !!selected.value?.summary && Object.keys(selected.value.summary.is.subs).length > 0)
const views = computed(() => ([{ key: 'returns', label: '收益' }, { key: 'groups', label: '分组' }, { key: 'ic', label: 'IC' }, ...(hasSubs.value ? [{ key: 'subs', label: '子池' }] : [])] as { key: View; label: string }[]))
</script>
<template>
  <SplitPane class="research-results" storage-key="research-rows" label="调整图表高度" axis="vertical" :initial=".64" :min-first="290" :max-first="900" :min-second="225">
    <template #first>
    <section class="panel chart-panel">
      <div class="panel-header">
        <button v-for="view in views" :key="view.key" class="panel-tab" :class="{ active: tab.view === view.key }" @click="emit('view', view.key)">{{ view.label }}</button>
        <span v-if="tab.view === 'returns' && tab.benchmark?.symbol && !tab.relative" class="benchmark-legend push"><i></i>{{ tab.benchmark.name }}</span>
        <select v-if="tab.view === 'returns' && tab.relative" class="curve-selection push" aria-label="超额收益曲线" :value="tab.returnSelection" @change="emit('returns', ($event.target as HTMLSelectElement).value as ReturnSelection)">
          <option value="both">多空 + 第 1 组</option><option value="long_short">多空</option><option value="long">第 1 组</option>
        </select>
        <select v-if="tab.view !== 'ic'" class="push" aria-label="收益累计方式" :value="`${tab.view === 'returns' && tab.relative ? 'relative-' : ''}${tab.compound ? 'compound' : 'sum'}`" @change="emit('mode', ($event.target as HTMLSelectElement).value)">
          <option value="sum">累加</option><option value="compound">复利</option>
          <option v-if="tab.view === 'returns'" value="relative-sum" :disabled="!tab.benchmark?.symbol">相对基准 · 累加</option>
          <option v-if="tab.view === 'returns'" value="relative-compound" :disabled="!tab.benchmark?.symbol">相对基准 · 复利</option>
        </select>
      </div>
      <ResultChart :data="tab.chart" :compound="tab.compound" :relative="tab.view === 'returns' && tab.relative" :raw="tab.view === 'ic'" :selected-id="tab.selected" :drawdown-line="tab.drawdown" />
    </section>
    </template><template #second><SplitPane class="results-bottom" storage-key="research-tables" label="调整参数组与检验宽度" :initial=".595" :min-first="350" :max-first="1200" :min-second="330"><template #first>
      <section class="panel"><div class="panel-header"><button v-for="[key,label] in [['parameters','参数组'],['years','分年'],['correlations','相关'],['exposure','风格暴露']]" :key="key" class="panel-tab" :class="{ active: tableView === key }" @click="tableView = key">{{ label }}</button></div>
        <div class="table-scroll">
          <table v-if="tableView === 'parameters'" class="data-table parameters"><thead><tr><th></th><th>参数</th><th>Sharpe</th><th>验证</th><th>年化收益</th><th>最大回撤</th><th>换手</th><th>Margin</th><th>t 值</th><th v-if="tab.benchmark?.symbol" :title="`相对${tab.benchmark.name}：多头第 1 组的年化超额收益`">年化超额</th><th v-if="tab.benchmark?.symbol" :title="`相对${tab.benchmark.name}的信息比率（IR）`">信息比率</th></tr></thead><tbody><tr v-for="run in tab.job?.runs ?? []" :key="run.id" :class="{ selected: run.id === tab.selected }" tabindex="0" @click="emit('select', run.id)" @keydown.enter="emit('select', run.id)"><td><input type="checkbox" :aria-label="`显示 ${variantLabel(run.label)}`" :checked="tab.visible.includes(run.id)" @click.stop @change="emit('visible', run.id)"></td><td class="nowrap"><i class="line-key" :style="{ background: colors[run.ordinal % colors.length] }"></i>{{ variantLabel(run.label) }}</td><td class="number">{{ fmt(run.summary?.is.sharpe) }}</td><td class="number">{{ fmt(run.summary?.validation.sharpe) }}</td><td class="number">{{ fmt(run.summary?.is.return, true,1) }}</td><td class="number">{{ fmt(run.summary?.is.maxdrawdown, true,1) }}</td><td class="number">{{ fmt(run.summary?.is.turnover, true,1) }}</td><td class="number">{{ fmt(run.summary?.is.margin, false, 1) }}</td><td class="number">{{ fmt(run.summary?.is.t) }}</td><td v-if="tab.benchmark?.symbol" class="number relative-value" :title="tab.benchmark.name">{{ fmt(tab.benchmark.relatives[run.id]?.is.excess,true,1) }}</td><td v-if="tab.benchmark?.symbol" class="number relative-value" :title="tab.benchmark.name">{{ fmt(tab.benchmark.relatives[run.id]?.is.information_ratio) }}</td></tr></tbody></table>
          <table v-else-if="tableView === 'years'" class="data-table"><thead><tr><th>{{ selected?.settings.by === 'quarter' ? '季度' : '年份' }}</th><th>Sharpe</th><th>收益</th><th>回撤</th><th>换手</th><th>Margin</th></tr></thead><tbody><tr v-for="period in [...(selected?.summary?.is.periods ?? []), ...(selected?.summary?.validation.periods ?? [])]" :key="period.period" :class="{ validation: period.period >= '2025' }"><td class="mono">{{ period.period }}{{ period.period >= '2025' ? ' 验证' : '' }}</td><td class="number">{{ fmt(period.sharpe) }}</td><td class="number">{{ fmt(period.return, true) }}</td><td class="number">{{ fmt(period.maxdrawdown, true) }}</td><td class="number">{{ fmt(period.turnover, true) }}</td><td class="number">{{ fmt(period.margin, false, 1) }}</td></tr></tbody></table>
          <table v-else-if="tableView === 'correlations'" class="data-table"><thead><tr><th>库内因子</th><th>相关</th><th>Sharpe</th></tr></thead><tbody><tr v-for="item in selected?.summary?.correlations ?? []" :key="item.entry"><td class="mono">{{ item.name }}</td><td class="number">{{ fmt(item.value) }}</td><td class="number">{{ fmt(item.sharpe) }}</td></tr></tbody></table>
          <table v-else class="data-table"><thead><tr><th>风格暴露</th><th>均值</th></tr></thead><tbody><tr v-for="(value,key) in selected?.summary?.is.exposure ?? {}" :key="key"><td>{{ ({size:'市值',reversal_20:'20 日反转',reversal_60:'60 日反转',momentum:'动量 240–20'} as Record<string,string>)[key] }}</td><td class="number">{{ fmt(value, false, 3) }}</td></tr></tbody></table>
        </div>
      </section></template><template #second>
      <section class="panel checks-panel"><div class="panel-header"><span>检验</span><span class="muted mono">{{ variantLabel(selected?.label ?? '') }}</span><span v-if="checks.length" class="push muted check-count">{{ counts[0] }} 项通过 · {{ counts[1] }} 项未通过 · {{ counts[2] }} 项提示</span></div><div class="table-scroll"><CheckTable :checks="checks" /></div></section>
    </template></SplitPane></template>
  </SplitPane>
</template>
<style scoped>
.chart-panel{min-height:0}.chart-panel>.result-chart{flex:1;height:0;min-height:0}.parameters td{height:29px;font-size:11px;padding-left:3px}.parameters th{font-size:10.5px;white-space:normal;overflow-wrap:anywhere;padding-left:3px}.parameters th:first-child{width:22px}.parameters th:nth-child(2){width:22%}.parameters td.number{padding-left:3px}.parameters .relative-value{font-size:10.5px}.line-key{display:inline-block;width:12px;height:3px;margin-right:5px;vertical-align:middle}.check-count{font-size:10px;white-space:nowrap}.validation{background:var(--shade)}.benchmark-legend{display:flex;align-items:center;gap:6px;font-size:11px;color:var(--ink3);font-weight:400;white-space:nowrap;max-width:190px;overflow:hidden;text-overflow:ellipsis}.benchmark-legend i{display:inline-block;width:16px;border-top:1px dashed var(--ink3)}.benchmark-legend+select,.curve-selection+select{margin-left:0!important}.checks-panel{container-type:inline-size}.checks-panel .panel-header{gap:8px}.checks-panel .panel-header>span:nth-child(2){font-size:11px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}@container(max-width:460px){.check-count{display:none}}
</style>
