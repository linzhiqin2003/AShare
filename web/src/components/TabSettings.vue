<script setup lang="ts">
import { computed, shallowRef } from 'vue'
import type { Settings, Scalar } from '../types'
import RebalanceControl from './RebalanceControl.vue'
import { UiMultiSelect, UiNumberInput, UiSelect, UiSwitch, vTip } from './ui'
// 因子检验视图自己的设置（摊在编辑器下方）；成交价、基准、样本区间在共用的「样本与成交」里。
const props = defineProps<{ modelValue: Settings; defaults: Settings }>()
const emit = defineEmits<{ 'update:modelValue': [settings: Settings] }>()
const saved = shallowRef(false)
let initial: Settings | null = null
try { initial = JSON.parse(localStorage.getItem('ashare.preset') ?? 'null') } catch { /* 损坏预设不影响编辑。 */ }
const preset = shallowRef<Settings | null>(initial)
// 衰减只作用于权重组合；扫描里含权重组合时仍可设。
const decayActive = computed(() => [props.modelValue.weights_portfolio].flat().includes(true))
function update(key: string, value: Scalar | Scalar[]) { emit('update:modelValue', { ...props.modelValue, [key]: value }); saved.value = false }
/** 预设与默认只换因子检验自己的设置，共用的成交价和基准保持不变。 */
function apply(source: Settings) { emit('update:modelValue', { ...source, trade_at: props.modelValue.trade_at, ...(props.modelValue.benchmark !== undefined ? { benchmark: props.modelValue.benchmark } : {}) }); saved.value = false }
function save() { preset.value = JSON.parse(JSON.stringify(props.modelValue)); localStorage.setItem('ashare.preset', JSON.stringify(preset.value)); saved.value = true }
</script>
<template>
  <section class="settings-section" aria-label="因子检验设置">
    <h5>信号处理</h5>
    <div class="setting-row"><span>股票池</span><UiSelect :model-value="modelValue.universe" aria-label="股票池" :options="[{value:'TOP800',label:'TOP800'},{value:'TOP1800',label:'TOP1800'},{value:'ALL',label:'全市场'}]" @update:model-value="update('universe',$event)" /></div>
    <div class="setting-row"><span>子池检验</span><UiSwitch :model-value="!!modelValue.sub_universes" aria-label="子池检验" @update:model-value="update('sub_universes',$event)" /></div>
    <div class="setting-row" :class="{ inactive: !decayActive }" v-tip="decayActive ? '' : '只对权重组合生效'"><span>衰减</span><UiNumberInput :model-value="Number(modelValue.decay ?? 0)" aria-label="衰减天数" :min="0" :max="2520" unit="天" :disabled="!decayActive" @update:model-value="update('decay',$event ?? 0)" /></div>
    <div class="setting-row"><span>中性化</span><UiMultiSelect :model-value="modelValue.group_neutralize" aria-label="中性化" :options="[{value:null,label:'无'},{value:'industry',label:'行业'},{value:'cap',label:'市值'},{value:'cap+industry',label:'市值+行业'}]" @update:model-value="update('group_neutralize',$event)" /></div>
    <h5>分组检验</h5>
    <div class="setting-row"><span>调仓周期</span><RebalanceControl :model-value="modelValue.rebalance" @update:model-value="update('rebalance',$event)" /></div>
    <div class="setting-row"><span>分组</span><UiSelect :model-value="modelValue.n_groups" aria-label="分组数" :options="[{value:5,label:'5'},{value:10,label:'10'}]" @update:model-value="update('n_groups',$event)" /></div>
    <div class="setting-row"><span>组合</span><UiSelect :model-value="modelValue.weights_portfolio" aria-label="组合" :options="[{value:false,label:'分组多空'},{value:true,label:'另算权重多空'}]" @update:model-value="update('weights_portfolio',$event)" /></div>
    <div class="setting-row"><span>分段</span><UiSelect :model-value="modelValue.by" aria-label="分段" :options="[{value:'year',label:'按年'},{value:'quarter',label:'按季'}]" @update:model-value="update('by',$event)" /></div>
    <div class="setting-row"><span>涨跌停与停牌</span><UiSwitch :model-value="!!modelValue.trade_limitation" aria-label="涨跌停与停牌" @update:model-value="update('trade_limitation',$event)" /></div>
    <div class="preset-row"><button class="text-button" :disabled="!preset" @click="preset && apply(preset)">使用预设</button><button class="text-button" @click="save">{{ saved ? '已存为预设' : '存为预设' }}</button><button class="text-button" @click="apply(defaults)">恢复默认</button></div>
  </section>
</template>
<style scoped>
.preset-row{display:flex;justify-content:flex-end;gap:16px;padding:8px 0 2px;font-size:11.5px;color:var(--ink3)}.preset-row .text-button:hover:not(:disabled){color:var(--ink)}
</style>
