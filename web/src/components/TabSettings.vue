<script setup lang="ts">
import { computed, onMounted, onUnmounted, shallowRef } from 'vue'
import { X, SlidersHorizontal, ChevronDown, Plus, Minus } from 'lucide-vue-next'
import type { Settings, Scalar } from '../types'
import SegmentedControl from './SegmentedControl.vue'
import RebalanceControl from './RebalanceControl.vue'
import BenchmarkSelect from './BenchmarkSelect.vue'
const props = defineProps<{ modelValue: Settings; name: string; defaults: Settings }>()
const emit = defineEmits<{ 'update:modelValue': [settings: Settings]; close: [] }>()
const root = shallowRef<HTMLElement | null>(null), saved = shallowRef(false)
let initial: Settings | null = null
try { initial = JSON.parse(localStorage.getItem('ashare.preset') ?? 'null') } catch { /* 损坏预设不影响编辑。 */ }
const preset = shallowRef<Settings | null>(initial)
// 衰减只作用于权重组合；扫描里含权重组合时仍可设。
const decayActive = computed(() => [props.modelValue.weights_portfolio].flat().includes(true))
function update(key: string, value: Scalar | Scalar[]) { emit('update:modelValue', { ...props.modelValue, [key]: value }); saved.value = false }
function save() { preset.value = JSON.parse(JSON.stringify(props.modelValue)); localStorage.setItem('ashare.preset', JSON.stringify(preset.value)); saved.value = true }
function outside(event: PointerEvent) { const target = event.target as HTMLElement; if (!root.value?.contains(target) && !target.closest('.settings-toggle')) emit('close') }
function escape(event: KeyboardEvent) { if (event.key === 'Escape') emit('close') }
onMounted(() => { document.addEventListener('pointerdown', outside); document.addEventListener('keydown', escape) })
onUnmounted(() => { document.removeEventListener('pointerdown', outside); document.removeEventListener('keydown', escape) })
</script>
<template>
  <section ref="root" class="settings-pop" role="dialog" aria-label="tab 设置">
    <header><SlidersHorizontal :size="14" /><b class="mono">{{ name }}</b><button class="preset-button push" :disabled="!preset" @click="preset && emit('update:modelValue', { ...preset })">{{ preset ? '使用预设' : '默认' }}<ChevronDown :size="11" /></button><button class="icon-button" aria-label="关闭设置" @click="emit('close')"><X :size="15" /></button></header>
    <div class="settings-body">
      <h5>样本</h5>
      <div class="setting-row"><span>股票池</span><SegmentedControl :model-value="modelValue.universe" label="股票池" :options="[{value:'TOP800',label:'TOP800'},{value:'TOP1800',label:'TOP1800'},{value:'ALL',label:'全市场'}]" @update:model-value="update('universe',$event)" /></div>
      <div class="setting-row"><span>样本内</span><span class="date-range">2016-01-01 至 2024-12-31</span></div>
      <div class="setting-row"><span>验证</span><span class="date-range">2025 年</span></div>
      <div class="setting-row"><span>基准</span><BenchmarkSelect :model-value="typeof modelValue.benchmark === 'string' ? modelValue.benchmark : undefined" @update:model-value="update('benchmark',$event)" /></div>
      <div class="setting-row"><span>子池检验</span><button class="switch" role="switch" aria-label="子池检验" :aria-checked="!!modelValue.sub_universes" @click="update('sub_universes',!modelValue.sub_universes)"><i /></button></div>
      <h5>信号处理</h5>
      <div class="setting-row" :class="{ inactive: !decayActive }" :title="decayActive ? undefined : '只对权重组合生效'"><span>衰减</span><div class="stepper"><button aria-label="减少衰减" :disabled="!decayActive" @click="update('decay',Math.max(0,Number(modelValue.decay)-1))"><Minus :size="11" /></button><input type="number" aria-label="衰减天数" :disabled="!decayActive" min="0" max="2520" :value="modelValue.decay" @change="update('decay',Number(($event.target as HTMLInputElement).value))"><button aria-label="增加衰减" :disabled="!decayActive" @click="update('decay',Math.min(2520,Number(modelValue.decay)+1))"><Plus :size="11" /></button><span>天</span></div></div>
      <div class="setting-row"><span>中性化</span><SegmentedControl :model-value="modelValue.group_neutralize" multiple label="中性化" :options="[{value:null,label:'无'},{value:'industry',label:'行业'},{value:'cap',label:'市值'},{value:'cap+industry',label:'市值+行业'}]" @update:model-value="update('group_neutralize',$event)" /></div>
      <h5>组合与成交</h5>
      <div class="setting-row"><span>调仓周期</span><RebalanceControl :model-value="modelValue.rebalance" @update:model-value="update('rebalance',$event)" /></div>
      <div class="setting-row"><span>成交</span><SegmentedControl :model-value="modelValue.trade_at" label="成交时点" :options="[{value:'open',label:'T+1 开盘'},{value:'close',label:'T+1 收盘'}]" @update:model-value="update('trade_at',$event)" /></div>
      <div class="setting-row"><span>涨跌停与停牌</span><button class="switch" role="switch" aria-label="涨跌停与停牌" :aria-checked="!!modelValue.trade_limitation" @click="update('trade_limitation',!modelValue.trade_limitation)"><i /></button></div>
      <div class="setting-row"><span>分组</span><SegmentedControl :model-value="modelValue.n_groups" label="分组数" :options="[{value:5,label:'5'},{value:10,label:'10'}]" @update:model-value="update('n_groups',$event)" /></div>
      <div class="setting-row"><span>组合</span><SegmentedControl :model-value="modelValue.weights_portfolio" label="组合" :options="[{value:false,label:'分组多空'},{value:true,label:'另算权重多空'}]" @update:model-value="update('weights_portfolio',$event)" /></div>
      <div class="setting-row"><span>分段</span><SegmentedControl :model-value="modelValue.by" label="分段" :options="[{value:'year',label:'按年'},{value:'quarter',label:'按季'}]" @update:model-value="update('by',$event)" /></div>
    </div>
    <footer><button class="button" @click="save">{{ saved ? '已存为预设' : '存为预设' }}</button><button class="button" @click="emit('update:modelValue',{...defaults})">恢复默认</button></footer>
  </section>
</template>
<style scoped>
.settings-pop{position:absolute;left:10px;right:10px;bottom:56px;max-height:calc(100% - 96px);display:flex;flex-direction:column;border:1px solid var(--line);background:white;border-radius:6px;box-shadow:0 -10px 36px #00000024;z-index:8}.settings-pop header{height:38px;min-height:38px;padding:0 12px;display:flex;align-items:center;gap:10px;border-bottom:1px solid var(--line)}.settings-pop header b{font-weight:600;font-size:12px;overflow:hidden;text-overflow:ellipsis}.preset-button{display:flex;align-items:center;gap:8px;border:1px solid var(--line);border-radius:3px;background:white;padding:2px 6px;font-size:11px;color:var(--ink3)}.settings-body{padding:0 12px 6px;min-height:0;overflow-y:auto;overflow-x:visible}.settings-body h5{margin:9px 0 0;font-size:11.5px;font-weight:500;color:var(--ink3);height:19px;border-bottom:1px solid var(--line)}.setting-row{display:flex;align-items:center;min-height:29px;border-bottom:1px solid var(--grid);justify-content:space-between;gap:5px;color:var(--ink2);font-size:12px}.date-range{color:var(--ink);font-size:11.5px}.switch{width:28px;height:16px;border:0;border-radius:10px;padding:2px;background:var(--line);flex:none}.switch i{display:block;width:12px;height:12px;background:white;border-radius:50%;box-shadow:0 0 1px #0003}.switch[aria-checked=true]{background:var(--ink)}.switch[aria-checked=true] i{transform:translateX(12px)}.stepper{display:flex;align-items:center;height:23px}.stepper input{width:45px;min-height:23px;height:23px;border-radius:0;text-align:center;font-family:var(--mono);appearance:textfield;padding:0;border-left:0;border-right:0}.stepper input::-webkit-inner-spin-button{appearance:none}.stepper button{height:23px;width:24px;display:grid;place-items:center;background:white;border:1px solid var(--line);color:var(--ink3)}.stepper button:first-child{border-radius:3px 0 0 3px}.stepper button:nth-child(3){border-radius:0 3px 3px 0}.stepper span{padding-left:8px;color:var(--ink3)}.setting-row.inactive>span,.setting-row.inactive .stepper{opacity:.4}.stepper :disabled{cursor:not-allowed}.settings-pop footer{display:flex;gap:8px;padding:9px 12px;background:var(--chrome);border-top:1px solid var(--line);border-radius:0 0 6px 6px;flex:none}
</style>
