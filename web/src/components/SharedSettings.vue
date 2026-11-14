<script setup lang="ts">
import type { Scalar, Settings } from '../types'
import BenchmarkSelect from './BenchmarkSelect.vue'
import { setShowValidation, showValidation } from '../validation'
import { UiSelect, UiSwitch } from './ui'
// 「样本与成交」：因子检验和策略回测共用的设置，只出现这一次。成交价和基准存在研究设置里，回测提交时用同一份。
const props = defineProps<{ modelValue: Settings }>()
const emit = defineEmits<{ 'update:modelValue': [settings: Settings] }>()
function update(key: string, value: Scalar | Scalar[]) { emit('update:modelValue', { ...props.modelValue, [key]: value }) }
</script>
<template>
  <section class="settings-section" aria-label="样本与成交">
    <h5>样本与成交</h5>
    <div class="setting-row"><span>样本内</span><span class="setting-value mono">2016-01-01 至 2024-12-31</span></div>
    <div class="setting-row"><span>验证 <span class="setting-value mono">2025 年</span></span><UiSwitch :model-value="showValidation" aria-label="显示验证期" @update:model-value="setShowValidation" /></div>
    <div class="setting-row"><span>成交</span><UiSelect :model-value="modelValue.trade_at" aria-label="成交时点" :options="[{value:'open',label:'T+1 开盘'},{value:'close',label:'T+1 收盘'}]" @update:model-value="update('trade_at',$event)" /></div>
    <div class="setting-row"><span>基准</span><BenchmarkSelect :model-value="typeof modelValue.benchmark === 'string' ? modelValue.benchmark : undefined" @update:model-value="update('benchmark',$event)" /></div>
  </section>
</template>
