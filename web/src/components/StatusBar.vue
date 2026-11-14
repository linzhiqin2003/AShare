<script setup lang="ts">
import { computed, onMounted, onUnmounted, shallowRef } from 'vue'
import { Clock } from 'lucide-vue-next'
import { api } from '../api'
import { heatColor, signedPercent, type MarketSummary } from '../market'
defineProps<{ elapsed: number | null }>()
const market = shallowRef<MarketSummary | null>(null)
const industries = computed(() => market.value?.industries.filter(i => i.change !== null) ?? [])
const scale = computed(() => Math.max(0.005, ...industries.value.map(i => Math.abs(i.change ?? 0))))
let timer: ReturnType<typeof setInterval> | undefined
async function load() { try { market.value = await api<MarketSummary>('/market') } catch { /* 状态栏取不到行情时只是不显示，不打扰。 */ } }
onMounted(() => { void load(); timer = setInterval(load, 300_000) })
onUnmounted(() => clearInterval(timer))
</script>
<template>
  <footer class="bottom-bar">
    <div v-if="market?.date" class="tape">
      <span class="closed"><i />已收盘 {{ market.date.slice(5) }}</span>
      <span v-for="item in market.indexes" :key="item.symbol" class="ix"><em>{{ item.name }}</em><b>{{ item.close.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) }}</b><b :class="item.change === null ? '' : item.change > 0 ? 'rise' : item.change < 0 ? 'fall' : ''">{{ signedPercent(item.change) }}</b></span>
      <span v-if="market.breadth" class="ix"><em>涨</em><b class="rise">{{ market.breadth.up }}</b><em>跌</em><b class="fall">{{ market.breadth.down }}</b></span>
      <span v-if="industries.length > 1" class="ix heat" :aria-label="`行业涨跌：${industries.map(i => i.name + ' ' + signedPercent(i.change)).join('，')}`">
        <em>{{ industries[0].name }}</em><b :class="(industries[0].change ?? 0) >= 0 ? 'rise' : 'fall'">{{ signedPercent(industries[0].change) }}</b>
        <span class="cells"><i v-for="item in industries" :key="item.name" :style="{ background: heatColor(item.change ?? 0, scale) }" /></span>
        <em>{{ industries[industries.length - 1].name }}</em><b :class="(industries[industries.length - 1].change ?? 0) >= 0 ? 'rise' : 'fall'">{{ signedPercent(industries[industries.length - 1].change) }}</b>
      </span>
    </div>
    <span v-if="elapsed !== null" class="last-run"><Clock :size="12" />上次运行 {{ elapsed.toFixed(1) }} s</span>
  </footer>
</template>
<style scoped>
.bottom-bar{height:24px;flex:none;border-top:1px solid var(--line);background:var(--panel);display:flex;align-items:center;gap:20px;padding:0 12px;color:var(--ink3);font-size:11.5px;overflow:hidden}
.tape{display:flex;align-items:center;gap:22px;white-space:nowrap;min-width:0;overflow:hidden}
.tape>span{flex:none;display:flex;align-items:center;gap:7px}
.closed{gap:6px!important}.closed i{width:7px;height:7px;border-radius:50%;background:var(--ink3);opacity:.6}
.tape em{font-style:normal;color:var(--ink2)}
.tape b{font-family:var(--mono);font-weight:500;color:var(--ink);font-size:11.5px}
.tape b.rise{color:var(--rise)}.tape b.fall{color:var(--fall)}
.cells{display:flex;gap:1px}.cells i{width:7px;height:13px;display:block}
.last-run{margin-left:auto;display:flex;align-items:center;gap:6px;flex:none}
@media (max-width:1380px){.heat .cells{display:none}}
</style>
