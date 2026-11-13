<script setup lang="ts">
import { computed, onUnmounted, shallowRef, watch } from 'vue'
import { api, friendly } from '../api'
import { fmt } from '../chart'
const props = defineProps<{ run?: string; date?: string; groups?: number }>()
interface Position { symbol:string; name:string; industry:string; factor:number|null; weight:number; previous_weight?:number; side:string; status:string }
interface Day { date:string; snapshot:string; signal_date:string; previous:string|null; items:Position[]; total:number }
const dates = shallowRef<string[]>([]), selected = shallowRef(''), compare = shallowRef(false), search = shallowRef(''), side = shallowRef('long')
const data = shallowRef<Day|null>(null), error = shallowRef('')
let sequence = 0, controller:AbortController|undefined
const items = computed(() => data.value?.items.filter(r => r.side === side.value && `${r.symbol} ${r.name} ${r.industry}`.toLowerCase().includes(search.value.toLowerCase())) ?? [])
function nearest(date?:string) { if (!date || !dates.value.length) return dates.value.at(-1) ?? ''; return dates.value.reduce((best,d) => Math.abs(Number(d)-Number(date)) < Math.abs(Number(best)-Number(date)) ? d : best, dates.value[0]) }
watch(() => props.run, async run => {
  const request = ++sequence; controller?.abort(); controller = new AbortController(); data.value = null; dates.value = []; selected.value = ''
  if (!run) return
  try { const result = await api<{dates:string[];date:string|null}>(`/runs/${run}/holdings`, {signal:controller.signal}); if (request === sequence) { dates.value = result.dates; selected.value = nearest(props.date) } } catch(e) { if (request === sequence) error.value = friendly(e).message }
}, {immediate:true})
watch(() => props.date, value => { selected.value = nearest(value) })
let daySequence = 0
watch(() => [selected.value,compare.value,props.run], async () => {
  const request = ++daySequence
  if (!selected.value || !props.run) return
  try { const result = await api<Day>(`/runs/${props.run}/holdings?date=${selected.value}&compare=${compare.value}`); if (request === daySequence) { data.value = result; error.value = '' } } catch(e) { if (request === daySequence) error.value = friendly(e).message }
})
onUnmounted(() => { sequence++; daySequence++; controller?.abort() })
</script>
<template><div class="holdings-view"><div class="holdings-toolbar"><select v-model="selected" aria-label="持仓日期"><option v-for="d in dates" :key="d" :value="d">{{ d.replace(/(\d{4})(\d{2})(\d{2})/,'$1-$2-$3') }}</option></select><select v-model="side" aria-label="持仓侧"><option value="long">第 1 组 · 多头</option><option value="short">第 {{ groups ?? 10 }} 组 · 空头</option></select><input v-model="search" aria-label="搜索持仓" placeholder="搜索持仓"><button class="comparison-toggle" role="switch" :aria-checked="compare" :title="data?.previous ? `比较 ${data.previous}` : '首次调仓'" @click="compare = !compare">对比前次<i /></button></div><div v-if="error" role="alert">{{ error }}</div><table class="data-table holdings-table"><thead><tr><th>代码</th><th>简称</th><th>行业</th><th :title="`因子值取自信号日 ${data?.signal_date ?? ''}`">因子值</th><th>权重</th><th v-if="compare">变动</th></tr></thead><tbody><tr v-for="item in items" :key="`${item.symbol}:${item.status}`" :class="item.status"><td class="mono">{{ item.symbol }}</td><td>{{ item.name }}</td><td>{{ item.industry }}</td><td class="number" :title="item.status === 'exit' ? '前次调仓信号日因子值' : '调仓信号日因子值'">{{ fmt(item.factor,false,4) }}</td><td class="number" :title="compare ? `前次权重 ${fmt(item.previous_weight,true,2)}` : ''">{{ fmt(item.weight,true,2) }}</td><td v-if="compare">{{ {new:'新进',exit:'调出',held:'留存'}[item.status as 'new'] }}</td></tr></tbody></table><div v-if="!items.length && !error" class="empty">{{ dates.length ? '没有符合条件的持仓' : '运行后查看持仓' }}</div></div></template>
<style scoped>
.holdings-view{min-width:0}.holdings-toolbar{position:sticky;top:0;display:flex;align-items:center;gap:6px;background:white;min-height:34px;padding:3px 0;flex-wrap:wrap}.holdings-toolbar select{font-size:11px;max-width:125px;height:24px;min-height:24px}.holdings-toolbar input{width:85px;min-width:50px;font-size:11px;min-height:24px;height:24px}.comparison-toggle{display:flex;align-items:center;gap:4px;background:none;border:0;font-size:11px;white-space:nowrap}.comparison-toggle i{width:23px;height:13px;background:var(--line);border-radius:10px;position:relative}.comparison-toggle i:before{content:'';position:absolute;left:2px;top:2px;width:9px;height:9px;border-radius:50%;background:white}.comparison-toggle[aria-checked=true] i{background:var(--ink)}.comparison-toggle[aria-checked=true] i:before{left:12px}.holdings-table{table-layout:fixed}.holdings-table td,.holdings-table th{padding:0 3px;height:27px;font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.new td:last-child{color:var(--green)}.exit{color:var(--ink3)}.exit td:last-child{color:var(--red)}.empty{padding:30px;text-align:center;color:var(--ink3)}
</style>
