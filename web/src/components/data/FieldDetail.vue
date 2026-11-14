<script setup lang="ts">
import { computed } from 'vue'
import { formatValue, integer, percent, ticks, yearTicks, type FieldDetail } from '../../datastatus'
const props = defineProps<{ field: FieldDetail }>()
const W = 390, H = 130, LEFT = 46, RIGHT = 380, TOP = 8, BOTTOM = 112
const months = computed(() => props.field.monthly.map(m => m[0]))
const coverage = computed(() => props.field.monthly.map(m => m[1]))
const coveragePath = computed(() => {
  const values = coverage.value, span = Math.max(1, values.length - 1)
  let path = '', open = false
  values.forEach((value, index) => {
    if (value === null) { open = false; return }
    path += `${open ? 'L' : 'M'}${(LEFT + index / span * (RIGHT - LEFT)).toFixed(1)} ${(BOTTOM - value * (BOTTOM - TOP)).toFixed(1)} `
    open = true
  })
  return path.trim()
})
const years = computed(() => yearTicks(months.value).map(t => ({ ...t, x: LEFT + t.index / Math.max(1, months.value.length - 1) * (RIGHT - LEFT) })))
const HW = 380, HH = 150, HL = 30, HR = 372, HB = 128, HT = 12
const histogram = computed(() => {
  const d = props.field.distribution
  if (!d) return null
  const peak = Math.max(1, ...d.counts), width = (HR - HL) / d.counts.length
  const bars = d.counts.map((count, index) => ({ x: HL + index * width + 0.75, width: Math.max(1, width - 1.5), height: count / peak * (HB - HT) }))
  const axis = ticks(d.low, d.high, 5).map(value => ({ x: HL + (value - d.low) / ((d.high - d.low) || 1) * (HR - HL), label: formatValue(value, props.field.unit) }))
  return { bars, axis, quantiles: d.quantiles.map(q => formatValue(q, props.field.unit)) }
})
const boolean = computed(() => props.field.dtype === 'bool')
</script>
<template>
  <div class="field-detail">
    <div class="kv">
      <span>单位</span><b>{{ field.unit === '比例' ? '%' : field.unit || '—' }}</b>
      <span>类型</span><b>{{ field.dtype }}</b>
      <span>起始</span><b>{{ field.start ?? '—' }}</b>
      <span>最新覆盖率（上市股票）</span><b>{{ percent(field.latest) }}</b>
      <span>全历史覆盖率</span><b>{{ percent(field.overall) }}</b>
      <span>最新有值的股票</span><b>{{ integer(field.latest_count) }}</b>
      <template v-if="boolean"><span>最新为真的比例</span><b>{{ percent(field.true_share) }}</b></template>
    </div>
    <h5>历史覆盖率</h5>
    <svg class="chart" :viewBox="`0 0 ${W} ${H}`" role="img" :aria-label="`${field.name} 历史覆盖率`">
      <line :x1="LEFT" :x2="RIGHT" :y1="BOTTOM" :y2="BOTTOM" class="axis" />
      <text :x="LEFT - 6" :y="BOTTOM + 3.5" text-anchor="end">0%</text>
      <line :x1="LEFT" :x2="RIGHT" :y1="(BOTTOM + TOP) / 2" :y2="(BOTTOM + TOP) / 2" class="grid" />
      <text :x="LEFT - 6" :y="(BOTTOM + TOP) / 2 + 3.5" text-anchor="end">50%</text>
      <line :x1="LEFT" :x2="RIGHT" :y1="TOP" :y2="TOP" class="grid" />
      <text :x="LEFT - 6" :y="TOP + 3.5" text-anchor="end">100%</text>
      <text v-for="tick in years" :key="tick.label" :x="tick.x" :y="H - 4" text-anchor="middle">{{ tick.label }}</text>
      <path :d="coveragePath" class="line" />
    </svg>
    <template v-if="histogram">
      <h5>最新一天的分布</h5>
      <svg class="chart" :viewBox="`0 0 ${HW} ${HH}`" role="img" :aria-label="`${field.name} 最新一天的分布`">
        <rect v-for="(bar, index) in histogram.bars" :key="index" :x="bar.x" :y="HB - bar.height" :width="bar.width" :height="bar.height" class="bar" />
        <line :x1="HL" :x2="HR" :y1="HB" :y2="HB" class="axis" />
        <text v-for="tick in histogram.axis" :key="tick.x" :x="tick.x" :y="HH - 6" text-anchor="middle">{{ tick.label }}</text>
      </svg>
      <div class="quantiles"><span>5%</span><span>25%</span><span>中位数</span><span>75%</span><span>95%</span><b v-for="(value, index) in histogram.quantiles" :key="index">{{ value }}</b></div>
    </template>
    <template v-if="field.factors.length">
      <h5>引用它的因子</h5>
      <div v-for="factor in field.factors" :key="factor.name" class="factor"><span class="mono">{{ factor.name }}</span><span v-if="factor.sharpe !== null" class="sharpe">Sharpe {{ factor.sharpe.toFixed(2) }}</span></div>
    </template>
  </div>
</template>
<style scoped>
.field-detail{padding:8px 10px;overflow:auto;min-height:0}
.kv{display:grid;grid-template-columns:1fr auto}
.kv>span{color:var(--ink3);height:24px;display:flex;align-items:center;border-bottom:1px solid var(--grid)}
.kv>b{font-weight:500;height:24px;display:flex;align-items:center;justify-content:flex-end;border-bottom:1px solid var(--grid);font-family:var(--mono);font-size:12px}
h5{font-size:11.5px;font-weight:600;color:var(--ink3);margin:12px 0 5px}
.chart{display:block;width:100%;max-width:390px;height:auto;font-family:'IBM Plex Sans','Noto Sans SC',sans-serif}
.chart text{font-size:10.5px;fill:var(--ink3)}
.chart .axis{stroke:var(--line)}.chart .grid{stroke:var(--grid)}
.chart .line{fill:none;stroke:var(--ink2);stroke-width:1.4;stroke-linejoin:round}
.chart .bar{fill:var(--ink2)}
.quantiles{display:grid;grid-template-columns:repeat(5,1fr);margin-top:4px;text-align:center}
.quantiles span{color:var(--ink3);height:24px;display:flex;align-items:center;justify-content:center}
.quantiles b{font-weight:500;font-family:var(--mono);font-size:12px;height:22px}
.factor{display:flex;align-items:center;gap:8px;height:24px;font-size:12px}
.sharpe{margin-left:auto;color:var(--ink3);font-family:var(--mono);font-size:11.5px}
</style>
