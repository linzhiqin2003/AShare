<script setup lang="ts">
import { CircleCheck, CircleX, CircleAlert } from 'lucide-vue-next'
import { fmt } from '../chart'
import type { Check } from '../types'
defineProps<{ checks: Check[] }>()
</script>
<template>
  <table class="data-table checks">
    <thead><tr><th>检验</th><th>值</th><th>阈值</th><th>结果</th></tr></thead>
    <tbody><tr v-for="check in checks" :key="check.key">
      <td class="check-label">{{ check.label }}</td>
      <td class="number">{{ fmt(check.value, ['turnover','coverage','positive_years'].includes(check.key)) }}</td>
      <td class="number muted nowrap">{{ check.threshold }}</td>
      <td><span class="check-status" :class="check.status"><component :is="check.status === 'pass' ? CircleCheck : check.status === 'fail' ? CircleX : CircleAlert" :size="13" />{{ check.status === 'pass' ? '通过' : check.status === 'fail' ? '未通过' : '提示' }}</span></td>
    </tr></tbody>
  </table>
</template>
<style scoped>
.checks{table-layout:auto}.checks td{height:24px;font-size:11px}.checks td.check-label{white-space:normal;overflow-wrap:anywhere;overflow:visible;text-overflow:clip;line-height:1.5;padding:4px 5px 4px 3px}.checks th:not(:first-child){width:1%;white-space:nowrap}.checks td.number{padding-left:4px;font-size:10.5px}.checks th{padding-left:3px!important;font-size:10px}.check-status{display:inline-flex;gap:3px;align-items:center;white-space:nowrap;font-size:10.5px}.check-status svg{width:11px}.pass{color:var(--green)}.fail{color:var(--red)}.hint{color:var(--warn)}
</style>
