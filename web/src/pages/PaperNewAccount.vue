<script setup lang="ts">
// 从回测新建模拟账户（2026-11-14）：选一个已完成的回测，给账户起名、定初始资金；
// 表达式与组合设置原样沿用回测。只写策略定义，账本由下一个交易日的运行器建立。
import { computed, nextTick, onMounted, onUnmounted, shallowRef } from 'vue'
import { api, friendly, post } from '../api'
import { UiNumberInput, UiSelect } from '../components/ui'
import type { PaperAccountItem, PaperBacktest } from './paper'
import { backtestLabel, suggestName, validName } from './paper'

const emit = defineEmits<{ created: [item: PaperAccountItem]; close: [] }>()
const backtests = shallowRef<PaperBacktest[] | null>(null)
const chosen = shallowRef<string | null>(null)
const name = shallowRef('')
const capital = shallowRef<number | null>(null)
const error = shallowRef('')
const busy = shallowRef(false)
const box = shallowRef<HTMLElement | null>(null)
let touched = false
function edited() { touched = true }

const options = computed(() => (backtests.value ?? []).map(b => ({ value: b.id as string | null, label: backtestLabel(b) })))
const current = computed(() => backtests.value?.find(b => b.id === chosen.value) ?? null)
const nameOk = computed(() => validName(name.value))
const ready = computed(() => !!chosen.value && nameOk.value && capital.value != null && capital.value >= 10 && !busy.value)

function pick(id: string | null) {
  chosen.value = id
  const b = current.value
  if (!b) return
  if (!touched) name.value = suggestName(b.name) || ''
  capital.value = b.capital ?? 1000
}
async function load() {
  try {
    const data = await api<{ items: PaperBacktest[] }>('/paper/backtests')
    backtests.value = data.items
    if (data.items.length) pick(data.items[0].id)
  } catch (e) { backtests.value = []; error.value = friendly(e).message }
}
async function submit() {
  if (!ready.value) return
  busy.value = true; error.value = ''
  try {
    const item = await post<PaperAccountItem>('/paper/accounts', { name: name.value, backtest: chosen.value, capital: capital.value })
    emit('created', item)
  } catch (e) { error.value = friendly(e).message } finally { busy.value = false }
}
function key(event: KeyboardEvent) { if (event.key === 'Escape' && !event.defaultPrevented) emit('close') }
onMounted(async () => { document.addEventListener('keydown', key); await load(); await nextTick(); box.value?.querySelector<HTMLElement>('button, input')?.focus() })
onUnmounted(() => document.removeEventListener('keydown', key))
</script>

<template>
  <div class="scrim" @pointerdown.self="emit('close')">
    <section ref="box" class="new-account panel" role="dialog" aria-modal="true" aria-labelledby="new-account-title">
      <div id="new-account-title" class="panel-header">从回测新建账户</div>
      <div class="body">
        <template v-if="backtests && backtests.length">
          <label class="field"><span>回测</span><UiSelect :model-value="chosen" :options="options" aria-label="回测" @update:model-value="pick" /></label>
          <label class="field"><span>账户名称</span><input v-model="name" :class="{ invalid: name && !nameOk }" maxlength="64" spellcheck="false" aria-label="账户名称" @input="edited"></label>
          <label class="field"><span>初始资金</span><UiNumberInput v-model="capital" :min="10" :max="1000000" :step="100" unit="万元" aria-label="初始资金" /></label>
        </template>
        <p v-else-if="backtests" class="none">还没有已完成的回测</p>
        <div v-else class="loading" aria-hidden="true" />
        <p v-if="error" class="error-text" role="alert">{{ error }}</p>
      </div>
      <div class="actions">
        <button type="button" class="button" @click="emit('close')">取消</button>
        <button type="button" class="button primary" :disabled="!ready" @click="submit">建立</button>
      </div>
    </section>
  </div>
</template>

<style scoped>
.scrim{position:fixed;inset:0;z-index:60;background:var(--overlay-shadow);display:grid;place-items:center}
.new-account{width:min(440px,calc(100vw - 32px));box-shadow:0 12px 40px var(--overlay-shadow)}
.body{padding:14px 14px 4px;display:grid;gap:10px}
.field{display:grid;grid-template-columns:72px minmax(0,1fr);align-items:center;gap:10px;font-size:12.5px}
.field>span{color:var(--ink3)}
.field :deep(.ui-select){height:28px;width:100%}
.field input{height:28px;width:100%}
.field input.invalid{border-color:var(--red)}
.none{margin:6px 0 10px;color:var(--ink3)}
.loading{height:28px;border-radius:3px;background:var(--grid)}
.error-text{color:var(--red);margin:0;font-size:12px}
.actions{display:flex;justify-content:flex-end;gap:8px;padding:10px 14px 14px}
</style>
