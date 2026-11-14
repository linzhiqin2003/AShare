<script setup lang="ts">
import { computed, onMounted, shallowRef } from 'vue'
import { Palette, SlidersHorizontal, Keyboard, Check } from 'lucide-vue-next'
import { api, friendly } from '../api'
import { rise, setRise, setTheme, theme, type Rise, type Theme } from '../appearance'
import UiSelect from '../components/ui/UiSelect.vue'
import UiNumberInput from '../components/ui/UiNumberInput.vue'
import UiSkeleton from '../components/ui/UiSkeleton.vue'
import { BACKTEST_FIELDS, validate, type BacktestDefaults, type PreferencePayload } from '../preferences'
import type { IndexList, Scalar } from '../types'

const sections = [
  { id: 'appearance', label: '外观', icon: Palette },
  { id: 'backtest', label: '默认回测设定', icon: SlidersHorizontal },
  { id: 'shortcuts', label: '快捷键', icon: Keyboard },
] as const
const active = shallowRef<string>('appearance')
function jump(id: string) { active.value = id; document.getElementById(`settings-${id}`)?.scrollIntoView({ block: 'nearest', behavior: 'smooth' }) }

const themes: { value: Theme; label: string }[] = [{ value: 'light', label: '浅色' }, { value: 'dark', label: '深色' }, { value: 'system', label: '跟随系统' }]
const defaults = shallowRef<BacktestDefaults | null>(null), indexes = shallowRef<IndexList['items']>([]), error = shallowRef(''), saving = shallowRef(false)
const columns = computed(() => { const half = Math.ceil(BACKTEST_FIELDS.length / 2); return [BACKTEST_FIELDS.slice(0, half), BACKTEST_FIELDS.slice(half)] })
const mac = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent)
const mod = mac ? '⌘' : 'Ctrl', shift = mac ? '⇧' : 'Shift'
const shortcuts = [
  { label: '运行当前表达式', keys: [mod, '↵'] },
  { label: '切换浅色与深色', keys: [mod, shift, 'L'] },
  { label: '关闭弹出的设置或菜单', keys: ['Esc'] },
  { label: '调整分栏宽度（分隔条获得焦点时）', keys: ['←', '→'] },
]

async function load() {
  try {
    const [payload, list] = await Promise.all([api<PreferencePayload>('/preferences'), api<IndexList>('/benchmarks')])
    defaults.value = payload.backtest; indexes.value = list.items; error.value = ''
  } catch (e) { error.value = friendly(e).message }
}
async function save(next: BacktestDefaults, method: 'PUT' | 'DELETE' = 'PUT') {
  const previous = defaults.value
  defaults.value = next; saving.value = true
  try {
    const payload = await api<PreferencePayload>('/preferences', method === 'PUT' ? { method, body: JSON.stringify(next) } : { method })
    defaults.value = payload.backtest; error.value = ''
  } catch (e) { defaults.value = previous; error.value = friendly(e).message }
  finally { saving.value = false }
}
function change(key: string, raw: Scalar) {
  // 数字框清空时控件发出 null：还在输入，不保存也不报错。
  if (!defaults.value || raw === null) return
  const field = BACKTEST_FIELDS.find(f => f.key === key)!
  const value = validate(field, raw)
  if (value === undefined) { error.value = field.invalid ?? '请输入有效的值'; return }
  const next: BacktestDefaults = { ...defaults.value, [key]: value }
  void save(Object.fromEntries(Object.entries(next).filter(([k]) => BACKTEST_FIELDS.some(f => f.key === k))) as BacktestDefaults)
}
function numeric(value: Scalar | undefined) { return typeof value === 'number' ? value : null }
const riseOptions = [{ value: 'red', label: '红涨绿跌' }, { value: 'green', label: '绿涨红跌' }]
const benchmarkOptions = computed(() => indexes.value.map(i => ({ value: i.symbol, label: i.name })))
onMounted(load)
</script>
<template>
  <main class="settings-page">
    <section class="panel settings-nav" aria-label="设置分类">
      <div class="panel-header">设置</div>
      <a v-for="item in sections" :key="item.id" :class="{ on: active === item.id }" href="#" @click.prevent="jump(item.id)"><component :is="item.icon" :size="14" />{{ item.label }}</a>
    </section>
    <div class="settings-column">
      <section id="settings-appearance" class="panel">
        <div class="panel-header">外观</div>
        <div class="settings-body">
          <h5>主题</h5>
          <div class="themes" role="radiogroup" aria-label="主题">
            <button v-for="item in themes" :key="item.value" class="theme-card" :class="{ on: theme === item.value }" role="radio" :aria-checked="theme === item.value" @click="setTheme(item.value)">
              <svg class="theme-swatch-light" v-if="item.value === 'light'" viewBox="0 0 120 70" width="150" height="88" aria-hidden="true"><rect width="120" height="70" rx="3" fill="var(--sw-bg)"/><rect x="4" y="10" width="72" height="34" rx="1.5" fill="var(--sw-panel)" stroke="var(--sw-line)"/><rect x="80" y="10" width="36" height="34" rx="1.5" fill="var(--sw-panel)" stroke="var(--sw-line)"/><rect x="4" y="48" width="112" height="18" rx="1.5" fill="var(--sw-panel)" stroke="var(--sw-line)"/><path d="M8 36 L22 30 L34 33 L48 24 L60 26 L72 18" fill="none" stroke="var(--sw-ink)" stroke-width="1.6"/><path d="M8 38 L22 36 L34 37 L48 33 L60 34 L72 30" fill="none" stroke="var(--sw-accent)" stroke-width="1.2"/><rect x="84" y="16" width="28" height="2" fill="var(--sw-ink3)"/><rect x="84" y="22" width="20" height="2" fill="var(--sw-ink3)"/><rect x="84" y="28" width="24" height="2" fill="var(--sw-ink3)"/><rect x="4" y="3" width="24" height="3" fill="var(--sw-ink)"/></svg>
              <svg class="theme-swatch-dark" v-else-if="item.value === 'dark'" viewBox="0 0 120 70" width="150" height="88" aria-hidden="true"><rect width="120" height="70" rx="3" fill="var(--sw-bg)"/><rect x="4" y="10" width="72" height="34" rx="1.5" fill="var(--sw-panel)" stroke="var(--sw-line)"/><rect x="80" y="10" width="36" height="34" rx="1.5" fill="var(--sw-panel)" stroke="var(--sw-line)"/><rect x="4" y="48" width="112" height="18" rx="1.5" fill="var(--sw-panel)" stroke="var(--sw-line)"/><path d="M8 36 L22 30 L34 33 L48 24 L60 26 L72 18" fill="none" stroke="var(--sw-ink)" stroke-width="1.6"/><path d="M8 38 L22 36 L34 37 L48 33 L60 34 L72 30" fill="none" stroke="var(--sw-accent)" stroke-width="1.2"/><rect x="84" y="16" width="28" height="2" fill="var(--sw-ink3)"/><rect x="84" y="22" width="20" height="2" fill="var(--sw-ink3)"/><rect x="84" y="28" width="24" height="2" fill="var(--sw-ink3)"/><rect x="4" y="3" width="24" height="3" fill="var(--sw-ink)"/></svg>
              <svg v-else viewBox="0 0 120 70" width="150" height="88" aria-hidden="true"><g class="theme-swatch-light"><rect width="60" height="70" fill="var(--sw-bg)"/><rect x="4" y="10" width="52" height="34" rx="1.5" fill="var(--sw-panel)" stroke="var(--sw-line)"/><path d="M8 36 L22 30 L34 33 L48 24" fill="none" stroke="var(--sw-ink)" stroke-width="1.6"/></g><g class="theme-swatch-dark"><rect x="60" width="60" height="70" fill="var(--sw-bg)"/><rect x="64" y="10" width="52" height="34" rx="1.5" fill="var(--sw-panel)" stroke="var(--sw-line)"/><path d="M68 34 L82 28 L96 30 L110 20" fill="none" stroke="var(--sw-ink)" stroke-width="1.6"/></g></svg>
              <span class="theme-label"><i class="tick"><Check v-if="theme === item.value" :size="10" :stroke-width="3" /></i>{{ item.label }}</span>
            </button>
          </div>
          <div class="two">
            <div class="sel"><span class="sel-label">涨跌颜色</span><UiSelect :model-value="rise" :options="riseOptions" aria-label="涨跌颜色" @update:model-value="setRise($event as Rise)" /></div>
          </div>
        </div>
      </section>
      <section id="settings-backtest" class="panel">
        <div class="panel-header">默认回测设定<span class="push header-note">新建标签页时套用</span></div>
        <div class="settings-body">
          <div v-if="defaults" class="two">
            <div v-for="(column, index) in columns" :key="index">
              <div v-for="field in column" :key="field.key" class="sel">
                <span class="sel-label">{{ field.label }}</span>
                <UiNumberInput v-if="field.kind === 'number'" :model-value="numeric(defaults[field.key])" :min="field.min" :max="field.max" :step="1" :unit="field.unit" :aria-label="field.label" @update:model-value="change(field.key, $event)" />
                <UiSelect v-else :model-value="defaults[field.key] ?? null" :options="field.key === 'benchmark' ? benchmarkOptions : field.options ?? []" :aria-label="field.label" @update:model-value="change(field.key, $event)" />
              </div>
            </div>
          </div>
          <div v-else-if="!error" class="two" aria-busy="true"><div v-for="(column, index) in columns" :key="index"><div v-for="field in column" :key="field.key" class="sel"><span class="sel-label">{{ field.label }}</span><UiSkeleton width="64px" /></div></div></div>
          <p v-if="error" class="settings-error" role="alert">{{ error }}</p>
          <button v-if="defaults" class="button reset" :disabled="saving" @click="save({}, 'DELETE')">恢复内置默认</button>
        </div>
      </section>
    </div>
    <div class="settings-column">
      <section id="settings-shortcuts" class="panel">
        <div class="panel-header">快捷键</div>
        <div class="settings-body">
          <div v-for="item in shortcuts" :key="item.label" class="shortcut"><span>{{ item.label }}</span><kbd>{{ item.keys.join(' ') }}</kbd></div>
        </div>
      </section>
    </div>
  </main>
</template>
<style scoped>
.settings-page{height:100%;padding:8px;display:grid;grid-template-columns:212px 1fr 1fr;gap:8px;min-height:0;overflow:auto}
.settings-column{display:flex;flex-direction:column;gap:8px;min-width:0}
.settings-page .panel-header{font-weight:600}
.header-note{font-weight:400;color:var(--ink3)}
.settings-nav a{display:flex;align-items:center;gap:9px;height:32px;padding:0 12px;color:var(--ink2);text-decoration:none}
.settings-nav a.on{background:var(--selected);color:var(--ink);font-weight:600}
.settings-nav a:hover{color:var(--ink)}
.settings-body{padding:8px 10px 10px}
h5{font-size:11.5px;font-weight:600;color:var(--ink3);margin:2px 0 5px}
.themes{display:flex;gap:14px}
.theme-card{flex:1 1 0;max-width:168px;min-width:0;border:1px solid var(--line);border-radius:4px;padding:8px 8px 6px;background:var(--panel);text-align:left;color:var(--ink)}
.theme-card.on{border-color:var(--accent)}
.theme-card>svg{display:block;width:100%;height:auto}
.theme-label{display:flex;align-items:center;gap:7px;margin-top:6px}
.tick{width:13px;height:13px;border-radius:2px;border:1px solid var(--ink3);display:grid;place-items:center}
.theme-card.on .tick{background:var(--ink);border-color:var(--ink);color:var(--panel)}
.two{display:grid;grid-template-columns:1fr 1fr;column-gap:24px;margin-top:10px}
.sel{display:flex;align-items:center;gap:5px;height:28px;border-bottom:1px solid var(--grid)}
.sel-label{color:var(--ink3);flex:1;white-space:nowrap}
.settings-error{margin:8px 0 0;color:var(--red);font-size:12px}
.reset{margin-top:12px}
.shortcut{display:flex;align-items:center;justify-content:space-between;height:28px;border-bottom:1px solid var(--grid)}
.shortcut:last-child{border-bottom:0}
kbd{font-family:var(--mono);font-size:11.5px;border:1px solid var(--line);border-radius:3px;padding:1px 7px;color:var(--ink2);background:var(--panel)}
@media (max-width:1100px){.settings-page{grid-template-columns:180px 1fr}.settings-page>.settings-column:last-child{grid-column:2}}
</style>
