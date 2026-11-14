<script setup lang="ts">
import { onMounted, onUnmounted, watch } from 'vue'
import { Database, LoaderCircle, X, Settings as SettingsIcon } from 'lucide-vue-next'
import { useResearch } from './useResearch'
import FactorLibrary from './components/FactorLibrary.vue'
import StatusBar from './components/StatusBar.vue'
import ResearchPage from './pages/ResearchPage.vue'
import DataPage from './pages/DataPage.vue'
import SettingsPage from './pages/SettingsPage.vue'
import PaperPage from './pages/PaperPage.vue'
import { PAGES, usePage } from './router'
import { themeShortcut } from './appearance'
const research = useResearch()
const { completions, metadata, elapsed, message } = research
const { page, navigate } = usePage()
let toastTimer: ReturnType<typeof setTimeout> | undefined
watch(message, value => { clearTimeout(toastTimer); if (value) toastTimer = setTimeout(() => message.value = '', 5000) })
onMounted(() => { window.addEventListener('keydown', themeShortcut); void research.init() })
onUnmounted(() => { clearTimeout(toastTimer); window.removeEventListener('keydown', themeShortcut) })
</script>
<template>
  <div class="app-shell">
    <header class="topbar"><span class="brand">AShare</span><nav aria-label="主导航"><button v-for="item in PAGES" :key="item.id" :class="{ active: page === item.id }" :aria-current="page === item.id ? 'page' : undefined" @click="navigate(item.id)">{{ item.label }}</button></nav><div class="top-status"><span><LoaderCircle :size="13" :class="{ spinning: metadata.running > 0 }" />{{ metadata.running }} 个任务运行中</span><span><Database :size="14" />数据截至 {{ metadata.data_date ?? '—' }}</span><button class="top-icon" :class="{ active: page === 'settings' }" aria-label="设置" title="设置" @click="navigate('settings')"><SettingsIcon :size="15" /></button></div></header>
    <div v-if="message" class="toast" role="alert">{{ message }}<button class="icon-button push" aria-label="关闭提示" @click="message = ''"><X :size="14" /></button></div>
    <div class="main-area">
      <ResearchPage v-show="page === 'research'" :research="research" />
      <FactorLibrary v-if="page === 'library'" :completions="completions" />
      <DataPage v-else-if="page === 'data'" :completions="completions" />
      <PaperPage v-else-if="page === 'paper'" />
      <SettingsPage v-else-if="page === 'settings'" />
    </div>
    <StatusBar :elapsed="elapsed" />
  </div>
</template>
