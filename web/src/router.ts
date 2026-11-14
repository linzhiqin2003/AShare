import { onMounted, onUnmounted, shallowRef } from 'vue'

// 顶栏页面与 hash 路由：`#/data` 这样的地址可直达，刷新停在当前页，前进后退可用。
export const PAGES = [
  { id: 'research', label: '研究台' },
  { id: 'library', label: '因子库' },
  { id: 'data', label: '数据' },
  { id: 'backtest', label: '回测' },
  { id: 'paper', label: '模拟盘' },
] as const
export type Page = (typeof PAGES)[number]['id'] | 'settings'
const KNOWN = new Set<string>([...PAGES.map(p => p.id), 'settings'])

export function pageFromHash(hash: string): Page {
  const name = hash.replace(/^#\/?/, '').split(/[/?]/)[0]
  return (KNOWN.has(name) ? name : 'research') as Page
}

export function usePage() {
  const page = shallowRef<Page>(pageFromHash(location.hash))
  const sync = () => { page.value = pageFromHash(location.hash) }
  function navigate(value: Page) { if (pageFromHash(location.hash) !== value || !location.hash) location.hash = '/' + value }
  onMounted(() => window.addEventListener('hashchange', sync))
  onUnmounted(() => window.removeEventListener('hashchange', sync))
  return { page, navigate }
}
