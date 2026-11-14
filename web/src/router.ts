import { onMounted, onUnmounted, shallowRef } from 'vue'

// 顶栏页面与 hash 路由：`#/data` 这样的地址可直达，刷新停在当前页，前进后退可用。
// 研究台和回测 2026-11-14 合并成「研究」页（因子检验 / 策略回测两个视图）；旧的 `#/backtest` 落到研究页，
// 由研究页切到策略回测视图（带 `?entry=` 时新开 tab 带入因子库条目），再把地址换成 `#/research`。
export const PAGES = [
  { id: 'research', label: '研究' },
  { id: 'library', label: '因子库' },
  { id: 'data', label: '数据' },
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
