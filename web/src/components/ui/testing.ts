// 测试辅助：自绘下拉的菜单挂在 body 上，wrapper.find 找不到。
import { nextTick } from 'vue'
import type { VueWrapper } from '@vue/test-utils'

export function options(): HTMLElement[] { return [...document.querySelectorAll<HTMLElement>('[role=listbox] [role=option]')] }

export async function pick(wrapper: VueWrapper, label: string, option: string) {
  await wrapper.get(`button[aria-label="${label}"]`).trigger('click')
  await nextTick()
  const target = options().find(o => o.textContent?.trim() === option)
  if (!target) throw new Error(`没有选项「${option}」：${options().map(o => o.textContent?.trim()).join(' / ')}`)
  target.click()
  await nextTick()
}

export function selectText(wrapper: VueWrapper, label: string) { return wrapper.get(`button[aria-label="${label}"]`).text() }
