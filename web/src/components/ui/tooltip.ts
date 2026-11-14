// 提示（2026-11-14 用户要求替换系统原生 title）：v-tip="文字"。
// 悬停或键盘聚焦 400 ms 后出现，挂在 body 上，按元素定位、下方放不下时翻到上方；文字为空时不显示。
import type { Directive } from 'vue'
import { place } from './position'
import './ui.css'

export const TIP_DELAY = 400
const TIP_ID = 'ui-tip'
interface State { text: string; timer?: ReturnType<typeof setTimeout>; show: () => void; hide: () => void }
const states = new WeakMap<Element, State>()
let owner: Element | null = null

function layer() {
  let element = document.getElementById(TIP_ID)
  if (!element) {
    element = document.createElement('div')
    element.id = TIP_ID; element.className = 'ui-tip'; element.setAttribute('role', 'tooltip')
    document.body.appendChild(element)
  }
  return element
}
function open(target: Element, text: string) {
  const element = layer()
  element.textContent = text; element.style.visibility = 'hidden'; element.style.display = 'block'
  const at = place(target.getBoundingClientRect(), { width: element.offsetWidth, height: element.offsetHeight }, 6)
  element.style.left = `${at.left}px`; element.style.top = `${at.top}px`; element.dataset.side = at.side; element.style.visibility = ''
  owner = target; target.setAttribute('aria-describedby', TIP_ID)
}
function close(target: Element) {
  if (owner !== target) return
  const element = document.getElementById(TIP_ID)
  if (element) element.style.display = 'none'
  target.removeAttribute('aria-describedby'); owner = null
}

export const vTip: Directive<HTMLElement | SVGElement, string | null | undefined> = {
  mounted(el, binding) {
    const state: State = {
      text: binding.value ?? '',
      show: () => { clearTimeout(state.timer); if (!state.text) return; state.timer = setTimeout(() => open(el, state.text), TIP_DELAY) },
      hide: () => { clearTimeout(state.timer); close(el) },
    }
    states.set(el, state)
    el.addEventListener('mouseenter', state.show); el.addEventListener('focusin', state.show)
    for (const name of ['mouseleave', 'focusout', 'pointerdown', 'keydown']) el.addEventListener(name, state.hide)
  },
  updated(el, binding) {
    const state = states.get(el)
    if (!state) return
    state.text = binding.value ?? ''
    if (owner === el) { if (state.text) open(el, state.text); else close(el) }
  },
  beforeUnmount(el) {
    const state = states.get(el)
    if (!state) return
    state.hide()
    el.removeEventListener('mouseenter', state.show); el.removeEventListener('focusin', state.show)
    for (const name of ['mouseleave', 'focusout', 'pointerdown', 'keydown']) el.removeEventListener(name, state.hide)
    states.delete(el)
  },
}
