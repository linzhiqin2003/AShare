import { shallowRef } from 'vue'
import type { Period, PlotPart } from './types'

// 「显示验证期」全局开关（2026-11-14 用户要求默认隐藏，避免看着验证期调参）。
// 与回测页共用同一个键：值为 '1' 表示显示，其他或不存在表示隐藏。
export const VALIDATION_KEY = 'ashare.showValidation'
export const VALIDATION_START = '2025'

function read(): boolean {
  try { return localStorage.getItem(VALIDATION_KEY) === '1' } catch { return false }
}

export const showValidation = shallowRef(read())

export function setShowValidation(value: boolean) {
  showValidation.value = value
  try { localStorage.setItem(VALIDATION_KEY, value ? '1' : '0') } catch { /* 无存储权限时只在本页生效。 */ }
}

// 其他标签页改了开关时同步。
if (typeof window !== 'undefined') window.addEventListener('storage', event => { if (event.key === VALIDATION_KEY) showValidation.value = read() })

/** 关闭时只保留样本内部分；图表据此只画到样本内末。 */
export function visibleParts(parts: PlotPart[], show = showValidation.value): PlotPart[] {
  return show ? parts : parts.filter(part => part.part !== 'validation')
}

/** 分年/分季表：关闭时去掉验证期的行。 */
export function visiblePeriods(periods: Period[], show = showValidation.value): Period[] {
  return show ? periods : periods.filter(period => period.period < VALIDATION_START)
}

/** 依赖验证期数值的检验：关闭时保留通过状态，不显示数值。 */
export const VALIDATION_CHECKS = new Set(['validation_ratio'])
