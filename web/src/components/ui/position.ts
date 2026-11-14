// 浮层定位：按触发器放在下方，下方放不下且上方更宽裕时翻到上方；水平方向贴边收回视口内。
export interface Placement { left: number; top: number; side: 'below' | 'above' }

export function place(anchor: DOMRect, size: { width: number; height: number }, gap = 4, margin = 8): Placement {
  const below = innerHeight - anchor.bottom - gap - margin, above = anchor.top - gap - margin
  const side = size.height > below && above > below ? 'above' : 'below'
  const top = side === 'below' ? anchor.bottom + gap : anchor.top - gap - size.height
  const left = Math.max(margin, Math.min(anchor.left, innerWidth - size.width - margin))
  return { left: Math.round(left), top: Math.round(Math.max(margin, top)), side }
}
