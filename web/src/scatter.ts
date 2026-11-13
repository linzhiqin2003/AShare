export interface Marker { id: string; x: number; y: number; radius: number }
export interface LabelSize { id: string; width: number; height: number }
export interface Rectangle { x: number; y: number; width: number; height: number }

function distance(a: Rectangle, b: Rectangle) {
  return Math.hypot(Math.max(a.x - b.x - b.width, b.x - a.x - a.width, 0), Math.max(a.y - b.y - b.height, b.y - a.y - a.height, 0))
}

/** 用实际文字包围盒寻找空侧；所有点都参与避让，密集时省略标签。 */
export function placeLabels(markers: Marker[], sizes: LabelSize[], bounds: Rectangle): Map<string, Rectangle> {
  const placed = new Map<string, Rectangle>()
  const obstacles = markers.map(m => ({ x: m.x - m.radius, y: m.y - m.radius, width: m.radius * 2, height: m.radius * 2 }))
  for (const size of sizes) {
    const marker = markers.find(m => m.id === size.id)
    if (!marker) continue
    const gap = marker.radius + 5
    const candidates: { box: Rectangle; clearance: number }[] = []
    for (const dx of [-1, 0, 1]) for (const dy of [-1, 0, 1]) {
      if (!dx && !dy) continue
      const box = {
        x: marker.x + (dx < 0 ? -gap - size.width : dx > 0 ? gap : -size.width / 2),
        y: marker.y + (dy < 0 ? -gap - size.height : dy > 0 ? gap : -size.height / 2),
        width: size.width, height: size.height,
      }
      if (box.x < bounds.x || box.y < bounds.y || box.x + box.width > bounds.x + bounds.width || box.y + box.height > bounds.y + bounds.height) continue
      const clearance = Math.min(...obstacles.map(other => distance(box, other)), ...[...placed.values()].map(other => distance(box, other)))
      if (clearance >= 3) candidates.push({ box, clearance })
    }
    candidates.sort((a, b) => b.clearance - a.clearance)
    if (candidates[0]) placed.set(size.id, candidates[0].box)
  }
  return placed
}
