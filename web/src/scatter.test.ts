import { describe, expect, it } from 'vitest'
import { placeLabels, type Rectangle } from './scatter'

const bounds = {x:0,y:0,width:200,height:100}
const marker = (id: string, x: number, y: number) => ({ id, x, y, radius: 5 })
const size = (id: string) => ({id,width:50,height:12})
const overlaps = (a: Rectangle,b: Rectangle) => a.x < b.x+b.width && a.x+a.width > b.x && a.y < b.y+b.height && a.y+a.height > b.y

describe('散点标签空位选择', () => {
  it('右侧邻点挡住时放到空侧，不使用固定右偏移', () => {
    const points = [marker('a',100,50),marker('b',120,25),marker('c',120,50),marker('d',120,75)]
    const box = placeLabels(points,[size('a')],bounds).get('a')!
    expect(box.x+box.width).toBeLessThan(100)
  })
  it('相邻标签避让所有点、已放置标签和图边界', () => {
    const points = [marker('a',80,50),marker('b',88,54),marker('c',150,80)]
    const boxes = [...placeLabels(points,points.map(p => size(p.id)),bounds).values()]
    expect(boxes.length).toBeGreaterThanOrEqual(2)
    for (const [i,box] of boxes.entries()) {
      expect(box.x).toBeGreaterThanOrEqual(0); expect(box.y).toBeGreaterThanOrEqual(0)
      expect(box.x+box.width).toBeLessThanOrEqual(200); expect(box.y+box.height).toBeLessThanOrEqual(100)
      for (const other of boxes.slice(i+1)) expect(overlaps(box,other)).toBe(false)
      for (const p of points) expect(overlaps(box,{x:p.x-5,y:p.y-5,width:10,height:10})).toBe(false)
    }
  })
  it('没有能容纳完整名称的空位时省略标签', () => {
    expect(placeLabels([marker('a',100,50)],[{id:'a',width:210,height:12}],bounds).size).toBe(0)
    const points = Array.from({length:231},(_,i) => marker(String(i),i%21*10,Math.floor(i/21)*10))
    expect(placeLabels(points,[size('115')],bounds).size).toBe(0)
  })
})
