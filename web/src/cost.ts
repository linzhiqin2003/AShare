// 研究台「扣费年化」估算（2026-11-14 用户要求）。研究台的分组回测不扣交易成本，这里按换手粗估。
// 换手是买卖两边合计（Σ|权重变化|，见 crates/panelbt-core/src/weights.rs），所以每单位换手扣一次单边成本：
// 佣金 2.5 bp + 滑点 5 bp + 印花税（只在卖出收 5 bp，摊到单边约 2.5 bp）≈ 10 bp。
// 年化收益 = 日均收益 × 240（groupbt PERIODS），日均换手同口径，所以扣费年化 = 年化 − 日均换手 × 240 × 单边成本，
// 与逐日扣费后再年化逐位等价（同一批交易日）。
export const COST_BPS = 10
export const PERIODS = 240

export function netReturn(annual: number | null | undefined, turnover: number | null | undefined, bps = COST_BPS): number | null {
  if (annual == null || turnover == null || !Number.isFinite(annual) || !Number.isFinite(turnover)) return null
  return annual - turnover * PERIODS * bps / 10000
}
