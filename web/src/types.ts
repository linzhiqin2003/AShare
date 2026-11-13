export type Scalar = string | number | boolean | null
export type Settings = Record<string, Scalar | Scalar[]>
export type View = 'returns' | 'groups' | 'ic' | 'subs'
export type ReturnSelection = 'both' | 'long_short' | 'long'
export interface ApiError { message: string; line?: number; column?: number }
export interface Completion { label: string; kind: 'operator' | 'field' | 'factor'; signature?: string; detail: string; hint: string }
export interface Check { key: string; label: string; value: number | null; threshold: string; status: 'pass' | 'fail' | 'hint' }
export interface Period { period: string; return: number | null; sharpe: number | null; turnover: number | null; maxdrawdown: number | null; margin: number | null }
export interface Stats extends Period { holding_days?:number|null; short_holding_days?:number|null; n: number; fitness: number | null; t: number | null; ic: number | null; coverage: number | null; exposure: Record<string, number | null>; periods: Period[]; years: Period[]; subs: Record<string, Stats> }
export interface Summary { is: Stats; validation: Stats; checks: Check[]; trial_count: number; correlations: { entry: string; name: string; value: number; sharpe: number | null }[] }
export interface Run { id: string; ordinal: number; label: string; status: string; settings: Settings; summary: Summary | null; elapsed: number | null }
export interface Job { id: string; status: string; total: number; elapsed: number | null; runs: Run[] }
export interface Packed { shape: number[]; data: string }
export interface PlotPart { part: string; dates: string[]; series: Record<string, Packed> }
export interface Line { id: string; name: string; color: string; values: number[]; benchmark?: boolean; kind?: 'long_short' | 'long' | 'weights' }
export interface ChartData { dates: string[]; lines: Line[] }
export interface Entry { holding_days?:number|null; id: string; name: string; source: string; settings: Settings; expression_id: string; status: string; sharpe: number | null; validation: number | null; turnover: number | null; fitness: number | null; annual_return: number | null; drawdown: number | null; coverage: number | null; t: number | null; margin: number | null; latest: string | null; summary?: Summary }
export interface LibraryData { items: Entry[]; total: number; counts: Record<string, number>; groups: { key: string; count: number }[]; offset: number; limit: number; facets?: Record<string,Record<string,number>> }
export interface IndexItem { symbol: string; name: string }
export interface IndexSelection { symbol: string | null; name: string; fallback: boolean; requested: string }
export interface IndexList { items: IndexItem[]; total: number; selected: IndexSelection; default: IndexSelection }
export interface Relative { excess: number | null; information_ratio: number | null; n: number }
export interface BenchmarkData extends IndexSelection { parts: PlotPart[]; relatives: Record<string, { is: Relative; validation: Relative }> }
