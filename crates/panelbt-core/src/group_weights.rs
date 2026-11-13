//! Streaming grouped research, sharing the pure-weight ledger and panelops rows.
use crate::weights::{Engine, Market, RowTarget};
use panelops_core::numpy_exact::RowOps;
use rayon::prelude::*;
use std::time::{Duration, Instant};

struct Clock {
    last: Option<Instant>,
    stages: [Duration; 5],
}
impl Clock {
    fn mark(&mut self, stage: usize) {
        if let Some(last) = self.last {
            let now = Instant::now();
            self.stages[stage] += now - last;
            self.last = Some(now);
        }
    }
}

#[derive(Clone, Copy)]
pub enum Numbers<'a> {
    F32(&'a [f32]),
    F64(&'a [f64]),
}
#[derive(Clone, Copy)]
pub enum Labels<'a> {
    U32(&'a [u32]),
    I64(&'a [i64]),
}
impl Labels<'_> {
    fn get(self, i: usize) -> i64 {
        match self {
            Self::U32(x) => i64::from(x[i]),
            Self::I64(x) => x[i],
        }
    }
}
impl Numbers<'_> {
    #[inline]
    fn get(self, i: usize) -> f64 {
        match self {
            Self::F32(x) => f64::from(x[i]),
            Self::F64(x) => x[i],
        }
    }
}

pub struct Input<'a> {
    pub rows: usize,
    pub cols: usize,
    pub signal_start: usize,
    pub execution_start: usize,
    pub factor: &'a [f64],
    pub logcap: &'a [f64],
    pub industry: Labels<'a>,
    pub industry_zero_missing: bool,
    pub vectors: Vec<&'a [f64]>,
    pub universes: Vec<&'a [bool]>,
    pub price: Numbers<'a>,
    pub close: Numbers<'a>,
    pub adjustment: Numbers<'a>,
    pub buy: &'a [bool],
    pub sell: &'a [bool],
    pub groups: usize,
    pub neutralize: usize,
    pub decay: usize,
    pub weighted: bool,
    pub profile: bool,
    pub rebalance: usize,
    pub record_holdings: bool,
}

#[derive(Default)]
pub struct Holdings {
    pub days: Vec<usize>,
    pub offsets: Vec<usize>,
    pub columns: Vec<u32>,
    pub weights: Vec<f64>,
}

pub struct Output {
    pub pnl: Vec<f64>,
    pub turnover: Vec<f64>,
    pub exposure: Vec<f64>,
    pub first_live: usize,
    pub bad_rows: Vec<usize>,
    pub stages: [f64; 5],
    pub holdings: Option<Holdings>,
}

struct Prices<'a> {
    raw: Numbers<'a>,
    adj: Numbers<'a>,
    cols: usize,
    next: usize,
    last: Vec<f64>,
    ring: Vec<Vec<f64>>,
}
impl<'a> Prices<'a> {
    fn new(raw: Numbers<'a>, adj: Numbers<'a>, cols: usize, history: usize, start: usize) -> Self {
        // Seed each column with its last valid value. Backward lookup avoids
        // evaluating every empty pre-window day, without truncating carry history.
        let mut last = vec![f64::NAN; cols];
        let mut missing: Vec<_> = (0..cols).collect();
        for day in (0..start).rev() {
            missing.retain(|&j| {
                let p = raw.get(day * cols + j) * adj.get(day * cols + j);
                if p.is_finite() && p > 0.0 {
                    last[j] = p;
                    false
                } else {
                    true
                }
            });
            if missing.is_empty() {
                break;
            }
        }
        Self {
            raw,
            adj,
            cols,
            next: start,
            last,
            ring: vec![vec![f64::NAN; cols]; history],
        }
    }
    fn advance(&mut self, until: usize) {
        while self.next <= until {
            let start = self.next * self.cols;
            for (j, v) in self.last.iter_mut().enumerate() {
                let p = self.raw.get(start + j) * self.adj.get(start + j);
                if p.is_finite() && p > 0.0 {
                    *v = p;
                }
            }
            let at = self.next % self.ring.len();
            self.ring[at].copy_from_slice(&self.last);
            self.next += 1;
        }
    }
    fn row(&self, index: usize) -> &[f64] {
        &self.ring[index % self.ring.len()]
    }
}

struct PoolDay {
    members: Vec<Vec<usize>>,
    normalized: Vec<f64>,
    bad: bool,
}

#[derive(Default)]
struct Cross {
    ops: RowOps,
    alpha: Vec<f64>,
    pool: Vec<f64>,
    labels: Vec<i64>,
    industry: Vec<i64>,
}
impl Cross {
    fn prepare(
        &mut self,
        input: &Input<'_>,
        s: usize,
        dot: &impl Fn(&[f64], &[f64]) -> f64,
    ) -> Vec<PoolDay> {
        let n = input.cols;
        let at = s * n;
        self.alpha.resize(n, f64::NAN);
        self.pool.resize(n, f64::NAN);
        self.labels.resize(n, -1);
        self.industry.resize(n, -1);
        for (v, &x) in self.alpha.iter_mut().zip(&input.factor[at..at + n]) {
            *v = if x.is_finite() { x } else { f64::NAN };
        }
        match input.neutralize {
            1 | 3 => {
                for (j, g) in self.industry.iter_mut().enumerate() {
                    let v = input.industry.get(at + j);
                    *g = if input.industry_zero_missing && v == 0 {
                        -1
                    } else {
                        v
                    };
                }
                self.ops.group_demean(&mut self.alpha, &self.industry);
            }
            2 => {
                self.ops.groups(
                    &input.logcap[at..at + n],
                    &mut self.labels,
                    10,
                    false,
                    false,
                );
                self.ops.group_demean(&mut self.alpha, &self.labels);
            }
            _ => (),
        }
        if input.neutralize == 3 {
            self.ops
                .residual(&mut self.alpha, &input.logcap[at..at + n], dot);
        }
        for v in &input.vectors {
            self.ops.residual(&mut self.alpha, &v[at..at + n], dot);
        }
        self.ops.demean(&mut self.alpha);
        input
            .universes
            .iter()
            .map(|universe| {
                for (j, v) in self.pool.iter_mut().enumerate() {
                    *v = if universe[at + j] {
                        self.alpha[j]
                    } else {
                        f64::NAN
                    };
                }
                self.ops.demean(&mut self.pool);
                let bad = !self
                    .ops
                    .groups(&self.pool, &mut self.labels, input.groups, true, true);
                let mut members = vec![Vec::new(); input.groups];
                for (j, &g) in self.labels.iter().enumerate() {
                    if g >= 0 {
                        members[g as usize].push(j);
                    }
                }
                let normalized = if input.weighted {
                    self.ops.normalize(&mut self.pool);
                    self.pool.clone()
                } else {
                    Vec::new()
                };
                PoolDay {
                    members,
                    normalized,
                    bad,
                }
            })
            .collect()
    }
}

struct Book {
    engine: Engine,
    pnl: Vec<f64>,
    turnover: Vec<f64>,
    held: Vec<f64>,
    active: Vec<bool>,
    first_live: usize,
    past: Vec<Vec<f64>>,
    decayed: Vec<f64>,
    holdings: Option<Holdings>,
}

/// Bounded blocks keep reductions inside a single worker, and every book in
/// chronological order. Independent books and cross sections can run in parallel.
pub fn run(input: &Input<'_>, dot: impl Fn(&[f64], &[f64]) -> f64 + Sync) -> Vec<Output> {
    const BLOCK: usize = 64;
    let n = input.cols;
    let t = input.rows - input.signal_start;
    let books = input.groups + 1 + usize::from(input.weighted);
    let pools = input.universes.len();
    let mut outputs: Vec<_> = (0..pools)
        .map(|_| Output {
            pnl: vec![0.0; books * t],
            turnover: vec![0.0; books * t],
            exposure: vec![f64::NAN; 4 * t],
            first_live: t,
            bad_rows: Vec::new(),
            stages: [0.0; 5],
            holdings: None,
        })
        .collect();
    if t == 0 || pools == 0 {
        return outputs;
    }
    let block = BLOCK.min(t);
    let mut states: Vec<_> = (0..pools * books)
        .map(|b| Book {
            engine: Engine::new(1, n),
            pnl: vec![0.0; t],
            turnover: vec![0.0; t],
            held: if b % books == input.groups {
                vec![0.0; block * n]
            } else {
                Vec::new()
            },
            active: vec![false; block],
            first_live: t,
            past: if input.weighted && b % books == books - 1 {
                vec![vec![0.0; n]; input.decay.max(1).min(t)]
            } else {
                Vec::new()
            },
            decayed: vec![0.0; n],
            holdings: (input.record_holdings && b == input.groups).then(|| Holdings {
                offsets: vec![0],
                ..Holdings::default()
            }),
        })
        .collect();
    let mut clock = Clock {
        last: input.profile.then(Instant::now),
        stages: [Duration::ZERO; 5],
    };
    let mut exec = Prices::new(
        input.price,
        input.adjustment,
        n,
        2,
        input.execution_start.min(input.rows),
    );
    let mut close = Prices::new(
        input.close,
        input.adjustment,
        n,
        241,
        input.signal_start.saturating_sub(240),
    );
    clock.mark(1);
    let mut returns = vec![0.0; block * n];
    let mut risk = vec![vec![f64::NAN; block * n]; 3];
    for begin in (0..t).step_by(block) {
        let end = (begin + block).min(t);
        let signals: Vec<_> = (begin..end)
            .into_par_iter()
            .map_init(Cross::default, |cross, s| cross.prepare(input, s, &dot))
            .collect();
        clock.mark(0);
        for s in begin..end {
            let signal = input.signal_start + s;
            let execution = input.execution_start + s;
            let at = (s - begin) * n;
            if execution >= input.rows {
                continue;
            }
            exec.advance((execution + 1).min(input.rows - 1));
            for (j, r) in returns[at..at + n].iter_mut().enumerate() {
                *r = if execution + 1 < input.rows {
                    exec.row(execution + 1)[j] / exec.row(execution)[j] - 1.0
                } else {
                    f64::NAN
                };
            }
            close.advance(signal);
            for (j, &p) in close.row(signal).iter().enumerate() {
                let d20 = if signal >= 20 {
                    close.row(signal - 20)[j]
                } else {
                    f64::NAN
                };
                let d60 = if signal >= 60 {
                    close.row(signal - 60)[j]
                } else {
                    f64::NAN
                };
                let d240 = if signal >= 240 {
                    close.row(signal - 240)[j]
                } else {
                    f64::NAN
                };
                risk[0][at + j] = 1.0 - p / d20;
                risk[1][at + j] = 1.0 - p / d60;
                risk[2][at + j] = d20 / d240 - 1.0;
            }
        }
        clock.mark(1);
        states.par_iter_mut().enumerate().for_each(|(b, state)| {
            let pool = b / books;
            let group = b % books;
            for (offset, day) in signals.iter().enumerate() {
                let s = begin + offset;
                let execution = input.execution_start + s;
                if execution >= input.rows {
                    continue;
                }
                let dest = execution - input.signal_start;
                let at = dest * n;
                let r = offset * n;
                let plan = &day[pool];
                let target = if group < input.groups {
                    RowTarget::Groups {
                        long: &plan.members[group],
                        short: &[],
                        long_budget: 1.0,
                        short_budget: 0.0,
                    }
                } else if group == input.groups {
                    RowTarget::Groups {
                        long: &plan.members[0],
                        short: &plan.members[input.groups - 1],
                        long_budget: 0.5,
                        short_budget: 0.5,
                    }
                } else if input.decay <= 1 {
                    RowTarget::Dense(&plan.normalized)
                } else {
                    let history = state.past.len();
                    for (v, &x) in state.past[s % history].iter_mut().zip(&plan.normalized) {
                        *v = if x.is_nan() { 0.0 } else { x };
                    }
                    state.decayed.fill(0.0);
                    for k in 0..input.decay.min(s + 1) {
                        for (v, &x) in state.decayed.iter_mut().zip(&state.past[(s - k) % history])
                        {
                            *v += (input.decay - k) as f64 * x;
                        }
                    }
                    let kk = (input.decay - 1).min(input.signal_start + s);
                    let den = ((kk + 1) * input.decay) as f64 - (kk * (kk + 1)) as f64 / 2.0;
                    for (v, &x) in state.decayed.iter_mut().zip(&plan.normalized) {
                        *v = if x.is_nan() { f64::NAN } else { *v / den };
                    }
                    RowTarget::Dense(&state.decayed)
                };
                let market = Market {
                    rows: 1,
                    cols: n,
                    returns: &returns[r..r + n],
                    buy: if input.buy.is_empty() {
                        &[]
                    } else {
                        &input.buy[at..at + n]
                    },
                    sell: if input.sell.is_empty() {
                        &[]
                    } else {
                        &input.sell[at..at + n]
                    },
                };
                let scheduled = s.is_multiple_of(input.rebalance);
                (state.pnl[dest], state.turnover[dest]) = state.engine.step(
                    0,
                    if scheduled { target } else { RowTarget::Hold },
                    &market,
                    0,
                    &dot,
                );
                if scheduled && let Some(snapshot) = &mut state.holdings {
                    snapshot.days.push(dest);
                    for (col, &weight) in state.engine.held().iter().enumerate() {
                        if weight != 0.0 {
                            snapshot.columns.push(col as u32);
                            snapshot.weights.push(weight);
                        }
                    }
                    snapshot.offsets.push(snapshot.columns.len());
                }
                if group == input.groups {
                    state.held[r..r + n].copy_from_slice(state.engine.held());
                    state.active[offset] = state.engine.active(0);
                    if state.active[offset] {
                        state.first_live = state.first_live.min(dest);
                    }
                }
            }
        });
        clock.mark(3);
        let exposures: Vec<_> = (0..(end - begin) * pools)
            .into_par_iter()
            .map_init(RowOps::default, |ops, item| {
                let pool = item % pools;
                let offset = item / pools;
                let s = begin + offset;
                let state = &states[pool * books + input.groups];
                let mut out = [f64::NAN; 4];
                if input.execution_start + s < input.rows && state.active[offset] {
                    let at = offset * n;
                    let held = &state.held[at..at + n];
                    let universe = &input.universes[pool][s * n..(s + 1) * n];
                    out[0] = ops.exposure(held, &input.logcap[s * n..(s + 1) * n], universe);
                    for (k, r) in risk.iter().enumerate() {
                        out[k + 1] = ops.exposure(held, &r[at..at + n], universe);
                    }
                }
                out
            })
            .collect();
        for (offset, day) in signals.iter().enumerate() {
            for (pool, plan) in day.iter().enumerate() {
                if plan.bad && (begin + offset).is_multiple_of(input.rebalance) {
                    outputs[pool]
                        .bad_rows
                        .push(input.signal_start + begin + offset);
                }
                let dest = input.execution_start + begin + offset - input.signal_start;
                if dest < t {
                    for (k, &value) in exposures[offset * pools + pool].iter().enumerate() {
                        outputs[pool].exposure[k * t + dest] = value;
                    }
                }
            }
        }
        clock.mark(4);
    }
    for (b, state) in states.into_iter().enumerate() {
        let pool = b / books;
        let group = b % books;
        outputs[pool].pnl[group * t..(group + 1) * t].copy_from_slice(&state.pnl);
        outputs[pool].turnover[group * t..(group + 1) * t].copy_from_slice(&state.turnover);
        if group == input.groups {
            outputs[pool].first_live = state.first_live;
            outputs[pool].holdings = state.holdings;
        }
    }
    outputs[0].stages = clock.stages.map(|v| v.as_secs_f64());
    outputs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(rows: usize, constant_factor: bool) -> Vec<Output> {
        let factor = if constant_factor {
            vec![1.0; rows * 4]
        } else {
            [4.0, 3.0, 2.0, 1.0].repeat(rows)
        };
        let logcap = [4.0, 3.0, 2.0, 1.0].repeat(rows);
        let price: Vec<_> = (0..rows).flat_map(|i| [1.0 + i as f64; 4]).collect();
        let adjustment = vec![1.0; rows * 4];
        let universe = vec![true; rows * 4];
        let input = Input {
            rows,
            cols: 4,
            signal_start: 0,
            execution_start: 1,
            factor: &factor,
            logcap: &logcap,
            industry: Labels::I64(&[]),
            industry_zero_missing: false,
            vectors: vec![],
            universes: vec![&universe, &universe],
            price: Numbers::F64(&price),
            close: Numbers::F64(&price),
            adjustment: Numbers::F64(&adjustment),
            buy: &[],
            sell: &[],
            groups: 2,
            neutralize: 0,
            decay: 3,
            weighted: true,
            profile: false,
            rebalance: 1,
            record_holdings: false,
        };
        run(&input, |x, y| {
            crate::weights::sum(&x.iter().zip(y).map(|(a, b)| a * b).collect::<Vec<_>>())
        })
    }

    #[test]
    fn every_row_survives_work_block_boundaries() {
        for t in [63, 64, 65, 128, 129, 193, 300] {
            let out = fixture(t, false);
            for pool in &out {
                assert!(pool.bad_rows.is_empty());
                assert_eq!(pool.first_live, 1);
                for day in 1..t {
                    let ret = if day + 1 < t {
                        (day + 2) as f64 / (day + 1) as f64 - 1.0
                    } else {
                        0.0
                    };
                    for group in 0..2 {
                        assert_eq!(pool.pnl[group * t + day].to_bits(), ret.to_bits());
                        assert_eq!(
                            pool.turnover[group * t + day],
                            if day == 1 { 1.0 } else { 0.0 }
                        );
                    }
                    // LS holdings [.25, .25, -.25, -.25] give size exposure 1.
                    assert_eq!(pool.exposure[day], 1.0);
                }
            }
            assert_eq!(out[0].pnl, out[1].pnl);
            assert_eq!(out[0].turnover, out[1].turnover);
        }
    }

    #[test]
    fn last_signal_is_checked_even_without_an_execution_row() {
        // Two distinct groups are impossible for an all-equal four-column row.
        let out = fixture(193, true);
        assert_eq!(out[0].bad_rows, (0..193).collect::<Vec<_>>());
        assert_eq!(out[0].first_live, 193);
    }

    #[test]
    fn empty_window_has_no_books_to_advance() {
        let out = fixture(0, false);
        assert_eq!(out.len(), 2);
        assert!(
            out.iter()
                .all(|p| p.pnl.is_empty() && p.bad_rows.is_empty())
        );
    }
}
