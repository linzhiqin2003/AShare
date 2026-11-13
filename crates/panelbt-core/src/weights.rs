//! Pure weight accounting. Targets are execution-day weights, never shifted here.
//! Preserve NumPy's compact column order and pairwise reductions exactly.

use tradecore::direction_allowed;

const TOL: f64 = 1e-12;

pub use panelops_core::numpy_exact::sum;

fn positive(x: f64) -> f64 {
    if x < 0.0 { 0.0 } else { x }
}

fn clean(x: f64) -> f64 {
    if x.is_nan() {
        0.0
    } else {
        x.clamp(-f64::MAX, f64::MAX)
    }
}

/// Dense targets or shared group labels. Missing labels are -1.
pub enum Plan<'a> {
    Dense(&'a [f64]),
    Groups {
        labels: &'a [i64],
        long: i64,
        short: i64,
        long_budget: f64,
        short_budget: f64,
    },
}

pub struct Market<'a> {
    pub rows: usize,
    pub cols: usize,
    pub returns: &'a [f64],
    pub buy: &'a [bool],
    pub sell: &'a [bool],
}

pub struct Output {
    pub weights: Option<Vec<f64>>,
    pub pnl: Vec<f64>,
    pub turnover: Vec<f64>,
}

#[derive(Default)]
struct Scratch {
    held: Vec<f64>,
    target: Vec<f64>,
    previous: Vec<f64>,
    current: Vec<f64>,
    returns: Vec<f64>,
    cols: Vec<usize>,
    frozen: Vec<bool>,
    select: Vec<f64>,
}

impl Scratch {
    fn push(&mut self, col: usize, want: f64, prev: f64, ret: f64) {
        if prev != 0.0 || (want != 0.0 && !want.is_nan()) {
            self.cols.push(col);
            self.target.push(clean(want));
            self.previous.push(prev);
            self.returns.push(clean(ret));
        }
    }

    fn selected_sum(&mut self, previous: bool, positive: bool, frozen: Option<bool>) -> f64 {
        let values = if previous {
            &self.previous
        } else {
            &self.target
        };
        self.select.clear();
        for (j, &v) in values.iter().enumerate() {
            if (if positive { v > 0.0 } else { v < 0.0 })
                && frozen.is_none_or(|state| self.frozen[j] == state)
            {
                self.select.push(v);
            }
        }
        let s = sum(&self.select);
        if positive { s } else { -s }
    }

    fn advance(&mut self, market: &Market<'_>, start: usize) {
        self.frozen.clear();
        self.frozen.resize(self.cols.len(), false);
        let long = self.selected_sum(false, true, None);
        let short = self.selected_sum(false, false, None);
        loop {
            let used_long = self.selected_sum(true, true, Some(true));
            let used_short = self.selected_sum(true, false, Some(true));
            let free_long = self.selected_sum(false, true, Some(false));
            let free_short = self.selected_sum(false, false, Some(false));
            let sl = if free_long > 0.0 {
                positive(long - used_long) / free_long
            } else {
                0.0
            };
            let ss = if free_short > 0.0 {
                positive(short - used_short) / free_short
            } else {
                0.0
            };
            let mut changed = false;
            for j in 0..self.cols.len() {
                let prev = self.previous[j];
                let target = self.target[j];
                let cand = if self.frozen[j] {
                    prev
                } else if target > 0.0 {
                    target * sl
                } else if target < 0.0 {
                    target * ss
                } else {
                    0.0
                };
                self.current[j] = cand;
                let k = start + self.cols[j];
                let buy = market.buy.is_empty() || market.buy[k];
                let sell = market.sell.is_empty() || market.sell[k];
                if !self.frozen[j]
                    && ((cand > prev + TOL && !direction_allowed(true, buy, sell))
                        || (cand < prev - TOL && !direction_allowed(false, buy, sell)))
                {
                    self.frozen[j] = true;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
    }
}

/// A target row, with group member columns in ascending order.
pub enum RowTarget<'a> {
    Hold,
    Dense(&'a [f64]),
    Groups {
        long: &'a [usize],
        short: &'a [usize],
        long_budget: f64,
        short_budget: f64,
    },
}

/// Shared weight ledger used by both dense batch and streaming research modes.
pub struct Engine {
    previous: Vec<Vec<f64>>,
    live: Vec<Vec<usize>>,
    scratch: Scratch,
}
impl Engine {
    pub fn new(books: usize, cols: usize) -> Self {
        Self {
            previous: vec![vec![0.0; cols]; books],
            live: vec![Vec::new(); books],
            scratch: Scratch::default(),
        }
    }
    pub fn held(&self) -> &[f64] {
        &self.scratch.held
    }
    pub fn active(&self, book: usize) -> bool {
        !self.live[book].is_empty()
    }
    pub fn step(
        &mut self,
        book: usize,
        plan: RowTarget<'_>,
        market: &Market<'_>,
        start: usize,
        dot: &impl Fn(&[f64], &[f64]) -> f64,
    ) -> (f64, f64) {
        let previous = &mut self.previous[book];
        let live = &mut self.live[book];
        let scratch = &mut self.scratch;
        scratch.cols.clear();
        scratch.target.clear();
        scratch.previous.clear();
        scratch.returns.clear();
        let hold = matches!(plan, RowTarget::Hold);
        match plan {
            RowTarget::Hold => {
                for &col in live.iter() {
                    scratch.push(
                        col,
                        previous[col],
                        previous[col],
                        market.returns[start + col],
                    );
                }
            }
            RowTarget::Dense(target) => {
                for (j, &want) in target.iter().enumerate() {
                    scratch.push(j, want, previous[j], market.returns[start + j]);
                }
            }
            RowTarget::Groups {
                long: l,
                short: s,
                long_budget,
                short_budget,
            } => {
                let wl = if l.is_empty() {
                    0.0
                } else {
                    long_budget * (1.0 / l.len() as f64)
                };
                let ws = if s.is_empty() {
                    0.0
                } else {
                    short_budget * (1.0 / s.len() as f64)
                };
                let (mut a, mut b, mut c) = (0, 0, 0);
                loop {
                    let il = l.get(a).copied().unwrap_or(usize::MAX);
                    let is = s.get(b).copied().unwrap_or(usize::MAX);
                    let ip = live.get(c).copied().unwrap_or(usize::MAX);
                    let col = il.min(is).min(ip);
                    if col == usize::MAX {
                        break;
                    }
                    let want =
                        (if col == il { wl } else { 0.0 }) - (if col == is { ws } else { 0.0 });
                    scratch.push(col, want, previous[col], market.returns[start + col]);
                    a += usize::from(col == il);
                    b += usize::from(col == is);
                    c += usize::from(col == ip);
                }
            }
        }
        scratch.held.resize(market.cols, 0.0);
        scratch.held.fill(0.0);
        if scratch.cols.is_empty() {
            return (0.0, 0.0);
        }
        scratch.current.resize(scratch.cols.len(), 0.0);
        if hold {
            scratch.current.copy_from_slice(&scratch.previous);
        } else if market.buy.is_empty() && market.sell.is_empty() {
            scratch.current.copy_from_slice(&scratch.target);
        } else {
            scratch.advance(market, start);
        }
        let pnl = dot(&scratch.current, &scratch.returns);
        scratch.select.clear();
        live.clear();
        for (j, &col) in scratch.cols.iter().enumerate() {
            let cur = scratch.current[j];
            scratch.select.push((cur - scratch.previous[j]).abs());
            previous[col] = cur;
            scratch.held[col] = cur;
            if cur != 0.0 {
                live.push(col);
            }
        }
        (pnl, sum(&scratch.select))
    }
}

/// All plans run inside one detached call. Dot uses NumPy's installed float64
/// kernel so the BLAS provider and its accumulation order cannot change.
pub fn run_batch(
    market: &Market<'_>,
    plans: &[(Plan<'_>, bool)],
    dot: impl Fn(&[f64], &[f64]) -> f64,
) -> Vec<Output> {
    struct Labels<'a> {
        labels: &'a [i64],
        keys: Vec<i64>,
        members: Vec<Vec<usize>>,
        direct: bool,
    }
    let mut labels: Vec<Labels<'_>> = Vec::new();
    for (plan, _) in plans {
        if let Plan::Groups {
            labels: values,
            long,
            short,
            ..
        } = plan
        {
            let index = labels
                .iter()
                .position(|c| c.labels.as_ptr() == values.as_ptr())
                .unwrap_or_else(|| {
                    labels.push(Labels {
                        labels: values,
                        keys: Vec::new(),
                        members: Vec::new(),
                        direct: false,
                    });
                    labels.len() - 1
                });
            labels[index].keys.push(*long);
            if *short >= 0 {
                labels[index].keys.push(*short);
            }
        }
    }
    for cache in &mut labels {
        cache.keys.sort_unstable();
        cache.keys.dedup();
        cache.direct = cache.keys.iter().enumerate().all(|(i, &k)| k == i as i64);
        cache.members.resize_with(cache.keys.len(), Vec::new);
    }
    let selections: Vec<_> = plans
        .iter()
        .map(|(p, _)| match p {
            Plan::Dense(_) => None,
            Plan::Groups {
                labels: values,
                long,
                short,
                ..
            } => {
                let i = labels
                    .iter()
                    .position(|c| c.labels.as_ptr() == values.as_ptr())
                    .unwrap();
                Some((
                    i,
                    labels[i].keys.binary_search(long).unwrap(),
                    labels[i].keys.binary_search(short).ok(),
                ))
            }
        })
        .collect();
    let mut engine = Engine::new(plans.len(), market.cols);
    let mut outputs: Vec<_> = plans
        .iter()
        .map(|(_, keep)| Output {
            weights: keep.then(|| vec![0.0; market.rows * market.cols]),
            pnl: vec![0.0; market.rows],
            turnover: vec![0.0; market.rows],
        })
        .collect();
    // Each label panel is scanned once per day, for all selected groups. Only
    // today's memberships are retained; memory is O(plans*N), not O(plans*T*N).
    for day in 0..market.rows {
        let start = day * market.cols;
        for cache in &mut labels {
            for row in &mut cache.members {
                row.clear();
            }
            for (j, &g) in cache.labels[start..start + market.cols].iter().enumerate() {
                let index = if cache.direct {
                    (g >= 0 && (g as usize) < cache.keys.len()).then_some(g as usize)
                } else {
                    cache.keys.binary_search(&g).ok()
                };
                if let Some(i) = index {
                    cache.members[i].push(j);
                }
            }
        }
        for (book, (plan, _)) in plans.iter().enumerate() {
            let out = &mut outputs[book];
            let target = match plan {
                Plan::Dense(target) => RowTarget::Dense(&target[start..start + market.cols]),
                Plan::Groups {
                    long_budget,
                    short_budget,
                    ..
                } => {
                    let (cache, long, short) = selections[book].unwrap();
                    RowTarget::Groups {
                        long: &labels[cache].members[long],
                        short: short.map_or(&[], |i| &labels[cache].members[i]),
                        long_budget: *long_budget,
                        short_budget: *short_budget,
                    }
                }
            };
            (out.pnl[day], out.turnover[day]) = engine.step(book, target, market, start, &dot);
            if let Some(held) = &mut out.weights {
                held[start..start + market.cols].copy_from_slice(engine.held());
            }
        }
    }
    outputs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buy_and_sell_freeze_changes_in_weight_not_position_sign() {
        // Covering a short is a buy; reducing a long is a sell. The reverse
        // trades remain allowed even on the same restricted columns.
        let market = Market {
            rows: 2,
            cols: 4,
            returns: &[0.0; 8],
            buy: &[true, true, true, true, false, true, false, true],
            sell: &[true, true, true, true, true, false, true, false],
        };
        let target = [-0.25, 0.25, 0.25, -0.25, 0.0, 0.0, 0.5, -0.5];
        let out = run_batch(&market, &[(Plan::Dense(&target), true)], |_, _| 0.0);
        assert_eq!(out[0].weights.as_ref().unwrap(), &target[..4].repeat(2));
        assert_eq!(out[0].turnover, [1.0, 0.0]);
        let reverse = [-0.25, 0.25, 0.25, -0.25, -0.5, 0.5, 0.0, 0.0];
        let out = run_batch(&market, &[(Plan::Dense(&reverse), true)], |_, _| 0.0);
        assert_eq!(out[0].weights.as_ref().unwrap(), &reverse);
        assert_eq!(out[0].turnover, [1.0, 1.0]);
    }

    #[test]
    fn frozen_longs_and_shorts_consume_separate_budgets() {
        let mut sell = [true; 12];
        sell[6] = false;
        let mut buy = [true; 12];
        buy[9] = false;
        let market = Market {
            rows: 2,
            cols: 6,
            returns: &[0.125; 12],
            buy: &buy,
            sell: &sell,
        };
        let target = [
            0.25, 0.25, 0.0, -0.125, -0.125, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, -0.25,
        ];
        let out = run_batch(&market, &[(Plan::Dense(&target), true)], |x, y| {
            sum(&x.iter().zip(y).map(|(a, b)| a * b).collect::<Vec<_>>())
        });
        assert_eq!(
            &out[0].weights.as_ref().unwrap()[6..],
            &[0.25, 0.0, 0.25, -0.125, 0.0, -0.125]
        );
        assert_eq!(out[0].turnover, [0.75, 0.75]);
        assert_eq!(out[0].pnl, [0.03125, 0.03125]);
    }

    #[test]
    fn frozen_weight_can_exceed_new_budget_without_reversing_free_side() {
        let market = Market {
            rows: 2,
            cols: 3,
            returns: &[0.0; 6],
            buy: &[],
            sell: &[true, true, true, false, true, true],
        };
        let target = [0.5, 0.0, -0.5, 0.0, 0.25, -0.25];
        let out = run_batch(&market, &[(Plan::Dense(&target), true)], |_, _| 0.0);
        assert_eq!(&out[0].weights.as_ref().unwrap()[3..], &[0.5, 0.0, -0.25]);
    }

    #[test]
    fn freezing_reallocates_remaining_budget() {
        let market = Market {
            rows: 2,
            cols: 4,
            returns: &[0.0; 8],
            buy: &[true, true, true, true, false, true, true, true],
            sell: &[true, true, true, true, false, true, true, true],
        };
        let target = [0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.5, 0.5];
        let out = run_batch(&market, &[(Plan::Dense(&target), true)], |_, _| 0.0);
        assert_eq!(
            out[0].weights.as_ref().unwrap(),
            &[0.25, 0.25, 0.25, 0.25, 0.25, 0.0, 0.375, 0.375]
        );
        assert_eq!(out[0].turnover, [1.0, 0.5]);
    }
}
