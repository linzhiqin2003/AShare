//! Deterministic daily spot portfolio engine; sparse holdings and target rows.
#![forbid(unsafe_code)]
use rayon::prelude::*;
use tradecore::{Account, FeeTerms, Fees, round_close, round_open as lot};
pub const BUILD_MARKER: &str = "panelbt-core-v2";
pub const DAILY: usize = 30;
pub const COUNTS: usize = 38;

pub struct Futures<'a> {
    pub kind: &'a [bool],
    pub multiplier: &'a [f64],
    pub expiry: &'a [i64],
    pub settlement: &'a [f64],
    pub margin_rate: &'a [f64],
    pub fee_rate: &'a [f64],
    pub fee_per_contract: &'a [f64],
    pub delivery_fee_rate: &'a [f64],
    pub delivery_fee_per_contract: &'a [f64],
}

pub struct Market<'a> {
    pub t: usize,
    pub n: usize,
    pub price: &'a [f64],
    pub close: &'a [f64],
    pub buy: &'a [bool],
    pub sell: &'a [bool],
    pub can_short: &'a [bool],
    pub volume: &'a [f64],
    pub split: &'a [f64],
    pub dividend: &'a [f64],
    pub first_delist: &'a [i64],
    pub action_indptr: &'a [i64],
    pub action_cols: &'a [i64],
    pub futures: Option<Futures<'a>>,
}
pub struct Targets<'a> {
    pub days: &'a [i64],
    pub indptr: &'a [i64],
    pub cols: &'a [i64],
    pub weights: &'a [f64],
    pub shares: bool,
}
pub struct Settings<'a> {
    pub vectors: Vec<&'a [f64]>,
    pub scalars: &'a [f64],
    pub flags: &'a [bool],
    pub slippage: &'a [f64],
    pub impact: &'a [f64],
    pub borrow: &'a [f64],
}
impl Settings<'_> {
    #[inline]
    fn v(&self, k: usize, day: usize) -> f64 {
        self.vectors[k][day]
    }
    #[inline]
    fn terms(&self, day: usize, buy: bool) -> FeeTerms {
        let d = usize::from(buy);
        FeeTerms {
            commission_rate: self.scalars[0],
            commission_min: self.scalars[1],
            commission_per_share: self.scalars[2],
            commission_max_rate: (self.scalars[3] >= 0.0).then_some(self.scalars[3]),
            tax_rate: self.v(d, day),
            fee_rate: self.v(2 + d, day),
            fee_per_share: self.v(4 + d, day),
        }
    }
}
pub type Trade = (usize, usize, f64, f64, f64, f64, f64, f64, f64);
pub type DelistEvent = (usize, usize, f64, f64, f64);
pub struct Output {
    pub daily: Vec<f64>,
    pub positions: Option<Vec<f64>>,
    pub trades: Option<Vec<Trade>>,
    pub events: Vec<DelistEvent>,
    pub counts: [u64; COUNTS],
    pub contribution: Vec<f64>,
    pub liquidations: Vec<(usize, usize, f64)>,
    pub bankruptcies: Vec<usize>,
    pub expiries: Vec<(usize, usize, f64, f64, f64)>,
}
#[inline]
fn cell(a: &[f64], day: usize, col: usize, m: &Market<'_>) -> f64 {
    if a.len() == 1 {
        a[0]
    } else if a.len() == m.t {
        a[day]
    } else {
        a[day * m.n + col]
    }
}
#[derive(Clone, Copy)]
struct Quote {
    price: f64,
    amount: f64,
    fees: Fees,
    slip: f64,
    impact: f64,
}
fn quote(q: f64, i: usize, day: usize, m: &Market<'_>, s: &Settings<'_>) -> Quote {
    let p = m.price[day * m.n + i];
    let slip = cell(s.slippage, day, i, m);
    let slipped = p * (1.0 + q.signum() * slip);
    let impact = if s.impact.is_empty() {
        0.0
    } else {
        let v = m.volume[day * m.n + i];
        if v > 0.0 && v.is_finite() {
            s.impact[day * m.n + i] * (q.abs() / v).powf(s.scalars[4])
        } else {
            f64::INFINITY
        }
    };
    let price = slipped * (1.0 + q.signum() * impact);
    Quote {
        price,
        amount: q.abs() * price,
        fees: s.terms(day, q > 0.0).charge(q.abs(), price),
        slip: q.abs() * p * slip,
        impact: q.abs() * (price - slipped).abs(),
    }
}
#[inline]
fn movement(q: f64, x: Quote) -> f64 {
    if q > 0.0 {
        -(x.amount + x.fees.commission + x.fees.tax + x.fees.other)
    } else {
        x.amount - x.fees.commission - x.fees.tax - x.fees.other
    }
}
struct State {
    cash: f64,
    shares: Vec<f64>,
    last: Vec<f64>,
    held: Vec<usize>,
}
impl State {
    fn account(&self, day: usize, m: &Market<'_>) -> Account {
        let mut a = Account {
            cash: self.cash,
            ..Default::default()
        };
        for &i in &self.held {
            let p = m.price[day * m.n + i];
            let x = self.shares[i] * if p.is_finite() { p } else { self.last[i] };
            if x > 0.0 { a.long += x } else { a.short -= x }
        }
        a
    }
}
#[allow(clippy::too_many_arguments)]
fn fill(
    day: usize,
    i: usize,
    q: f64,
    carry: bool,
    forced: bool,
    m: &Market<'_>,
    s: &Settings<'_>,
    st: &mut State,
    out: &mut Output,
) {
    let x = quote(q, i, day, m, s);
    let before = st.shares[i];
    st.cash += movement(q, x);
    st.shares[i] += q;
    // Cash-flow attribution plus terminal mark equals total security P&L,
    // including execution slippage, impact, distributions and borrow charges.
    out.contribution[i] += movement(q, x);
    for (k, v) in [
        (2, x.amount),
        (3, x.fees.commission),
        (4, x.fees.tax),
        (5, x.fees.other),
        (6, x.slip),
        (9, x.impact),
    ] {
        out.daily[k * m.t + day] += v;
    }
    out.counts[0] += 1;
    if x.fees.commission > x.amount * s.scalars[0] + q.abs() * s.scalars[2] {
        out.counts[4] += 1;
    }
    if carry {
        out.counts[8] += 1;
    }
    if x.impact > 0.0 {
        out.counts[15] += 1;
    }
    if s.scalars[2] > 0.0 || s.v(if q > 0.0 { 5 } else { 4 }, day) > 0.0 {
        out.counts[16] += 1;
    }
    if s.scalars[3] >= 0.0
        && x.fees.commission < (x.amount * s.scalars[0] + q.abs() * s.scalars[2]).max(s.scalars[1])
    {
        out.counts[17] += 1;
    }
    if st.shares[i] < before.min(0.0) {
        out.counts[18] += 1;
    }
    if before < 0.0 && q > 0.0 {
        out.counts[19] += 1;
    }
    if forced {
        out.counts[24] += 1;
        out.liquidations.push((day, i, q));
    }
    if let Some(trades) = out.trades.as_mut() {
        trades.push((
            day,
            i,
            q,
            x.price,
            x.fees.commission,
            x.fees.tax,
            x.fees.other,
            x.slip,
            x.impact,
        ));
    }
    if before == 0.0 && st.shares[i] != 0.0 {
        st.held.push(i);
    }
}
fn past_close(m: &Market<'_>, day: usize, i: usize) -> f64 {
    for d in (0..day).rev() {
        let p = m.close[d * m.n + i];
        if p.is_finite() {
            let mut mark = p;
            for j in d + 1..=day {
                if !m.dividend.is_empty() {
                    mark = (mark - m.dividend[j * m.n + i]).max(0.0);
                }
                if !m.split.is_empty() {
                    mark /= m.split[j * m.n + i];
                }
            }
            return mark;
        }
    }
    0.0
}
fn capacity(m: &Market<'_>, s: &Settings<'_>, day: usize, i: usize, step: f64) -> f64 {
    if s.scalars[6] > 0.0 && !m.volume.is_empty() {
        let v = m.volume[day * m.n + i];
        if !v.is_finite() || v <= 0.0 {
            return 0.0;
        }
        let q = v * s.scalars[6];
        if step == 0.0 {
            q
        } else {
            (q / step).floor() * step
        }
    } else {
        f64::INFINITY
    }
}
#[inline]
fn quantity_tolerance(a: f64, b: f64) -> f64 {
    1e-9 * a.abs().max(b.abs()).max(1.0)
}
#[inline]
fn feasible(a: Account, s: &Settings<'_>, day: usize) -> bool {
    a.margin(s.v(11, day), s.v(12, day), 0.0) <= a.equity()
        && (s.flags[3] || a.own_cash(s.flags[4]) >= 0.0)
}
fn projected(
    mut a: Account,
    i: usize,
    q: f64,
    current: f64,
    day: usize,
    m: &Market<'_>,
    s: &Settings<'_>,
) -> Account {
    if q == 0.0 {
        return a;
    }
    a.cash += movement(q, quote(q, i, day, m, s));
    let p = m.price[day * m.n + i];
    a.long += ((current + q).max(0.0) - current.max(0.0)) * p;
    a.short += ((current + q).min(0.0) - current.min(0.0)) * -p;
    a
}
#[allow(clippy::too_many_arguments)]
fn affordable(
    cash: f64,
    desired: f64,
    i: usize,
    day: usize,
    step: f64,
    min: f64,
    m: &Market<'_>,
    s: &Settings<'_>,
) -> f64 {
    let cost = |q: f64| {
        if q == 0.0 {
            0.0
        } else {
            -movement(q, quote(q, i, day, m, s))
        }
    };
    let terms = s.terms(day, true);
    let mut q = if s.impact.is_empty()
        && terms.commission_per_share == 0.0
        && terms.fee_per_share == 0.0
        && terms.commission_max_rate.is_none()
    {
        let p = m.price[day * m.n + i] * (1.0 + cell(s.slippage, day, i, m));
        let limit = (cash / (p * (1.0 + terms.tax_rate + terms.fee_rate + terms.commission_rate)))
            .min((cash - terms.commission_min) / (p * (1.0 + terms.tax_rate + terms.fee_rate)))
            .max(0.0);
        desired.min(lot(limit, min, step))
    } else {
        let (mut lo, mut hi) = (0.0, desired);
        for _ in 0..64 {
            let mid = (lo + hi) * 0.5;
            if cost(mid) <= cash {
                lo = mid
            } else {
                hi = mid
            }
        }
        lot(lo, min, step)
    };
    while q > 0.0 && cost(q) > cash {
        q = if step == 0.0 {
            q.next_down()
        } else {
            lot(q - step, min, step)
        }
    }
    q
}

pub fn run(m: &Market<'_>, targets: &Targets<'_>, s: &Settings<'_>) -> Output {
    if m.futures.is_some() {
        futures::run(m, targets, s)
    } else if s.flags[6] {
        run_impl::<true>(m, targets, s)
    } else {
        run_impl::<false>(m, targets, s)
    }
}
fn run_impl<const CLASSIC: bool>(
    m: &Market<'_>,
    targets: &Targets<'_>,
    s: &Settings<'_>,
) -> Output {
    let mut out = Output {
        daily: vec![0.0; DAILY * m.t],
        positions: s.flags[1].then(|| vec![0.0; m.t * m.n]),
        trades: s.flags[1].then(Vec::new),
        events: Vec::new(),
        counts: [0; COUNTS],
        contribution: vec![0.0; m.n],
        liquidations: Vec::new(),
        bankruptcies: Vec::new(),
        expiries: Vec::new(),
    };
    let mut st = State {
        cash: s.scalars[5],
        shares: vec![0.0; m.n],
        last: vec![0.0; m.n],
        held: Vec::new(),
    };
    let mut pending: Vec<(usize, f64)> = Vec::new();
    let mut commands = Vec::new();
    let mut orders = Vec::new();
    let mut weights = vec![0.0; m.n];
    let mut cursor = 0;
    let (mut previous, mut unit) = (s.scalars[5], 1.0);
    let (mut call, mut bankrupt) = (false, false);
    let mut forced_quantities = vec![0.0; m.n];
    for day in 0..m.t {
        let base = day * m.n;
        // External flow is applied before corporate actions. Restricted proceeds
        // cannot be withdrawn; debt is never increased by a withdrawal.
        let available = if CLASSIC {
            st.cash.max(0.0)
        } else {
            let mut a = Account {
                cash: st.cash,
                ..Default::default()
            };
            for &i in &st.held {
                a.short += (-st.shares[i]).max(0.0) * st.last[i]
            }
            a.own_cash(s.flags[4]).max(0.0)
        };
        let requested = s.v(10, day);
        let flow = requested.max(-available);
        st.cash += flow;
        out.daily[12 * m.t + day] = flow;
        out.daily[13 * m.t + day] = flow - requested;
        if flow != 0.0 {
            out.counts[12] += 1;
        }
        if flow > requested {
            out.counts[13] += 1;
        }
        let action_start = m.action_indptr[day] as usize;
        let action_end = m.action_indptr[day + 1] as usize;
        let mut removed = false;
        for &col in &m.action_cols[action_start..action_end] {
            let i = col as usize;
            if st.shares[i] == 0.0 {
                continue;
            }
            if !m.dividend.is_empty() && m.dividend[base + i] != 0.0 {
                let d = st.shares[i] * m.dividend[base + i];
                let tax = if d > 0.0 { d * s.v(8, day) } else { 0.0 };
                if st.last[i] > 0.0 {
                    st.last[i] = (st.last[i] - m.dividend[base + i]).max(0.0);
                }
                st.cash += d - tax;
                out.contribution[i] += d - tax;
                out.daily[10 * m.t + day] += d;
                out.daily[11 * m.t + day] += tax;
                if m.dividend[base + i] > 0.0 {
                    out.counts[6] += 1;
                }
                if tax > 0.0 {
                    out.counts[11] += 1;
                }
                if d < 0.0 {
                    out.counts[22] += 1;
                }
            }
            if !m.split.is_empty() && m.split[base + i] != 1.0 {
                let f = m.split[base + i];
                st.shares[i] *= f;
                st.last[i] /= f;
                if f != 1.0 {
                    out.counts[7] += 1;
                }
            }
            if m.first_delist[i] <= day as i64 {
                let p = st.last[i] * s.scalars[7];
                let proceeds = st.shares[i] * p;
                st.cash += proceeds;
                out.contribution[i] += proceeds;
                out.events.push((day, i, st.shares[i], p, proceeds));
                out.counts[5] += 1;
                st.shares[i] = 0.0;
                removed = true;
            }
        }
        if removed {
            st.held.retain(|&i| st.shares[i] != 0.0);
        }
        let rebalance = cursor < targets.days.len() && targets.days[cursor] as usize == day;
        let forced = !CLASSIC && ((call && s.flags[5]) || bankrupt);
        let mut unfilled_liquidation = false;
        commands.clear();
        if rebalance {
            pending.clear();
            for &i in &st.held {
                weights[i] = 0.0;
                commands.push((i, 0.0));
            }
            for j in targets.indptr[cursor] as usize..targets.indptr[cursor + 1] as usize {
                let i = targets.cols[j] as usize;
                weights[i] = targets.weights[j];
                commands.push((i, weights[i]));
            }
            commands.sort_unstable_by_key(|x| x.0);
            commands.dedup_by_key(|x| x.0);
            for c in &mut commands {
                c.1 = weights[c.0];
            }
            cursor += 1;
        } else if s.flags[0] {
            std::mem::swap(&mut commands, &mut pending);
            pending.clear();
        }
        let mut account = if !commands.is_empty() || forced {
            st.account(day, m)
        } else {
            Account::default()
        };
        let value = account.equity();
        if forced {
            pending.clear();
            let requested_commands = commands.clone();
            commands.clear();
            let after_liquidation = |keep: f64| {
                let mut a = account;
                for &i in &st.held {
                    let current = st.shares[i];
                    let step = if targets.shares { 0.0 } else { s.vectors[7][i] };
                    let minimum = s.vectors[6][i];
                    let magnitude =
                        round_close(current.abs() * (1.0 - keep), current.abs(), minimum, step);
                    let q = -current.signum() * magnitude;
                    if q != 0.0 && m.price[base + i].is_finite() {
                        let x = quote(q, i, day, m, s);
                        if x.price > 0.0 && x.price.is_finite() {
                            a = projected(a, i, q, current, day, m, s);
                        }
                    }
                }
                a
            };
            let keep = if bankrupt || value <= 0.0 || !feasible(after_liquidation(0.0), s, day) {
                0.0
            } else if feasible(after_liquidation(1.0), s, day) {
                1.0
            } else {
                let (mut lo, mut hi) = (0.0, 1.0);
                for _ in 0..64 {
                    let mid = (lo + hi) * 0.5;
                    if feasible(after_liquidation(mid), s, day) {
                        lo = mid
                    } else {
                        hi = mid
                    }
                }
                lo
            };
            for &i in &st.held {
                let current = st.shares[i];
                let step = if targets.shares { 0.0 } else { s.vectors[7][i] };
                let minimum = s.vectors[6][i];
                let mut amount =
                    round_close(current.abs() * (1.0 - keep), current.abs(), minimum, step);
                if rebalance && let Some(&(_, w)) = requested_commands.iter().find(|x| x.0 == i) {
                    let p = m.price[base + i];
                    let mark = if p.is_finite() { p } else { st.last[i] };
                    let desired = if targets.shares {
                        w
                    } else if mark > 0.0 {
                        w * value.max(0.0) * (1.0 - s.scalars[8]) / mark
                    } else {
                        current
                    };
                    let risk_reduction = if current > 0.0 {
                        current - desired.max(0.0)
                    } else {
                        desired.min(0.0) - current
                    };
                    amount = amount.max(round_close(risk_reduction, current.abs(), minimum, step));
                }
                forced_quantities[i] = -current.signum() * amount;
                commands.push((i, current + forced_quantities[i]));
            }
        }
        orders.clear();
        if !CLASSIC {
            commands.sort_by_key(|&(i, _)| (st.shares[i] < 0.0, i));
        }
        for &(i, w) in &commands {
            if m.first_delist[i] <= day as i64 {
                continue;
            }
            if st.shares[i] == 0.0 {
                st.last[i] = past_close(m, day, i);
            }
            let p = m.price[base + i];
            let mark = if p.is_finite() { p } else { st.last[i] };
            let step = if targets.shares { 0.0 } else { s.vectors[7][i] };
            let minimum = s.vectors[6][i];
            let mut target = if forced {
                w
            } else if targets.shares {
                out.counts[9] += 1;
                w
            } else if mark > 0.0 {
                let raw = w * value.max(0.0) * (1.0 - s.scalars[8]) / mark;
                // A nonzero residual below the opening minimum is not a close.
                if raw.abs() < minimum && st.shares[i] * raw > 0.0 {
                    raw
                } else {
                    raw.signum() * lot(raw.abs(), minimum, step)
                }
            } else {
                0.0
            };
            if (target - st.shares[i]).abs() <= quantity_tolerance(target, st.shares[i]) {
                target = st.shares[i];
            }
            if !forced
                && !targets.shares
                && w != 0.0
                && st.shares[i] != 0.0
                && value > 0.0
                && (w - st.shares[i] * mark / value).abs() < s.scalars[9]
            {
                out.counts[10] += 1;
                continue;
            }
            let current = st.shares[i];
            let cap = capacity(m, s, day, i, step);
            let mut used = 0.0;
            let mut blocked_buy = false;
            let mut blocked_sell = false;
            // Reductions, including the closing half of a sign reversal.
            let reduction = if forced {
                forced_quantities[i]
            } else if current > 0.0 && target < current {
                -round_close(current - target.max(0.0), current, minimum, step)
            } else if current < 0.0 && target > current {
                round_close(target.min(0.0) - current, -current, minimum, step)
            } else {
                0.0
            };
            if reduction != 0.0 {
                let allowed = p.is_finite()
                    && if reduction > 0.0 {
                        m.buy[base + i]
                    } else {
                        m.sell[base + i]
                    };
                let mut q = if allowed {
                    reduction.signum()
                        * round_close(reduction.abs().min(cap), current.abs(), minimum, step)
                } else {
                    0.0
                };
                if q != reduction {
                    if reduction > 0.0 {
                        blocked_buy = true
                    } else {
                        blocked_sell = true
                    }
                }
                if !CLASSIC
                    && q != 0.0
                    && quote(q, i, day, m, s).price > 0.0
                    && quote(q, i, day, m, s).price.is_finite()
                    && !s.flags[3]
                    && account.own_cash(s.flags[4]) >= -1e-10 * s.scalars[5]
                    && projected(account, i, q, st.shares[i], day, m, s).own_cash(s.flags[4]) < 0.0
                {
                    let requested_quantity = q;
                    let (mut lo, mut hi) = (0.0, q.abs());
                    for _ in 0..64 {
                        let mid = (lo + hi) * 0.5;
                        let x = q.signum() * round_close(mid, current.abs(), minimum, step);
                        if projected(account, i, x, st.shares[i], day, m, s).own_cash(s.flags[4])
                            >= 0.0
                        {
                            lo = mid
                        } else {
                            hi = mid
                        }
                    }
                    q = q.signum() * round_close(lo, current.abs(), minimum, step);
                    if (q - requested_quantity).abs() > quantity_tolerance(q, requested_quantity) {
                        out.daily[24 * m.t + day] += 1.0;
                        out.counts[27] += 1;
                    }
                }
                if q != 0.0 {
                    let x = quote(q, i, day, m, s);
                    let candidate = if CLASSIC {
                        account
                    } else {
                        projected(account, i, q, st.shares[i], day, m, s)
                    };
                    // Reductions remain possible in a deficient margin account.
                    // The cash-only special case retains its exact legacy check.
                    if x.price > 0.0
                        && x.price.is_finite()
                        && (!CLASSIC
                            || st.cash + x.amount >= x.fees.commission + x.fees.tax + x.fees.other)
                    {
                        fill(day, i, q, !rebalance, forced, m, s, &mut st, &mut out);
                        account = candidate;
                        used = q.abs();
                    } else if !x.price.is_finite() || x.price <= 0.0 {
                        out.daily[25 * m.t + day] += 1.0;
                        out.counts[28] += 1;
                    }
                }
                if forced && reduction.abs() - used > quantity_tolerance(reduction, current) {
                    unfilled_liquidation = true;
                }
            }
            // Openings are delayed until every reduction has been executed.
            let after = st.shares[i];
            let opening = if target > 0.0 && after >= 0.0 && target > after && target > current {
                lot(target - after, minimum, step)
            } else if target < 0.0 && after <= 0.0 && target < after && target < current {
                -lot(after - target, minimum, step)
            } else {
                0.0
            };
            if opening.abs() > quantity_tolerance(target, after) && !forced && !bankrupt {
                let allowed = p.is_finite()
                    && if opening > 0.0 {
                        m.buy[base + i]
                    } else {
                        m.sell[base + i]
                            && s.flags[2]
                            && (m.can_short.is_empty() || m.can_short[base + i])
                    };
                let mut q = if allowed {
                    opening.signum() * lot(opening.abs().min((cap - used).max(0.0)), minimum, step)
                } else {
                    0.0
                };
                if q != 0.0 {
                    let x = quote(q, i, day, m, s);
                    if !x.price.is_finite() || x.price <= 0.0 {
                        q = 0.0;
                        out.daily[25 * m.t + day] += 1.0;
                        out.counts[28] += 1;
                    }
                }
                if opening != q {
                    if opening > 0.0 {
                        blocked_buy = true
                    } else {
                        blocked_sell = true
                    }
                }
                if q != 0.0 {
                    if current * q < 0.0 && used > 0.0 {
                        out.counts[30] += 1;
                    }
                    orders.push((i, q));
                }
            }
            if mark == 0.0 && w != 0.0 && target >= current {
                if w > 0.0 {
                    blocked_buy = true
                } else {
                    blocked_sell = true
                }
            }
            if blocked_buy {
                out.daily[7 * m.t + day] += 1.0;
                out.counts[1] += 1;
            }
            if blocked_sell {
                out.daily[8 * m.t + day] += 1.0;
                out.counts[2] += 1;
            }
            if (blocked_buy || blocked_sell) && s.flags[0] && !forced {
                pending.push((i, w));
            }
        }
        if !orders.is_empty() {
            if CLASSIC {
                let total: f64 = orders
                    .iter()
                    .map(|&(i, q)| -movement(q, quote(q, i, day, m, s)))
                    .sum();
                let scale = if total > st.cash {
                    out.counts[3] += 1;
                    st.cash / total
                } else {
                    1.0
                };
                for &(i, desired) in &orders {
                    let step = if targets.shares { 0.0 } else { s.vectors[7][i] };
                    let min = s.vectors[6][i];
                    let mut q = if scale < 1.0 {
                        lot(desired * scale, min, step)
                    } else {
                        desired
                    };
                    if q > 0.0 && -movement(q, quote(q, i, day, m, s)) > st.cash {
                        q = affordable(st.cash, q, i, day, step, min, m, s);
                    }
                    if q > quantity_tolerance(st.shares[i] + desired, st.shares[i]) {
                        fill(day, i, q, !rebalance, false, m, s, &mut st, &mut out);
                    }
                }
            } else {
                orders.sort_by_key(|&(i, q)| (q > 0.0, i));
                let trial = |scale: f64| {
                    let mut a = account;
                    for &(i, q) in &orders {
                        let step = if targets.shares { 0.0 } else { s.vectors[7][i] };
                        let q = q.signum() * lot(q.abs() * scale, s.vectors[6][i], step);
                        if q != 0.0 {
                            a = projected(a, i, q, st.shares[i], day, m, s);
                        }
                    }
                    a
                };
                let scale = if feasible(trial(1.0), s, day) {
                    1.0
                } else {
                    out.counts[20] += 1;
                    let (mut lo, mut hi) = (0.0, 1.0);
                    for _ in 0..64 {
                        let mid = (lo + hi) * 0.5;
                        if feasible(trial(mid), s, day) {
                            lo = mid
                        } else {
                            hi = mid
                        }
                    }
                    lo
                };
                for &(i, desired) in &orders {
                    let step = if targets.shares { 0.0 } else { s.vectors[7][i] };
                    let mut q =
                        desired.signum() * lot(desired.abs() * scale, s.vectors[6][i], step);
                    // An already deficient own-cash balance can make an early
                    // short opening infeasible even when the whole basket fits.
                    if q != 0.0
                        && !feasible(projected(account, i, q, st.shares[i], day, m, s), s, day)
                    {
                        let (mut lo, mut hi) = (0.0, q.abs());
                        for _ in 0..64 {
                            let mid = (lo + hi) * 0.5;
                            let x = q.signum() * lot(mid, s.vectors[6][i], step);
                            if feasible(projected(account, i, x, st.shares[i], day, m, s), s, day) {
                                lo = mid;
                            } else {
                                hi = mid;
                            }
                        }
                        q = q.signum() * lot(lo, s.vectors[6][i], step);
                    }
                    if q.abs() > quantity_tolerance(st.shares[i] + desired, st.shares[i]) {
                        account = projected(account, i, q, st.shares[i], day, m, s);
                        fill(day, i, q, !rebalance, false, m, s, &mut st, &mut out);
                    }
                }
            }
        }
        if !commands.is_empty() {
            st.held.sort_unstable();
            st.held.dedup();
            st.held.retain(|&i| st.shares[i] != 0.0);
        }
        let (mut long, mut short) = (0.0, 0.0);
        for &i in &st.held {
            let p = m.close[base + i];
            if p.is_finite() {
                st.last[i] = p;
            } else if !CLASSIC && st.last[i] == 0.0 && m.price[base + i].is_finite() {
                st.last[i] = m.price[base + i];
            }
            let v = st.shares[i] * st.last[i];
            if CLASSIC || v > 0.0 {
                long += v
            } else {
                short -= v;
                let fee = -v * cell(s.borrow, day, i, m);
                st.cash -= fee;
                out.contribution[i] -= fee;
                out.daily[19 * m.t + day] += fee;
                if fee > 0.0 {
                    out.counts[21] += 1;
                }
            }
        }
        let own = st.cash - if s.flags[4] { 0.0 } else { short };
        let interest = own.max(0.0) * s.v(9, day);
        let financing = (-own).max(0.0) * s.v(15, day);
        st.cash += interest - financing;
        if interest > 1e-10 {
            out.counts[14] += 1;
        }
        if financing > 1e-10 {
            out.counts[26] += 1;
        }
        let equity = st.cash + long - short;
        let denom = previous + flow;
        let ret = if denom > 8.0 * f64::EPSILON * s.scalars[5].max(previous.abs()).max(flow.abs()) {
            equity / denom - 1.0
        } else {
            0.0
        };
        unit *= 1.0 + ret;
        let required = s.v(11, day) * long + s.v(12, day) * short;
        let tolerance = 1e-10 * equity.abs().max(long).max(short).max(1.0);
        call = !CLASSIC
            && (s.v(13, day) * long + s.v(14, day) * short > equity + tolerance
                || unfilled_liquidation);
        if call {
            out.counts[23] += 1;
        }
        let valued = !CLASSIC && st.held.iter().all(|&i| st.last[i] > 0.0);
        if !CLASSIC
            && equity <= 0.0
            && !bankrupt
            && valued
            && (!st.held.is_empty() || st.cash < 0.0)
        {
            bankrupt = true;
            out.bankruptcies.push(day);
            out.counts[25] += 1;
            if short > 0.0 || own < -tolerance {
                out.counts[29] += 1;
            }
        }
        for (k, v) in [
            (0, equity / s.scalars[5]),
            (1, st.cash),
            (14, interest),
            (15, ret),
            (16, unit),
            (17, long),
            (18, short),
            (20, financing),
            (
                21,
                if equity > 0.0 {
                    required / equity
                } else if required > 0.0 {
                    f64::INFINITY
                } else {
                    0.0
                },
            ),
            (22, if call { 1.0 } else { 0.0 }),
            (23, equity),
        ] {
            out.daily[k * m.t + day] = v;
        }
        out.daily[2 * m.t + day] = if value > 0.0 {
            out.daily[2 * m.t + day] * 0.5 / value
        } else {
            0.0
        };
        previous = equity;
        if let Some(p) = out.positions.as_mut() {
            p[base..base + m.n].copy_from_slice(&st.shares);
        }
    }
    for &i in &st.held {
        out.contribution[i] += st.shares[i] * st.last[i];
    }
    out
}

mod futures;
pub fn run_batch(
    m: &Market<'_>,
    targets: &[Targets<'_>],
    s: &Settings<'_>,
    threads: usize,
) -> Result<Vec<Output>, rayon::ThreadPoolBuildError> {
    if threads == 1 || targets.len() <= 1 {
        return Ok(targets.iter().map(|t| run(m, t, s)).collect());
    }
    let workers = if threads == 0 {
        std::thread::available_parallelism().map_or(1, usize::from)
    } else {
        threads
    };
    rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .map(|p| p.install(|| targets.par_iter().map(|t| run(m, t, s)).collect()))
}
