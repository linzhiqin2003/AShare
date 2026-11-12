//! Deterministic daily spot portfolio engine; sparse holdings and target rows.
#![forbid(unsafe_code)]
use rayon::prelude::*;
use tradecore::{Account, FeeTerms, Fees, round_close, round_open as lot};
pub const BUILD_MARKER: &str = "panelbt-core-v2";
pub const DAILY: usize = 30;
pub const COUNTS: usize = 39;

pub struct Futures<'a> {
    /// Futures-column flags read as bytes: any nonzero byte is a future.
    pub kind: &'a [u8],
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
    /// Boolean mask read as bytes: any nonzero byte is true.
    pub buy: &'a [u8],
    pub sell: &'a [u8],
    pub can_short: &'a [u8],
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
    match (s.flags[6], m.futures.is_some()) {
        (true, false) => run_impl::<true, false>(m, targets, s),
        (false, false) => run_impl::<false, false>(m, targets, s),
        (true, true) => run_impl::<true, true>(m, targets, s),
        (false, true) => run_impl::<false, true>(m, targets, s),
    }
}
#[inline]
fn is_future<const FUT: bool>(m: &Market<'_>, i: usize) -> bool {
    FUT && m.futures.as_ref().is_some_and(|f| f.kind[i] != 0)
}
/// (minimum, step) for a column: futures trade whole contracts.
#[inline]
fn lot_terms<const FUT: bool>(
    m: &Market<'_>,
    s: &Settings<'_>,
    targets: &Targets<'_>,
    i: usize,
) -> (f64, f64) {
    if is_future::<FUT>(m, i) {
        (1.0, 1.0)
    } else {
        (
            s.vectors[6][i],
            if targets.shares { 0.0 } else { s.vectors[7][i] },
        )
    }
}
/// Initial-margin and own-cash test; `fm` is the futures collateral (0 for spot).
#[inline]
fn feasible(a: Account, fm: f64, s: &Settings<'_>, day: usize) -> bool {
    a.margin(s.v(11, day), s.v(12, day), fm) <= a.equity()
        && (s.flags[3] || a.own_cash(s.flags[4]) - fm >= 0.0)
}
#[allow(clippy::too_many_arguments)]
#[inline]
fn project_any<const FUT: bool>(
    a: Account,
    fm: f64,
    i: usize,
    q: f64,
    shares: &[f64],
    basis: &[f64],
    day: usize,
    m: &Market<'_>,
    s: &Settings<'_>,
) -> (Account, f64) {
    if is_future::<FUT>(m, i) {
        futures::project(a, fm, i, q, shares[i], basis[i], day, m, s)
    } else {
        (projected(a, i, q, shares[i], day, m, s), fm)
    }
}
#[inline]
fn quote_any<const FUT: bool>(
    q: f64,
    i: usize,
    day: usize,
    m: &Market<'_>,
    s: &Settings<'_>,
) -> Quote {
    if is_future::<FUT>(m, i) {
        futures::quote(q, i, day, m, s)
    } else {
        quote(q, i, day, m, s)
    }
}
/// Common-factor opening basket with a prefix feasibility check. Used for every
/// margin-account opening and, in the cash specialization, for futures openings
/// that reserve collateral before the legacy cash-funded spot basket.
#[allow(clippy::too_many_arguments)]
fn open_basket<const FUT: bool>(
    orders: &mut [(usize, f64)],
    account: &mut Account,
    fm: &mut f64,
    count_spot_scaling: bool,
    rebalance: bool,
    day: usize,
    m: &Market<'_>,
    targets: &Targets<'_>,
    s: &Settings<'_>,
    st: &mut State,
    book: &mut futures::Book,
    out: &mut Output,
) {
    if FUT {
        // Hedge openings reserve collateral before short sales and spot buys.
        orders.sort_by_key(|&(i, q)| (!is_future::<FUT>(m, i), q > 0.0, i));
    } else {
        orders.sort_by_key(|&(i, q)| (q > 0.0, i));
    }
    let trial = |scale: f64| {
        let (mut a, mut margin) = (*account, *fm);
        for &(i, q) in orders.iter() {
            let (minimum, step) = lot_terms::<FUT>(m, s, targets, i);
            let q = q.signum() * lot(q.abs() * scale, minimum, step);
            if q != 0.0 {
                (a, margin) =
                    project_any::<FUT>(a, margin, i, q, &st.shares, &book.basis, day, m, s);
            }
        }
        (a, margin)
    };
    let full = trial(1.0);
    let scale = if feasible(full.0, full.1, s, day) {
        1.0
    } else {
        if count_spot_scaling {
            out.counts[20] += 1;
        }
        if FUT && orders.iter().any(|&(i, _)| is_future::<FUT>(m, i)) {
            out.counts[33] += 1;
        }
        let (mut lo, mut hi) = (0.0, 1.0);
        for _ in 0..64 {
            let mid = (lo + hi) * 0.5;
            let x = trial(mid);
            if feasible(x.0, x.1, s, day) {
                lo = mid
            } else {
                hi = mid
            }
        }
        lo
    };
    for &(i, desired) in orders.iter() {
        let (minimum, step) = lot_terms::<FUT>(m, s, targets, i);
        let mut q = desired.signum() * lot(desired.abs() * scale, minimum, step);
        // An already deficient own-cash balance can make an early
        // short opening infeasible even when the whole basket fits.
        if q != 0.0 {
            let x = project_any::<FUT>(*account, *fm, i, q, &st.shares, &book.basis, day, m, s);
            if !feasible(x.0, x.1, s, day) {
                let (mut lo, mut hi) = (0.0, q.abs());
                for _ in 0..64 {
                    let mid = (lo + hi) * 0.5;
                    let c = q.signum() * lot(mid, minimum, step);
                    let y =
                        project_any::<FUT>(*account, *fm, i, c, &st.shares, &book.basis, day, m, s);
                    if feasible(y.0, y.1, s, day) {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                q = q.signum() * lot(lo, minimum, step);
            }
        }
        if q.abs() > quantity_tolerance(st.shares[i] + desired, st.shares[i]) {
            (*account, *fm) =
                project_any::<FUT>(*account, *fm, i, q, &st.shares, &book.basis, day, m, s);
            if is_future::<FUT>(m, i) {
                book.execute(
                    &mut st.shares,
                    &mut st.cash,
                    i,
                    q,
                    day,
                    !rebalance,
                    false,
                    m,
                    s,
                    out,
                );
            } else {
                fill(day, i, q, !rebalance, false, m, s, st, out);
            }
        }
    }
}
/// One daily account loop for every account type. `CLASSIC` selects the exact
/// long-only cash specialization; `FUT` compiles in futures columns. With
/// `FUT = true`, a futures column that is never held or targeted executes no
/// futures arithmetic, so spot results equal the `FUT = false` build bit for bit.
fn run_impl<const CLASSIC: bool, const FUT: bool>(
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
    let mut book = futures::Book {
        basis: if FUT { vec![0.0; m.n] } else { Vec::new() },
        held: Vec::new(),
        seen: false,
    };
    let mut pending: Vec<(usize, f64)> = Vec::new();
    let mut commands = Vec::new();
    let mut orders = Vec::new();
    let mut future_orders = Vec::new();
    let mut weights = vec![0.0; m.n];
    let mut cursor = 0;
    let (mut previous, mut unit) = (s.scalars[5], 1.0);
    let (mut call, mut bankrupt) = (false, false);
    let mut forced_quantities = vec![0.0; m.n];
    for day in 0..m.t {
        let base = day * m.n;
        // Futures collateral at the last settlement and today's rate.
        let fm_before = if FUT {
            futures::held_margin(&book.held, &st.shares, &book.basis, day, m)
        } else {
            0.0
        };
        // External flow is applied before corporate actions. Restricted proceeds
        // and futures collateral cannot be withdrawn; debt is never increased.
        let available = if CLASSIC {
            if FUT {
                (st.cash - fm_before).max(0.0)
            } else {
                st.cash.max(0.0)
            }
        } else {
            let mut a = Account {
                cash: st.cash,
                ..Default::default()
            };
            for &i in &st.held {
                a.short += (-st.shares[i]).max(0.0) * st.last[i]
            }
            if FUT {
                (a.own_cash(s.flags[4]) - fm_before).max(0.0)
            } else {
                a.own_cash(s.flags[4]).max(0.0)
            }
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
        // A cash account without futures never records calls or insolvency.
        let forced = (!CLASSIC || FUT) && ((call && s.flags[5]) || bankrupt);
        let mut unfilled_liquidation = false;
        commands.clear();
        if rebalance {
            pending.clear();
            for &i in st.held.iter().chain(&book.held) {
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
        let mut fm = fm_before;
        let value = account.equity();
        if forced {
            pending.clear();
            let requested_commands = commands.clone();
            commands.clear();
            if FUT {
                // Liquidation replaces the plan, but a nonzero target for an
                // expired contract is still reported rather than dropped.
                let f = m.futures.as_ref().unwrap();
                for &(i, w) in &requested_commands {
                    if f.kind[i] != 0 && day as i64 > f.expiry[i] && w != 0.0 {
                        out.daily[29 * m.t + day] += 1.0;
                        out.counts[37] += 1;
                    }
                }
            }
            let after_liquidation = |keep: f64| {
                let (mut a, mut margin) = (account, fm);
                for &i in book.held.iter().chain(&st.held) {
                    let current = st.shares[i];
                    let (minimum, step) = lot_terms::<FUT>(m, s, targets, i);
                    let magnitude =
                        round_close(current.abs() * (1.0 - keep), current.abs(), minimum, step);
                    let q = -current.signum() * magnitude;
                    if q != 0.0 && m.price[base + i].is_finite() {
                        let x = quote_any::<FUT>(q, i, day, m, s);
                        if x.price > 0.0 && x.price.is_finite() {
                            (a, margin) = project_any::<FUT>(
                                a,
                                margin,
                                i,
                                q,
                                &st.shares,
                                &book.basis,
                                day,
                                m,
                                s,
                            );
                        }
                    }
                }
                (a, margin)
            };
            let all = after_liquidation(0.0);
            let none = after_liquidation(1.0);
            let keep = if bankrupt || value <= 0.0 || !feasible(all.0, all.1, s, day) {
                0.0
            } else if feasible(none.0, none.1, s, day) {
                1.0
            } else {
                let (mut lo, mut hi) = (0.0, 1.0);
                for _ in 0..64 {
                    let mid = (lo + hi) * 0.5;
                    let x = after_liquidation(mid);
                    if feasible(x.0, x.1, s, day) {
                        lo = mid
                    } else {
                        hi = mid
                    }
                }
                lo
            };
            for &i in book.held.iter().chain(&st.held) {
                let current = st.shares[i];
                let future = is_future::<FUT>(m, i);
                let (minimum, step) = lot_terms::<FUT>(m, s, targets, i);
                let mut amount =
                    round_close(current.abs() * (1.0 - keep), current.abs(), minimum, step);
                if rebalance && let Some(&(_, w)) = requested_commands.iter().find(|x| x.0 == i) {
                    let p = m.price[base + i];
                    let mark = if p.is_finite() {
                        p
                    } else if future {
                        book.basis[i]
                    } else {
                        st.last[i]
                    };
                    let desired = if targets.shares {
                        w
                    } else if mark > 0.0 {
                        let raw = w * value.max(0.0) * (1.0 - s.scalars[8]);
                        if future {
                            raw / (mark * m.futures.as_ref().unwrap().multiplier[i])
                        } else {
                            raw / mark
                        }
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
        future_orders.clear();
        if !CLASSIC {
            if FUT {
                // Futures reductions release collateral first; spot sales then covers.
                commands.sort_by_key(|&(i, _)| (!is_future::<FUT>(m, i), st.shares[i] < 0.0, i));
            } else {
                commands.sort_by_key(|&(i, _)| (st.shares[i] < 0.0, i));
            }
        } else if FUT {
            // Stable: the legacy spot order is unchanged.
            commands.sort_by_key(|&(i, _)| !is_future::<FUT>(m, i));
        }
        for &(i, w) in &commands {
            let future = is_future::<FUT>(m, i);
            if m.first_delist[i] <= day as i64 {
                continue;
            }
            if future && day as i64 > m.futures.as_ref().unwrap().expiry[i] {
                if w != 0.0 {
                    out.daily[29 * m.t + day] += 1.0;
                    out.counts[37] += 1;
                }
                continue;
            }
            if !future && st.shares[i] == 0.0 {
                st.last[i] = past_close(m, day, i);
            }
            let p = m.price[base + i];
            let mark = if p.is_finite() {
                p
            } else if future {
                // Held: the last settlement basis. Unheld: the previous row's
                // published settlement, as unheld spot uses the previous close.
                if st.shares[i] != 0.0 {
                    book.basis[i]
                } else if day > 0 {
                    m.futures.as_ref().unwrap().settlement[base - m.n + i]
                } else {
                    0.0
                }
            } else {
                st.last[i]
            };
            let (minimum, step) = lot_terms::<FUT>(m, s, targets, i);
            let mut rounded_to_zero = false;
            let mut target = if forced {
                w
            } else if targets.shares {
                out.counts[9] += 1;
                w
            } else if mark > 0.0 {
                if future {
                    let raw = w * value.max(0.0) * (1.0 - s.scalars[8])
                        / (mark * m.futures.as_ref().unwrap().multiplier[i]);
                    // Whole contracts, rounded toward zero.
                    let contracts = raw.signum() * raw.abs().floor();
                    rounded_to_zero = raw != 0.0 && contracts == 0.0;
                    contracts
                } else {
                    let raw = w * value.max(0.0) * (1.0 - s.scalars[8]) / mark;
                    // A nonzero residual below the opening minimum is not a close.
                    if raw.abs() < minimum && st.shares[i] * raw > 0.0 {
                        raw
                    } else {
                        raw.signum() * lot(raw.abs(), minimum, step)
                    }
                }
            } else {
                0.0
            };
            if (target - st.shares[i]).abs() <= quantity_tolerance(target, st.shares[i]) {
                target = st.shares[i];
            }
            let held_notional = if future {
                st.shares[i] * mark * m.futures.as_ref().unwrap().multiplier[i]
            } else {
                st.shares[i] * mark
            };
            if !forced
                && !targets.shares
                && w != 0.0
                && st.shares[i] != 0.0
                && value > 0.0
                && (w - held_notional / value).abs() < s.scalars[9]
            {
                out.counts[10] += 1;
                continue;
            }
            if FUT && rounded_to_zero {
                out.counts[38] += 1;
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
                        m.buy[base + i] != 0
                    } else {
                        m.sell[base + i] != 0
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
                if future {
                    // A contract reduction only releases collateral; its loss is
                    // settled at the close whether or not it is executed, so it is
                    // never trimmed by the own-cash rule. Fees may create debt.
                    if q != 0.0 {
                        let x = futures::quote(q, i, day, m, s);
                        if x.price > 0.0 && x.price.is_finite() {
                            (account, fm) = futures::project(
                                account,
                                fm,
                                i,
                                q,
                                st.shares[i],
                                book.basis[i],
                                day,
                                m,
                                s,
                            );
                            book.execute(
                                &mut st.shares,
                                &mut st.cash,
                                i,
                                q,
                                day,
                                !rebalance,
                                forced,
                                m,
                                s,
                                &mut out,
                            );
                            used = q.abs();
                        } else {
                            out.daily[25 * m.t + day] += 1.0;
                            out.counts[28] += 1;
                        }
                    }
                } else {
                    if !CLASSIC
                        && q != 0.0
                        && quote(q, i, day, m, s).price > 0.0
                        && quote(q, i, day, m, s).price.is_finite()
                        && !s.flags[3]
                        && account.own_cash(s.flags[4]) - fm >= -1e-10 * s.scalars[5]
                        && projected(account, i, q, st.shares[i], day, m, s).own_cash(s.flags[4])
                            - fm
                            < 0.0
                    {
                        let requested_quantity = q;
                        let (mut lo, mut hi) = (0.0, q.abs());
                        for _ in 0..64 {
                            let mid = (lo + hi) * 0.5;
                            let x = q.signum() * round_close(mid, current.abs(), minimum, step);
                            if projected(account, i, x, st.shares[i], day, m, s)
                                .own_cash(s.flags[4])
                                - fm
                                >= 0.0
                            {
                                lo = mid
                            } else {
                                hi = mid
                            }
                        }
                        q = q.signum() * round_close(lo, current.abs(), minimum, step);
                        if (q - requested_quantity).abs()
                            > quantity_tolerance(q, requested_quantity)
                        {
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
                        // The cash-only special case retains its exact legacy check
                        // while cash is nonnegative. Once a futures loss has left
                        // debt, a sale only has to pay its own fees: it must never
                        // deepen the debt, but it need not repay it in one fill.
                        if x.price > 0.0
                            && x.price.is_finite()
                            && (!CLASSIC
                                || if st.cash >= 0.0 {
                                    st.cash + x.amount
                                        >= x.fees.commission + x.fees.tax + x.fees.other
                                } else {
                                    x.amount >= x.fees.commission + x.fees.tax + x.fees.other
                                })
                        {
                            fill(day, i, q, !rebalance, forced, m, s, &mut st, &mut out);
                            account = candidate;
                            used = q.abs();
                        } else if !x.price.is_finite() || x.price <= 0.0 {
                            out.daily[25 * m.t + day] += 1.0;
                            out.counts[28] += 1;
                        }
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
                        m.buy[base + i] != 0
                    } else {
                        // Future shorts borrow no securities.
                        m.sell[base + i] != 0
                            && (future
                                || (s.flags[2]
                                    && (m.can_short.is_empty() || (m.can_short[base + i] != 0))))
                    };
                let mut q = if allowed {
                    opening.signum() * lot(opening.abs().min((cap - used).max(0.0)), minimum, step)
                } else {
                    0.0
                };
                if q != 0.0 {
                    let x = quote_any::<FUT>(q, i, day, m, s);
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
                    if CLASSIC && future {
                        future_orders.push((i, q));
                    } else {
                        orders.push((i, q));
                    }
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
        if CLASSIC && FUT && !future_orders.is_empty() {
            // Cash account: contracts reserve collateral from the current cash
            // (with long_initial = 1 the test is collateral <= cash).
            let mut a = Account {
                cash: st.cash,
                ..Default::default()
            };
            open_basket::<FUT>(
                &mut future_orders,
                &mut a,
                &mut fm,
                false,
                rebalance,
                day,
                m,
                targets,
                s,
                &mut st,
                &mut book,
                &mut out,
            );
        }
        if !orders.is_empty() {
            if CLASSIC {
                // Spot purchases may use cash not reserved as futures collateral.
                let free = |cash: f64| if FUT { cash - fm } else { cash };
                let total: f64 = orders
                    .iter()
                    .map(|&(i, q)| -movement(q, quote(q, i, day, m, s)))
                    .sum();
                let scale = if total > free(st.cash) {
                    out.counts[3] += 1;
                    free(st.cash) / total
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
                    if q > 0.0 && -movement(q, quote(q, i, day, m, s)) > free(st.cash) {
                        q = affordable(free(st.cash), q, i, day, step, min, m, s);
                    }
                    if q > quantity_tolerance(st.shares[i] + desired, st.shares[i]) {
                        fill(day, i, q, !rebalance, false, m, s, &mut st, &mut out);
                    }
                }
            } else {
                open_basket::<FUT>(
                    &mut orders,
                    &mut account,
                    &mut fm,
                    true,
                    rebalance,
                    day,
                    m,
                    targets,
                    s,
                    &mut st,
                    &mut book,
                    &mut out,
                );
            }
        }
        if !commands.is_empty() {
            st.held.sort_unstable();
            st.held.dedup();
            st.held.retain(|&i| st.shares[i] != 0.0);
            if FUT {
                book.held.sort_unstable();
                book.held.dedup();
                book.held.retain(|&i| st.shares[i] != 0.0);
            }
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
        let closing_fm = if FUT {
            book.settle(&mut st.shares, &mut st.cash, day, m, &mut out)
        } else {
            0.0
        };
        let own = st.cash - if s.flags[4] { 0.0 } else { short };
        let own = if FUT { own - closing_fm } else { own };
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
        let mut required = s.v(11, day) * long + s.v(12, day) * short;
        let mut maintenance = s.v(13, day) * long + s.v(14, day) * short;
        let mut tolerance = 1e-10 * equity.abs().max(long).max(short).max(1.0);
        if FUT {
            required += closing_fm;
            maintenance += closing_fm;
            tolerance = tolerance.max(1e-10 * closing_fm);
        }
        // A cash account is monitored only after it has held a contract.
        let monitored = !CLASSIC || (FUT && book.seen);
        call = monitored && (maintenance > equity + tolerance || unfilled_liquidation);
        if call {
            out.counts[23] += 1;
            if FUT && closing_fm > 0.0 {
                out.counts[34] += 1;
            }
        }
        let valued = monitored && st.held.iter().all(|&i| st.last[i] > 0.0);
        if monitored
            && equity <= 0.0
            && !bankrupt
            && valued
            && (!st.held.is_empty() || (FUT && !book.held.is_empty()) || st.cash < 0.0)
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
        if FUT {
            out.daily[27 * m.t + day] = closing_fm;
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
pub mod validate;
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
