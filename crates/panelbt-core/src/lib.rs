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
    /// Limit price per entry (NaN: ordinary target); empty when none.
    pub limits: &'a [f64],
    pub shares: bool,
}
pub struct Settings<'a> {
    pub vectors: Vec<&'a [f64]>,
    pub scalars: &'a [f64],
    pub flags: &'a [bool],
    pub slippage: &'a [f64],
    pub impact: &'a [f64],
    pub borrow: &'a [f64],
    /// Declarative exit rules and limit-order settings (spot only).
    pub rules: Option<rules::Rules<'a>>,
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
#[derive(Clone)]
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
    /// (trade index, reason) for rule and limit fills; other fills rebalance.
    pub reasons: Vec<(usize, u8)>,
    /// Rule counters; empty unless rules are configured.
    pub rule_counts: Vec<u64>,
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
#[inline]
fn quote(q: f64, i: usize, day: usize, m: &Market<'_>, s: &Settings<'_>) -> Quote {
    quote_at(q, i, day, m.price[day * m.n + i], m, s)
}
/// Quote a fill at an explicit raw (pre-slippage) price `p`.
#[inline]
fn quote_at(q: f64, i: usize, day: usize, p: f64, m: &Market<'_>, s: &Settings<'_>) -> Quote {
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
/// Account state. The per-security vectors are borrowed so that the stepping
/// interface can keep them in caller-visible arrays without copying.
pub struct State<'s> {
    pub cash: f64,
    pub shares: &'s mut [f64],
    pub last: &'s mut [f64],
    pub held: Vec<usize>,
}
/// Per-position bookkeeping maintained only when `T` (tracking) is compiled
/// in: the stepping interface and declarative rules. Empty slices otherwise.
pub struct Track<'s> {
    /// Quantity-weighted raw execution price of the open position.
    pub cost: &'s mut [f64],
    /// Row on which the open position was entered; -1 when flat.
    pub entry: &'s mut [i64],
    /// Highest high (long) or lowest low (short) since entry, through the
    /// previous row; the entry price on the entry row.
    pub peak: &'s mut [f64],
}
impl Track<'static> {
    pub fn none() -> Self {
        Track {
            cost: &mut [],
            entry: &mut [],
            peak: &mut [],
        }
    }
}
#[inline]
fn track_fill(
    trk: &mut Track<'_>,
    i: usize,
    before: f64,
    after: f64,
    q: f64,
    raw: f64,
    day: usize,
) {
    if after == 0.0 {
        trk.cost[i] = 0.0;
        trk.entry[i] = -1;
        trk.peak[i] = 0.0;
    } else if before == 0.0 || before.signum() != after.signum() {
        trk.cost[i] = raw;
        trk.entry[i] = day as i64;
        trk.peak[i] = raw;
    } else if after.abs() > before.abs() {
        trk.cost[i] = (before.abs() * trk.cost[i] + q.abs() * raw) / after.abs();
    }
}
impl State<'_> {
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
#[inline]
fn fill<const T: bool>(
    day: usize,
    i: usize,
    q: f64,
    raw: f64,
    carry: bool,
    forced: bool,
    m: &Market<'_>,
    s: &Settings<'_>,
    st: &mut State<'_>,
    trk: &mut Track<'_>,
    rb: &mut rules::RuleBook,
    out: &mut Output,
) {
    let x = quote_at(q, i, day, raw, m, s);
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
    if T {
        track_fill(trk, i, before, st.shares[i], q, raw, day);
        rb.on_fill(i, before, q, st.shares[i]);
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
    // Rules and limit targets compile in position tracking. With futures
    // columns only exit rules are accepted at input (they act on spot columns).
    let track = s.rules.is_some() || !targets.limits.is_empty();
    match (s.flags[6], m.futures.is_some(), track) {
        (true, false, false) => run_impl::<true, false, false>(m, targets, s),
        (false, false, false) => run_impl::<false, false, false>(m, targets, s),
        (true, true, false) => run_impl::<true, true, false>(m, targets, s),
        (false, true, false) => run_impl::<false, true, false>(m, targets, s),
        (true, false, true) => run_impl::<true, false, true>(m, targets, s),
        (false, false, true) => run_impl::<false, false, true>(m, targets, s),
        (true, true, true) => run_impl::<true, true, true>(m, targets, s),
        (false, true, true) => run_impl::<false, true, true>(m, targets, s),
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
    shares: bool,
    i: usize,
) -> (f64, f64) {
    if is_future::<FUT>(m, i) {
        (1.0, 1.0)
    } else {
        (s.vectors[6][i], if shares { 0.0 } else { s.vectors[7][i] })
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
fn open_basket<const FUT: bool, const T: bool>(
    orders: &mut [(usize, f64)],
    account: &mut Account,
    fm: &mut f64,
    count_spot_scaling: bool,
    rebalance: bool,
    day: usize,
    m: &Market<'_>,
    shares: bool,
    s: &Settings<'_>,
    st: &mut State<'_>,
    book: &mut futures::Book<'_>,
    trk: &mut Track<'_>,
    rb: &mut rules::RuleBook,
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
            let (minimum, step) = lot_terms::<FUT>(m, s, shares, i);
            let q = q.signum() * lot(q.abs() * scale, minimum, step);
            if q != 0.0 {
                (a, margin) = project_any::<FUT>(a, margin, i, q, st.shares, book.basis, day, m, s);
            }
        }
        (a, margin)
    };
    let full = trial(1.0);
    let scale = if feasible(full.0, full.1, s, day) {
        1.0
    } else {
        if count_spot_scaling && !(T && rb.limit_scaled[1]) {
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
        let (minimum, step) = lot_terms::<FUT>(m, s, shares, i);
        let mut q = desired.signum() * lot(desired.abs() * scale, minimum, step);
        // An already deficient own-cash balance can make an early
        // short opening infeasible even when the whole basket fits.
        if q != 0.0 {
            let x = project_any::<FUT>(*account, *fm, i, q, st.shares, book.basis, day, m, s);
            if !feasible(x.0, x.1, s, day) {
                let (mut lo, mut hi) = (0.0, q.abs());
                for _ in 0..64 {
                    let mid = (lo + hi) * 0.5;
                    let c = q.signum() * lot(mid, minimum, step);
                    let y =
                        project_any::<FUT>(*account, *fm, i, c, st.shares, book.basis, day, m, s);
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
                project_any::<FUT>(*account, *fm, i, q, st.shares, book.basis, day, m, s);
            if is_future::<FUT>(m, i) {
                book.execute(
                    st.shares,
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
                fill::<T>(
                    day,
                    i,
                    q,
                    m.price[day * m.n + i],
                    !rebalance,
                    false,
                    m,
                    s,
                    st,
                    trk,
                    rb,
                    out,
                );
            }
        }
    }
}
/// A target row: sorted unique columns and their weights or quantities.
#[derive(Clone, Copy)]
pub struct Row<'a> {
    pub cols: &'a [i64],
    pub weights: &'a [f64],
    /// Limit prices (NaN: ordinary target); empty when none.
    pub limits: &'a [f64],
}
/// The daily account loop, split into a pre-open phase (external flow,
/// corporate actions, delisting) and a trading-and-close phase. `run` drives
/// both phases for every row; the stepping interface drives them one row at a
/// time. Everything that persists between rows lives here or in the borrowed
/// `State`, `futures::Book` and `Track`.
#[derive(Clone)]
pub struct Core {
    pub out: Output,
    pending: Vec<(usize, f64)>,
    commands: Vec<(usize, f64)>,
    orders: Vec<(usize, f64)>,
    future_orders: Vec<(usize, f64)>,
    weights: Vec<f64>,
    forced_quantities: Vec<f64>,
    previous: f64,
    unit: f64,
    call: bool,
    bankrupt: bool,
    fm_before: f64,
    flow: f64,
    pub rb: rules::RuleBook,
}
impl Core {
    /// `track` allocates the rule state (stepping interface, rules).
    pub fn new(m: &Market<'_>, s: &Settings<'_>, track: bool) -> Core {
        Core {
            out: Output {
                daily: vec![0.0; DAILY * m.t],
                positions: s.flags[1].then(|| vec![0.0; m.t * m.n]),
                trades: s.flags[1].then(Vec::new),
                events: Vec::new(),
                counts: [0; COUNTS],
                contribution: vec![0.0; m.n],
                liquidations: Vec::new(),
                bankruptcies: Vec::new(),
                expiries: Vec::new(),
                reasons: Vec::new(),
                rule_counts: Vec::new(),
            },
            pending: Vec::new(),
            commands: Vec::new(),
            orders: Vec::new(),
            future_orders: Vec::new(),
            weights: vec![0.0; m.n],
            forced_quantities: vec![0.0; m.n],
            previous: s.scalars[5],
            unit: 1.0,
            call: false,
            bankrupt: false,
            fm_before: 0.0,
            flow: 0.0,
            rb: if track {
                rules::RuleBook::new(m.n)
            } else {
                rules::RuleBook::default()
            },
        }
    }
    /// Pre-open phase of `day`: external flow, corporate actions, delisting.
    #[inline(always)]
    pub fn open<const CLASSIC: bool, const FUT: bool, const T: bool>(
        &mut self,
        day: usize,
        m: &Market<'_>,
        s: &Settings<'_>,
        st: &mut State<'_>,
        book: &mut futures::Book<'_>,
        trk: &mut Track<'_>,
    ) {
        let Core {
            out,
            fm_before,
            flow,
            rb,
            ..
        } = self;
        let base = day * m.n;
        // Futures collateral at the last settlement and today's rate.
        *fm_before = if FUT {
            futures::held_margin(&book.held, st.shares, book.basis, day, m)
        } else {
            0.0
        };
        // External flow is applied before corporate actions. Restricted proceeds
        // and futures collateral cannot be withdrawn; debt is never increased.
        let available = if CLASSIC {
            if FUT {
                (st.cash - *fm_before).max(0.0)
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
                (a.own_cash(s.flags[4]) - *fm_before).max(0.0)
            } else {
                a.own_cash(s.flags[4]).max(0.0)
            }
        };
        let requested = s.v(10, day);
        *flow = requested.max(-available);
        st.cash += *flow;
        out.daily[12 * m.t + day] = *flow;
        out.daily[13 * m.t + day] = *flow - requested;
        if *flow != 0.0 {
            out.counts[12] += 1;
        }
        if *flow > requested {
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
                if T && s.rules.as_ref().is_some_and(|r| r.div_adjust) {
                    trk.cost[i] = (trk.cost[i] - m.dividend[base + i]).max(0.0);
                    trk.peak[i] = (trk.peak[i] - m.dividend[base + i]).max(0.0);
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
                if T {
                    trk.cost[i] /= f;
                    trk.peak[i] /= f;
                }
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
                if T {
                    trk.cost[i] = 0.0;
                    trk.entry[i] = -1;
                    trk.peak[i] = 0.0;
                    if !rb.due.is_empty() {
                        rb.due[i] = 0;
                    }
                }
            }
        }
        if removed {
            st.held.retain(|&i| st.shares[i] != 0.0);
        }
    }
    /// Trading and closing phase of `day`; `row` is the day's new target row.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub fn trade<const CLASSIC: bool, const FUT: bool, const T: bool>(
        &mut self,
        day: usize,
        row: Option<Row<'_>>,
        shares: bool,
        m: &Market<'_>,
        s: &Settings<'_>,
        st: &mut State<'_>,
        book: &mut futures::Book<'_>,
        trk: &mut Track<'_>,
    ) {
        let Core {
            out,
            pending,
            commands,
            orders,
            future_orders,
            weights,
            forced_quantities,
            previous,
            unit,
            call,
            bankrupt,
            fm_before,
            flow,
            rb,
        } = self;
        let base = day * m.n;
        let rebalance = row.is_some();
        // Rules are compiled in only with tracking; `rules_on` is then a
        // runtime switch shared by the whole row.
        let rules = if T { s.rules.as_ref() } else { None };
        if let Some(r) = rules {
            // `cash_scaled`/`margin_scaled` count a row once whether an
            // ordinary purchase, a limit order or both were cut.
            rb.row_scaled = [out.counts[3], out.counts[20]];
            rb.limit_scaled = [false, false];
            rb.entry_fresh = r.timing != 0;
            if let Some(row) = row {
                rb.begin_row(day, row.cols, row.weights, row.limits, r);
            }
        }
        // A cash account without futures never records calls or insolvency.
        let forced = (!CLASSIC || FUT) && ((*call && s.flags[5]) || *bankrupt);
        let mut unfilled_liquidation = false;
        commands.clear();
        if rebalance {
            pending.clear();
            for &i in st.held.iter().chain(&book.held) {
                weights[i] = 0.0;
                commands.push((i, 0.0));
            }
            let row = row.unwrap();
            for j in 0..row.cols.len() {
                let i = row.cols[j] as usize;
                weights[i] = row.weights[j];
                commands.push((i, weights[i]));
            }
            commands.sort_unstable_by_key(|x| x.0);
            commands.dedup_by_key(|x| x.0);
            for c in commands.iter_mut() {
                c.1 = weights[c.0];
            }
        } else if s.flags[0] {
            std::mem::swap(commands, pending);
            pending.clear();
        }
        if let Some(r) = rules
            && r.timing != 0
            && !forced
        {
            rules::intraday::<CLASSIC>(
                day, shares, *bankrupt, *fm_before, m, s, r, st, trk, rb, out,
            );
        }
        let mut account = if !commands.is_empty() || forced || rules.is_some() {
            st.account(day, m)
        } else {
            Account::default()
        };
        let mut fm = *fm_before;
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
                    let (minimum, step) = lot_terms::<FUT>(m, s, shares, i);
                    let magnitude =
                        round_close(current.abs() * (1.0 - keep), current.abs(), minimum, step);
                    let q = -current.signum() * magnitude;
                    if q != 0.0 && m.price[base + i].is_finite() {
                        let x = quote_any::<FUT>(q, i, day, m, s);
                        if x.price > 0.0 && x.price.is_finite() {
                            (a, margin) = project_any::<FUT>(
                                a, margin, i, q, st.shares, book.basis, day, m, s,
                            );
                        }
                    }
                }
                (a, margin)
            };
            let all = after_liquidation(0.0);
            let none = after_liquidation(1.0);
            let keep = if *bankrupt || value <= 0.0 || !feasible(all.0, all.1, s, day) {
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
                let (minimum, step) = lot_terms::<FUT>(m, s, shares, i);
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
                    let desired = if shares {
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
        for &(i, w) in commands.iter() {
            let future = is_future::<FUT>(m, i);
            if m.first_delist[i] <= day as i64 {
                continue;
            }
            // A rule exit, a due exit or a limit entry owns the column today.
            if rules.is_some() && !forced && (rb.exited[i] || rb.due[i] != 0 || rb.limit_today[i]) {
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
            let (minimum, step) = lot_terms::<FUT>(m, s, shares, i);
            let mut rounded_to_zero = false;
            let mut target = if forced {
                w
            } else if shares {
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
                && !shares
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
                    && tradecore::direction_allowed(
                        reduction > 0.0,
                        m.buy[base + i] != 0,
                        m.sell[base + i] != 0,
                    );
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
                                st.shares,
                                &mut st.cash,
                                i,
                                q,
                                day,
                                !rebalance,
                                forced,
                                m,
                                s,
                                out,
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
                            fill::<T>(day, i, q, p, !rebalance, forced, m, s, st, trk, rb, out);
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
            let reentry = if let Some(r) = rules {
                opening.abs() > quantity_tolerance(target, after)
                    && !forced
                    && !*bankrupt
                    && rb.blocks(i, day, r)
            } else {
                false
            };
            if reentry {
                rb.counts[5] += 1;
            } else if opening.abs() > quantity_tolerance(target, after) && !forced && !*bankrupt {
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
            open_basket::<FUT, T>(
                future_orders,
                &mut a,
                &mut fm,
                false,
                rebalance,
                day,
                m,
                shares,
                s,
                st,
                book,
                trk,
                rb,
                out,
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
                    if !(T && rb.limit_scaled[0]) {
                        out.counts[3] += 1;
                    }
                    free(st.cash) / total
                } else {
                    1.0
                };
                for &(i, desired) in orders.iter() {
                    let step = if shares { 0.0 } else { s.vectors[7][i] };
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
                        fill::<T>(
                            day,
                            i,
                            q,
                            m.price[base + i],
                            !rebalance,
                            false,
                            m,
                            s,
                            st,
                            trk,
                            rb,
                            out,
                        );
                    }
                }
            } else {
                open_basket::<FUT, T>(
                    orders,
                    &mut account,
                    &mut fm,
                    true,
                    rebalance,
                    day,
                    m,
                    shares,
                    s,
                    st,
                    book,
                    trk,
                    rb,
                    out,
                );
            }
        }
        if let Some(r) = rules
            && r.timing == 0
            && !forced
        {
            rules::intraday::<CLASSIC>(day, shares, *bankrupt, fm, m, s, r, st, trk, rb, out);
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
        if let Some(r) = rules {
            // Today's range extends the peak only of positions that saw all of it.
            for &i in &st.held {
                if !rb.fresh[i] {
                    let k = base + i;
                    if st.shares[i] > 0.0 && r.high[k].is_finite() {
                        trk.peak[i] = trk.peak[i].max(r.high[k]);
                    } else if st.shares[i] < 0.0 && r.low[k].is_finite() {
                        trk.peak[i] = trk.peak[i].min(r.low[k]);
                    }
                }
            }
            rb.end_row(r);
        }
        let closing_fm = if FUT {
            book.settle(st.shares, &mut st.cash, day, m, out)
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
        let denom = *previous + *flow;
        let ret = if denom > 8.0 * f64::EPSILON * s.scalars[5].max(previous.abs()).max(flow.abs()) {
            equity / denom - 1.0
        } else {
            0.0
        };
        *unit *= 1.0 + ret;
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
        *call = monitored && (maintenance > equity + tolerance || unfilled_liquidation);
        if *call {
            out.counts[23] += 1;
            if FUT && closing_fm > 0.0 {
                out.counts[34] += 1;
            }
        }
        let valued = monitored && st.held.iter().all(|&i| st.last[i] > 0.0);
        if monitored
            && equity <= 0.0
            && !*bankrupt
            && valued
            && (!st.held.is_empty() || (FUT && !book.held.is_empty()) || st.cash < 0.0)
        {
            *bankrupt = true;
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
            (16, *unit),
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
            (22, if *call { 1.0 } else { 0.0 }),
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
        *previous = equity;
        if let Some(p) = out.positions.as_mut() {
            p[base..base + m.n].copy_from_slice(st.shares);
        }
    }
    /// Final marks: contribution of the positions still held after the last row.
    pub fn finish(&mut self, st: &State<'_>, s: &Settings<'_>) {
        for &i in &st.held {
            self.out.contribution[i] += st.shares[i] * st.last[i];
        }
        if s.rules.is_some() {
            self.out.rule_counts = self.rb.counts.to_vec();
        }
    }
    /// A maintenance call at the last close (liquidation is due on this row).
    pub fn margin_call(&self) -> bool {
        self.call
    }
    pub fn bankrupt(&self) -> bool {
        self.bankrupt
    }
    /// Equity at the last close (initial cash before the first row).
    pub fn equity(&self) -> f64 {
        self.previous
    }
    /// Carried price-, mask- or volume-blocked orders: (column, target).
    pub fn pending(&self) -> &[(usize, f64)] {
        &self.pending
    }
}
/// One daily account loop for every account type. `CLASSIC` selects the exact
/// long-only cash specialization; `FUT` compiles in futures columns. With
/// `FUT = true`, a futures column that is never held or targeted executes no
/// futures arithmetic, so spot results equal the `FUT = false` build bit for bit.
/// `run` drives the same two phases as the stepping interface, with position
/// tracking compiled out (`T = false`).
fn run_impl<const CLASSIC: bool, const FUT: bool, const T: bool>(
    m: &Market<'_>,
    targets: &Targets<'_>,
    s: &Settings<'_>,
) -> Output {
    let mut shares = vec![0.0; m.n];
    let mut last = vec![0.0; m.n];
    let mut basis = if FUT { vec![0.0; m.n] } else { Vec::new() };
    let mut st = State {
        cash: s.scalars[5],
        shares: &mut shares,
        last: &mut last,
        held: Vec::new(),
    };
    let mut book = futures::Book {
        basis: &mut basis,
        held: Vec::new(),
        seen: false,
    };
    let (mut cost, mut peak, mut entry) = if T {
        (vec![0.0; m.n], vec![0.0; m.n], vec![-1; m.n])
    } else {
        (Vec::new(), Vec::new(), Vec::new())
    };
    let mut trk = Track {
        cost: &mut cost,
        entry: &mut entry,
        peak: &mut peak,
    };
    let mut core = Core::new(m, s, T);
    let mut cursor = 0;
    for day in 0..m.t {
        core.open::<CLASSIC, FUT, T>(day, m, s, &mut st, &mut book, &mut trk);
        let row = if cursor < targets.days.len() && targets.days[cursor] as usize == day {
            let (a, b) = (
                targets.indptr[cursor] as usize,
                targets.indptr[cursor + 1] as usize,
            );
            cursor += 1;
            Some(Row {
                cols: &targets.cols[a..b],
                weights: &targets.weights[a..b],
                limits: if targets.limits.is_empty() {
                    &[]
                } else {
                    &targets.limits[a..b]
                },
            })
        } else {
            None
        };
        core.trade::<CLASSIC, FUT, T>(day, row, targets.shares, m, s, &mut st, &mut book, &mut trk);
    }
    core.finish(&st, s);
    core.out
}

pub mod futures;
pub mod group_weights;
pub mod rules;
pub mod validate;
pub mod weights;
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

pub mod codec;
use codec::Codec;

/// Caller-owned per-security vectors of a stepped simulation. `basis` holds
/// futures settlement bases; `cost`, `entry` and `peak` are the position
/// bookkeeping of `Track`.
pub struct Vectors<'s> {
    pub shares: &'s mut [f64],
    pub last: &'s mut [f64],
    pub basis: &'s mut [f64],
    pub cost: &'s mut [f64],
    pub peak: &'s mut [f64],
    pub entry: &'s mut [i64],
}

/// Everything a stepped simulation owns apart from its `Vectors`.
///
/// `begin` runs the pre-open phase of row 0. Each `step` runs the trading and
/// closing phase of the current row and then the pre-open phase of the next
/// one, so between steps the visible state is exactly what is known before the
/// next row's trading: the previous close plus that row's cash flow, corporate
/// actions and delisting. The phases and their order are those of `run`.
#[derive(Clone)]
pub struct Stepper {
    pub core: Core,
    pub cash: f64,
    pub held: Vec<usize>,
    pub book_held: Vec<usize>,
    pub seen: bool,
    pub day: usize,
}

/// Format version 2: a CRC-32 of everything before it closes the blob.
const SNAPSHOT_MAGIC: &[u8; 8] = b"PBTSTEP2";

impl Stepper {
    /// A placeholder with no rows; replaced by `new` before use.
    pub fn new_empty() -> Stepper {
        Stepper {
            core: Core {
                out: Output {
                    daily: Vec::new(),
                    positions: None,
                    trades: None,
                    events: Vec::new(),
                    counts: [0; COUNTS],
                    contribution: Vec::new(),
                    liquidations: Vec::new(),
                    bankruptcies: Vec::new(),
                    expiries: Vec::new(),
                    reasons: Vec::new(),
                    rule_counts: Vec::new(),
                },
                pending: Vec::new(),
                commands: Vec::new(),
                orders: Vec::new(),
                future_orders: Vec::new(),
                weights: Vec::new(),
                forced_quantities: Vec::new(),
                previous: 0.0,
                unit: 1.0,
                call: false,
                bankrupt: false,
                fm_before: 0.0,
                flow: 0.0,
                rb: rules::RuleBook::default(),
            },
            cash: 0.0,
            held: Vec::new(),
            book_held: Vec::new(),
            seen: false,
            day: 0,
        }
    }
    pub fn new(m: &Market<'_>, s: &Settings<'_>) -> Stepper {
        Stepper {
            core: Core::new(m, s, true),
            cash: s.scalars[5],
            held: Vec::new(),
            book_held: Vec::new(),
            seen: false,
            day: 0,
        }
    }
    fn with<R>(
        &mut self,
        v: Vectors<'_>,
        f: impl FnOnce(&mut Core, &mut State<'_>, &mut futures::Book<'_>, &mut Track<'_>) -> R,
    ) -> R {
        let mut st = State {
            cash: self.cash,
            shares: v.shares,
            last: v.last,
            held: std::mem::take(&mut self.held),
        };
        let mut book = futures::Book {
            basis: v.basis,
            held: std::mem::take(&mut self.book_held),
            seen: self.seen,
        };
        let mut trk = Track {
            cost: v.cost,
            entry: v.entry,
            peak: v.peak,
        };
        let r = f(&mut self.core, &mut st, &mut book, &mut trk);
        self.cash = st.cash;
        self.held = st.held;
        self.book_held = book.held;
        self.seen = book.seen;
        r
    }
    /// Pre-open phase of row 0.
    pub fn begin(&mut self, v: Vectors<'_>, m: &Market<'_>, s: &Settings<'_>) {
        match (s.flags[6], m.futures.is_some()) {
            (true, false) => self.with(v, |c, st, b, k| {
                c.open::<true, false, true>(0, m, s, st, b, k)
            }),
            (false, false) => self.with(v, |c, st, b, k| {
                c.open::<false, false, true>(0, m, s, st, b, k)
            }),
            (true, true) => self.with(v, |c, st, b, k| {
                c.open::<true, true, true>(0, m, s, st, b, k)
            }),
            (false, true) => self.with(v, |c, st, b, k| {
                c.open::<false, true, true>(0, m, s, st, b, k)
            }),
        }
    }
    /// Trade and close the current row, then open the next one.
    pub fn step(
        &mut self,
        v: Vectors<'_>,
        row: Option<Row<'_>>,
        shares: bool,
        m: &Market<'_>,
        s: &Settings<'_>,
    ) {
        let day = self.day;
        assert!(day < m.t, "simulation already finished");
        match (s.flags[6], m.futures.is_some()) {
            (true, false) => self.with(v, |c, st, b, k| {
                step_day::<true, false>(c, day, row, shares, m, s, st, b, k)
            }),
            (false, false) => self.with(v, |c, st, b, k| {
                step_day::<false, false>(c, day, row, shares, m, s, st, b, k)
            }),
            (true, true) => self.with(v, |c, st, b, k| {
                step_day::<true, true>(c, day, row, shares, m, s, st, b, k)
            }),
            (false, true) => self.with(v, |c, st, b, k| {
                step_day::<false, true>(c, day, row, shares, m, s, st, b, k)
            }),
        }
        self.day += 1;
    }
    /// The finished output, with final marks added; the stepper is unchanged.
    pub fn result(&self, shares: &[f64], last: &[f64], rules: bool) -> Output {
        let mut out = self.core.out.clone();
        for &i in &self.held {
            out.contribution[i] += shares[i] * last[i];
        }
        if rules {
            out.rule_counts = self.core.rb.counts.to_vec();
        }
        out
    }
    /// Serialize; `header` fingerprints the inputs the snapshot belongs to.
    pub fn encode(&self, header: &[u64], v: &Vectors<'_>) -> Vec<u8> {
        let mut w = SNAPSHOT_MAGIC.to_vec();
        header.to_vec().put(&mut w);
        let c = &self.core;
        let o = &c.out;
        (
            o.daily.clone(),
            o.positions.clone(),
            o.trades.clone(),
            o.events.clone(),
        )
            .put(&mut w);
        o.counts.put(&mut w);
        (
            o.contribution.clone(),
            o.liquidations.clone(),
            o.bankruptcies.clone(),
            o.expiries.clone(),
            c.pending.clone(),
        )
            .put(&mut w);
        (
            c.weights.clone(),
            c.forced_quantities.clone(),
            c.previous,
            c.unit,
            c.call,
        )
            .put(&mut w);
        (
            c.bankrupt,
            c.fm_before,
            c.flow,
            self.cash,
            self.held.clone(),
        )
            .put(&mut w);
        (self.book_held.clone(), self.seen, self.day).put(&mut w);
        (
            v.shares.to_vec(),
            v.last.to_vec(),
            v.basis.to_vec(),
            v.cost.to_vec(),
            v.peak.to_vec(),
        )
            .put(&mut w);
        v.entry.to_vec().put(&mut w);
        (o.reasons.clone(), o.rule_counts.clone()).put(&mut w);
        let b = &c.rb;
        (
            b.bought.clone(),
            b.used.clone(),
            b.fresh.clone(),
            b.exited.clone(),
            b.marked.clone(),
        )
            .put(&mut w);
        (
            b.touched.clone(),
            b.due.clone(),
            b.block.clone(),
            b.blocked.clone(),
            b.limits.clone(),
        )
            .put(&mut w);
        (
            b.limit_today.clone(),
            b.limit_cols.clone(),
            b.entry_fresh,
            b.counts,
        )
            .put(&mut w);
        let crc = codec::crc32(&w);
        w.extend_from_slice(&crc.to_le_bytes());
        w
    }
    /// Restore a snapshot into `self` and `v`; the header must match exactly.
    pub fn decode(&mut self, blob: &[u8], header: &[u64], v: Vectors<'_>) -> Result<(), String> {
        if blob.starts_with(b"PBTSTEP1") {
            return Err(
                "snapshot: format version 1 is no longer supported; create a new snapshot".into(),
            );
        }
        let body = blob
            .strip_prefix(SNAPSHOT_MAGIC.as_slice())
            .ok_or("snapshot: not a panelbt simulator snapshot (or an older format)")?;
        if body.len() < 4 {
            return Err("snapshot: truncated".into());
        }
        let (mut r, crc) = body.split_at(body.len() - 4);
        if codec::crc32(&blob[..blob.len() - 4]).to_le_bytes() != crc {
            return Err("snapshot: checksum mismatch (corrupted or truncated)".into());
        }
        if Vec::<u64>::take(&mut r)? != header {
            return Err("snapshot: market shape, account type or settings differ".into());
        }
        #[allow(clippy::type_complexity)]
        let (daily, positions, trades, events): (
            Vec<f64>,
            Option<Vec<f64>>,
            Option<Vec<Trade>>,
            Vec<DelistEvent>,
        ) = Codec::take(&mut r)?;
        let counts = Codec::take(&mut r)?;
        #[allow(clippy::type_complexity)]
        let (contribution, liquidations, bankruptcies, expiries, pending): (
            Vec<f64>,
            Vec<(usize, usize, f64)>,
            Vec<usize>,
            Vec<(usize, usize, f64, f64, f64)>,
            Vec<(usize, f64)>,
        ) = Codec::take(&mut r)?;
        let (weights, forced_quantities, previous, unit, call): (
            Vec<f64>,
            Vec<f64>,
            f64,
            f64,
            bool,
        ) = Codec::take(&mut r)?;
        let (bankrupt, fm_before, flow, cash, held): (bool, f64, f64, f64, Vec<usize>) =
            Codec::take(&mut r)?;
        let (book_held, seen, day): (Vec<usize>, bool, usize) = Codec::take(&mut r)?;
        #[allow(clippy::type_complexity)]
        let (shares, last, basis, cost, peak): (
            Vec<f64>,
            Vec<f64>,
            Vec<f64>,
            Vec<f64>,
            Vec<f64>,
        ) = Codec::take(&mut r)?;
        let entry: Vec<i64> = Codec::take(&mut r)?;
        let (reasons, rule_counts): (Vec<(usize, u8)>, Vec<u64>) = Codec::take(&mut r)?;
        #[allow(clippy::type_complexity)]
        let (bought, used, fresh, exited, marked): (
            Vec<f64>,
            Vec<f64>,
            Vec<bool>,
            Vec<bool>,
            Vec<bool>,
        ) = Codec::take(&mut r)?;
        #[allow(clippy::type_complexity)]
        let (touched, due, block, blocked, limits): (
            Vec<usize>,
            Vec<u8>,
            Vec<i64>,
            Vec<usize>,
            Vec<(usize, f64, f64)>,
        ) = Codec::take(&mut r)?;
        let (limit_today, limit_cols, entry_fresh, book_counts): (
            Vec<bool>,
            Vec<usize>,
            bool,
            [u64; rules::RULE_COUNTS],
        ) = Codec::take(&mut r)?;
        let book = &self.core.rb;
        let sized = bought.len() == book.bought.len()
            && used.len() == book.used.len()
            && fresh.len() == book.fresh.len()
            && exited.len() == book.exited.len()
            && marked.len() == book.marked.len()
            && due.len() == book.due.len()
            && block.len() == book.block.len()
            && limit_today.len() == book.limit_today.len();
        let n0 = v.shares.len();
        let inside = |c: &[usize]| c.iter().all(|&i| i < n0);
        if !sized
            || !inside(&touched)
            || !inside(&blocked)
            || !inside(&limit_cols)
            || limits.iter().any(|x| x.0 >= n0)
        {
            return Err("snapshot: inconsistent sizes".into());
        }
        if !r.is_empty() {
            return Err("snapshot: trailing bytes".into());
        }
        let o = &self.core.out;
        let n = v.shares.len();
        let ok = |a: &Vec<f64>, b: &[f64]| a.len() == b.len();
        let held_ok = |h: &[usize]| h.iter().all(|&i| i < n);
        if !ok(&daily, &o.daily)
            || positions.as_ref().map(Vec::len) != o.positions.as_ref().map(Vec::len)
            || trades.is_some() != o.trades.is_some()
            || !ok(&contribution, &o.contribution)
            || !ok(&weights, &self.core.weights)
            || !ok(&forced_quantities, &self.core.forced_quantities)
            || !ok(&shares, v.shares)
            || !ok(&last, v.last)
            || !ok(&basis, v.basis)
            || !ok(&cost, v.cost)
            || !ok(&peak, v.peak)
            || entry.len() != v.entry.len()
            || !held_ok(&held)
            || !held_ok(&book_held)
            || pending.iter().any(|&(i, _)| i >= n)
            || day > o.daily.len() / DAILY
        {
            return Err("snapshot: inconsistent sizes".into());
        }
        self.core.out = Output {
            daily,
            positions,
            trades,
            events,
            counts,
            contribution,
            liquidations,
            bankruptcies,
            expiries,
            reasons,
            rule_counts,
        };
        let c = &mut self.core;
        c.rb = rules::RuleBook {
            bought,
            used,
            fresh,
            exited,
            marked,
            touched,
            due,
            block,
            blocked,
            limits,
            limit_today,
            limit_cols,
            entry_fresh,
            counts: book_counts,
            ..Default::default()
        };
        (c.pending, c.weights, c.forced_quantities) = (pending, weights, forced_quantities);
        (c.previous, c.unit, c.call, c.bankrupt) = (previous, unit, call, bankrupt);
        (c.fm_before, c.flow) = (fm_before, flow);
        (self.cash, self.held, self.book_held, self.seen, self.day) =
            (cash, held, book_held, seen, day);
        v.shares.copy_from_slice(&shares);
        v.last.copy_from_slice(&last);
        v.basis.copy_from_slice(&basis);
        v.cost.copy_from_slice(&cost);
        v.peak.copy_from_slice(&peak);
        v.entry.copy_from_slice(&entry);
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn step_day<const CLASSIC: bool, const FUT: bool>(
    c: &mut Core,
    day: usize,
    row: Option<Row<'_>>,
    shares: bool,
    m: &Market<'_>,
    s: &Settings<'_>,
    st: &mut State<'_>,
    book: &mut futures::Book<'_>,
    trk: &mut Track<'_>,
) {
    c.trade::<CLASSIC, FUT, true>(day, row, shares, m, s, st, book, trk);
    if day + 1 < m.t {
        c.open::<CLASSIC, FUT, true>(day + 1, m, s, st, book, trk);
    }
}
