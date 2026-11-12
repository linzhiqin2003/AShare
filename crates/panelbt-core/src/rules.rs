//! Declarative exit rules and limit targets (spot only), compiled in only
//! when position tracking (`T`) is on. See SPEC "Declarative exit rules".

use super::{
    Account, Market, Output, Settings, State, Track, capacity, fill, lot, lot_terms, movement,
    past_close, quantity_tolerance, quote_at, round_close,
};

/// Rule configuration. Percentages have length 0 (off), 1, N or T×N.
pub struct Rules<'a> {
    pub open: &'a [f64],
    pub high: &'a [f64],
    pub low: &'a [f64],
    pub stop: &'a [f64],
    pub take: &'a [f64],
    pub trail: &'a [f64],
    /// Rows; 0 disables.
    pub max_hold: i64,
    /// 0: exec_price at the open; 1: at the close; 2: an average price.
    pub timing: u8,
    pub both_target: bool,
    /// Rows a re-entry stays blocked after an exit; -1: until a later target row.
    pub cooldown: i64,
    pub div_adjust: bool,
    pub sell_lock: bool,
    pub until_next: bool,
    /// Limit orders fill when the price only touches the limit (`low == L`
    /// for a buy); by default the range must cross it (`low < L`).
    pub limit_touch: bool,
}

pub const RULE_COUNTS: usize = 9;
const STOP: u8 = 1;
const TAKE: u8 = 2;
const TRAIL: u8 = 3;
const HOLD: u8 = 4;
pub const LIMIT: u8 = 5;
const DEFERRED: usize = 4;
const REENTRY: usize = 5;
const LIMIT_FILLED: usize = 6;
const LIMIT_UNFILLED: usize = 7;
const LIMIT_BLOCKED: usize = 8;

/// Owned rule state; empty unless tracking is compiled in.
#[derive(Clone, Default)]
pub struct RuleBook {
    /// Shares bought on the current row (sell lock).
    pub bought: Vec<f64>,
    /// Absolute quantity traded on the current row (participation cap).
    pub used: Vec<f64>,
    /// Entered on the current row without seeing the whole row's range.
    pub fresh: Vec<bool>,
    /// A rule fill or trigger on the current row.
    pub exited: Vec<bool>,
    pub marked: Vec<bool>,
    pub touched: Vec<usize>,
    /// Reason of an unfinished exit; 0 when none.
    pub due: Vec<u8>,
    /// Row of the last exit trigger; -1 when re-entry is not blocked.
    pub block: Vec<i64>,
    pub blocked: Vec<usize>,
    /// Active limit orders (column, target, limit price), ascending columns.
    pub limits: Vec<(usize, f64, f64)>,
    /// Columns with a limit entry in the current row's target.
    pub limit_today: Vec<bool>,
    pub limit_cols: Vec<usize>,
    /// Whether an ordinary entry on the current row is fresh.
    pub entry_fresh: bool,
    pub counts: [u64; RULE_COUNTS],
    pub scratch: Vec<usize>,
    pub plan: Vec<(usize, f64, f64, bool, bool)>,
    /// `cash_scaled`/`margin_scaled` at the start of the row, and whether a
    /// limit order counted them this row (transient, reset every row).
    pub row_scaled: [u64; 2],
    pub limit_scaled: [bool; 2],
}

impl RuleBook {
    pub fn new(n: usize) -> RuleBook {
        RuleBook {
            bought: vec![0.0; n],
            used: vec![0.0; n],
            fresh: vec![false; n],
            exited: vec![false; n],
            marked: vec![false; n],
            due: vec![0; n],
            block: vec![-1; n],
            limit_today: vec![false; n],
            ..Default::default()
        }
    }
    #[inline]
    pub fn mark(&mut self, i: usize) {
        if !self.marked[i] {
            self.marked[i] = true;
            self.touched.push(i);
        }
    }
    /// Bookkeeping of one fill (called from `fill` when tracking).
    #[inline]
    pub fn on_fill(&mut self, i: usize, before: f64, q: f64, after: f64) {
        if self.used.is_empty() {
            return;
        }
        self.mark(i);
        self.used[i] += q.abs();
        if after == 0.0 {
            self.due[i] = 0;
        }
        if q > 0.0 && after > 0.0 {
            self.bought[i] += q.min(after);
        }
        if before == 0.0 || before.signum() != after.signum() {
            self.fresh[i] = self.entry_fresh;
        }
    }
    /// Whether an opening of column `i` is blocked on `day`.
    #[inline]
    pub fn blocks(&self, i: usize, day: usize, r: &Rules<'_>) -> bool {
        self.due[i] != 0
            || (self.block[i] >= 0 && (r.cooldown < 0 || day as i64 <= self.block[i] + r.cooldown))
    }
    /// A new target row: replace limit orders and lift `cooldown=None` blocks.
    pub fn begin_row(
        &mut self,
        day: usize,
        cols: &[i64],
        weights: &[f64],
        limits: &[f64],
        r: &Rules<'_>,
    ) {
        if r.cooldown < 0 {
            let block = &mut self.block;
            self.blocked.retain(|&c| {
                if block[c] < day as i64 {
                    block[c] = -1;
                    false
                } else {
                    true
                }
            });
        }
        self.limits.clear();
        if !limits.is_empty() {
            for j in 0..cols.len() {
                if limits[j].is_finite() {
                    let i = cols[j] as usize;
                    self.limits.push((i, weights[j], limits[j]));
                    self.limit_today[i] = true;
                    self.limit_cols.push(i);
                }
            }
        }
    }
    /// End of a row: per-row flags back to their defaults.
    pub fn end_row(&mut self, r: &Rules<'_>) {
        for &i in &self.touched {
            self.bought[i] = 0.0;
            self.used[i] = 0.0;
            self.fresh[i] = false;
            self.exited[i] = false;
            self.marked[i] = false;
        }
        self.touched.clear();
        for &i in &self.limit_cols {
            self.limit_today[i] = false;
        }
        self.limit_cols.clear();
        if !r.until_next {
            self.limits.clear();
        }
    }
}

#[inline]
fn pct(a: &[f64], day: usize, i: usize, m: &Market<'_>) -> f64 {
    match a.len() {
        0 => f64::NAN,
        1 => a[0],
        x if x == m.t * m.n => a[day * m.n + i],
        _ => a[i],
    }
}

/// Exit trigger of a held, non-fresh position: (reason, raw price).
fn evaluate(
    i: usize,
    day: usize,
    cur: f64,
    m: &Market<'_>,
    r: &Rules<'_>,
    trk: &Track<'_>,
) -> Option<(u8, f64)> {
    let k = day * m.n + i;
    let hold = r.max_hold > 0 && trk.entry[i] >= 0 && day as i64 - trk.entry[i] >= r.max_hold;
    if hold && r.timing == 0 {
        return Some((HOLD, m.price[k]));
    }
    let (o, h, l) = (r.open[k], r.high[k], r.low[k]);
    if o.is_finite() && h.is_finite() && l.is_finite() {
        let (sl, tr, tp) = (
            pct(r.stop, day, i, m),
            pct(r.trail, day, i, m),
            pct(r.take, day, i, m),
        );
        let (cost, peak) = (trk.cost[i], trk.peak[i]);
        let hit = if cur > 0.0 {
            let s = if sl.is_finite() {
                cost * (1.0 - sl)
            } else {
                f64::NEG_INFINITY
            };
            let t = if tr.is_finite() {
                peak * (1.0 - tr)
            } else {
                f64::NEG_INFINITY
            };
            let (stop, why) = if t > s { (t, TRAIL) } else { (s, STOP) };
            let target = if tp.is_finite() {
                cost * (1.0 + tp)
            } else {
                f64::INFINITY
            };
            if o <= stop {
                Some((why, o))
            } else if o >= target {
                Some((TAKE, o))
            } else {
                match (l <= stop, h >= target) {
                    (true, true) if r.both_target => Some((TAKE, target)),
                    (true, _) => Some((why, stop)),
                    (false, true) => Some((TAKE, target)),
                    (false, false) => None,
                }
            }
        } else {
            let s = if sl.is_finite() {
                cost * (1.0 + sl)
            } else {
                f64::INFINITY
            };
            let t = if tr.is_finite() {
                peak * (1.0 + tr)
            } else {
                f64::INFINITY
            };
            let (stop, why) = if t < s { (t, TRAIL) } else { (s, STOP) };
            let target = if tp.is_finite() && tp < 1.0 {
                cost * (1.0 - tp)
            } else {
                f64::NEG_INFINITY
            };
            if o >= stop {
                Some((why, o))
            } else if o <= target {
                Some((TAKE, o))
            } else {
                match (h >= stop, l <= target) {
                    (true, true) if r.both_target => Some((TAKE, target)),
                    (true, _) => Some((why, stop)),
                    (false, true) => Some((TAKE, target)),
                    (false, false) => None,
                }
            }
        };
        if hit.is_some() {
            return hit;
        }
    }
    if hold { Some((HOLD, m.price[k])) } else { None }
}

/// Account valued at the carried `last` marks. A close followed by a reopening
/// lists a column twice in `held`, so the list is deduplicated first.
fn marked_account(st: &mut State<'_>) -> Account {
    st.held.sort_unstable();
    st.held.dedup();
    let mut a = Account {
        cash: st.cash,
        ..Default::default()
    };
    for &i in &st.held {
        let x = st.shares[i] * st.last[i];
        if x > 0.0 { a.long += x } else { a.short -= x }
    }
    a
}

#[allow(clippy::too_many_arguments)]
fn projected_at(
    mut a: Account,
    i: usize,
    q: f64,
    current: f64,
    raw: f64,
    day: usize,
    m: &Market<'_>,
    s: &Settings<'_>,
) -> Account {
    if q == 0.0 {
        return a;
    }
    a.cash += movement(q, quote_at(q, i, day, raw, m, s));
    a.long += ((current + q).max(0.0) - current.max(0.0)) * raw;
    a.short += ((current + q).min(0.0) - current.min(0.0)) * -raw;
    a
}

/// Risk-reducing fill toward zero at `raw`, under masks, sell lock, the
/// participation cap and the reduction cash rules. Returns the filled quantity.
#[allow(clippy::too_many_arguments)]
fn reduce<const CLASSIC: bool>(
    i: usize,
    request: f64,
    raw: f64,
    reason: u8,
    day: usize,
    shares: bool,
    m: &Market<'_>,
    s: &Settings<'_>,
    r: &Rules<'_>,
    st: &mut State<'_>,
    trk: &mut Track<'_>,
    rb: &mut RuleBook,
    out: &mut Output,
) -> f64 {
    let k = day * m.n + i;
    let cur = st.shares[i];
    let (minimum, step) = lot_terms::<false>(m, s, shares, i);
    let allowed = raw.is_finite()
        && raw > 0.0
        && if request > 0.0 {
            m.buy[k] != 0
        } else {
            m.sell[k] != 0
        };
    if !allowed {
        return 0.0;
    }
    let mut avail = request.abs();
    if r.sell_lock && cur > 0.0 {
        avail = avail.min((cur - rb.bought[i]).max(0.0));
    }
    let cap = (capacity(m, s, day, i, step) - rb.used[i]).max(0.0);
    let mut q = request.signum() * round_close(avail.min(cap), cur.abs(), minimum, step);
    if q == 0.0 {
        return 0.0;
    }
    let x = quote_at(q, i, day, raw, m, s);
    if !(x.price > 0.0 && x.price.is_finite()) {
        return 0.0;
    }
    let fees = x.fees.commission + x.fees.tax + x.fees.other;
    if CLASSIC {
        let ok = if st.cash >= 0.0 {
            st.cash + x.amount >= fees
        } else {
            x.amount >= fees
        };
        if !ok {
            return 0.0;
        }
    } else if !s.flags[3] {
        let a = marked_account(st);
        let own = |b: Account| b.own_cash(s.flags[4]);
        if own(a) >= -1e-10 * s.scalars[5] && own(projected_at(a, i, q, cur, raw, day, m, s)) < 0.0
        {
            let (mut lo, mut hi) = (0.0, q.abs());
            for _ in 0..64 {
                let mid = (lo + hi) * 0.5;
                let y = q.signum() * round_close(mid, cur.abs(), minimum, step);
                if own(projected_at(a, i, y, cur, raw, day, m, s)) >= 0.0 {
                    lo = mid
                } else {
                    hi = mid
                }
            }
            q = q.signum() * round_close(lo, cur.abs(), minimum, step);
            if q == 0.0 {
                return 0.0;
            }
        }
    }
    fill::<true>(day, i, q, raw, false, false, m, s, st, trk, rb, out);
    if let Some(t) = out.trades.as_ref() {
        out.reasons.push((t.len() - 1, reason));
    }
    q
}

/// Rule exits, then limit orders. Not called on forced-liquidation rows.
#[allow(clippy::too_many_arguments)]
pub(super) fn intraday<const CLASSIC: bool>(
    day: usize,
    shares: bool,
    bankrupt: bool,
    m: &Market<'_>,
    s: &Settings<'_>,
    r: &Rules<'_>,
    st: &mut State<'_>,
    trk: &mut Track<'_>,
    rb: &mut RuleBook,
    out: &mut Output,
) {
    let base = day * m.n;
    // The ordinary rebalance may have left duplicate or closed entries.
    st.held.sort_unstable();
    st.held.dedup();
    st.held.retain(|&i| st.shares[i] != 0.0);
    // (a) Exits, ascending columns held at the start of the phase.
    let mut cols = std::mem::take(&mut rb.scratch);
    cols.clear();
    cols.extend_from_slice(&st.held);
    for &i in &cols {
        let cur = st.shares[i];
        if cur == 0.0 {
            rb.due[i] = 0;
            continue;
        }
        let (reason, raw) = if rb.due[i] != 0 {
            (rb.due[i], m.price[base + i])
        } else if rb.fresh[i] {
            continue;
        } else {
            match evaluate(i, day, cur, m, r, trk) {
                Some((why, raw)) => {
                    rb.counts[(why - 1) as usize] += 1;
                    (why, raw)
                }
                None => continue,
            }
        };
        rb.mark(i);
        rb.exited[i] = true;
        if rb.block[i] < 0 {
            rb.blocked.push(i);
        }
        rb.block[i] = day as i64;
        reduce::<CLASSIC>(i, -cur, raw, reason, day, shares, m, s, r, st, trk, rb, out);
        if st.shares[i] != 0.0 {
            rb.due[i] = reason;
            rb.counts[DEFERRED] += 1;
        } else {
            rb.due[i] = 0;
        }
    }
    rb.scratch = cols;
    // (b) Limit orders.
    if !rb.limits.is_empty() {
        limit_orders::<CLASSIC>(day, shares, bankrupt, m, s, r, st, trk, rb, out);
    }
    st.held.sort_unstable();
    st.held.dedup();
    st.held.retain(|&i| st.shares[i] != 0.0);
}

#[allow(clippy::too_many_arguments)]
fn limit_orders<const CLASSIC: bool>(
    day: usize,
    shares: bool,
    bankrupt: bool,
    m: &Market<'_>,
    s: &Settings<'_>,
    r: &Rules<'_>,
    st: &mut State<'_>,
    trk: &mut Track<'_>,
    rb: &mut RuleBook,
    out: &mut Output,
) {
    let base = day * m.n;
    let mut equity = st.cash;
    for &i in &st.held {
        equity += st.shares[i] * st.last[i];
    }
    // Plan: (column, target quantity, raw fill price, short of target for a
    // reason other than the account, cut to what the account affords).
    let mut plan = std::mem::take(&mut rb.plan);
    plan.clear();
    let mut keep = Vec::with_capacity(rb.limits.len());
    for &(i, w, limit) in &rb.limits {
        if m.first_delist[i] <= day as i64 {
            continue;
        }
        if rb.exited[i] || rb.due[i] != 0 {
            keep.push((i, w, limit));
            continue;
        }
        let (minimum, step) = lot_terms::<false>(m, s, shares, i);
        let cur = st.shares[i];
        let mut target = if shares {
            w
        } else {
            let raw = w * equity.max(0.0) * (1.0 - s.scalars[8]) / limit;
            if raw.abs() < minimum && cur * raw > 0.0 {
                raw
            } else {
                raw.signum() * lot(raw.abs(), minimum, step)
            }
        };
        if (target - cur).abs() <= quantity_tolerance(target, cur) {
            target = cur;
        }
        if target == cur {
            continue;
        }
        let (o, h, l) = (r.open[base + i], r.high[base + i], r.low[base + i]);
        let valid = o.is_finite() && h.is_finite() && l.is_finite();
        let (touched, raw) = if target > cur {
            (
                valid && (l < limit || (r.limit_touch && l == limit)),
                o.min(limit),
            )
        } else {
            (
                valid && (h > limit || (r.limit_touch && h == limit)),
                o.max(limit),
            )
        };
        keep.push((i, w, limit));
        if !touched {
            rb.counts[LIMIT_UNFILLED] += 1;
            continue;
        }
        plan.push((i, target, raw, false, false));
    }
    rb.limits = keep;
    // Reductions, ascending columns.
    for p in plan.iter_mut() {
        let (i, target, raw) = (p.0, p.1, p.2);
        let cur = st.shares[i];
        let (minimum, step) = lot_terms::<false>(m, s, shares, i);
        let request = if cur > 0.0 && target < cur {
            -round_close(cur - target.max(0.0), cur, minimum, step)
        } else if cur < 0.0 && target > cur {
            round_close(target.min(0.0) - cur, -cur, minimum, step)
        } else {
            0.0
        };
        if request != 0.0 {
            let q = reduce::<CLASSIC>(
                i, request, raw, LIMIT, day, shares, m, s, r, st, trk, rb, out,
            );
            if q != 0.0 {
                rb.counts[LIMIT_FILLED] += 1;
            }
            p.3 = q != request;
        }
    }
    // Openings: short openings first, then long, by column.
    plan.sort_by_key(|p| (p.1 > st.shares[p.0], p.0));
    rb.entry_fresh = true;
    for p in plan.iter_mut() {
        let (i, target, raw) = (p.0, p.1, p.2);
        let k = base + i;
        let after = st.shares[i];
        let (minimum, step) = lot_terms::<false>(m, s, shares, i);
        let opening = if target > 0.0 && after >= 0.0 && target > after {
            lot(target - after, minimum, step)
        } else if target < 0.0 && after <= 0.0 && target < after {
            -lot(after - target, minimum, step)
        } else {
            0.0
        };
        if opening.abs() <= quantity_tolerance(target, after) {
            continue;
        }
        if bankrupt {
            p.3 = true;
            continue;
        }
        if rb.blocks(i, day, r) {
            rb.counts[REENTRY] += 1;
            p.3 = true;
            continue;
        }
        let allowed = if opening > 0.0 {
            m.buy[k] != 0
        } else {
            m.sell[k] != 0 && s.flags[2] && (m.can_short.is_empty() || m.can_short[k] != 0)
        };
        let cap = (capacity(m, s, day, i, step) - rb.used[i]).max(0.0);
        let mut q = if allowed {
            opening.signum() * lot(opening.abs().min(cap), minimum, step)
        } else {
            0.0
        };
        if q != 0.0 {
            let x = quote_at(q, i, day, raw, m, s);
            if !(x.price > 0.0 && x.price.is_finite()) {
                q = 0.0;
            }
        }
        // Short of the opening for a mask, the cap, a block or an invalid quote.
        let short = q != opening;
        let pre = q;
        if q != 0.0 {
            if CLASSIC {
                let cost = |y: f64| {
                    if y == 0.0 {
                        0.0
                    } else {
                        -movement(y, quote_at(y, i, day, raw, m, s))
                    }
                };
                if cost(q) > st.cash {
                    // Like an ordinary purchase: cut to what cash affords
                    // including fees, counted as `cash_scaled`, not blocked.
                    if !rb.limit_scaled[0] && out.counts[3] == rb.row_scaled[0] {
                        out.counts[3] += 1;
                    }
                    rb.limit_scaled[0] = true;
                    p.4 = true;
                    let (mut lo, mut hi) = (0.0, q);
                    for _ in 0..64 {
                        let mid = (lo + hi) * 0.5;
                        if cost(mid) <= st.cash {
                            lo = mid
                        } else {
                            hi = mid
                        }
                    }
                    q = lot(lo, minimum, step);
                    while q > 0.0 && cost(q) > st.cash {
                        q = if step == 0.0 {
                            q.next_down()
                        } else {
                            lot(q - step, minimum, step)
                        };
                    }
                }
            } else {
                let a = marked_account(st);
                let ok = |y: f64| {
                    super::feasible(projected_at(a, i, y, after, raw, day, m, s), 0.0, s, day)
                };
                if !ok(q) {
                    if !rb.limit_scaled[1] && out.counts[20] == rb.row_scaled[1] {
                        out.counts[20] += 1; // margin_scaled
                    }
                    rb.limit_scaled[1] = true;
                    p.4 = true;
                    let (mut lo, mut hi) = (0.0, q.abs());
                    for _ in 0..64 {
                        let mid = (lo + hi) * 0.5;
                        if ok(q.signum() * lot(mid, minimum, step)) {
                            lo = mid
                        } else {
                            hi = mid
                        }
                    }
                    q = q.signum() * lot(lo, minimum, step);
                }
            }
        }
        // A cut to what the account affords completes the order only when the
        // shortfall is at most one lot, or for fractional quantities at most
        // the quantity the order's own costs amount to; a zero fill never.
        let cut_short = q != pre && {
            let x = quote_at(pre, i, day, raw, m, s);
            let costs =
                pre.abs() * (x.price - raw).abs() + x.fees.commission + x.fees.tax + x.fees.other;
            let allow = step.max(costs / raw);
            q == 0.0 || (pre - q).abs() > allow + quantity_tolerance(pre, q)
        };
        if q.abs() > quantity_tolerance(target, after) {
            if after == 0.0 {
                st.last[i] = past_close(m, day, i);
            }
            fill::<true>(day, i, q, raw, false, false, m, s, st, trk, rb, out);
            if let Some(t) = out.trades.as_ref() {
                out.reasons.push((t.len() - 1, LIMIT));
            }
            rb.counts[LIMIT_FILLED] += 1;
        }
        if short || cut_short {
            p.3 = true;
        }
    }
    rb.entry_fresh = r.timing != 0;
    // Completed orders are removed; partially filled ones counted. An order
    // cut by the account to within one lot (fees) of its size is complete.
    for p in &plan {
        if p.3 {
            rb.counts[LIMIT_BLOCKED] += 1;
        }
    }
    let done: Vec<usize> = plan
        .iter()
        .filter(|p| {
            (p.4 && !p.3) || (p.1 - st.shares[p.0]).abs() <= quantity_tolerance(p.1, st.shares[p.0])
        })
        .map(|p| p.0)
        .collect();
    rb.limits.retain(|o| !done.contains(&o.0));
    rb.plan = plan;
}
