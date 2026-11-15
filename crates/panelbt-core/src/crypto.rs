//! Opt-in hourly, single-collateral futures liquidation semantics.
//! The legacy row implementation is left intact when liquidation is absent.

use super::{Core, Market, Row, Settings, State, futures};
use futures::{Book, rate};
use tradecore::{futures_charge, futures_margin, round_close, round_open};

/// The two existing floating-point evaluation orders are intentional: retain
/// both when checking the basket and each individual opening. Preview and
/// execution must call these same methods, without a collateral epsilon.
#[derive(Clone, Copy)]
struct OpeningBudget {
    equity: f64,
    margin: f64,
}

struct Opening {
    quantity: f64,
    multiplier: f64,
    mark: f64,
    price: f64,
    fee: f64,
    rate: f64,
}

impl OpeningBudget {
    fn project(&mut self, x: &Opening) {
        self.equity += x.quantity * x.multiplier * (x.mark - x.price) - x.fee;
        self.margin += x.quantity.abs() * x.multiplier * x.mark * x.rate;
    }

    fn allows(&self, x: &Opening) -> bool {
        self.equity > 0.0
            && self.equity + x.quantity * x.multiplier * (x.mark - x.price) - x.fee
                >= self.margin + x.quantity.abs() * x.multiplier * x.mark * x.rate
    }
}

/// Collateral-only preview at caller-supplied reference marks/execution prices.
/// Reductions precede the opening basket. Rates/fees are today's known terms;
/// no current bar prices, masks, volume or liquidation stress are consulted.
/// Targets are already quantized contracts. This does not promise execution
/// through liquidity, quantity filters, expiry, price gaps or liquidation.
pub fn opening_feasible(
    day: usize,
    cash: f64,
    shares: &[f64],
    basis: &[f64],
    targets: &[f64],
    prices: &[f64],
    m: &Market<'_>,
) -> bool {
    let f = m.futures.as_ref().unwrap();
    let mut balance = cash;
    let mut remaining = shares.to_vec();
    let mut orders = Vec::new();
    for i in 0..m.n {
        let delta = targets[i] - shares[i];
        let close = if shares[i] * delta < 0.0 {
            -shares[i].signum() * shares[i].abs().min(delta.abs())
        } else {
            0.0
        };
        let fee = |q| {
            futures_charge(
                q,
                f.multiplier[i],
                prices[i],
                rate(f.fee_rate, day, i, m),
                rate(f.fee_per_contract, day, i, m),
            )
        };
        if close != 0.0 {
            balance += futures::realized(shares[i], close, basis[i], prices[i], f.multiplier[i])
                - fee(close);
            remaining[i] += close;
        }
        let quantity = targets[i] - remaining[i];
        if quantity != 0.0 {
            orders.push(Opening {
                quantity,
                multiplier: f.multiplier[i],
                mark: prices[i],
                price: prices[i],
                fee: fee(quantity),
                rate: rate(f.margin_rate, day, i, m),
            });
        }
    }
    let mut budget = OpeningBudget {
        equity: balance,
        margin: 0.0,
    };
    for i in 0..m.n {
        if remaining[i] != 0.0 {
            budget.equity += remaining[i] * f.multiplier[i] * (prices[i] - basis[i]);
            budget.margin += futures_margin(
                remaining[i],
                f.multiplier[i],
                prices[i],
                rate(f.margin_rate, day, i, m),
            );
        }
    }
    let mut basket = budget;
    for order in &orders {
        basket.project(order);
    }
    if basket.equity < basket.margin {
        return false;
    }
    for order in &orders {
        if !budget.allows(order) {
            return false;
        }
        budget.project(order);
    }
    true
}

pub struct Inputs<'a> {
    pub funding: &'a [f64],
    pub maint: &'a [f64],
    pub step: &'a [f64],
    pub minimum: &'a [f64],
    pub notional: &'a [f64],
    pub open: &'a [f64],
    pub high: &'a [f64],
    pub low: &'a [f64],
    pub clearance: &'a [f64],
    pub cap: &'a [f64],
}
impl<'a> Inputs<'a> {
    pub fn new(a: &[&'a [f64]]) -> Self {
        Self {
            funding: a[0],
            maint: a[1],
            step: a[2],
            minimum: a[3],
            notional: a[4],
            open: a[5],
            high: a[6],
            low: a[7],
            clearance: a[8],
            cap: a[9],
        }
    }
    pub fn liquidation(&self) -> bool {
        !self.open.is_empty()
    }
}

pub fn check_shapes(a: &[&[f64]], t: usize, n: usize) -> Result<(), String> {
    if a.len() != 10
        || a.iter().enumerate().any(|(i, x)| {
            if [1, 8, 9].contains(&i) {
                ![0, 1, n, t * n].contains(&x.len())
            } else {
                !x.is_empty() && x.len() != t * n
            }
        })
        || a[5..].iter().any(|x| x.is_empty() != a[5].is_empty())
    {
        return Err("crypto: invalid panel lengths".into());
    }
    Ok(())
}

pub fn validate(kind: &[u8], margin: &[f64], a: &[&[f64]], t: usize) -> Result<(), String> {
    let n = kind.len();
    check_shapes(a, t, n)?;
    if ![1, n, t * n].contains(&margin.len()) {
        return Err("margin_rate: invalid shape".into());
    }
    let c = Inputs::new(a);
    if c.liquidation() && kind.contains(&0) {
        return Err("liquidation requires all futures".into());
    }
    let names = [
        "funding",
        "maint_rate",
        "qty_step",
        "min_qty",
        "min_notional",
        "mark_open",
        "mark_high",
        "mark_low",
        "clearance_fee_rate",
        "tier_cap",
    ];
    for (j, xs) in a.iter().enumerate() {
        for (k, &x) in xs.iter().enumerate() {
            if xs.len() != 1 && kind[k % n] == 0 {
                continue;
            }
            let bad = match j {
                0 => x.is_infinite(),
                2 => !x.is_finite() || x < 1.0 || x.fract() != 0.0,
                3 => !x.is_finite() || x < 0.0 || x.fract() != 0.0,
                5..=7 => x.is_infinite() || x <= 0.0,
                9 => !x.is_finite() || x <= 0.0,
                _ => !x.is_finite() || x < 0.0,
            };
            if bad {
                return Err(format!("{}: invalid value at cell {k}", names[j]));
            }
        }
    }
    if c.liquidation() {
        for k in 0..t * n {
            if c.low[k] > c.open[k] || c.open[k] > c.high[k] || c.low[k] > c.high[k] {
                return Err(format!(
                    "mark OHLC: expected low <= open <= high at cell {k}"
                ));
            }
        }
    }
    Ok(())
}

#[inline]
pub(super) fn quantity_rules(m: &Market<'_>) -> bool {
    m.futures
        .as_ref()
        .and_then(|f| f.crypto.as_ref())
        .is_some_and(|c| !c.step.is_empty() || !c.minimum.is_empty())
}

#[inline]
pub(super) fn terms(day: usize, i: usize, m: &Market<'_>) -> (f64, f64) {
    let c = m.futures.as_ref().unwrap().crypto.as_ref();
    let k = day * m.n + i;
    let minimum = c
        .filter(|c| !c.minimum.is_empty())
        .map_or(1.0, |c| c.minimum[k]);
    let step = c.filter(|c| !c.step.is_empty()).map_or(1.0, |c| c.step[k]);
    ((minimum / step).ceil() * step, step)
}

pub(super) fn maintenance(q: f64, price: f64, day: usize, i: usize, m: &Market<'_>) -> f64 {
    let f = m.futures.as_ref().unwrap();
    let r = f
        .crypto
        .as_ref()
        .filter(|c| !c.maint.is_empty())
        .map_or(f.margin_rate, |c| c.maint);
    q.abs() * f.multiplier[i] * price * rate(r, day, i, m)
}

fn tidy(book: &mut Book<'_>, st: &State<'_>) {
    book.held.sort_unstable();
    book.held.dedup();
    book.held.retain(|&i| st.shares[i] != 0.0);
}

fn marks(day: usize, m: &Market<'_>, held: &[usize]) -> Result<(), String> {
    let c = m.futures.as_ref().unwrap().crypto.as_ref().unwrap();
    for &i in held {
        let k = day * m.n + i;
        if [c.open[k], c.high[k], c.low[k]]
            .iter()
            .any(|p| !p.is_finite() || *p <= 0.0)
        {
            return Err(format!("mark: missing held value at row {day}, column {i}"));
        }
    }
    Ok(())
}

/// Mark-to-market equity, maintenance, notional and initial collateral.
fn value(
    day: usize,
    close: bool,
    m: &Market<'_>,
    st: &State<'_>,
    book: &Book<'_>,
) -> (f64, f64, f64, f64) {
    let f = m.futures.as_ref().unwrap();
    let c = f.crypto.as_ref().unwrap();
    let (mut equity, mut mm, mut nominal, mut initial) = (st.cash, 0.0, 0.0, 0.0);
    for &i in &book.held {
        let p = if close {
            f.settlement[day * m.n + i]
        } else {
            c.open[day * m.n + i]
        };
        let q = st.shares[i] * f.multiplier[i];
        equity += q * (p - book.basis[i]);
        nominal += q.abs() * p;
        mm += maintenance(st.shares[i], p, day, i, m);
        initial += futures::margin(st.shares[i], p, day, i, m);
    }
    (equity, mm, nominal, initial)
}

fn exceeded(day: usize, mode: u8, m: &Market<'_>, st: &State<'_>, book: &Book<'_>) -> bool {
    let f = m.futures.as_ref().unwrap();
    let c = f.crypto.as_ref().unwrap();
    book.held.iter().any(|&i| {
        let k = day * m.n + i;
        let p = match mode {
            0 => c.open[k],
            1 => c.high[k],
            _ => f.settlement[k],
        };
        st.shares[i].abs() * f.multiplier[i] * p > rate(c.cap, day, i, m)
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "Explicit row and account context"
)]
fn liquidate(
    core: &mut Core,
    day: usize,
    phase: usize,
    lambda: f64,
    m: &Market<'_>,
    s: &Settings<'_>,
    st: &mut State<'_>,
    book: &mut Book<'_>,
) -> Result<(), String> {
    let f = m.futures.as_ref().unwrap();
    let c = f.crypto.as_ref().unwrap();
    let mut dues = Vec::with_capacity(book.held.len());
    // Validate every forced quote before changing the account. No mask or cap
    // blocks exchange liquidation, but impact still needs a real denominator.
    for &i in &book.held {
        let k = day * m.n + i;
        let raw = if phase == 44 {
            f.settlement[k]
        } else if phase == 43 {
            let extreme = if st.shares[i] > 0.0 {
                c.low[k]
            } else {
                c.high[k]
            };
            c.open[k] + lambda * (extreme - c.open[k])
        } else {
            c.open[k]
        };
        let x = futures::quote_at(-st.shares[i], i, day, raw, m, s);
        if !x.price.is_finite() || x.price <= 0.0 {
            return Err(format!(
                "liquidation quote: invalid at row {day}, column {i}"
            ));
        }
        dues.push((i, x, rate(c.clearance, day, i, m) * x.amount));
    }
    core.out.counts[phase] += 1;
    for &(i, x, _) in &dues {
        let q = -st.shares[i];
        book.execute_quote(
            st.shares,
            &mut st.cash,
            i,
            q,
            day,
            false,
            true,
            m,
            &mut core.out,
            x,
        );
    }
    let due: f64 = dues.iter().map(|x| x.2).sum();
    let fee = due.min(st.cash.max(0.0));
    if fee < due {
        core.out.counts[50] += 1;
    }
    if due > 0.0 {
        for &(i, _, d) in &dues {
            core.out.contribution[i] -= fee * (d / due);
        }
    }
    st.cash -= fee;
    if let Some(w) = core.out.wallet.as_mut() {
        w.balance -= fee;
    }
    core.out.daily[31 * m.t + day] += fee;
    core.out.daily[5 * m.t + day] += fee;
    if st.cash < 0.0 {
        core.out.daily[32 * m.t + day] += -st.cash;
        if let Some(w) = core.out.wallet.as_mut() {
            w.balance += -st.cash;
        }
        core.out.counts[51] += 1;
        if !core.bankrupt {
            core.out.bankruptcies.push(day);
            core.out.counts[25] += 1;
            core.out.counts[29] += 1;
        }
        core.bankrupt = true;
        st.cash = 0.0;
    }
    book.held.clear();
    core.pending.clear();
    core.commands.clear();
    core.orders.clear();
    core.future_orders.clear();
    core.weights.fill(0.0);
    core.call = false;
    Ok(())
}

fn rounded(day: usize, i: usize, qty: f64, m: &Market<'_>) -> f64 {
    let (minimum, step) = terms(day, i, m);
    let q = round_open(qty.abs(), minimum, step);
    let f = m.futures.as_ref().unwrap();
    let c = f.crypto.as_ref().unwrap();
    if !c.notional.is_empty()
        && q * f.multiplier[i] * c.open[day * m.n + i] < c.notional[day * m.n + i]
    {
        0.0
    } else {
        qty.signum() * q
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "Explicit row and account context"
)]
pub(super) fn trade(
    core: &mut Core,
    day: usize,
    row: Option<Row<'_>>,
    shares: bool,
    m: &Market<'_>,
    s: &Settings<'_>,
    st: &mut State<'_>,
    book: &mut Book<'_>,
) -> Result<(), String> {
    let f = m.futures.as_ref().unwrap();
    let c = f.crypto.as_ref().unwrap();
    let base = day * m.n;
    marks(day, m, &book.held)?;
    let mut tier = exceeded(day, 0, m, st, book);
    let opening = value(day, false, m, st, book);
    let value_before = opening.0;
    let mut cleared = false;
    if !book.held.is_empty() && opening.0 - opening.1 <= opening.2 * 1e-12 {
        liquidate(core, day, 41, 0.0, m, s, st, book)?;
        cleared = true;
    }
    core.commands.clear();
    if !cleared && !core.bankrupt {
        if let Some(row) = row {
            core.pending.clear();
            for &i in &book.held {
                core.weights[i] = 0.0;
                core.commands.push((i, 0.0));
            }
            for (&i, &w) in row.cols.iter().zip(row.weights) {
                core.weights[i as usize] = w;
                core.commands.push((i as usize, w));
            }
            core.commands.sort_unstable_by_key(|x| x.0);
            core.commands.dedup_by_key(|x| x.0);
            for x in &mut core.commands {
                x.1 = core.weights[x.0];
            }
        } else if s.flags[0] {
            std::mem::swap(&mut core.commands, &mut core.pending);
            core.pending.clear();
        }
    }
    core.orders.clear();
    for &(i, w) in &core.commands {
        if day as i64 > f.expiry[i] {
            if w != 0.0 {
                core.out.counts[37] += 1;
                core.out.daily[29 * m.t + day] += 1.0;
            }
            continue;
        }
        let p = m.price[base + i];
        // Validate executable intent before a missing weight denominator can
        // turn it into zero. Unheld, masked/out-of-life columns stay unread.
        if st.shares[i] == 0.0
            && w != 0.0
            && p.is_finite()
            && if w > 0.0 {
                m.buy[base + i] != 0
            } else {
                m.sell[base + i] != 0
            }
        {
            marks(day, m, &[i])?;
        }
        let mark = if c.open[base + i].is_finite() {
            c.open[base + i]
        } else if day > 0 {
            f.settlement[base - m.n + i]
        } else {
            // Inactive first-row columns still need a sizing placeholder to
            // classify blocked targets. Executable intents were checked above.
            f.settlement[base + i]
        };
        let raw = if shares {
            core.out.counts[9] += 1;
            w
        } else if mark > 0.0 {
            w * value_before.max(0.0) * (1.0 - s.scalars[8]) / (mark * f.multiplier[i])
        } else {
            0.0
        };
        let target = if shares {
            raw
        } else {
            raw.signum() * raw.abs().floor()
        };
        if raw != 0.0 && target == 0.0 {
            core.out.counts[38] += 1;
        }
        let before = st.shares[i];
        if !shares
            && w != 0.0
            && before != 0.0
            && value_before > 0.0
            && (w - before * mark * f.multiplier[i] / value_before).abs() < s.scalars[9]
        {
            core.out.counts[10] += 1;
            continue;
        }
        let (minimum, step) = terms(day, i, m);
        // Full closes retain odd lots even after a rule change. Capacity is
        // applied before rounding so a cap large enough to close is not lost.
        let cap = if s.scalars[6] <= 0.0 || m.volume.is_empty() {
            f64::INFINITY
        } else {
            let v = m.volume[base + i];
            if v.is_finite() && v > 0.0 {
                v * s.scalars[6]
            } else {
                0.0
            }
        };
        let reducing = if before > 0.0 && target < before {
            -(before - target.max(0.0))
        } else if before < 0.0 && target > before {
            target.min(0.0) - before
        } else {
            0.0
        };
        let mut used = 0.0;
        let mut blocked = [false, false];
        if reducing != 0.0 {
            let allowed = p.is_finite()
                && if reducing > 0.0 {
                    m.buy[base + i] != 0
                } else {
                    m.sell[base + i] != 0
                };
            let q = if allowed {
                reducing.signum()
                    * round_close(reducing.abs().min(cap), before.abs(), minimum, step)
            } else {
                0.0
            };
            // Quantity filtering is final; only market obstacles can carry.
            if !allowed || reducing.abs() > cap {
                blocked[usize::from(reducing < 0.0)] = true;
            }
            if q != 0.0 {
                let x = futures::quote(q, i, day, m, s);
                if x.price.is_finite() && x.price > 0.0 {
                    book.execute_quote(
                        st.shares,
                        &mut st.cash,
                        i,
                        q,
                        day,
                        row.is_none(),
                        false,
                        m,
                        &mut core.out,
                        x,
                    );
                    used = q.abs();
                    if q.abs() == before.abs() {
                        core.out.counts[56] += 1;
                    }
                    if !c.notional.is_empty()
                        && q.abs() * f.multiplier[i] * mark < c.notional[base + i]
                    {
                        core.out.counts[55] += 1;
                    }
                } else {
                    blocked[usize::from(reducing < 0.0)] = true;
                    core.out.counts[28] += 1;
                    core.out.daily[25 * m.t + day] += 1.0;
                }
            } else if allowed {
                core.out.counts[38] += 1;
            }
        }
        let after = st.shares[i];
        let desired = if (target > after && target > before && after >= 0.0)
            || (target < after && target < before && after <= 0.0)
        {
            target - after
        } else {
            0.0
        };
        if desired != 0.0 {
            let allowed = p.is_finite()
                && if desired > 0.0 {
                    m.buy[base + i] != 0
                } else {
                    m.sell[base + i] != 0
                };
            if allowed {
                marks(day, m, &[i])?;
                let requested = desired.signum() * desired.abs().min((cap - used).max(0.0));
                let q = rounded(day, i, requested, m);
                if q.abs() < desired.abs() {
                    core.out.counts[53] += 1;
                }
                if q == 0.0 {
                    core.out.counts[38] += 1;
                    if !c.notional.is_empty()
                        && round_open(requested.abs(), minimum, step) * f.multiplier[i] * mark
                            < c.notional[base + i]
                    {
                        core.out.counts[54] += 1;
                    }
                } else {
                    let x = futures::quote(q, i, day, m, s);
                    if x.price.is_finite() && x.price > 0.0 {
                        if before * q < 0.0 && used > 0.0 {
                            core.out.counts[30] += 1;
                        }
                        core.orders.push((i, q));
                    } else {
                        blocked[usize::from(desired < 0.0)] = true;
                        core.out.counts[28] += 1;
                        core.out.daily[25 * m.t + day] += 1.0;
                    }
                }
                if requested.abs() < desired.abs() {
                    blocked[usize::from(desired < 0.0)] = true;
                }
            } else {
                blocked[usize::from(desired < 0.0)] = true;
            }
        }
        for (j, b) in blocked.iter().enumerate() {
            if *b {
                core.out.counts[1 + j] += 1;
                core.out.daily[(7 + j) * m.t + day] += 1.0;
            }
        }
        if blocked.iter().any(|b| *b) && s.flags[0] {
            core.pending.push((i, w));
        }
    }
    tidy(book, st);
    // Independent opening basket evaluated at mark_open: fees and the
    // execution-to-mark loss change equity, not the accounting cash basis.
    let v = value(day, false, m, st, book);
    let project = |orders: &[(usize, f64)], scale: f64| {
        let mut budget = OpeningBudget {
            equity: v.0,
            margin: v.3,
        };
        for &(i, qty) in orders {
            let q = rounded(day, i, qty * scale, m);
            if q == 0.0 {
                continue;
            }
            let x = futures::quote(q, i, day, m, s);
            let mark = c.open[base + i];
            budget.project(&Opening {
                quantity: q,
                multiplier: f.multiplier[i],
                mark,
                price: x.price,
                fee: x.fees.commission,
                rate: rate(f.margin_rate, day, i, m),
            });
        }
        budget.equity >= budget.margin
    };
    // Candidate execution gains cannot bootstrap an empty collateral account,
    // including one just exhausted by ordinary reductions.
    let scale = if v.0 <= 0.0 && !core.orders.is_empty() {
        core.out.counts[33] += 1;
        0.0
    } else if project(&core.orders, 1.0) {
        1.0
    } else {
        core.out.counts[33] += 1;
        let (mut lo, mut hi) = (0.0, 1.0);
        for _ in 0..64 {
            let mid = (lo + hi) * 0.5;
            if project(&core.orders, mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    };
    let mut budget = OpeningBudget {
        equity: v.0,
        margin: v.3,
    };
    for &(i, qty) in &core.orders {
        let mut q = rounded(day, i, qty * scale, m);
        let feasible = |q: f64| {
            let x = futures::quote(q, i, day, m, s);
            budget.allows(&Opening {
                quantity: q,
                multiplier: f.multiplier[i],
                mark: c.open[base + i],
                price: x.price,
                fee: x.fees.commission,
                rate: rate(f.margin_rate, day, i, m),
            })
        };
        if q != 0.0 && !feasible(q) {
            let (mut lo, mut hi) = (0.0, q.abs());
            for _ in 0..64 {
                let mid = (lo + hi) * 0.5;
                let q0 = rounded(day, i, qty.signum() * mid, m);
                if q0 == 0.0 || feasible(q0) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            q = rounded(day, i, qty.signum() * lo, m);
        }
        if q.abs() < qty.abs() {
            core.out.counts[53] += 1;
        }
        if q == 0.0 {
            core.out.counts[38] += 1;
            continue;
        }
        let x = futures::quote(q, i, day, m, s);
        budget.project(&Opening {
            quantity: q,
            multiplier: f.multiplier[i],
            mark: c.open[base + i],
            price: x.price,
            fee: x.fees.commission,
            rate: rate(f.margin_rate, day, i, m),
        });
        if core.out.counts[41..45].iter().sum::<u64>() > 0 {
            core.out.counts[57] += 1;
        }
        book.execute_quote(
            st.shares,
            &mut st.cash,
            i,
            q,
            day,
            row.is_none(),
            false,
            m,
            &mut core.out,
            x,
        );
    }
    tidy(book, st);
    tier |= exceeded(day, 0, m, st, book);
    let v = value(day, false, m, st, book);
    if !book.held.is_empty() && v.0 - v.1 <= v.2 * 1e-12 {
        liquidate(core, day, 42, 0.0, m, s, st, book)?;
    }
    tier |= exceeded(day, 1, m, st, book);
    if !book.held.is_empty() {
        let v = value(day, false, m, st, book);
        let mut slope = 0.0;
        for &i in &book.held {
            let q = st.shares[i] * f.multiplier[i];
            let k = base + i;
            let end = if q > 0.0 { c.low[k] } else { c.high[k] };
            let r = rate(
                if c.maint.is_empty() {
                    f.margin_rate
                } else {
                    c.maint
                },
                day,
                i,
                m,
            );
            slope += (q - r * q.abs()) * (end - c.open[k]);
        }
        if slope >= 0.0 {
            core.out.counts[49] += 1;
        } else if v.0 - v.1 + slope <= v.2 * 1e-12 {
            let lambda = (-(v.0 - v.1) / slope).min(1.0);
            core.out.counts[if book.held.len() == 1 { 45 } else { 46 }] += 1;
            if book.held.iter().any(|&i| st.shares[i] > 0.0)
                && book.held.iter().any(|&i| st.shares[i] < 0.0)
            {
                core.out.counts[47] += 1;
            }
            if (lambda - 1.0).abs() <= 1e-12 {
                core.out.counts[48] += 1;
            }
            // Intraday exit: the high may precede the exit, so tier coverage
            // is conservatively invalid regardless of the interpolated price.
            tier = true;
            liquidate(core, day, 43, lambda, m, s, st, book)?;
        }
    }
    // Capture expiry notional before cash delivery removes its position.
    tier |= exceeded(day, 2, m, st, book);
    let mut fm = book.settle::<true>(st.shares, &mut st.cash, day, m, &mut core.out);
    if let Some(error) = &core.out.error {
        return Err(error.clone());
    }
    let v = value(day, true, m, st, book);
    if !book.held.is_empty() && v.0 - v.1 <= v.2 * 1e-12 {
        liquidate(core, day, 44, 0.0, m, s, st, book)?;
        fm = 0.0;
    }
    core.out.counts[52] += u64::from(tier);
    let own = st.cash - fm;
    let interest = own.max(0.0) * s.v(9, day);
    let financing = (-own).max(0.0) * s.v(15, day);
    st.cash += interest - financing;
    if let Some(w) = core.out.wallet.as_mut() {
        w.balance += interest - financing;
        w.close(day, st.shares, m);
    }
    if interest > 1e-10 {
        core.out.counts[14] += 1;
    }
    if financing > 1e-10 {
        core.out.counts[26] += 1;
    }
    // Terminal delivery fees / interest are debt, not an exchange liquidation
    // credit. Still latch failure so a later deposit cannot revive trading.
    if st.cash < 0.0 && !core.bankrupt {
        core.bankrupt = true;
        core.out.bankruptcies.push(day);
        core.out.counts[25] += 1;
        core.out.counts[29] += 1;
    }
    let denom = core.previous + core.flow;
    let ret = if denom
        > 8.0 * f64::EPSILON * s.scalars[5].max(core.previous.abs()).max(core.flow.abs())
    {
        st.cash / denom - 1.0
    } else {
        0.0
    };
    core.unit *= 1.0 + ret;
    core.call = false;
    for (k, v) in [
        (0, st.cash / s.scalars[5]),
        (1, st.cash),
        (14, interest),
        (15, ret),
        (16, core.unit),
        (20, financing),
        (
            21,
            if st.cash > 0.0 {
                fm / st.cash
            } else if fm > 0.0 {
                f64::INFINITY
            } else {
                0.0
            },
        ),
        (23, st.cash),
        (27, fm),
    ] {
        core.out.daily[k * m.t + day] = v;
    }
    core.out.daily[2 * m.t + day] = if value_before > 0.0 {
        core.out.daily[2 * m.t + day] * 0.5 / value_before
    } else {
        0.0
    };
    core.previous = st.cash;
    if let Some(p) = core.out.positions.as_mut() {
        p[base..base + m.n].copy_from_slice(st.shares);
    }
    Ok(())
}
