//! Coupled spot/futures account. The spot-only runner stays separately compiled.

use super::{
    COUNTS, DAILY, Market, Output, Quote, Settings, State, Targets, capacity, cell, fill,
    past_close, projected, quantity_tolerance, quote,
};
use tradecore::{
    Account, Fees, futures_charge, futures_margin, futures_variation, round_close, round_open,
};

#[inline]
fn rate(a: &[f64], day: usize, col: usize, m: &Market<'_>) -> f64 {
    if a.len() == 1 {
        a[0]
    } else if a.len() == m.n {
        a[col]
    } else {
        a[day * m.n + col]
    }
}

#[inline]
fn fmargin(q: f64, p: f64, day: usize, i: usize, m: &Market<'_>) -> f64 {
    let f = m.futures.as_ref().unwrap();
    futures_margin(q, f.multiplier[i], p, rate(f.margin_rate, day, i, m))
}

fn future_quote(q: f64, i: usize, day: usize, m: &Market<'_>, s: &Settings<'_>) -> Quote {
    let f = m.futures.as_ref().unwrap();
    let raw = m.price[day * m.n + i];
    let slip = cell(s.slippage, day, i, m);
    let slipped = raw * (1.0 + q.signum() * slip);
    let impact = if s.impact.is_empty() {
        0.0
    } else {
        let v = m.volume[day * m.n + i];
        if v.is_finite() && v > 0.0 {
            s.impact[day * m.n + i] * (q.abs() / v).powf(s.scalars[4])
        } else {
            f64::INFINITY
        }
    };
    let price = slipped * (1.0 + q.signum() * impact);
    let amount = q.abs() * f.multiplier[i] * price;
    Quote {
        price,
        amount,
        fees: Fees {
            commission: futures_charge(
                q,
                f.multiplier[i],
                price,
                rate(f.fee_rate, day, i, m),
                rate(f.fee_per_contract, day, i, m),
            ),
            ..Default::default()
        },
        slip: q.abs() * f.multiplier[i] * raw * slip,
        impact: q.abs() * f.multiplier[i] * (price - slipped).abs(),
    }
}

#[inline]
fn realized(before: f64, q: f64, basis: f64, price: f64, multiplier: f64) -> f64 {
    if before * q < 0.0 {
        futures_variation(
            before.signum() * before.abs().min(q.abs()),
            multiplier,
            basis,
            price,
        )
    } else {
        0.0
    }
}

#[inline]
fn feasible(a: Account, fm: f64, s: &Settings<'_>, day: usize) -> bool {
    a.margin(s.v(11, day), s.v(12, day), fm) <= a.equity()
        && (s.flags[3] || a.own_cash(s.flags[4]) - fm >= 0.0)
}

struct Portfolio {
    spot: State,
    basis: Vec<f64>,
    futures_held: Vec<usize>,
}

impl Portfolio {
    fn futures_margin(&self, day: usize, m: &Market<'_>) -> f64 {
        self.futures_held
            .iter()
            .map(|&i| fmargin(self.spot.shares[i], self.basis[i], day, i, m))
            .sum()
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Explicit account and market state"
    )]
    fn project(
        &self,
        mut a: Account,
        mut fm: f64,
        i: usize,
        q: f64,
        day: usize,
        m: &Market<'_>,
        s: &Settings<'_>,
    ) -> (Account, f64) {
        if q == 0.0 {
            return (a, fm);
        }
        if m.futures.as_ref().unwrap().kind[i] {
            let x = future_quote(q, i, day, m, s);
            let before = self.spot.shares[i];
            let mult = m.futures.as_ref().unwrap().multiplier[i];
            a.cash += realized(before, q, self.basis[i], x.price, mult) - x.fees.commission;
            fm -= fmargin(before, self.basis[i], day, i, m);
            fm += fmargin(before + q, x.price, day, i, m);
        } else {
            a = projected(a, i, q, self.spot.shares[i], day, m, s);
        }
        (a, fm)
    }

    #[expect(clippy::too_many_arguments, reason = "Explicit fill context")]
    fn execute(
        &mut self,
        i: usize,
        q: f64,
        day: usize,
        rebalance: bool,
        forced: bool,
        m: &Market<'_>,
        s: &Settings<'_>,
        out: &mut Output,
    ) {
        if !m.futures.as_ref().unwrap().kind[i] {
            fill(day, i, q, !rebalance, forced, m, s, &mut self.spot, out);
            return;
        }
        let x = future_quote(q, i, day, m, s);
        let before = self.spot.shares[i];
        let after = before + q;
        let mult = m.futures.as_ref().unwrap().multiplier[i];
        let pnl = realized(before, q, self.basis[i], x.price, mult);
        self.spot.cash += pnl - x.fees.commission;
        out.contribution[i] += pnl - x.fees.commission;
        self.spot.shares[i] = after;
        self.basis[i] = if after == 0.0 {
            0.0
        } else if before == 0.0 || before.signum() != after.signum() {
            x.price
        } else if before.signum() == q.signum() {
            (before.abs() * self.basis[i] + q.abs() * x.price) / after.abs()
        } else {
            self.basis[i]
        };
        if before == 0.0 && after != 0.0 {
            self.futures_held.push(i);
        }
        out.daily[2 * m.t + day] += x.amount;
        out.daily[3 * m.t + day] += x.fees.commission;
        out.daily[6 * m.t + day] += x.slip;
        out.daily[9 * m.t + day] += x.impact;
        out.daily[26 * m.t + day] += pnl;
        out.counts[0] += 1;
        if !rebalance {
            out.counts[8] += 1;
        }
        if x.impact > 0.0 {
            out.counts[15] += 1;
        }
        if before.abs() < after.abs() {
            out.counts[31] += 1;
        } else {
            out.counts[32] += 1;
        }
        if forced {
            out.counts[24] += 1;
            out.counts[35] += 1;
            out.liquidations.push((day, i, q));
        }
        if let Some(trades) = out.trades.as_mut() {
            trades.push((
                day,
                i,
                q,
                x.price,
                x.fees.commission,
                0.0,
                0.0,
                x.slip,
                x.impact,
            ));
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "Explicit target and market context"
)]
fn target_quantity(
    i: usize,
    w: f64,
    value: f64,
    day: usize,
    m: &Market<'_>,
    s: &Settings<'_>,
    portfolio: &Portfolio,
    shares: bool,
) -> f64 {
    if shares {
        return w;
    }
    let future = m.futures.as_ref().unwrap().kind[i];
    let p = m.price[day * m.n + i];
    let mark = if p.is_finite() {
        p
    } else if future {
        portfolio.basis[i]
    } else {
        portfolio.spot.last[i]
    };
    if mark <= 0.0 {
        return 0.0;
    }
    let denom = mark
        * if future {
            m.futures.as_ref().unwrap().multiplier[i]
        } else {
            1.0
        };
    let raw = w * value.max(0.0) * (1.0 - s.scalars[8]) / denom;
    if future {
        raw.signum() * raw.abs().floor()
    } else {
        let min = s.vectors[6][i];
        if raw.abs() < min && portfolio.spot.shares[i] * raw > 0.0 {
            raw
        } else {
            raw.signum() * round_open(raw.abs(), min, s.vectors[7][i])
        }
    }
}

fn lot_params(i: usize, m: &Market<'_>, s: &Settings<'_>, shares: bool) -> (f64, f64) {
    if m.futures.as_ref().unwrap().kind[i] {
        (1.0, 1.0)
    } else {
        (s.vectors[6][i], if shares { 0.0 } else { s.vectors[7][i] })
    }
}

pub(super) fn run(m: &Market<'_>, targets: &Targets<'_>, s: &Settings<'_>) -> Output {
    let f = m.futures.as_ref().unwrap();
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
    let mut portfolio = Portfolio {
        spot: State {
            cash: s.scalars[5],
            shares: vec![0.0; m.n],
            last: vec![0.0; m.n],
            held: Vec::new(),
        },
        basis: vec![0.0; m.n],
        futures_held: Vec::new(),
    };
    let mut pending: Vec<(usize, f64)> = Vec::new();
    let mut commands: Vec<(usize, f64)> = Vec::new();
    let mut orders: Vec<(usize, f64)> = Vec::new();
    let mut weights = vec![0.0; m.n];
    let mut planned = vec![0.0; m.n];
    let mut cursor = 0;
    let (mut previous, mut unit) = (s.scalars[5], 1.0);
    let (mut call, mut bankrupt) = (false, false);
    for day in 0..m.t {
        let base = day * m.n;
        let fm_before = portfolio.futures_margin(day, m);
        let restricted_short: f64 = if s.flags[4] {
            0.0
        } else {
            portfolio
                .spot
                .held
                .iter()
                .map(|&i| (-portfolio.spot.shares[i]).max(0.0) * portfolio.spot.last[i])
                .sum()
        };
        let available = (portfolio.spot.cash - restricted_short - fm_before).max(0.0);
        let requested = s.v(10, day);
        let flow = requested.max(-available);
        portfolio.spot.cash += flow;
        out.daily[12 * m.t + day] = flow;
        out.daily[13 * m.t + day] = flow - requested;
        out.counts[12] += u64::from(flow != 0.0);
        out.counts[13] += u64::from(flow > requested);
        let mut removed = false;
        for &col in &m.action_cols[m.action_indptr[day] as usize..m.action_indptr[day + 1] as usize]
        {
            let i = col as usize;
            if portfolio.spot.shares[i] == 0.0 {
                continue;
            }
            if !m.dividend.is_empty() && m.dividend[base + i] != 0.0 {
                let d = portfolio.spot.shares[i] * m.dividend[base + i];
                let tax = if d > 0.0 { d * s.v(8, day) } else { 0.0 };
                if portfolio.spot.last[i] > 0.0 {
                    portfolio.spot.last[i] =
                        (portfolio.spot.last[i] - m.dividend[base + i]).max(0.0);
                }
                portfolio.spot.cash += d - tax;
                out.contribution[i] += d - tax;
                out.daily[10 * m.t + day] += d;
                out.daily[11 * m.t + day] += tax;
                out.counts[6] += 1;
                out.counts[11] += u64::from(tax > 0.0);
                out.counts[22] += u64::from(d < 0.0);
            }
            if !m.split.is_empty() && m.split[base + i] != 1.0 {
                let factor = m.split[base + i];
                portfolio.spot.shares[i] *= factor;
                portfolio.spot.last[i] /= factor;
                out.counts[7] += 1;
            }
            if m.first_delist[i] <= day as i64 {
                let p = portfolio.spot.last[i] * s.scalars[7];
                let proceeds = portfolio.spot.shares[i] * p;
                portfolio.spot.cash += proceeds;
                out.contribution[i] += proceeds;
                out.events
                    .push((day, i, portfolio.spot.shares[i], p, proceeds));
                out.counts[5] += 1;
                portfolio.spot.shares[i] = 0.0;
                removed = true;
            }
        }
        if removed {
            portfolio
                .spot
                .held
                .retain(|&i| portfolio.spot.shares[i] != 0.0);
        }
        let rebalance = cursor < targets.days.len() && targets.days[cursor] as usize == day;
        let forced = (call && s.flags[5]) || bankrupt;
        let mut unfilled_liquidation = false;
        commands.clear();
        if rebalance {
            pending.clear();
            for &i in portfolio.spot.held.iter().chain(&portfolio.futures_held) {
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
            portfolio.spot.account(day, m)
        } else {
            Account {
                cash: portfolio.spot.cash,
                ..Default::default()
            }
        };
        let mut fm = fm_before;
        let value = account.equity();
        if forced {
            pending.clear();
            let requested_commands = commands.clone();
            commands.clear();
            let after_liquidation = |keep: f64| {
                let (mut a, mut margin) = (account, fm);
                for &i in portfolio.futures_held.iter().chain(&portfolio.spot.held) {
                    let current = portfolio.spot.shares[i];
                    let (minimum, step) = lot_params(i, m, s, targets.shares);
                    let qty =
                        round_close(current.abs() * (1.0 - keep), current.abs(), minimum, step);
                    let q = -current.signum() * qty;
                    if q != 0.0 && m.price[base + i].is_finite() {
                        let x = if f.kind[i] {
                            future_quote(q, i, day, m, s)
                        } else {
                            quote(q, i, day, m, s)
                        };
                        if x.price > 0.0 && x.price.is_finite() {
                            (a, margin) = portfolio.project(a, margin, i, q, day, m, s);
                        }
                    }
                }
                (a, margin)
            };
            let all = after_liquidation(0.0);
            let keep = if bankrupt || value <= 0.0 || !feasible(all.0, all.1, s, day) {
                0.0
            } else if feasible(account, fm, s, day) {
                1.0
            } else {
                let (mut lo, mut hi) = (0.0, 1.0);
                for _ in 0..64 {
                    let mid = (lo + hi) * 0.5;
                    let trial = after_liquidation(mid);
                    if feasible(trial.0, trial.1, s, day) {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                lo
            };
            for &i in portfolio.futures_held.iter().chain(&portfolio.spot.held) {
                let current = portfolio.spot.shares[i];
                let (minimum, step) = lot_params(i, m, s, targets.shares);
                let mut qty =
                    round_close(current.abs() * (1.0 - keep), current.abs(), minimum, step);
                if rebalance && let Some(&(_, w)) = requested_commands.iter().find(|x| x.0 == i) {
                    let desired =
                        target_quantity(i, w, value, day, m, s, &portfolio, targets.shares);
                    let reduction = if current > 0.0 {
                        current - desired.max(0.0)
                    } else {
                        desired.min(0.0) - current
                    };
                    qty = qty.max(round_close(reduction, current.abs(), minimum, step));
                }
                planned[i] = -current.signum() * qty;
                commands.push((i, current + planned[i]));
            }
        }
        // Futures reductions release collateral first; spot sales then covers.
        commands.sort_by_key(|&(i, _)| (!f.kind[i], portfolio.spot.shares[i] < 0.0, i));
        orders.clear();
        for &(i, w) in &commands {
            if !f.kind[i] && m.first_delist[i] <= day as i64 {
                continue;
            }
            if f.kind[i] && day as i64 > f.expiry[i] {
                if w != 0.0 {
                    out.daily[29 * m.t + day] += 1.0;
                    out.counts[37] += 1;
                }
                continue;
            }
            if !f.kind[i] && portfolio.spot.shares[i] == 0.0 {
                portfolio.spot.last[i] = past_close(m, day, i);
            }
            let p = m.price[base + i];
            let mark = if p.is_finite() {
                p
            } else if f.kind[i] {
                portfolio.basis[i]
            } else {
                portfolio.spot.last[i]
            };
            let current = portfolio.spot.shares[i];
            let (minimum, step) = lot_params(i, m, s, targets.shares);
            let mut target = if forced {
                w
            } else {
                if targets.shares {
                    out.counts[9] += 1;
                }
                target_quantity(i, w, value, day, m, s, &portfolio, targets.shares)
            };
            if (target - current).abs() <= quantity_tolerance(target, current) {
                target = current;
            }
            let target_notional = current * mark * if f.kind[i] { f.multiplier[i] } else { 1.0 };
            if !forced
                && !targets.shares
                && w != 0.0
                && current != 0.0
                && value > 0.0
                && (w - target_notional / value).abs() < s.scalars[9]
            {
                out.counts[10] += 1;
                continue;
            }
            let cap = capacity(m, s, day, i, step);
            let mut used = 0.0;
            let mut blocked_buy = false;
            let mut blocked_sell = false;
            let reduction = if forced {
                planned[i]
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
                if q != 0.0 {
                    let x = if f.kind[i] {
                        future_quote(q, i, day, m, s)
                    } else {
                        quote(q, i, day, m, s)
                    };
                    if x.price > 0.0 && x.price.is_finite() {
                        let (candidate, candidate_fm) =
                            portfolio.project(account, fm, i, q, day, m, s);
                        let own_before = account.own_cash(s.flags[4]) - fm;
                        let own_after = candidate.own_cash(s.flags[4]) - candidate_fm;
                        if !s.flags[3] && own_before >= -1e-10 * s.scalars[5] && own_after < 0.0 {
                            let requested_quantity = q;
                            let (mut lo, mut hi) = (0.0, q.abs());
                            for _ in 0..64 {
                                let mid = (lo + hi) * 0.5;
                                let trial_q =
                                    q.signum() * round_close(mid, current.abs(), minimum, step);
                                let (trial_a, trial_fm) =
                                    portfolio.project(account, fm, i, trial_q, day, m, s);
                                if trial_a.own_cash(s.flags[4]) - trial_fm >= 0.0 {
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
                            let (candidate, candidate_fm) =
                                portfolio.project(account, fm, i, q, day, m, s);
                            if f.kind[i]
                                || portfolio.spot.cash + x.amount
                                    >= x.fees.commission + x.fees.tax + x.fees.other
                            {
                                portfolio.execute(i, q, day, rebalance, forced, m, s, &mut out);
                                account = candidate;
                                fm = candidate_fm;
                                used = q.abs();
                            }
                        }
                    } else {
                        out.daily[25 * m.t + day] += 1.0;
                        out.counts[28] += 1;
                    }
                }
                if forced && reduction.abs() - used > quantity_tolerance(reduction, current) {
                    unfilled_liquidation = true;
                }
            }
            let after = portfolio.spot.shares[i];
            let opening = if target > 0.0 && after >= 0.0 && target > after && target > current {
                round_open(target - after, minimum, step)
            } else if target < 0.0 && after <= 0.0 && target < after && target < current {
                -round_open(after - target, minimum, step)
            } else {
                0.0
            };
            if opening.abs() > quantity_tolerance(target, after) && !forced && !bankrupt {
                let allowed = p.is_finite()
                    && if opening > 0.0 {
                        m.buy[base + i]
                    } else {
                        m.sell[base + i]
                            && (f.kind[i]
                                || (s.flags[2]
                                    && (m.can_short.is_empty() || m.can_short[base + i])))
                    };
                let mut q = if allowed {
                    opening.signum()
                        * round_open(opening.abs().min((cap - used).max(0.0)), minimum, step)
                } else {
                    0.0
                };
                if q != 0.0 {
                    let x = if f.kind[i] {
                        future_quote(q, i, day, m, s)
                    } else {
                        quote(q, i, day, m, s)
                    };
                    if x.price <= 0.0 || !x.price.is_finite() {
                        q = 0.0;
                        out.daily[25 * m.t + day] += 1.0;
                        out.counts[28] += 1;
                    }
                }
                if q != opening {
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
            // Hedge openings reserve collateral before cash-funded spot purchases.
            orders.sort_by_key(|&(i, q)| (!f.kind[i], q > 0.0, i));
            let trial = |scale: f64| {
                let (mut a, mut margin) = (account, fm);
                for &(i, desired) in &orders {
                    let (minimum, step) = lot_params(i, m, s, targets.shares);
                    let q = desired.signum() * round_open(desired.abs() * scale, minimum, step);
                    if q != 0.0 {
                        (a, margin) = portfolio.project(a, margin, i, q, day, m, s);
                    }
                }
                (a, margin)
            };
            let full = trial(1.0);
            let scale = if feasible(full.0, full.1, s, day) {
                1.0
            } else {
                out.counts[20] += 1;
                if orders.iter().any(|&(i, _)| f.kind[i]) {
                    out.counts[33] += 1;
                }
                let (mut lo, mut hi) = (0.0, 1.0);
                for _ in 0..64 {
                    let mid = (lo + hi) * 0.5;
                    let test = trial(mid);
                    if feasible(test.0, test.1, s, day) {
                        lo = mid
                    } else {
                        hi = mid
                    }
                }
                lo
            };
            for &(i, desired) in &orders {
                let (minimum, step) = lot_params(i, m, s, targets.shares);
                let mut q = desired.signum() * round_open(desired.abs() * scale, minimum, step);
                if q != 0.0 {
                    let test = portfolio.project(account, fm, i, q, day, m, s);
                    if !feasible(test.0, test.1, s, day) {
                        let (mut lo, mut hi) = (0.0, q.abs());
                        for _ in 0..64 {
                            let mid = (lo + hi) * 0.5;
                            let candidate = q.signum() * round_open(mid, minimum, step);
                            let trial = portfolio.project(account, fm, i, candidate, day, m, s);
                            if feasible(trial.0, trial.1, s, day) {
                                lo = mid
                            } else {
                                hi = mid
                            }
                        }
                        q = q.signum() * round_open(lo, minimum, step);
                    }
                }
                if q.abs()
                    > quantity_tolerance(
                        portfolio.spot.shares[i] + desired,
                        portfolio.spot.shares[i],
                    )
                {
                    (account, fm) = portfolio.project(account, fm, i, q, day, m, s);
                    portfolio.execute(i, q, day, rebalance, false, m, s, &mut out);
                }
            }
        }
        if !commands.is_empty() {
            portfolio.spot.held.sort_unstable();
            portfolio.spot.held.dedup();
            portfolio
                .spot
                .held
                .retain(|&i| portfolio.spot.shares[i] != 0.0);
            portfolio.futures_held.sort_unstable();
            portfolio.futures_held.dedup();
            portfolio
                .futures_held
                .retain(|&i| portfolio.spot.shares[i] != 0.0);
        }
        let (mut long, mut short) = (0.0, 0.0);
        for &i in &portfolio.spot.held {
            let p = m.close[base + i];
            if p.is_finite() {
                portfolio.spot.last[i] = p;
            } else if portfolio.spot.last[i] == 0.0 && m.price[base + i].is_finite() {
                portfolio.spot.last[i] = m.price[base + i];
            }
            let v = portfolio.spot.shares[i] * portfolio.spot.last[i];
            if v > 0.0 {
                long += v;
            } else {
                short -= v;
                let fee = -v * cell(s.borrow, day, i, m);
                portfolio.spot.cash -= fee;
                out.contribution[i] -= fee;
                out.daily[19 * m.t + day] += fee;
                out.counts[21] += u64::from(fee > 0.0);
            }
        }
        let mut closing_fm = 0.0;
        for &i in &portfolio.futures_held {
            let settlement = f.settlement[base + i];
            let pnl = futures_variation(
                portfolio.spot.shares[i],
                f.multiplier[i],
                portfolio.basis[i],
                settlement,
            );
            portfolio.spot.cash += pnl;
            out.contribution[i] += pnl;
            out.daily[26 * m.t + day] += pnl;
            portfolio.basis[i] = settlement;
            if day as i64 == f.expiry[i] {
                let qty = portfolio.spot.shares[i];
                let fee = futures_charge(
                    qty,
                    f.multiplier[i],
                    settlement,
                    rate(f.delivery_fee_rate, day, i, m),
                    rate(f.delivery_fee_per_contract, day, i, m),
                );
                portfolio.spot.cash -= fee;
                out.contribution[i] -= fee;
                out.daily[5 * m.t + day] += fee;
                out.daily[28 * m.t + day] += fee;
                out.expiries.push((day, i, qty, settlement, fee));
                out.counts[36] += 1;
                portfolio.spot.shares[i] = 0.0;
                portfolio.basis[i] = 0.0;
            } else {
                closing_fm += fmargin(portfolio.spot.shares[i], settlement, day, i, m);
            }
        }
        portfolio
            .futures_held
            .retain(|&i| portfolio.spot.shares[i] != 0.0);
        let own = portfolio.spot.cash - if s.flags[4] { 0.0 } else { short } - closing_fm;
        let interest = own.max(0.0) * s.v(9, day);
        let financing = (-own).max(0.0) * s.v(15, day);
        portfolio.spot.cash += interest - financing;
        out.counts[14] += u64::from(interest > 1e-10);
        out.counts[26] += u64::from(financing > 1e-10);
        let equity = portfolio.spot.cash + long - short;
        let denom = previous + flow;
        let ret = if denom > 8.0 * f64::EPSILON * s.scalars[5].max(previous.abs()).max(flow.abs()) {
            equity / denom - 1.0
        } else {
            0.0
        };
        unit *= 1.0 + ret;
        let required = s.v(11, day) * long + s.v(12, day) * short + closing_fm;
        let tolerance = 1e-10 * equity.abs().max(long).max(short).max(closing_fm).max(1.0);
        call = s.v(13, day) * long + s.v(14, day) * short + closing_fm > equity + tolerance
            || unfilled_liquidation;
        if call {
            out.counts[23] += 1;
            if closing_fm > 0.0 {
                out.counts[34] += 1;
            }
        }
        let valued = portfolio
            .spot
            .held
            .iter()
            .all(|&i| portfolio.spot.last[i] > 0.0);
        if equity <= 0.0
            && !bankrupt
            && valued
            && (!portfolio.spot.held.is_empty()
                || !portfolio.futures_held.is_empty()
                || portfolio.spot.cash < 0.0)
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
            (1, portfolio.spot.cash),
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
            (27, closing_fm),
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
            p[base..base + m.n].copy_from_slice(&portfolio.spot.shares);
        }
    }
    for &i in &portfolio.spot.held {
        out.contribution[i] += portfolio.spot.shares[i] * portfolio.spot.last[i];
    }
    out
}
