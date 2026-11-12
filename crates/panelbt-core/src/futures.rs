//! Futures primitives used by the single daily runner in `lib.rs`.
//!
//! There is deliberately no second account loop here. The runner is compiled
//! with `FUT = false` for spot-only markets and `FUT = true` when a `Futures`
//! specification is present; spot rules are the same source in both cases.

use super::{Market, Output, Quote, Settings, cell};
use tradecore::{Account, Fees, futures_charge, futures_margin, futures_variation};

/// Scalar, per-column (stored as 1×N) or T×N futures rate.
#[inline]
pub(super) fn rate(a: &[f64], day: usize, col: usize, m: &Market<'_>) -> f64 {
    if a.len() == 1 {
        a[0]
    } else if a.len() == m.n {
        a[col]
    } else {
        a[day * m.n + col]
    }
}

#[inline]
pub(super) fn margin(q: f64, p: f64, day: usize, i: usize, m: &Market<'_>) -> f64 {
    let f = m.futures.as_ref().unwrap();
    futures_margin(q, f.multiplier[i], p, rate(f.margin_rate, day, i, m))
}

/// Collateral of all held contracts at their current basis and today's rate.
pub(super) fn held_margin(
    held: &[usize],
    shares: &[f64],
    basis: &[f64],
    day: usize,
    m: &Market<'_>,
) -> f64 {
    let mut total = 0.0;
    for &i in held {
        total += margin(shares[i], basis[i], day, i, m);
    }
    total
}

pub(super) fn quote(q: f64, i: usize, day: usize, m: &Market<'_>, s: &Settings<'_>) -> Quote {
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

/// Realized P&L of the contracts closed by `q` against their settlement basis.
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

/// Account and total collateral after a hypothetical contract fill.
#[expect(clippy::too_many_arguments, reason = "Explicit account state")]
pub(super) fn project(
    mut a: Account,
    mut fm: f64,
    i: usize,
    q: f64,
    before: f64,
    basis: f64,
    day: usize,
    m: &Market<'_>,
    s: &Settings<'_>,
) -> (Account, f64) {
    if q == 0.0 {
        return (a, fm);
    }
    let x = quote(q, i, day, m, s);
    let mult = m.futures.as_ref().unwrap().multiplier[i];
    a.cash += realized(before, q, basis, x.price, mult) - x.fees.commission;
    fm -= margin(before, basis, day, i, m);
    fm += margin(before + q, x.price, day, i, m);
    (a, fm)
}

/// Contract state owned by the runner when `FUT = true`.
pub struct Book<'s> {
    pub basis: &'s mut [f64],
    pub held: Vec<usize>,
    /// Sticky: a cash account monitors maintenance and insolvency once it has
    /// held a contract; a cash account without futures never does.
    pub seen: bool,
}

impl Book<'_> {
    /// Execute a contract fill. The runner never reverses a position in one
    /// fill: reversals are a complete close followed by a separate opening.
    #[expect(clippy::too_many_arguments, reason = "Explicit fill context")]
    pub(super) fn execute(
        &mut self,
        shares: &mut [f64],
        cash: &mut f64,
        i: usize,
        q: f64,
        day: usize,
        carry: bool,
        forced: bool,
        m: &Market<'_>,
        s: &Settings<'_>,
        out: &mut Output,
    ) {
        let x = quote(q, i, day, m, s);
        let before = shares[i];
        let after = before + q;
        debug_assert!(before == 0.0 || after == 0.0 || before.signum() == after.signum());
        let mult = m.futures.as_ref().unwrap().multiplier[i];
        let pnl = realized(before, q, self.basis[i], x.price, mult);
        *cash += pnl - x.fees.commission;
        out.contribution[i] += pnl - x.fees.commission;
        shares[i] = after;
        self.basis[i] = if after == 0.0 {
            0.0
        } else if before == 0.0 {
            x.price
        } else if before.signum() == q.signum() {
            (before.abs() * self.basis[i] + q.abs() * x.price) / after.abs()
        } else {
            self.basis[i]
        };
        if before == 0.0 && after != 0.0 {
            self.held.push(i);
            self.seen = true;
        }
        out.daily[2 * m.t + day] += x.amount;
        out.daily[3 * m.t + day] += x.fees.commission;
        out.daily[6 * m.t + day] += x.slip;
        out.daily[9 * m.t + day] += x.impact;
        out.daily[26 * m.t + day] += pnl;
        out.counts[0] += 1;
        if carry {
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

    /// Closing variation, expiry cash settlement and closing collateral.
    pub(super) fn settle(
        &mut self,
        shares: &mut [f64],
        cash: &mut f64,
        day: usize,
        m: &Market<'_>,
        out: &mut Output,
    ) -> f64 {
        let f = m.futures.as_ref().unwrap();
        let base = day * m.n;
        let mut closing = 0.0;
        for &i in &self.held {
            let settlement = f.settlement[base + i];
            let pnl = futures_variation(shares[i], f.multiplier[i], self.basis[i], settlement);
            *cash += pnl;
            out.contribution[i] += pnl;
            out.daily[26 * m.t + day] += pnl;
            self.basis[i] = settlement;
            if day as i64 == f.expiry[i] {
                let qty = shares[i];
                let fee = futures_charge(
                    qty,
                    f.multiplier[i],
                    settlement,
                    rate(f.delivery_fee_rate, day, i, m),
                    rate(f.delivery_fee_per_contract, day, i, m),
                );
                *cash -= fee;
                out.contribution[i] -= fee;
                out.daily[5 * m.t + day] += fee;
                out.daily[28 * m.t + day] += fee;
                out.expiries.push((day, i, qty, settlement, fee));
                out.counts[36] += 1;
                shares[i] = 0.0;
                self.basis[i] = 0.0;
            } else {
                closing += margin(shares[i], settlement, day, i, m);
            }
        }
        self.held.retain(|&i| shares[i] != 0.0);
        closing
    }
}
