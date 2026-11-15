//! Parallel exchange wallet ledger. Never feeds the legacy trading arithmetic.

use crate::{Market, codec::Codec};

#[derive(Clone)]
pub struct Ledger {
    pub balance: f64,
    pub unrealized: f64,
    pub entry: Vec<f64>,
    pub balances: Vec<f64>,
    pub unrealized_rows: Vec<f64>,
}

impl Ledger {
    pub fn new(initial: f64, t: usize, n: usize) -> Self {
        Self {
            balance: initial,
            unrealized: 0.0,
            entry: vec![0.0; n],
            balances: vec![0.0; t],
            unrealized_rows: vec![0.0; t],
        }
    }

    pub fn fill(&mut self, i: usize, before: f64, q: f64, price: f64, mult: f64, fee: f64) {
        if before * q < 0.0 {
            let closed = before.signum() * before.abs().min(q.abs());
            self.balance += closed * mult * (price - self.entry[i]);
        }
        self.balance -= fee;
        let after = before + q;
        if after == 0.0 {
            self.entry[i] = 0.0;
        } else if before == 0.0 || before.signum() != after.signum() {
            self.entry[i] = price;
        } else if before * q > 0.0 {
            self.entry[i] = (before.abs() * self.entry[i] + q.abs() * price) / after.abs();
        }
    }

    pub fn close(&mut self, day: usize, shares: &[f64], m: &Market<'_>) {
        let f = m.futures.as_ref().unwrap();
        self.unrealized = 0.0;
        for (i, &q) in shares.iter().enumerate() {
            if q != 0.0 {
                self.unrealized +=
                    q * f.multiplier[i] * (f.settlement[day * m.n + i] - self.entry[i]);
            }
        }
        self.balances[day] = self.balance;
        self.unrealized_rows[day] = self.unrealized;
    }
}

impl Codec for Ledger {
    fn put(&self, w: &mut Vec<u8>) {
        self.balance.put(w);
        self.unrealized.put(w);
        self.entry.put(w);
        self.balances.put(w);
        self.unrealized_rows.put(w);
    }
    fn take(r: &mut &[u8]) -> Result<Self, String> {
        Ok(Self {
            balance: Codec::take(r)?,
            unrealized: Codec::take(r)?,
            entry: Codec::take(r)?,
            balances: Codec::take(r)?,
            unrealized_rows: Codec::take(r)?,
        })
    }
}
