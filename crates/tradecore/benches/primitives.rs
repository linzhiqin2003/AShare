//! Paired cross-crate versus equivalent in-module inlining microbenchmark.
use std::{hint::black_box, time::Instant};
use tradecore::FeeTerms;
#[inline]
fn local(terms: FeeTerms, q: f64, p: f64) -> f64 {
    let a = q * p;
    if a <= 0.0 || q <= 0.0 {
        return 0.0;
    }
    let mut c =
        (terms.commission_rate * a + terms.commission_per_share * q).max(terms.commission_min);
    if let Some(cap) = terms.commission_max_rate {
        c = c.min(cap * a);
    }
    c + terms.tax_rate * a + (terms.fee_rate * a + terms.fee_per_share * q)
}
fn sample<F: Fn(FeeTerms, f64, f64) -> f64>(f: F) -> f64 {
    let start = Instant::now();
    let mut total = 0.0;
    for i in 0..1_000_000 {
        let terms = black_box(FeeTerms {
            commission_rate: 0.00025,
            commission_per_share: 0.001,
            commission_min: 5.0,
            commission_max_rate: Some(0.01),
            tax_rate: 0.0005,
            fee_rate: 0.00001,
            fee_per_share: 0.001,
        });
        total += f(terms, black_box((i % 1000 + 1) as f64), black_box(20.0));
    }
    black_box(total);
    start.elapsed().as_secs_f64()
}
fn main() {
    let (mut external, mut internal) = (Vec::new(), Vec::new());
    for _ in 0..11 {
        external.push(sample(|terms, q, p| {
            let x = terms.charge(q, p);
            x.commission + x.tax + x.other
        }));
        internal.push(sample(local));
    }
    println!(
        "{{\"iterations\":1000000,\"cross_crate_samples_s\":{external:?},\"local_samples_s\":{internal:?}}}"
    );
}
