//! Deterministic kernels. Each reduction belongs to exactly one worker.
#![forbid(unsafe_code)]
use rayon::prelude::*;
use std::collections::BTreeMap;

pub mod element;
pub mod evaluation;
pub mod numpy_exact;

pub const VERSION: &str = "0.1.0";

#[derive(Clone, Copy)]
pub struct Panel<'a> {
    pub x: &'a [f64],
    pub y: &'a [f64],
    pub mask: &'a [bool],
    pub t: usize,
    pub n: usize,
}
impl Panel<'_> {
    fn valid(&self, i: usize) -> bool {
        self.mask.is_empty() || self.mask[i]
    }
    fn x(&self, i: usize) -> f64 {
        if self.valid(i) && self.x[i].is_finite() {
            self.x[i]
        } else {
            f64::NAN
        }
    }
    fn y(&self, i: usize) -> f64 {
        if self.valid(i) && self.y[i].is_finite() {
            self.y[i]
        } else {
            f64::NAN
        }
    }
}

#[derive(Clone, Copy)]
pub struct Options {
    pub window: usize,
    pub min_periods: usize,
    pub ddof: usize,
    pub q: f64,
    pub normalize: bool,
    pub alpha: f64,
    pub periods: usize,
    /// Extremum position operators break ties toward the oldest observation.
    pub earliest: bool,
    /// Rolling sum, mean and sumif evaluate every window directly with a correctly
    /// rounded sum, so each cell depends only on the window's multiset of values.
    pub exact: bool,
}

#[derive(Default, Clone, Copy)]
struct Sum {
    main: f64,
    correction: f64,
}
impl Sum {
    fn add(&mut self, x: f64) {
        let next = self.main + x;
        self.correction += if self.main.abs() >= x.abs() {
            (self.main - next) + x
        } else {
            (x - next) + self.main
        };
        self.main = next;
    }
    fn value(self) -> f64 {
        self.main + self.correction
    }
}
fn sum(values: impl Iterator<Item = f64>) -> f64 {
    let mut s = Sum::default();
    for x in values {
        s.add(x);
    }
    s.value()
}
#[cfg(test)]
fn mean(v: &[f64]) -> f64 {
    sum(v.iter().copied()) / v.len() as f64
}
fn quantile(sorted: &[f64], q: f64) -> f64 {
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}
fn order_key(value: f64) -> u64 {
    // Canonicalize signed zero: equality and all tie conventions identify +/-0.
    let bits = if value == 0. {
        0.0f64.to_bits()
    } else {
        value.to_bits()
    };
    if bits >> 63 == 0 {
        bits ^ (1 << 63)
    } else {
        !bits
    }
}
fn rank_in_place(v: &mut [f64], ids: &mut Vec<(u64, usize)>, normalize: bool) {
    ids.clear();
    ids.extend(v.iter().enumerate().map(|(i, &x)| (order_key(x), i)));
    // Keys are contiguous. Sorting does not gather the value vector at each comparison.
    ids.sort_unstable_by_key(|p| p.0);
    let mut start = 0;
    while start < ids.len() {
        let mut end = start + 1;
        while end < ids.len() && ids[start].0 == ids[end].0 {
            end += 1;
        }
        let rank = (start + end + 1) as f64 / 2.;
        let rank = if normalize {
            if ids.len() == 1 {
                0.5
            } else {
                (rank - 1.) / (ids.len() - 1) as f64
            }
        } else {
            rank
        };
        for &(_, i) in &ids[start..end] {
            v[i] = rank;
        }
        start = end;
    }
}
// Nonnegative partial sums have no cancellation; <=256 terms bound their
// relative rounding error well below 2e-10. Compensate the partial totals.
fn positive_sum(v: &[f64], transform: impl Fn(f64) -> f64) -> f64 {
    sum(v
        .chunks(256)
        .map(|chunk| chunk.iter().map(|&x| transform(x)).sum::<f64>()))
}
fn correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() < 2 {
        return f64::NAN;
    }
    let mx = sum(x.iter().map(|v| v - x[0])) / x.len() as f64;
    let my = sum(y.iter().map(|v| v - y[0])) / y.len() as f64;
    let (mut xx, mut yy, mut xy) = (Sum::default(), Sum::default(), Sum::default());
    for (&a, &b) in x.iter().zip(y) {
        let dx = (a - x[0]) - mx;
        let dy = (b - y[0]) - my;
        xx.add(dx * dx);
        yy.add(dy * dy);
        xy.add(dx * dy);
    }
    if xx.value() <= 0. || yy.value() <= 0. {
        f64::NAN
    } else {
        xy.value() / xx.value().sqrt() / yy.value().sqrt()
    }
}

fn select_quantile(v: &mut [f64], q: f64) -> f64 {
    let at = q * (v.len() - 1) as f64;
    let lo = at.floor() as usize;
    let (_, pivot, tail) = v.select_nth_unstable_by(lo, f64::total_cmp);
    let a = *pivot;
    if at == lo as f64 {
        a
    } else {
        let b = tail.iter().copied().min_by(f64::total_cmp).unwrap();
        a + (b - a) * (at - lo as f64)
    }
}

#[derive(Default)]
struct CrossScratch {
    ids: Vec<usize>,
    values: Vec<f64>,
    other: Vec<f64>,
    order: Vec<(u64, usize)>,
}

#[cfg(test)]
fn cell(op: &str, x: &[f64], y: &[f64], idx: &[usize], t: usize, p: Options) -> f64 {
    let c = x.len();
    if c < p.min_periods {
        return f64::NAN;
    }
    match op {
        "min" | "max" | "argmin" | "argmax" => {
            let maximum = op == "max" || op == "argmax";
            let mut best = 0;
            for i in 1..c {
                if (maximum && x[i] >= x[best]) || (!maximum && x[i] <= x[best]) {
                    best = i;
                }
            }
            if op.starts_with("arg") {
                (t - idx[best]) as f64
            } else {
                x[best]
            }
        }
        "rank" => {
            if idx[c - 1] != t {
                return f64::NAN;
            }
            let today = x[c - 1];
            let less = x.iter().filter(|&&v| v < today).count();
            let equal = x.iter().filter(|&&v| v == today).count();
            let r = less as f64 + (equal + 1) as f64 / 2.;
            if p.normalize {
                if c == 1 {
                    0.5
                } else {
                    (r - 1.) / (c - 1) as f64
                }
            } else {
                r
            }
        }
        "quantile" => {
            let mut v = x.to_vec();
            v.sort_unstable_by(f64::total_cmp);
            quantile(&v, p.q)
        }
        "product" => {
            if x.contains(&0.) {
                0.
            } else {
                x.iter().product()
            }
        }
        "decay_linear" => {
            let weights: Vec<f64> = idx
                .iter()
                .map(|&i| p.window as f64 - (t - i) as f64)
                .collect();
            sum(x.iter().zip(&weights).map(|(a, b)| a * b)) / sum(weights.into_iter())
        }
        _ => {
            let anchor = x[0];
            let shifted: Vec<f64> = x.iter().map(|v| v - anchor).collect();
            let x = shifted.as_slice();
            let m = mean(x);
            let variance = sum(x.iter().map(|v| (v - m) * (v - m)));
            if op == "std" {
                return if c > p.ddof {
                    (variance / (c - p.ddof) as f64).sqrt()
                } else {
                    f64::NAN
                };
            }
            if op == "skew" || op == "kurt" {
                let power = if op == "skew" { 3 } else { 4 };
                if c < power || variance <= 0. {
                    return f64::NAN;
                }
                return sum(x.iter().map(|v| (v - m).powi(power as i32)))
                    / c as f64
                    / (variance / c as f64).powf(power as f64 / 2.)
                    - if power == 4 { 3. } else { 0. };
            }
            let my = mean(y);
            let vy = sum(y.iter().map(|v| (v - my) * (v - my)));
            let cross = sum(x.iter().zip(y).map(|(a, b)| (a - m) * (b - my)));
            if op == "cov" {
                return if c > p.ddof {
                    cross / (c - p.ddof) as f64
                } else {
                    f64::NAN
                };
            }
            if c < 2 || vy <= 0. {
                return f64::NAN;
            }
            if op == "corr" {
                return if variance > 0. {
                    cross / variance.sqrt() / vy.sqrt()
                } else {
                    f64::NAN
                };
            }
            let slope = cross / vy;
            if op.ends_with("residual") {
                if idx[c - 1] == t {
                    x[c - 1] - m - slope * (y[c - 1] - my)
                } else {
                    f64::NAN
                }
            } else {
                slope
            }
        }
    }
}

mod time;
pub use time::Op as TimeOp;
pub fn ts(pan: Panel<'_>, op: &str, p: Options, _layout: usize) -> Vec<f64> {
    time::run(pan, TimeOp::parse(op).expect("validated time operator"), p)
}
#[cfg(test)]
fn simple_series(get: impl Fn(usize) -> f64, length: usize, op: &str, p: Options) -> Vec<f64> {
    let x: Vec<f64> = (0..length).map(get).collect();
    ts(
        Panel {
            x: &x,
            y: &[],
            mask: &[],
            t: length,
            n: 1,
        },
        op,
        p,
        0,
    )
}
#[cfg(test)]
fn higher_series(get: impl Fn(usize) -> f64, length: usize, op: &str, p: Options) -> Vec<f64> {
    simple_series(get, length, op, p)
}

pub struct CrossOptions<'a> {
    pub normalize: bool,
    pub ddof: usize,
    pub lower: f64,
    pub upper: f64,
    pub scale: f64,
    pub groups: &'a [i64],
    pub exposures: Vec<&'a [f64]>,
    pub weights: &'a [f64],
    pub rcond: f64,
}

fn residual(x: &[f64], mut columns: Vec<Vec<f64>>, weights: &[f64], rcond: f64) -> Vec<f64> {
    let mut residual: Vec<f64> = x.iter().zip(weights).map(|(v, w)| v * w.sqrt()).collect();
    for col in &mut columns {
        for (v, w) in col.iter_mut().zip(weights) {
            *v *= w.sqrt();
        }
        let norm = sum(col.iter().map(|v| v * v)).sqrt();
        if norm > 0. {
            for v in col {
                *v /= norm;
            }
        }
    }
    // Pivoted, twice-reorthogonalized modified Gram-Schmidt. Dependent columns
    // are removed; singleton groups remain legitimate basis vectors.
    for k in 0..columns.len() {
        let best = (k..columns.len())
            .max_by(|&a, &b| {
                sum(columns[a].iter().map(|v| v * v))
                    .total_cmp(&sum(columns[b].iter().map(|v| v * v)))
            })
            .unwrap();
        columns.swap(k, best);
        let norm = sum(columns[k].iter().map(|v| v * v)).sqrt();
        if norm <= rcond {
            break;
        }
        for v in &mut columns[k] {
            *v /= norm;
        }
        let (head, tail) = columns.split_at_mut(k + 1);
        let basis = &head[k];
        for _ in 0..2 {
            let projection = sum(residual.iter().zip(basis).map(|(a, b)| a * b));
            for (v, b) in residual.iter_mut().zip(basis) {
                *v -= projection * b;
            }
            for col in tail.iter_mut() {
                let projection = sum(col.iter().zip(basis).map(|(a, b)| a * b));
                for (v, b) in col.iter_mut().zip(basis) {
                    *v -= projection * b;
                }
            }
        }
    }
    residual
        .iter()
        .zip(weights)
        .map(|(v, w)| v / w.sqrt())
        .collect()
}

#[derive(Clone, Copy, PartialEq)]
enum CrossOp {
    Rank,
    Zscore,
    Demean,
    Scale,
    WinsorQuantile,
    WinsorMad,
    GroupDemean,
    Neutralize,
}

pub fn cs(pan: Panel<'_>, op: &str, p: &CrossOptions<'_>) -> Vec<f64> {
    let op = match op {
        "rank" => CrossOp::Rank,
        "zscore" => CrossOp::Zscore,
        "demean" => CrossOp::Demean,
        "scale" => CrossOp::Scale,
        "winsor_quantile" => CrossOp::WinsorQuantile,
        "winsor_mad" => CrossOp::WinsorMad,
        "group_demean" => CrossOp::GroupDemean,
        "neutralize" => CrossOp::Neutralize,
        _ => unreachable!(),
    };
    if pan.n == 0 {
        return Vec::new();
    }
    if matches!(op, CrossOp::Zscore | CrossOp::Demean | CrossOp::Scale) {
        let mut out = vec![0.; pan.t * pan.n];
        out.par_chunks_mut(pan.n).enumerate().for_each(|(t, row)| {
            let input = &pan.x[t * pan.n..(t + 1) * pan.n];
            // Masking is the only preprocessing needed for these three reductions.
            let masked: Vec<f64>;
            let v = if pan.mask.is_empty() {
                input
            } else {
                masked = input
                    .iter()
                    .enumerate()
                    .map(|(j, &x)| {
                        if pan.valid(t * pan.n + j) {
                            x
                        } else {
                            f64::NAN
                        }
                    })
                    .collect();
                &masked
            };
            if op == CrossOp::Scale {
                let denominator = positive_sum(v, |v| if v.is_finite() { v.abs() } else { 0. });
                for (dest, &value) in row.iter_mut().zip(v) {
                    *dest = if value.is_finite() && denominator > 0. {
                        value / denominator
                    } else {
                        f64::NAN
                    };
                }
                return;
            }
            let origin = v.iter().copied().find(|x| x.is_finite()).unwrap_or(0.);
            let mut count = 0usize;
            let m = sum(v.iter().map(|&x| {
                if x.is_finite() {
                    count += 1;
                    x - origin
                } else {
                    0.
                }
            })) / count.max(1) as f64;
            let sd = if op == CrossOp::Zscore {
                let variance = positive_sum(v, |value| {
                    if value.is_finite() {
                        let d = (value - origin) - m;
                        d * d
                    } else {
                        0.
                    }
                });
                if count > p.ddof {
                    (variance / (count - p.ddof) as f64).sqrt()
                } else {
                    f64::NAN
                }
            } else {
                1.
            };
            for (dest, &value) in row.iter_mut().zip(v) {
                let centered = (value - origin) - m;
                *dest = if !value.is_finite() {
                    f64::NAN
                } else if op == CrossOp::Demean {
                    centered
                } else if sd > 0. {
                    centered / sd
                } else {
                    f64::NAN
                };
            }
        });
        return out;
    }
    let mut out = vec![f64::NAN; pan.t * pan.n];
    out.par_chunks_mut(pan.n).enumerate().for_each_init(
        CrossScratch::default,
        |scratch, (t, row)| {
            let CrossScratch {
                ids,
                values: v,
                other,
                order,
            } = scratch;
            ids.clear();
            ids.extend((t * pan.n..(t + 1) * pan.n).filter(|&i| {
                pan.x(i).is_finite()
                    && (p.groups.is_empty() || p.groups[i] >= 0)
                    && p.exposures.iter().all(|e| e[i].is_finite())
                    && (p.weights.is_empty() || (p.weights[i].is_finite() && p.weights[i] > 0.))
            }));
            if ids.is_empty() {
                return;
            }
            v.clear();
            v.extend(ids.iter().map(|&i| pan.x[i]));
            if op == CrossOp::Rank {
                rank_in_place(v, order, p.normalize);
                for (&i, &value) in ids.iter().zip(v.iter()) {
                    row[i % pan.n] = value;
                }
                return;
            }
            let z = match op {
                CrossOp::WinsorQuantile | CrossOp::WinsorMad => {
                    other.clear();
                    other.extend_from_slice(v);
                    let (lo, hi) = if op == CrossOp::WinsorQuantile {
                        (
                            select_quantile(other, p.lower),
                            select_quantile(other, p.upper),
                        )
                    } else {
                        let med = select_quantile(other, 0.5);
                        for value in other.iter_mut() {
                            *value = (*value - med).abs();
                        }
                        let mad = select_quantile(other, 0.5);
                        (med - p.scale * mad, med + p.scale * mad)
                    };
                    v.iter().map(|v| v.clamp(lo, hi)).collect()
                }
                CrossOp::GroupDemean => {
                    let mut groups: BTreeMap<i64, (Sum, usize)> = BTreeMap::new();
                    for (&i, &v) in ids.iter().zip(v.iter()) {
                        let e = groups.entry(p.groups[i]).or_default();
                        e.0.add(v);
                        e.1 += 1;
                    }
                    ids.iter()
                        .zip(v.iter())
                        .map(|(&i, &v)| {
                            let e = groups[&p.groups[i]];
                            v - e.0.value() / e.1 as f64
                        })
                        .collect()
                }
                CrossOp::Neutralize => {
                    // Frisch-Waugh: remove the weighted industry intercept first.
                    let mut group_ids = BTreeMap::new();
                    let mut inv = Vec::with_capacity(ids.len());
                    for &i in ids.iter() {
                        let g = if p.groups.is_empty() { 0 } else { p.groups[i] };
                        let next = group_ids.len();
                        let entry = group_ids.entry(g).or_insert(next);
                        inv.push(*entry);
                    }
                    let weights: Vec<f64> = ids
                        .iter()
                        .map(|&i| {
                            if p.weights.is_empty() {
                                1.
                            } else {
                                p.weights[i]
                            }
                        })
                        .collect();
                    let demean = |values: &[f64]| {
                        let mut anchors = vec![f64::NAN; group_ids.len()];
                        let mut totals = vec![Sum::default(); group_ids.len()];
                        let mut denoms = vec![Sum::default(); group_ids.len()];
                        for ((&value, &g), &w) in values.iter().zip(&inv).zip(&weights) {
                            if anchors[g].is_nan() {
                                anchors[g] = value;
                            }
                            totals[g].add((value - anchors[g]) * w);
                            denoms[g].add(w);
                        }
                        values
                            .iter()
                            .zip(&inv)
                            .map(|(&value, &g)| {
                                (value - anchors[g]) - totals[g].value() / denoms[g].value()
                            })
                            .collect::<Vec<f64>>()
                    };
                    let dependent = demean(v);
                    let design = p
                        .exposures
                        .iter()
                        .map(|e| demean(&ids.iter().map(|&i| e[i]).collect::<Vec<_>>()))
                        .collect();
                    residual(&dependent, design, &weights, p.rcond)
                }
                CrossOp::Rank | CrossOp::Zscore | CrossOp::Demean | CrossOp::Scale => {
                    unreachable!()
                }
            };
            for (&i, &v) in ids.iter().zip(&z) {
                row[i % pan.n] = v;
            }
        },
    );
    out
}

pub fn ic(pan: Panel<'_>, rank: bool) -> Vec<f64> {
    (0..pan.t)
        .into_par_iter()
        .map_init(CrossScratch::default, |scratch, t| {
            let CrossScratch {
                values: x,
                other: y,
                order,
                ..
            } = scratch;
            x.clear();
            y.clear();
            for i in t * pan.n..(t + 1) * pan.n {
                if pan.x(i).is_finite() && pan.y(i).is_finite() {
                    x.push(pan.x[i]);
                    y.push(pan.y[i]);
                }
            }
            if rank {
                rank_in_place(x, order, false);
                rank_in_place(y, order, false);
            }
            correlation(x, y)
        })
        .collect()
}

fn labels(pan: Panel<'_>, t: usize, q: usize) -> Vec<usize> {
    let mut ids: Vec<(u64, usize)> = (0..pan.n)
        .filter(|&j| pan.x(t * pan.n + j).is_finite())
        .map(|j| (order_key(pan.x[t * pan.n + j]), j))
        .collect();
    ids.sort_unstable();
    let mut out = vec![usize::MAX; pan.n];
    for (i, &(_, j)) in ids.iter().enumerate() {
        out[j] = i * q / ids.len();
    }
    out
}
pub fn grouped(
    pan: Panel<'_>,
    q: usize,
    weights: &[f64],
    periods: usize,
    turnover: bool,
) -> Vec<f64> {
    if turnover {
        return evaluation::turnover(pan, q, periods);
    }
    let mut out = vec![f64::NAN; pan.t * q];
    out.par_chunks_mut(q).enumerate().for_each(|(t, row)| {
        let now = labels(pan, t, q);
        let mut sums = vec![Sum::default(); q];
        let mut denom = vec![Sum::default(); q];
        for (j, &g) in now.iter().enumerate() {
            if g == usize::MAX {
                continue;
            }
            let i = t * pan.n + j;
            let y = pan.y(i);
            let w = if weights.is_empty() { 1. } else { weights[i] };
            if y.is_finite() && w.is_finite() && w > 0. {
                sums[g].add(y * w);
                denom[g].add(w);
            }
        }
        for g in 0..q {
            if denom[g].value() > 0. {
                let v = sums[g].value() / denom[g].value();
                row[g] = v;
            }
        }
    });
    out
}
pub fn targets(
    pan: Panel<'_>,
    days: &[i64],
    k: usize,
    fraction: f64,
) -> (Vec<i64>, Vec<i64>, Vec<f64>) {
    let selected: Vec<Vec<i64>> = days
        .par_iter()
        .map(|&t| {
            let start = t as usize * pan.n;
            let mut ids: Vec<usize> = (0..pan.n)
                .filter(|&j| pan.x(start + j).is_finite())
                .collect();
            let count = if k > 0 {
                k.min(ids.len())
            } else {
                (ids.len() as f64 * fraction).ceil() as usize
            };
            let compare = |&a: &usize, &b: &usize| {
                pan.x[start + b]
                    .partial_cmp(&pan.x[start + a])
                    .unwrap()
                    .then(a.cmp(&b))
            };
            if count < ids.len() {
                ids.select_nth_unstable_by(count, compare);
                ids.truncate(count);
            }
            ids.sort_unstable();
            ids.into_iter().map(|v| v as i64).collect()
        })
        .collect();
    let mut ptr = vec![0];
    let mut cols = Vec::new();
    let mut weights = Vec::new();
    for row in selected {
        let count = row.len();
        weights.extend(std::iter::repeat_n(1. / count as f64, count));
        cols.extend(row);
        ptr.push(cols.len() as i64);
    }
    (ptr, cols, weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incremental_decay_and_product_match_windows() {
        let values: Vec<f64> = (0..1100)
            .map(|i| {
                if i % 17 == 0 {
                    f64::NAN
                } else {
                    1. + (i as f64 * 0.31).sin() * 0.2
                }
            })
            .collect();
        for op in ["decay_linear", "product"] {
            for window in [1, 5, 20, 60, 250] {
                let p = Options {
                    window,
                    min_periods: 1,
                    ddof: 1,
                    q: 0.5,
                    normalize: false,
                    alpha: 0.5,
                    periods: 1,
                    earliest: false,
                    exact: false,
                };
                let actual = simple_series(|i| values[i], values.len(), op, p);
                for (t, &a) in actual.iter().enumerate() {
                    let start = (t + 1).saturating_sub(window);
                    let ids: Vec<usize> = (start..=t).filter(|&i| values[i].is_finite()).collect();
                    let v: Vec<f64> = ids.iter().map(|&i| values[i]).collect();
                    let expected = cell(op, &v, &[], &ids, t, p);
                    assert!(
                        (a.is_nan() && expected.is_nan())
                            || (a - expected).abs() <= 2e-10 + 2e-10 * expected.abs(),
                        "{op} W={window} t={t}: {a} != {expected}"
                    );
                }
            }
        }
        for values in [[1e308, 1e308, 1e-308, 0.], [1e-308, 1e-308, 1e308, -1.]] {
            let p = Options {
                window: 3,
                min_periods: 1,
                ddof: 1,
                q: 0.5,
                normalize: false,
                alpha: 0.5,
                periods: 1,
                earliest: false,
                exact: false,
            };
            let actual = simple_series(|i| values[i], values.len(), "product", p);
            assert_eq!(actual[2], values[..3].iter().product::<f64>());
        }
    }

    #[test]
    fn guarded_higher_moments_match_centered_windows() {
        for base in [0., 1e10] {
            let values: Vec<f64> = (0..1100)
                .map(|i| {
                    if (400..700).contains(&i) {
                        base + 1.
                    } else {
                        base + i as f64 * 0.001
                    }
                })
                .collect();
            for op in ["skew", "kurt"] {
                for window in [5, 60, 250] {
                    let p = Options {
                        window,
                        min_periods: 4,
                        ddof: 1,
                        q: 0.5,
                        normalize: false,
                        alpha: 0.5,
                        periods: 1,
                        earliest: false,
                        exact: false,
                    };
                    let result = higher_series(|i| values[i], values.len(), op, p);
                    for (t, &actual) in result.iter().enumerate() {
                        let start = (t + 1).saturating_sub(window);
                        let ids: Vec<usize> = (start..=t).collect();
                        let expected = cell(op, &values[start..=t], &[], &ids, t, p);
                        assert!(
                            (actual.is_nan() && expected.is_nan())
                                || (actual - expected).abs() <= 2e-10 + 2e-10 * expected.abs(),
                            "{op} base={base} window={window} t={t} actual={actual} expected={expected}"
                        );
                    }
                }
            }
        }
    }
}
