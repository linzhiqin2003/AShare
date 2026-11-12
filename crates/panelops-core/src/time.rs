//! Row-streaming kernels. Output column blocks are disjoint safe borrows.
use super::{Options, Panel, Sum};
use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
#[repr(usize)]
pub enum Op {
    Delay = 0,
    Delta = 1,
    Sum = 2,
    Mean = 3,
    Std = 4,
    Min = 5,
    Max = 6,
    Rank = 7,
    Quantile = 8,
    Argmin = 9,
    Argmax = 10,
    Product = 11,
    Corr = 12,
    Cov = 13,
    DecayLinear = 14,
    Ema = 15,
    Skew = 16,
    Kurt = 17,
    Slope = 18,
    Residual = 19,
    RegressionSlope = 20,
    RegressionResidual = 21,
    Var = 22,
    Zscore = 23,
    Rsquare = 24,
    Median = 25,
    Mad = 26,
    Count = 27,
    Sumif = 28,
    Ffill = 29,
    Idxmin = 30,
    Idxmax = 31,
}
impl Op {
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "delay" => Self::Delay,
            "delta" => Self::Delta,
            "sum" => Self::Sum,
            "mean" => Self::Mean,
            "std" => Self::Std,
            "min" => Self::Min,
            "max" => Self::Max,
            "rank" => Self::Rank,
            "quantile" => Self::Quantile,
            "argmin" => Self::Argmin,
            "argmax" => Self::Argmax,
            "product" => Self::Product,
            "corr" => Self::Corr,
            "cov" => Self::Cov,
            "decay_linear" => Self::DecayLinear,
            "ema" => Self::Ema,
            "skew" => Self::Skew,
            "kurt" => Self::Kurt,
            "slope" => Self::Slope,
            "residual" => Self::Residual,
            "regression_slope" => Self::RegressionSlope,
            "regression_residual" => Self::RegressionResidual,
            "var" => Self::Var,
            "zscore" => Self::Zscore,
            "rsquare" => Self::Rsquare,
            "median" => Self::Median,
            "mad" => Self::Mad,
            "count" => Self::Count,
            "sumif" => Self::Sumif,
            "ffill" => Self::Ffill,
            "idxmin" => Self::Idxmin,
            "idxmax" => Self::Idxmax,
            _ => return None,
        })
    }
}
const DELAY: usize = 0;
const DELTA: usize = 1;
const SUM: usize = 2;
const MEAN: usize = 3;
const STD: usize = 4;
const MIN: usize = 5;
const MAX: usize = 6;
const RANK: usize = 7;
const QUANTILE: usize = 8;
const ARGMIN: usize = 9;
const ARGMAX: usize = 10;
const PRODUCT: usize = 11;
const CORR: usize = 12;
const COV: usize = 13;
const DECAY_LINEAR: usize = 14;
const EMA: usize = 15;
const SKEW: usize = 16;
const KURT: usize = 17;
const SLOPE: usize = 18;
const RESIDUAL: usize = 19;
const REGRESSION_SLOPE: usize = 20;
const REGRESSION_RESIDUAL: usize = 21;
const VAR: usize = 22;
const ZSCORE: usize = 23;
const RSQUARE: usize = 24;
const MEDIAN: usize = 25;
const MAD: usize = 26;
const COUNT: usize = 27;
const SUMIF: usize = 28;
const FFILL: usize = 29;
const IDXMIN: usize = 30;
const IDXMAX: usize = 31;

pub fn run(pan: Panel<'_>, op: Op, p: Options) -> Vec<f64> {
    match op {
        Op::Delay => dispatch::<DELAY>(pan, p),
        Op::Delta => dispatch::<DELTA>(pan, p),
        Op::Sum => dispatch::<SUM>(pan, p),
        Op::Mean => dispatch::<MEAN>(pan, p),
        Op::Std => dispatch::<STD>(pan, p),
        Op::Min => dispatch::<MIN>(pan, p),
        Op::Max => dispatch::<MAX>(pan, p),
        Op::Rank => dispatch::<RANK>(pan, p),
        Op::Quantile => dispatch::<QUANTILE>(pan, p),
        Op::Argmin => dispatch::<ARGMIN>(pan, p),
        Op::Argmax => dispatch::<ARGMAX>(pan, p),
        Op::Product if p.window == 1 => dispatch::<DELAY>(pan, Options { periods: 0, ..p }),
        Op::Product => dispatch::<PRODUCT>(pan, p),
        Op::Corr => dispatch::<CORR>(pan, p),
        Op::Cov => dispatch::<COV>(pan, p),
        Op::DecayLinear => dispatch::<DECAY_LINEAR>(pan, p),
        Op::Ema => dispatch::<EMA>(pan, p),
        Op::Skew => dispatch::<SKEW>(pan, p),
        Op::Kurt => dispatch::<KURT>(pan, p),
        Op::Slope => dispatch::<SLOPE>(pan, p),
        Op::Residual => dispatch::<RESIDUAL>(pan, p),
        Op::RegressionSlope => dispatch::<REGRESSION_SLOPE>(pan, p),
        Op::RegressionResidual => dispatch::<REGRESSION_RESIDUAL>(pan, p),
        Op::Var => dispatch::<VAR>(pan, p),
        Op::Zscore => dispatch::<ZSCORE>(pan, p),
        Op::Rsquare => dispatch::<RSQUARE>(pan, p),
        Op::Median => dispatch::<MEDIAN>(pan, p),
        Op::Mad => dispatch::<MAD>(pan, p),
        Op::Count => dispatch::<COUNT>(pan, p),
        Op::Sumif => dispatch::<SUMIF>(pan, p),
        Op::Ffill => dispatch::<FFILL>(pan, p),
        Op::Idxmin => dispatch::<IDXMIN>(pan, p),
        Op::Idxmax => dispatch::<IDXMAX>(pan, p),
    }
}

fn dispatch<const O: usize>(pan: Panel<'_>, p: Options) -> Vec<f64> {
    let mut out = vec![0.; pan.t * pan.n];
    if out.is_empty() {
        return out;
    }
    if O == DELAY || O == DELTA {
        let lag = p.periods.saturating_mul(pan.n);
        if lag >= out.len() {
            out.fill(f64::NAN);
            return out;
        }
        out[..lag].fill(f64::NAN);
        out[lag..]
            .par_chunks_mut(pan.n)
            .enumerate()
            .for_each(|(t, row)| {
                let old = &pan.x[t * pan.n..(t + 1) * pan.n];
                let now = &pan.x[lag + t * pan.n..lag + (t + 1) * pan.n];
                if pan.mask.is_empty() {
                    if O == DELAY {
                        for (dest, &old) in row.iter_mut().zip(old) {
                            *dest = if old.is_finite() { old } else { f64::NAN };
                        }
                    } else {
                        for ((dest, &old), &now) in row.iter_mut().zip(old).zip(now) {
                            *dest = if old.is_finite() && now.is_finite() {
                                now - old
                            } else {
                                f64::NAN
                            };
                        }
                    }
                } else {
                    for (j, v) in row.iter_mut().enumerate() {
                        *v = f64::NAN;
                        let i = t * pan.n + j;
                        if pan.valid(i) && pan.valid(i + lag) && old[j].is_finite() {
                            *v = if O == DELAY {
                                old[j]
                            } else if now[j].is_finite() {
                                now[j] - old[j]
                            } else {
                                f64::NAN
                            };
                        }
                    }
                }
            });
        return out;
    }
    // A block owns one slice per output row, never a second output panel.
    // Its width affects scheduling only; each column keeps the same arithmetic.
    let workers = rayon::current_num_threads().min(pan.n);
    let width = (pan.n.div_ceil(workers).div_ceil(64) * 64).min(1024);
    let mut blocks: Vec<Vec<&mut [f64]>> = (0..pan.n.div_ceil(width))
        .map(|_| Vec::with_capacity(pan.t))
        .collect();
    for row in out.chunks_mut(pan.n) {
        for (b, slice) in row.chunks_mut(width).enumerate() {
            blocks[b].push(slice);
        }
    }
    blocks.into_par_iter().enumerate().for_each(|(b, rows)| {
        let first = b * width;
        // The fast path changes validity branches only, never arithmetic order.
        let dense = pan.mask.is_empty()
            && (0..pan.t).all(|t| {
                pan.x[t * pan.n + first..t * pan.n + first + rows[0].len()]
                    .iter()
                    .all(|v| v.is_finite())
                    && (pan.y.is_empty()
                        || pan.y[t * pan.n + first..t * pan.n + first + rows[0].len()]
                            .iter()
                            .all(|v| v.is_finite()))
            });
        if dense {
            block::<O, true>(pan, p, first, rows);
        } else {
            block::<O, false>(pan, p, first, rows);
        }
    });
    out
}

#[inline(always)]
fn values<const O: usize, const D: bool>(pan: Panel<'_>, t: usize, j: usize) -> (f64, f64, bool) {
    let i = t * pan.n + j;
    let x = pan.x[i];
    let paired =
        O == CORR || O == COV || O == REGRESSION_SLOPE || O == REGRESSION_RESIDUAL || O == SUMIF;
    let y = if paired {
        pan.y[i]
    } else if O == SLOPE || O == RESIDUAL || O == RSQUARE {
        t as f64
    } else {
        0.
    };
    (
        x,
        y,
        D || (pan.valid(i) && x.is_finite() && (!paired || y.is_finite())),
    )
}

fn block<const O: usize, const D: bool>(
    pan: Panel<'_>,
    p: Options,
    first: usize,
    rows: Vec<&mut [f64]>,
) {
    if O == SUM || O == MEAN || O == COUNT || O == SUMIF {
        sum_rows::<O, D>(pan, p, first, rows);
        return;
    }
    if O == STD || O == VAR || O == ZSCORE {
        std_rows::<O, D>(pan, p, first, rows);
        return;
    }
    if O == DECAY_LINEAR {
        sums::<O, D>(pan, p, first, rows);
        return;
    }
    if O == EMA || O == FFILL {
        let n = rows[0].len();
        let mut state = vec![f64::NAN; n];
        let mut counts = vec![0usize; n];
        let mut last = vec![0; n];
        for (t, row) in rows.into_iter().enumerate() {
            for (k, dest) in row.iter_mut().enumerate() {
                *dest = f64::NAN;
                let (x, _, ok) = values::<O, D>(pan, t, first + k);
                if ok {
                    state[k] = if O == FFILL || counts[k] == 0 {
                        x
                    } else {
                        p.alpha * x + (1. - p.alpha) * state[k]
                    };
                    counts[k] += 1;
                    last[k] = t;
                }
                if (D || pan.valid(t * pan.n + first + k))
                    && counts[k] >= if O == EMA { p.min_periods } else { 1 }
                    && (O != FFILL || p.periods == 0 || t - last[k] <= p.periods)
                {
                    *dest = state[k];
                }
            }
        }
        return;
    }
    if O == MIN || O == MAX || O == ARGMIN || O == ARGMAX || O == IDXMIN || O == IDXMAX {
        extrema::<O, D>(pan, p, first, rows);
        return;
    }
    if O == RANK || O == QUANTILE || O == MEDIAN || O == MAD || O == PRODUCT {
        ordered::<O, D>(pan, p, first, rows);
        return;
    }
    moments::<O, D>(pan, p, first, rows);
}

fn sums<const O: usize, const D: bool>(
    pan: Panel<'_>,
    p: Options,
    first: usize,
    rows: Vec<&mut [f64]>,
) {
    let n = rows[0].len();
    let mut total = vec![Sum::default(); n];
    let mut weighted = vec![Sum::default(); n];
    let mut count = vec![0usize; n];
    let mut denom = vec![0.; n];
    let mut peak = vec![0f64; n];
    let mut weighted_error = vec![0f64; n];
    for (t, row) in rows.into_iter().enumerate() {
        let start = (t + 1).saturating_sub(p.window);
        for (k, dest) in row.iter_mut().enumerate() {
            *dest = f64::NAN;
            let j = first + k;
            if O == DECAY_LINEAR {
                weighted[k].add(-total[k].main);
                weighted[k].add(-total[k].correction);
                denom[k] -= count[k] as f64;
            }
            if t >= p.window {
                let (x, y, ok) = values::<O, D>(pan, t - p.window, j);
                if ok {
                    count[k] -= 1;
                    total[k].add(-if O == COUNT {
                        f64::from(x != 0.)
                    } else if O == SUMIF && y == 0. {
                        0.
                    } else {
                        x
                    });
                }
            }
            let (x, y, ok) = values::<O, D>(pan, t, j);
            if ok {
                count[k] += 1;
                let v = if O == COUNT {
                    f64::from(x != 0.)
                } else if O == SUMIF && y == 0. {
                    0.
                } else {
                    x
                };
                total[k].add(v);
                if O == DECAY_LINEAR {
                    weighted[k].add(x * p.window as f64);
                    weighted_error[k] += 8. * f64::EPSILON * (x * p.window as f64).abs();
                    denom[k] += p.window as f64;
                }
            }
            peak[k] = peak[k].max(total[k].main.abs());
            // Double-compensation can retain errors after a large observation exits.
            weighted_error[k] += 8. * f64::EPSILON.powi(2) * peak[k] * p.window as f64;
            let event = peak[k] * f64::EPSILON * 8. > total[k].value().abs() + 1e-300
                || (O == DECAY_LINEAR
                    && weighted_error[k] > 1e-11 * (weighted[k].value().abs() + denom[k]));
            if t % 128 == 0
                || event
                || !total[k].value().is_finite()
                || (O == DECAY_LINEAR && !weighted[k].value().is_finite())
            {
                total[k] = Sum::default();
                weighted[k] = Sum::default();
                count[k] = 0;
                denom[k] = 0.;
                peak[k] = 0.;
                weighted_error[k] = 0.;
                let mut magnitude = 0f64;
                let mut weighted_magnitude = 0f64;
                for r in start..=t {
                    let (x, y, ok) = values::<O, D>(pan, r, j);
                    if ok {
                        count[k] += 1;
                        let v = if O == COUNT {
                            f64::from(x != 0.)
                        } else if O == SUMIF && y == 0. {
                            0.
                        } else {
                            x
                        };
                        total[k].add(v);
                        magnitude = magnitude.max(v.abs());
                        if O == DECAY_LINEAR {
                            let w = (p.window - (t - r)) as f64;
                            weighted[k].add(x * w);
                            weighted_magnitude = weighted_magnitude.max((x * w).abs());
                            denom[k] += w;
                        }
                    }
                }
                if 8. * f64::EPSILON * magnitude > 1e-11 * (total[k].value().abs() + 1.) {
                    total[k] = Sum {
                        main: accurate_sum((start..=t).filter_map(|r| {
                            let (x, _, ok) = values::<O, D>(pan, r, j);
                            ok.then_some(x)
                        })),
                        correction: 0.,
                    };
                }
                if 8. * f64::EPSILON * weighted_magnitude
                    > 1e-11 * (weighted[k].value().abs() + denom[k])
                {
                    weighted[k] = Sum {
                        main: accurate_sum((start..=t).filter_map(|r| {
                            let (x, _, ok) = values::<O, D>(pan, r, j);
                            ok.then_some(x * (p.window - (t - r)) as f64)
                        })),
                        correction: 0.,
                    };
                }
                peak[k] = total[k].main.abs().max(magnitude);
            }
            if count[k] >= p.min_periods && (D || pan.valid(t * pan.n + j)) {
                *dest = if O == MEAN {
                    total[k].value() / count[k] as f64
                } else if O == DECAY_LINEAR {
                    weighted[k].value() / denom[k]
                } else {
                    total[k].value()
                };
            }
        }
    }
}

#[derive(Clone)]
struct Moments<const M: usize> {
    sums: [Sum; M],
    error: [f64; M],
    ox: f64,
    oy: f64,
    count: usize,
    last_x: f64,
    last_y: f64,
    run_x: usize,
    run_y: usize,
}
impl<const M: usize> Default for Moments<M> {
    fn default() -> Self {
        Self {
            sums: [Sum::default(); M],
            error: [0.; M],
            ox: 0.,
            oy: 0.,
            count: 0,
            last_x: f64::NAN,
            last_y: f64::NAN,
            run_x: 0,
            run_y: 0,
        }
    }
}
impl<const M: usize> Moments<M> {
    #[inline(always)]
    fn update<const O: usize>(&mut self, x: f64, y: f64, positive: bool) {
        if positive {
            if self.count == 0 {
                self.ox = x;
                self.oy = y;
                self.sums = [Sum::default(); M];
                self.error = [0.; M];
            }
            self.count += 1;
            self.run_x = if x == self.last_x { self.run_x + 1 } else { 1 };
            self.last_x = x;
            self.run_y = if y == self.last_y { self.run_y + 1 } else { 1 };
            self.last_y = y;
        } else {
            self.count -= 1;
        }
        let dx = x - self.ox;
        let dy = y - self.oy;
        let sign = if positive { 1. } else { -1. };
        let z = if M == 2 {
            [dx, dx * dx, 0., 0., 0.]
        } else if O == SKEW || O == KURT {
            [dx, dx * dx, dx * dx * dx, (dx * dx) * (dx * dx), 0.]
        } else {
            [dx, dy, dx * dx, dy * dy, dx * dy]
        };
        for (i, &z) in z.iter().take(M).enumerate() {
            self.sums[i].add(sign * z);
            self.error[i] += 8. * f64::EPSILON * z.abs();
        }
    }
    fn constant_x(&self) -> bool {
        self.count > 0 && self.run_x >= self.count
    }
    fn constant_y(&self) -> bool {
        self.count > 0 && self.run_y >= self.count
    }
}

fn rebuild<const O: usize, const D: bool, const M: usize>(
    s: &mut Moments<M>,
    pan: Panel<'_>,
    p: Options,
    t: usize,
    j: usize,
) {
    *s = Moments::default();
    for r in (t + 1).saturating_sub(p.window)..=t {
        let (x, y, ok) = values::<O, D>(pan, r, j);
        if ok {
            s.update::<O>(x, y, true);
        }
    }
}

fn moments<const O: usize, const D: bool>(
    pan: Panel<'_>,
    p: Options,
    first: usize,
    rows: Vec<&mut [f64]>,
) {
    if O == STD || O == VAR || O == ZSCORE {
        moment_block::<O, D, 2>(pan, p, first, rows);
    } else if O == SKEW || O == KURT {
        moment_block::<O, D, 4>(pan, p, first, rows);
    } else {
        moment_block::<O, D, 5>(pan, p, first, rows);
    }
}
fn moment_block<const O: usize, const D: bool, const M: usize>(
    pan: Panel<'_>,
    p: Options,
    first: usize,
    rows: Vec<&mut [f64]>,
) {
    let mut states = vec![Moments::<M>::default(); rows[0].len()];
    for (t, row) in rows.into_iter().enumerate() {
        for (k, dest) in row.iter_mut().enumerate() {
            *dest = f64::NAN;
            let j = first + k;
            let s = &mut states[k];
            if t >= p.window {
                let (x, y, ok) = values::<O, D>(pan, t - p.window, j);
                if ok {
                    s.update::<O>(x, y, false);
                }
            }
            let (x, y, ok) = values::<O, D>(pan, t, j);
            if ok {
                s.update::<O>(x, y, true);
            }
            // The run counter proves exact constancy without rescanning a window.
            let constant = if M == 5 {
                s.constant_y() || s.constant_x()
            } else {
                s.constant_x()
            };
            if constant && s.count >= p.min_periods {
                let z = constant_result::<O, M>(s, p, ok);
                if D || pan.valid(t * pan.n + j) {
                    *dest = z;
                }
                continue;
            }
            let (mut z, risk) = moment_result::<O, M>(s, p, x, y, ok);
            if t % 128 == 0 || risk {
                rebuild::<O, D, M>(s, pan, p, t, j);
                let (rebuilt, remaining_risk) = moment_result::<O, M>(s, p, x, y, ok);
                z = rebuilt;
                if s.constant_x() || (M == 5 && s.constant_y()) {
                    z = constant_result::<O, M>(s, p, ok);
                } else if (remaining_risk || !z.is_finite()) && s.count >= p.min_periods {
                    z = direct::<O, D>(pan, p, t, j);
                }
            }
            if D || pan.valid(t * pan.n + j) {
                *dest = z;
            }
        }
    }
}
fn constant_result<const O: usize, const M: usize>(s: &Moments<M>, p: Options, today: bool) -> f64 {
    if s.count < p.min_periods {
        return f64::NAN;
    }
    if O == STD || O == VAR || O == COV {
        return if s.count > p.ddof { 0. } else { f64::NAN };
    }
    if M == 5 && !s.constant_y() && s.count >= 2 && O != CORR && O != RSQUARE {
        return if (O == RESIDUAL || O == REGRESSION_RESIDUAL) && !today {
            f64::NAN
        } else {
            0.
        };
    }
    f64::NAN
}
fn moment_result<const O: usize, const M: usize>(
    s: &Moments<M>,
    p: Options,
    x: f64,
    y: f64,
    today: bool,
) -> (f64, bool) {
    if s.count < p.min_periods || s.count == 0 {
        return (f64::NAN, false);
    }
    if s.count == 2 && !s.constant_y() && (O == RESIDUAL || O == REGRESSION_RESIDUAL) {
        return (if today { 0. } else { f64::NAN }, false);
    }
    let c = s.count as f64;
    let a = s.sums.map(Sum::value);
    let vx = if M == 5 {
        a[2] - a[0] * a[0] / c
    } else {
        a[1] - a[0] * a[0] / c
    };
    let ex = if M == 5 { s.error[2] } else { s.error[1] }
        + 2. * (a[0] / c).abs() * s.error[0]
        + 8. * f64::EPSILON * a[0] * a[0] / c;
    let mut risk = !vx.is_finite() || vx <= 0. || ex > 1e-11 * vx.abs();
    let z = if M == 2 {
        let variance = if s.count > p.ddof {
            vx / (s.count - p.ddof) as f64
        } else {
            f64::NAN
        };
        if O == VAR {
            variance
        } else if O == ZSCORE {
            if today {
                (x - s.ox - a[0] / c) / variance.sqrt()
            } else {
                f64::NAN
            }
        } else {
            variance.sqrt()
        }
    } else if M == 4 {
        let m = a[0] / c;
        let v = vx / c;
        let m3 = a[2] / c - 3. * m * a[1] / c + 2. * m * m * m;
        let m4 = a[3] / c - 4. * m * a[2] / c + 6. * m * m * a[1] / c - 3. * m * m * m * m;
        let denominator = if O == SKEW { v * v.sqrt() } else { v * v };
        let magnitude = if O == SKEW {
            a[2].abs() / c + 3. * (m * a[1]).abs() / c + 2. * m.abs().powi(3)
        } else {
            a[3].abs() / c
                + 4. * (m * a[2]).abs() / c
                + 6. * m * m * a[1].abs() / c
                + 3. * m.powi(4)
        };
        let err = if O == SKEW {
            s.error[2] + 3. * m.abs() * s.error[1] + (3. * a[1].abs() / c + 6. * m * m) * s.error[0]
        } else {
            s.error[3]
                + 4. * m.abs() * s.error[2]
                + 6. * m * m * s.error[1]
                + (4. * a[2].abs() / c + 12. * m.abs() * a[1].abs() / c + 12. * m.abs().powi(3))
                    * s.error[0]
        };
        risk |= err / c + 32. * f64::EPSILON * magnitude > 1e-11 * denominator.abs();
        if s.count < if O == SKEW { 3 } else { 4 } {
            f64::NAN
        } else if O == SKEW {
            m3 / denominator
        } else {
            m4 / denominator - 3.
        }
    } else {
        let vy = a[3] - a[1] * a[1] / c;
        let cross = a[4] - a[0] * a[1] / c;
        let ey =
            s.error[3] + 2. * (a[1] / c).abs() * s.error[1] + 8. * f64::EPSILON * a[1] * a[1] / c;
        let ec = s.error[4]
            + (a[1] * s.error[0]).abs() / c
            + (a[0] * s.error[1]).abs() / c
            + 8. * f64::EPSILON * (a[0] * a[1] / c).abs();
        risk |=
            !vy.is_finite() || vy <= 0. || ey > 1e-11 * vy.abs() || ec > 1e-11 * (cross.abs() + 1.);
        if O == COV {
            if s.count > p.ddof {
                cross / (s.count - p.ddof) as f64
            } else {
                f64::NAN
            }
        } else if s.count < 2 || vy <= 0. {
            f64::NAN
        } else if O == CORR || O == RSQUARE {
            let r = cross / vx.sqrt() / vy.sqrt();
            if O == RSQUARE { r * r } else { r }
        } else {
            let slope = cross / vy;
            if O == RESIDUAL || O == REGRESSION_RESIDUAL {
                if today {
                    x - s.ox - a[0] / c - slope * (y - s.oy - a[1] / c)
                } else {
                    f64::NAN
                }
            } else {
                slope
            }
        }
    };
    if O == RESIDUAL || O == REGRESSION_RESIDUAL {
        let magnitude = (x - s.ox).abs() + (a[0] / c).abs() + z.abs();
        risk |= 16. * f64::EPSILON * magnitude > 1e-11 * (z.abs() + 1.);
    }
    (z, risk)
}

// Allocation-free shifted two-pass fallback for extreme centered powers.
fn direct<const O: usize, const D: bool>(pan: Panel<'_>, p: Options, t: usize, j: usize) -> f64 {
    let start = (t + 1).saturating_sub(p.window);
    let mut s = Moments::<5>::default();
    for r in start..=t {
        let (x, y, ok) = values::<O, D>(pan, r, j);
        if ok {
            s.update::<COV>(x, y, true);
        }
    }
    if s.count < p.min_periods {
        return f64::NAN;
    }
    let mx = s.sums[0].value() / s.count as f64;
    let my = s.sums[1].value() / s.count as f64;
    let mut powers = [Sum::default(); 5];
    for r in start..=t {
        let (x, y, ok) = values::<O, D>(pan, r, j);
        if ok {
            let dx = (x - s.ox) - mx;
            let dy = (y - s.oy) - my;
            for (sum, v) in
                powers
                    .iter_mut()
                    .zip([dx * dx, dy * dy, dx * dy, dx.powi(3), dx.powi(4)])
            {
                sum.add(v);
            }
        }
    }
    let [vx, vy, cross, m3, m4] = powers.map(Sum::value);
    let c = s.count as f64;
    let (x, y, today) = values::<O, D>(pan, t, j);
    if O == SKEW {
        if s.count >= 3 && vx > 0. {
            (m3 / c) / (vx / c).powf(1.5)
        } else {
            f64::NAN
        }
    } else if O == KURT {
        if s.count >= 4 && vx > 0. {
            (m4 / c) / (vx / c).powi(2) - 3.
        } else {
            f64::NAN
        }
    } else if O == STD || O == VAR || O == ZSCORE {
        let v = if s.count > p.ddof {
            vx / (s.count - p.ddof) as f64
        } else {
            f64::NAN
        };
        if O == VAR {
            v
        } else if O == STD {
            v.sqrt()
        } else if today && v > 0. {
            ((x - s.ox) - mx) / v.sqrt()
        } else {
            f64::NAN
        }
    } else if O == COV {
        if s.count > p.ddof {
            cross / (s.count - p.ddof) as f64
        } else {
            f64::NAN
        }
    } else if O == CORR || O == RSQUARE {
        if vx > 0. && vy > 0. {
            let r = cross / vx.sqrt() / vy.sqrt();
            if O == RSQUARE { r * r } else { r }
        } else {
            f64::NAN
        }
    } else if vy > 0. {
        let slope = cross / vy;
        if O == RESIDUAL || O == REGRESSION_RESIDUAL {
            if today {
                (x - s.ox) - mx - slope * ((y - s.oy) - my)
            } else {
                f64::NAN
            }
        } else {
            slope
        }
    } else {
        f64::NAN
    }
}

fn extrema<const O: usize, const D: bool>(
    pan: Panel<'_>,
    p: Options,
    first: usize,
    rows: Vec<&mut [f64]>,
) {
    let n = rows[0].len();
    let w = p.window.min(pan.t);
    // A suffix for the preceding block plus the current prefix is the window.
    // Only W rows of state are stored; there is no T-by-N intermediate.
    let mut suffix = vec![0.; w * n];
    let mut indices = vec![usize::MAX; w * n];
    let mut counts = vec![0usize; w * n];
    let mut prefix = vec![0.; n];
    let mut prefix_ids = vec![usize::MAX; n];
    let mut prefix_counts = vec![0usize; n];
    let maximum = O == MAX || O == ARGMAX || O == IDXMAX;
    for (t, row) in rows.into_iter().enumerate() {
        let offset = t % p.window;
        if offset == 0 {
            prefix_ids.fill(usize::MAX);
            prefix_counts.fill(0);
            if t >= p.window {
                for r in (0..w).rev() {
                    for k in 0..n {
                        let (x, _, ok) = values::<O, D>(pan, t - p.window + r, first + k);
                        let i = r * n + k;
                        if r + 1 < w {
                            suffix[i] = suffix[i + n];
                            indices[i] = indices[i + n];
                            counts[i] = counts[i + n];
                        } else {
                            indices[i] = usize::MAX;
                            counts[i] = 0;
                        }
                        if ok {
                            counts[i] += 1;
                            // Keep the newer suffix position on a tie.
                            if indices[i] == usize::MAX
                                || (if maximum {
                                    x > suffix[i]
                                } else {
                                    x < suffix[i]
                                })
                            {
                                suffix[i] = x;
                                indices[i] = t - p.window + r;
                            }
                        }
                    }
                }
            }
        }
        for (k, dest) in row.iter_mut().enumerate() {
            *dest = f64::NAN;
            let (x, _, ok) = values::<O, D>(pan, t, first + k);
            if ok {
                prefix_counts[k] += 1;
                if prefix_ids[k] == usize::MAX
                    || (if maximum {
                        x >= prefix[k]
                    } else {
                        x <= prefix[k]
                    })
                {
                    prefix[k] = x;
                    prefix_ids[k] = t;
                }
            }
            let (mut count, mut best, mut index) = (prefix_counts[k], prefix[k], prefix_ids[k]);
            if t >= p.window && offset + 1 < p.window {
                let i = (offset + 1) * n + k;
                count += counts[i];
                if indices[i] != usize::MAX
                    && (index == usize::MAX
                        || (if maximum {
                            suffix[i] > best
                        } else {
                            suffix[i] < best
                        }))
                {
                    best = suffix[i];
                    index = indices[i];
                }
            }
            if count >= p.min_periods && (D || pan.valid(t * pan.n + first + k)) {
                *dest = if O == ARGMIN || O == ARGMAX {
                    (t - index) as f64
                } else if O == IDXMIN || O == IDXMAX {
                    (p.window - (t - index)) as f64
                } else {
                    best
                };
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Node {
    value: f64,
    row: usize,
    priority: u64,
    left: usize,
    right: usize,
    size: usize,
}
const NONE: usize = usize::MAX;
#[derive(Clone)]
struct OrderTree {
    nodes: Vec<Node>,
    root: usize,
}
impl OrderTree {
    fn new(window: usize) -> Self {
        Self {
            nodes: vec![
                Node {
                    value: 0.,
                    row: 0,
                    priority: 0,
                    left: NONE,
                    right: NONE,
                    size: 0
                };
                window
            ],
            root: NONE,
        }
    }
    fn size(&self, r: usize) -> usize {
        if r == NONE { 0 } else { self.nodes[r].size }
    }
    fn update(&mut self, r: usize) {
        self.nodes[r].size = 1 + self.size(self.nodes[r].left) + self.size(self.nodes[r].right);
    }
    fn less(&self, a: usize, b: usize) -> bool {
        let x = self.nodes[a];
        let y = self.nodes[b];
        x.value < y.value || (x.value == y.value && x.row < y.row)
    }
    fn merge(&mut self, a: usize, b: usize) -> usize {
        if a == NONE {
            return b;
        }
        if b == NONE {
            return a;
        }
        if self.nodes[a].priority > self.nodes[b].priority {
            let r = self.merge(self.nodes[a].right, b);
            self.nodes[a].right = r;
            self.update(a);
            a
        } else {
            let l = self.merge(a, self.nodes[b].left);
            self.nodes[b].left = l;
            self.update(b);
            b
        }
    }
    fn remove(&mut self, r: usize, old: usize) -> usize {
        if r == old {
            return self.merge(self.nodes[r].left, self.nodes[r].right);
        }
        if self.less(old, r) {
            let l = self.remove(self.nodes[r].left, old);
            self.nodes[r].left = l;
        } else {
            let right = self.remove(self.nodes[r].right, old);
            self.nodes[r].right = right;
        }
        self.update(r);
        r
    }
    fn insert(&mut self, r: usize, new: usize) -> usize {
        if r == NONE {
            return new;
        }
        if self.less(new, r) {
            let l = self.insert(self.nodes[r].left, new);
            self.nodes[r].left = l;
            if self.nodes[l].priority > self.nodes[r].priority {
                self.nodes[r].left = self.nodes[l].right;
                self.nodes[l].right = r;
                self.update(r);
                self.update(l);
                return l;
            }
        } else {
            let right = self.insert(self.nodes[r].right, new);
            self.nodes[r].right = right;
            if self.nodes[right].priority > self.nodes[r].priority {
                self.nodes[r].right = self.nodes[right].left;
                self.nodes[right].left = r;
                self.update(r);
                self.update(right);
                return right;
            }
        }
        self.update(r);
        r
    }
    fn advance(&mut self, t: usize, value: f64) {
        let i = t % self.nodes.len();
        if t >= self.nodes.len() && self.nodes[i].size != 0 {
            self.root = self.remove(self.root, i);
        }
        self.nodes[i].size = 0;
        if value.is_finite() {
            // Deterministic SplitMix priorities keep tree shape thread-independent.
            let mut z = (t as u64).wrapping_add(0x9e3779b97f4a7c15);
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            self.nodes[i] = Node {
                value,
                row: t,
                priority: z ^ (z >> 31),
                left: NONE,
                right: NONE,
                size: 1,
            };
            self.root = self.insert(self.root, i);
        }
    }
    fn at(&self, mut k: usize) -> f64 {
        let mut r = self.root;
        loop {
            let n = self.size(self.nodes[r].left);
            if k == n {
                return self.nodes[r].value;
            }
            if k < n {
                r = self.nodes[r].left;
            } else {
                k -= n + 1;
                r = self.nodes[r].right;
            }
        }
    }
    fn quantile(&self, q: f64) -> f64 {
        let at = q * (self.size(self.root) - 1) as f64;
        let lo = at.floor() as usize;
        let hi = at.ceil() as usize;
        let a = self.at(lo);
        a + (self.at(hi) - a) * (at - lo as f64)
    }
}

fn ordered<const O: usize, const D: bool>(
    pan: Panel<'_>,
    p: Options,
    first: usize,
    rows: Vec<&mut [f64]>,
) {
    let n = rows[0].len();
    let capacity = p.window.min(pan.t);
    let tree = (O == QUANTILE || O == MEDIAN) && p.window > 64;
    let mut trees = if tree {
        vec![OrderTree::new(capacity); n]
    } else {
        Vec::new()
    };
    let mut ring = if tree {
        Vec::new()
    } else {
        vec![f64::NAN; capacity * n]
    };
    let mut counts = vec![0usize; n];
    let mut scratch = Vec::with_capacity(capacity);
    let mut sorted = if O == QUANTILE || O == MEDIAN {
        vec![vec![f64::INFINITY; capacity]; n]
    } else {
        Vec::new()
    };
    for (t, row) in rows.into_iter().enumerate() {
        for (k, dest) in row.iter_mut().enumerate() {
            *dest = f64::NAN;
            let j = first + k;
            let (x, _, ok) = values::<O, D>(pan, t, j);
            let value = if ok { x } else { f64::NAN };
            if tree {
                let tr = &mut trees[k];
                tr.advance(t, value);
                if tr.size(tr.root) >= p.min_periods && (D || pan.valid(t * pan.n + j)) {
                    *dest = tr.quantile(if O == MEDIAN { 0.5 } else { p.q });
                }
                continue;
            }
            let values = &mut ring[k * capacity..(k + 1) * capacity];
            let slot = t % capacity;
            let old = values[slot];
            values[slot] = value;
            counts[k] += usize::from(ok);
            counts[k] -= usize::from(old.is_finite());
            if O == QUANTILE || O == MEDIAN {
                let order = &mut sorted[k];
                let old = if old.is_finite() { old } else { f64::INFINITY };
                let value = if ok { value } else { f64::INFINITY };
                let at = order.iter().filter(|&&v| v < old).count();
                let to = order.iter().filter(|&&v| v < value).count();
                if old < value {
                    for i in at..to - 1 {
                        order[i] = order[i + 1];
                    }
                    order[to - 1] = value;
                } else if old > value {
                    for i in (to..at).rev() {
                        order[i + 1] = order[i];
                    }
                    order[to] = value;
                }
            }
            if counts[k] < p.min_periods || (!D && !pan.valid(t * pan.n + j)) {
                continue;
            }
            *dest = if O == RANK {
                if !ok {
                    continue;
                }
                let less = values.iter().filter(|&&v| v < x).count();
                let equal = values.iter().filter(|&&v| v == x).count();
                let rank = less as f64 + (equal + 1) as f64 / 2.;
                if p.normalize {
                    if counts[k] == 1 {
                        0.5
                    } else {
                        (rank - 1.) / (counts[k] - 1) as f64
                    }
                } else {
                    rank
                }
            } else if O == PRODUCT {
                if values.contains(&0.) {
                    0.
                } else {
                    values[slot + 1..]
                        .iter()
                        .chain(&values[..=slot])
                        .filter(|v| v.is_finite())
                        .product()
                }
            } else if O == QUANTILE || O == MEDIAN {
                super::quantile(&sorted[k][..counts[k]], if O == MEDIAN { 0.5 } else { p.q })
            } else {
                scratch.clear();
                scratch.extend(values.iter().copied().filter(|v| v.is_finite()));
                let median = select(&mut scratch, 0.5);
                for v in &mut scratch {
                    *v = (*v - median).abs();
                }
                select(&mut scratch, 0.5)
            };
        }
    }
}
fn select(v: &mut [f64], q: f64) -> f64 {
    let pos = q * (v.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let (_, pivot, tail) = v.select_nth_unstable_by(lo, f64::total_cmp);
    let a = *pivot;
    if lo == hi {
        a
    } else {
        let b = tail.iter().copied().min_by(f64::total_cmp).unwrap();
        a + (b - a) * (pos - lo as f64)
    }
}

// A short floating expansion preserves small terms across multiple cancellations.
// This rare-path sum visits the original window in increasing time order.
fn accurate_sum(values: impl Iterator<Item = f64>) -> f64 {
    let mut partials: Vec<f64> = Vec::new();
    for mut x in values {
        let mut length = 0;
        for index in 0..partials.len() {
            let mut y = partials[index];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let high = x + y;
            if !high.is_finite() {
                return high;
            }
            let low = y - (high - x);
            if low != 0. {
                partials[length] = low;
                length += 1;
            }
            x = high;
        }
        partials.truncate(length);
        partials.push(x);
    }
    partials.into_iter().rev().sum()
}
#[inline(always)]
fn add(main: &mut f64, correction: &mut f64, value: f64) {
    let adjusted = value + *correction;
    let next = *main + adjusted;
    *correction = (*main - next) + adjusted;
    *main = next;
}

fn sum_rows<const O: usize, const D: bool>(
    pan: Panel<'_>,
    p: Options,
    first: usize,
    rows: Vec<&mut [f64]>,
) {
    let n = rows[0].len();
    let (mut main, mut correction, mut peak) = (vec![0.; n], vec![0.; n], vec![0f64; n]);
    let mut counts = vec![0usize; n];
    for (t, row) in rows.into_iter().enumerate() {
        if t % 512 == 0 {
            main.fill(0.);
            correction.fill(0.);
            counts.fill(0);
            peak.fill(0.);
            for r in (t + 1).saturating_sub(p.window)..=t {
                let input = &pan.x[r * pan.n + first..r * pan.n + first + n];
                for k in 0..n {
                    let x = input[k];
                    let y = if O == SUMIF {
                        pan.y[r * pan.n + first + k]
                    } else {
                        0.
                    };
                    let ok = D
                        || (x.is_finite()
                            && (O != SUMIF || y.is_finite())
                            && pan.valid(r * pan.n + first + k));
                    let a = if ok {
                        if O == COUNT {
                            f64::from(x != 0.)
                        } else if O == SUMIF && y == 0. {
                            0.
                        } else {
                            x
                        }
                    } else {
                        0.
                    };
                    add(&mut main[k], &mut correction[k], a);
                    peak[k] = peak[k].max(a.abs());
                    counts[k] += usize::from(ok);
                }
            }
        } else {
            if t >= p.window {
                let input =
                    &pan.x[(t - p.window) * pan.n + first..(t - p.window) * pan.n + first + n];
                for k in 0..n {
                    let x = input[k];
                    let y = if O == SUMIF {
                        pan.y[(t - p.window) * pan.n + first + k]
                    } else {
                        0.
                    };
                    let ok = D
                        || (x.is_finite()
                            && (O != SUMIF || y.is_finite())
                            && pan.valid((t - p.window) * pan.n + first + k));
                    let a = if ok {
                        if O == COUNT {
                            f64::from(x != 0.)
                        } else if O == SUMIF && y == 0. {
                            0.
                        } else {
                            x
                        }
                    } else {
                        0.
                    };
                    add(&mut main[k], &mut correction[k], -a);
                    counts[k] -= usize::from(ok);
                }
            }
            let input = &pan.x[t * pan.n + first..t * pan.n + first + n];
            for k in 0..n {
                let x = input[k];
                let y = if O == SUMIF {
                    pan.y[t * pan.n + first + k]
                } else {
                    0.
                };
                let ok = D
                    || (x.is_finite()
                        && (O != SUMIF || y.is_finite())
                        && pan.valid(t * pan.n + first + k));
                let a = if ok {
                    if O == COUNT {
                        f64::from(x != 0.)
                    } else if O == SUMIF && y == 0. {
                        0.
                    } else {
                        x
                    }
                } else {
                    0.
                };
                add(&mut main[k], &mut correction[k], a);
                counts[k] += usize::from(ok);
            }
        }
        for (k, dest) in row.iter_mut().enumerate() {
            *dest = f64::NAN;
            let mut value = main[k] + correction[k];
            peak[k] = peak[k].max(main[k].abs());
            if !value.is_finite() || 8. * f64::EPSILON * peak[k] > 1e-11 * (value.abs() + 1.) {
                let mut contributions = Vec::with_capacity(p.window.min(t + 1));
                counts[k] = 0;
                peak[k] = 0.;
                for r in (t + 1).saturating_sub(p.window)..=t {
                    let x = pan.x[r * pan.n + first + k];
                    let y = if O == SUMIF {
                        pan.y[r * pan.n + first + k]
                    } else {
                        0.
                    };
                    let ok = D
                        || (x.is_finite()
                            && (O != SUMIF || y.is_finite())
                            && pan.valid(r * pan.n + first + k));
                    if ok {
                        counts[k] += 1;
                        contributions.push(if O == COUNT {
                            f64::from(x != 0.)
                        } else if O == SUMIF && y == 0. {
                            0.
                        } else {
                            x
                        });
                    }
                }
                peak[k] = contributions.iter().fold(0f64, |m, x| m.max(x.abs()));
                value = accurate_sum(contributions.into_iter());
                main[k] = value;
                correction[k] = 0.;
            }
            if counts[k] >= p.min_periods && (D || pan.valid(t * pan.n + first + k)) {
                *dest = if O == MEAN {
                    value / counts[k] as f64
                } else {
                    value
                };
            }
        }
    }
}

fn std_rows<const O: usize, const D: bool>(
    pan: Panel<'_>,
    p: Options,
    first: usize,
    rows: Vec<&mut [f64]>,
) {
    let n = rows[0].len();
    let (mut s1, mut s2, mut peak, mut origin) =
        (vec![0.; n], vec![0.; n], vec![0f64; n], vec![0.; n]);
    let (mut count, mut run, mut last) = (vec![0usize; n], vec![0usize; n], vec![f64::NAN; n]);
    let mut last_reset = vec![0usize; n];
    for (t, row) in rows.into_iter().enumerate() {
        if t >= p.window {
            let input = &pan.x[(t - p.window) * pan.n + first..(t - p.window) * pan.n + first + n];
            if pan.mask.is_empty() {
                for (((((&x, &o), s), ss), c), p) in input
                    .iter()
                    .zip(&origin)
                    .zip(&mut s1)
                    .zip(&mut s2)
                    .zip(&mut count)
                    .zip(&mut peak)
                {
                    let ok = D || x.is_finite();
                    let d = if ok { x - o } else { 0. };
                    *s -= d;
                    *ss -= d * d;
                    *c -= usize::from(ok);
                    *p = p.max(d.abs());
                }
            } else {
                for k in 0..n {
                    let x = input[k];
                    let ok = D || (x.is_finite() && pan.valid((t - p.window) * pan.n + first + k));
                    let d = if ok { x - origin[k] } else { 0. };
                    let d2 = d * d;
                    s1[k] -= d;
                    s2[k] -= d2;
                    count[k] -= usize::from(ok);
                    peak[k] = peak[k].max(d.abs());
                }
            }
        }
        let input = &pan.x[t * pan.n + first..t * pan.n + first + n];
        if pan.mask.is_empty() {
            for (((((((&x, &o), s), ss), c), p), run), last) in input
                .iter()
                .zip(&origin)
                .zip(&mut s1)
                .zip(&mut s2)
                .zip(&mut count)
                .zip(&mut peak)
                .zip(&mut run)
                .zip(&mut last)
            {
                let ok = D || x.is_finite();
                let d = if ok { x - o } else { 0. };
                *s += d;
                *ss += d * d;
                *c += usize::from(ok);
                *p = p.max(d.abs());
                if ok {
                    *run = if *last == x { *run + 1 } else { 1 };
                    *last = x;
                }
            }
        } else {
            for k in 0..n {
                let x = input[k];
                let ok = D || (x.is_finite() && pan.valid(t * pan.n + first + k));
                let d = if ok { x - origin[k] } else { 0. };
                let d2 = d * d;
                s1[k] += d;
                s2[k] += d2;
                count[k] += usize::from(ok);
                peak[k] = peak[k].max(d.abs());
                if ok {
                    run[k] = if last[k] == x { run[k] + 1 } else { 1 };
                    last[k] = x;
                }
            }
        }
        for (k, dest) in row.iter_mut().enumerate() {
            *dest = f64::NAN;
            let mut c = count[k];
            let j = first + k;
            if c < p.min_periods || c <= p.ddof {
                continue;
            }
            if run[k] >= c {
                last_reset[k] = t;
                origin[k] = last[k];
                s1[k] = 0.;
                s2[k] = 0.;
                peak[k] = 0.;
                if D || pan.valid(t * pan.n + j) {
                    *dest = if O == ZSCORE { f64::NAN } else { 0. };
                }
                continue;
            }
            let mut mean = s1[k] / c as f64;
            let mut variance = s2[k] - s1[k] * mean;
            let error = 8.
                * f64::EPSILON
                * (2. * (t - last_reset[k]) as f64 + p.window as f64)
                * (peak[k] * peak[k] + 2. * mean.abs() * peak[k])
                + 8. * f64::EPSILON * (s1[k] * mean).abs();
            // Include rounding accumulated by ordinary updates, not just cancellation.
            let risk = error > 1e-11 * variance.abs() || variance <= 0. || !variance.is_finite();
            if t % 512 == 0 || risk {
                last_reset[k] = t;
                let mut anchor = f64::NAN;
                let mut sum1 = Sum::default();
                let mut sum2 = Sum::default();
                c = 0;
                peak[k] = 0.;
                for r in (t + 1).saturating_sub(p.window)..=t {
                    let (x, _, ok) = values::<O, D>(pan, r, j);
                    if ok {
                        if anchor.is_nan() {
                            anchor = x;
                        }
                        let d = x - anchor;
                        sum1.add(d);
                        sum2.add(d * d);
                        peak[k] = peak[k].max(d.abs());
                        c += 1;
                    }
                }
                origin[k] = anchor;
                s1[k] = sum1.value();
                s2[k] = sum2.value();

                mean = s1[k] / c as f64;
                variance = s2[k] - s1[k] * mean;
            }
            if D || pan.valid(t * pan.n + j) {
                let v = variance / (c - p.ddof) as f64;
                *dest = if O == VAR {
                    v
                } else if O == STD {
                    v.sqrt()
                } else {
                    let (x, _, ok) = values::<O, D>(pan, t, j);
                    if ok {
                        (x - origin[k] - mean) / v.sqrt()
                    } else {
                        f64::NAN
                    }
                };
            }
        }
    }
}
