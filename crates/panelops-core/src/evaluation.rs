//! Evaluation kernels reuse lagged row orderings without materializing lagged panels.
use super::{Panel, correlation};
use rayon::prelude::*;
use std::collections::VecDeque;

fn order(pan: Panel<'_>, t: usize) -> Vec<usize> {
    let mut pairs: Vec<_> = (0..pan.n)
        .filter(|&j| pan.x(t * pan.n + j).is_finite())
        .map(|j| (super::order_key(pan.x[t * pan.n + j]), j))
        .collect();
    pairs.sort_unstable();
    pairs.into_iter().map(|(_, j)| j).collect()
}
fn labels(pan: Panel<'_>, t: usize, q: usize) -> Vec<usize> {
    let ids = order(pan, t);
    let mut labels = vec![usize::MAX; pan.n];
    for (i, &j) in ids.iter().enumerate() {
        labels[j] = i * q / ids.len();
    }
    labels
}
fn rank_subset(pan: Panel<'_>, t: usize, order: &[usize], common: &[bool], out: &mut [f64]) {
    let mut start = 0;
    let mut preceding = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && pan.x[t * pan.n + order[start]] == pan.x[t * pan.n + order[end]]
        {
            end += 1;
        }
        let count = order[start..end].iter().filter(|&&j| common[j]).count();
        let rank = preceding as f64 + (count + 1) as f64 / 2.;
        for &j in &order[start..end] {
            if common[j] {
                out[j] = rank;
            }
        }
        preceding += count;
        start = end;
    }
}

pub fn autocorr(pan: Panel<'_>, periods: usize, rank: bool) -> Vec<f64> {
    let mut out = vec![f64::NAN; pan.t];
    if pan.t == 0 || pan.n == 0 || periods >= pan.t {
        return out;
    }
    let chunk = pan.t.div_ceil(rayon::current_num_threads());
    out.par_chunks_mut(chunk)
        .enumerate()
        .for_each(|(block, dest)| {
            let start = block * chunk;
            let cached = rank && periods > 0 && periods <= 32.min(chunk);
            let mut orders = VecDeque::new();
            if cached {
                for t in start.saturating_sub(periods)..start {
                    orders.push_back(order(pan, t));
                }
            }
            let mut common = vec![false; pan.n];
            let (mut rx, mut ry) = (vec![0.; pan.n], vec![0.; pan.n]);
            let (mut x, mut y) = (Vec::with_capacity(pan.n), Vec::with_capacity(pan.n));
            for (offset, z) in dest.iter_mut().enumerate() {
                let t = start + offset;
                let now = if rank { order(pan, t) } else { Vec::new() };
                if t >= periods {
                    let old = t - periods;
                    for (j, c) in common.iter_mut().enumerate() {
                        *c = pan.x(t * pan.n + j).is_finite() && pan.x(old * pan.n + j).is_finite();
                    }
                    if rank {
                        let previous;
                        let order_old = if periods == 0 {
                            &now
                        } else if cached {
                            orders.front().unwrap()
                        } else {
                            previous = order(pan, old);
                            &previous
                        };
                        rank_subset(pan, t, &now, &common, &mut rx);
                        rank_subset(pan, old, order_old, &common, &mut ry);
                    }
                    x.clear();
                    y.clear();
                    for (j, &c) in common.iter().enumerate() {
                        if c {
                            x.push(if rank { rx[j] } else { pan.x[t * pan.n + j] });
                            y.push(if rank { ry[j] } else { pan.x[old * pan.n + j] });
                        }
                    }
                    *z = correlation(&x, &y);
                }
                if cached {
                    if orders.len() == periods {
                        orders.pop_front();
                    }
                    orders.push_back(now);
                }
            }
        });
    out
}

pub fn turnover(pan: Panel<'_>, q: usize, periods: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; pan.t * q];
    if pan.t == 0 || pan.n == 0 || periods >= pan.t {
        return out;
    }
    let chunk = pan.t.div_ceil(rayon::current_num_threads());
    out.par_chunks_mut(chunk * q)
        .enumerate()
        .for_each(|(block, rows)| {
            let start = block * chunk;
            let cached = periods > 0 && periods <= 32.min(chunk);
            let mut past = VecDeque::new();
            if cached {
                for t in start.saturating_sub(periods)..start {
                    past.push_back(labels(pan, t, q));
                }
            }
            let (mut count, mut retained) = (vec![0usize; q], vec![0usize; q]);
            for (offset, row) in rows.chunks_mut(q).enumerate() {
                let t = start + offset;
                let now = labels(pan, t, q);
                if t >= periods {
                    let previous;
                    let old = if periods == 0 {
                        &now
                    } else if cached {
                        past.front().unwrap()
                    } else {
                        previous = labels(pan, t - periods, q);
                        &previous
                    };
                    count.fill(0);
                    retained.fill(0);
                    for (j, &g) in now.iter().enumerate() {
                        if g != usize::MAX {
                            count[g] += 1;
                            retained[g] += usize::from(old[j] == g);
                        }
                    }
                    for (g, z) in row.iter_mut().enumerate() {
                        if count[g] > 0 {
                            *z = 1. - retained[g] as f64 / count[g] as f64;
                        }
                    }
                }
                if cached {
                    if past.len() == periods {
                        past.pop_front();
                    }
                    past.push_back(now);
                }
            }
        });
    out
}
