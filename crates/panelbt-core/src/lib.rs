//! 与 Python 无关的逐日组合模拟器；只遍历持仓和目标的活动列。
#![forbid(unsafe_code)]

use rayon::prelude::*;

pub const BUILD_MARKER: &str = "panelbt-core-v1";

pub struct Market<'a> {
    pub t: usize,
    pub n: usize,
    pub price: &'a [f64],
    pub close: &'a [f64],
    pub buy: &'a [bool],
    pub sell: &'a [bool],
    pub volume: Option<&'a [f64]>,
    pub split: Option<&'a [f64]>,
    pub dividend: Option<&'a [f64]>,
    pub delist: Option<&'a [bool]>,
}

pub struct Targets<'a> {
    pub days: &'a [i64],
    pub indptr: &'a [i64],
    pub cols: &'a [i64],
    pub weights: &'a [f64],
}

pub struct Settings<'a> {
    pub commission: f64,
    pub min_commission: f64,
    pub transfer: f64,
    pub slippage: f64,
    pub initial: f64,
    pub participation: f64,
    pub recovery: f64,
    pub buffer: f64,
    pub tax_sell: &'a [f64],
    pub tax_buy: &'a [f64],
    pub min_qty: &'a [f64],
    pub step: &'a [f64],
    pub carry: bool,
    pub record: bool,
}

pub type Trade = (usize, usize, f64, f64, f64, f64, f64, f64);
pub type DelistEvent = (usize, usize, f64, f64, f64);

pub struct Output {
    pub daily: Vec<f64>,
    pub positions: Option<Vec<f64>>,
    pub trades: Option<Vec<Trade>>,
    pub events: Vec<DelistEvent>,
    pub counts: [u64; 9],
}

fn lot(x: f64, minimum: f64, step: f64) -> f64 {
    if step == 0.0 {
        x.max(0.0)
    } else if x < minimum {
        0.0
    } else {
        minimum + ((x - minimum) / step).floor() * step
    }
}

fn fees(amount: f64, rate: f64, s: &Settings<'_>) -> (f64, f64, f64) {
    if amount <= 0.0 {
        (0.0, 0.0, 0.0)
    } else {
        (
            (amount * s.commission).max(s.min_commission),
            amount * rate,
            amount * s.transfer,
        )
    }
}

fn needed(q: f64, price: f64, rate: f64, s: &Settings<'_>) -> f64 {
    let amount = q * price;
    let (c, t, f) = fees(amount, rate, s);
    amount + c + t + f
}

fn affordable(cash: f64, price: f64, rate: f64, minimum: f64, step: f64, s: &Settings<'_>) -> f64 {
    let limit = (cash / (price * (1.0 + rate + s.transfer + s.commission)))
        .min((cash - s.min_commission) / (price * (1.0 + rate + s.transfer)))
        .max(0.0);
    let mut q = lot(limit, minimum, step);
    // 浮点乘加在边界可能向上舍入一个 ulp，继续减量而不是把现金截断为零。
    while q > 0.0 && needed(q, price, rate, s) > cash {
        q = if step == 0.0 {
            q.next_down()
        } else {
            lot(q - step, minimum, step)
        };
    }
    q
}

struct Fill<'a> {
    day: usize,
    col: usize,
    qty: f64,
    price: f64,
    carry: bool,
    market: &'a Market<'a>,
    settings: &'a Settings<'a>,
}

fn execute(fill: Fill<'_>, cash: &mut f64, shares: &mut [f64], out: &mut Output) {
    let Fill {
        day,
        col,
        qty,
        price,
        carry,
        market: m,
        settings: s,
    } = fill;
    let amount = qty.abs() * price;
    let rate = if qty > 0.0 {
        s.tax_buy[day]
    } else {
        s.tax_sell[day]
    };
    let (commission, tax, transfer) = fees(amount, rate, s);
    let slip = qty.abs() * m.price[day * m.n + col] * s.slippage;
    if qty > 0.0 {
        *cash -= amount + commission + tax + transfer;
    } else {
        *cash += amount - commission - tax - transfer;
    }
    shares[col] += qty;
    out.daily[2 * m.t + day] += amount;
    out.daily[3 * m.t + day] += commission;
    out.daily[4 * m.t + day] += tax;
    out.daily[5 * m.t + day] += transfer;
    out.daily[6 * m.t + day] += slip;
    out.counts[0] += 1;
    if commission > amount * s.commission {
        out.counts[4] += 1;
    }
    if carry {
        out.counts[8] += 1;
    }
    if let Some(trades) = out.trades.as_mut() {
        trades.push((day, col, qty, price, commission, tax, transfer, slip));
    }
}

fn past_close(m: &Market<'_>, day: usize, col: usize) -> f64 {
    for row in (0..day).rev() {
        let p = m.close[row * m.n + col];
        if p.is_finite() {
            return p;
        }
    }
    0.0
}

fn cap(m: &Market<'_>, s: &Settings<'_>, offset: usize, step: f64) -> f64 {
    if s.participation > 0.0
        && let Some(volume) = m.volume
    {
        let v = volume[offset];
        if !v.is_finite() || v <= 0.0 {
            return 0.0;
        }
        let x = v * s.participation;
        return if step == 0.0 {
            x
        } else {
            (x / step).floor() * step
        };
    }
    f64::INFINITY
}

pub fn run(m: &Market<'_>, targets: &Targets<'_>, s: &Settings<'_>) -> Output {
    let mut out = Output {
        daily: vec![0.0; 9 * m.t],
        positions: s.record.then(|| vec![0.0; m.t * m.n]),
        trades: s.record.then(Vec::new),
        events: Vec::new(),
        counts: [0; 9],
    };
    let mut cash = s.initial;
    let mut shares = vec![0.0; m.n];
    let mut last = vec![0.0; m.n];
    let mut dead = vec![false; m.n];
    let mut held: Vec<usize> = Vec::new();
    let mut pending: Vec<(usize, f64)> = Vec::new();
    let mut commands: Vec<(usize, f64)> = Vec::new();
    let mut buys: Vec<(usize, f64)> = Vec::new();
    let mut weights = vec![0.0; m.n];
    let mut cursor = 0;
    for day in 0..m.t {
        let base = day * m.n;
        // 公司行为使用除权前股数；退市只用此前已知收盘价。
        for &i in &held {
            if let Some(dividend) = m.dividend {
                let div = dividend[base + i];
                cash += shares[i] * div;
                if div > 0.0 {
                    out.counts[6] += 1;
                }
            }
            if let Some(split) = m.split {
                let factor = split[base + i];
                shares[i] *= factor;
                if factor != 1.0 {
                    out.counts[7] += 1;
                }
            }
            if m.delist.is_some_and(|d| d[base + i]) {
                let price = last[i] * s.recovery;
                let proceeds = shares[i] * price;
                cash += proceeds;
                out.events.push((day, i, shares[i], price, proceeds));
                out.counts[5] += 1;
                shares[i] = 0.0;
                dead[i] = true;
            }
        }
        held.retain(|&i| shares[i] > 0.0);
        let rebalance = cursor < targets.days.len() && targets.days[cursor] as usize == day;
        commands.clear();
        if rebalance {
            pending.clear();
            for &i in &held {
                weights[i] = 0.0;
                commands.push((i, 0.0));
            }
            let start = targets.indptr[cursor] as usize;
            let end = targets.indptr[cursor + 1] as usize;
            for j in start..end {
                let i = targets.cols[j] as usize;
                weights[i] = targets.weights[j];
                commands.push((i, weights[i]));
            }
            commands.sort_unstable_by_key(|x| x.0);
            commands.dedup_by_key(|x| x.0);
            for item in &mut commands {
                item.1 = weights[item.0];
            }
            cursor += 1;
        } else if s.carry {
            std::mem::swap(&mut commands, &mut pending);
            pending.clear();
        }
        let value = cash
            + held
                .iter()
                .map(|&i| {
                    let p = m.price[base + i];
                    shares[i] * if p.is_finite() { p } else { last[i] }
                })
                .sum::<f64>();
        buys.clear();
        for &(i, weight) in &commands {
            if shares[i] == 0.0 {
                last[i] = past_close(m, day, i);
                if let Some(d) = m.delist {
                    dead[i] |= (0..=day).any(|row| d[row * m.n + i]);
                }
            }
            if dead[i] {
                continue;
            }
            let p = m.price[base + i];
            let valuation = if p.is_finite() { p } else { last[i] };
            let step = s.step[i];
            let minimum = s.min_qty[i];
            let target = if valuation > 0.0 {
                lot(weight * value * (1.0 - s.buffer) / valuation, minimum, step)
            } else {
                0.0
            };
            let capacity = cap(m, s, base + i, step);
            let mut blocked = false;
            if shares[i] > target {
                let wanted = if target == 0.0 || step == 0.0 {
                    shares[i] - target
                } else {
                    ((shares[i] - target) / step).floor() * step
                };
                let mut q = wanted.min(capacity);
                if !p.is_finite() || !m.sell[base + i] {
                    q = 0.0;
                }
                if wanted > q {
                    blocked = true;
                    out.daily[8 * m.t + day] += 1.0;
                    out.counts[2] += 1;
                }
                if q > 0.0 {
                    // 卖出不能让极端最低佣金把现金打成负数；不满足者视为现金不足。
                    let price = p * (1.0 - s.slippage);
                    let amount = q * price;
                    let (c, tax, transfer) = fees(amount, s.tax_sell[day], s);
                    if cash + amount >= c + tax + transfer {
                        execute(
                            Fill {
                                day,
                                col: i,
                                qty: -q,
                                price,
                                carry: !rebalance,
                                market: m,
                                settings: s,
                            },
                            &mut cash,
                            &mut shares,
                            &mut out,
                        );
                    }
                }
            } else {
                let wanted = lot(target - shares[i], minimum, step);
                let mut q = lot(wanted.min(capacity), minimum, step);
                if !p.is_finite() || !m.buy[base + i] {
                    q = 0.0;
                }
                if wanted > q || (valuation == 0.0 && weight > 0.0) {
                    blocked = true;
                    out.daily[7 * m.t + day] += 1.0;
                    out.counts[1] += 1;
                }
                if q > 0.0 {
                    buys.push((i, q));
                }
            }
            if blocked && s.carry {
                pending.push((i, weight));
            }
        }
        let total: f64 = buys
            .iter()
            .map(|&(i, q)| needed(q, m.price[base + i] * (1.0 + s.slippage), s.tax_buy[day], s))
            .sum();
        let scale = if total > cash {
            out.counts[3] += 1;
            cash / total
        } else {
            1.0
        };
        // commands 按列号排列，buys 保留顺序；卖出已全部完成。
        for &(i, desired) in &buys {
            let price = m.price[base + i] * (1.0 + s.slippage);
            let mut q = if scale < 1.0 {
                lot(desired * scale, s.min_qty[i], s.step[i])
            } else {
                desired
            };
            if needed(q, price, s.tax_buy[day], s) > cash {
                q = q.min(affordable(
                    cash,
                    price,
                    s.tax_buy[day],
                    s.min_qty[i],
                    s.step[i],
                    s,
                ));
            }
            if q > 0.0 {
                execute(
                    Fill {
                        day,
                        col: i,
                        qty: q,
                        price,
                        carry: !rebalance,
                        market: m,
                        settings: s,
                    },
                    &mut cash,
                    &mut shares,
                    &mut out,
                );
                held.push(i);
            }
        }
        held.sort_unstable();
        held.dedup();
        held.retain(|&i| shares[i] > 0.0);
        for &i in &held {
            let close = m.close[base + i];
            if close.is_finite() {
                last[i] = close;
            }
        }
        out.daily[day] =
            (cash + held.iter().map(|&i| shares[i] * last[i]).sum::<f64>()) / s.initial;
        out.daily[m.t + day] = cash;
        out.daily[2 * m.t + day] = if value > 0.0 {
            out.daily[2 * m.t + day] * 0.5 / value
        } else {
            0.0
        };
        if let Some(positions) = out.positions.as_mut() {
            positions[base..base + m.n].copy_from_slice(&shares);
        }
    }
    out
}

pub fn run_batch(
    m: &Market<'_>,
    targets: &[Targets<'_>],
    s: &Settings<'_>,
    threads: usize,
) -> Result<Vec<Output>, rayon::ThreadPoolBuildError> {
    if threads == 1 || targets.len() <= 1 {
        return Ok(targets.iter().map(|target| run(m, target, s)).collect());
    }
    // threads=0 明确使用全部可用核，不受 RAYON_NUM_THREADS 环境覆盖。
    let workers = if threads == 0 {
        std::thread::available_parallelism().map_or(1, usize::from)
    } else {
        threads
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()?;
    Ok(pool.install(|| targets.par_iter().map(|target| run(m, target, s)).collect()))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lots() {
        assert_eq!(lot(199.0, 200.0, 1.0), 0.0);
        assert_eq!(lot(201.9, 200.0, 1.0), 201.0);
        assert_eq!(lot(75.5, 100.0, 0.0), 75.5);
        assert_eq!(lot(299.9, 100.0, 100.0), 200.0);
    }
}
