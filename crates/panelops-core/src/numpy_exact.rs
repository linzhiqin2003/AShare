//! NumPy-order row kernels; deliberately distinct from compensated operators.

/// NumPy's contiguous float64 add reduction (eight lanes, 128-element leaves).
/// The leading positive zero is the ufunc reduction identity.
pub fn sum(x: &[f64]) -> f64 {
    fn pair(x: &[f64]) -> f64 {
        let n = x.len();
        if n < 8 {
            return x.iter().fold(-0.0, |a, b| a + b);
        }
        if n <= 128 {
            let mut r: [f64; 8] = x[..8].try_into().unwrap();
            let end = n - n % 8;
            for block in x[8..end].chunks_exact(8) {
                for j in 0..8 {
                    r[j] += block[j];
                }
            }
            let mut value = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
            for v in &x[end..] {
                value += v;
            }
            value
        } else {
            let half = (n / 2) / 8 * 8;
            pair(&x[..half]) + pair(&x[half..])
        }
    }
    0.0 + pair(x)
}

/// Reusable scratch for exact, single-row NumPy-compatible operations.
#[derive(Default)]
pub struct RowOps {
    temp: Vec<f64>,
    x: Vec<f64>,
    y: Vec<f64>,
    ids: Vec<usize>,
    order: Vec<(u64, usize)>,
    totals: Vec<f64>,
    counts: Vec<usize>,
    touched: Vec<usize>,
}

impl RowOps {
    pub fn demean(&mut self, row: &mut [f64]) {
        self.temp.clear();
        let mut count = 0;
        for &v in row.iter() {
            self.temp.push(if v.is_nan() {
                0.0
            } else {
                count += 1;
                v
            });
        }
        let mean = if count > 0 {
            sum(&self.temp) / count as f64
        } else {
            f64::NAN
        };
        for v in row {
            if !v.is_nan() {
                *v -= mean;
            }
        }
    }

    /// Sequential bincount accumulation, preserving column order within groups.
    pub fn group_demean(&mut self, row: &mut [f64], groups: &[i64]) {
        let max = groups.iter().copied().max().unwrap_or(-1);
        if max < 0 {
            row.fill(f64::NAN);
            return;
        }
        let base = groups
            .iter()
            .copied()
            .filter(|&g| g >= 0)
            .min()
            .unwrap_or(0);
        if max - base > 2_000_000 {
            // Large arbitrary labels must not cause an allocation proportional
            // to the label value. This branch keeps each group's addition order.
            let mut groupsums = std::collections::BTreeMap::<i64, (f64, usize)>::new();
            for (&v, &g) in row.iter().zip(groups) {
                if !v.is_nan() && g >= 0 {
                    let e = groupsums.entry(g).or_default();
                    e.0 += v;
                    e.1 += 1;
                }
            }
            for (v, &g) in row.iter_mut().zip(groups) {
                *v = if !v.is_nan() && g >= 0 {
                    let e = groupsums[&g];
                    *v - e.0 / e.1 as f64
                } else {
                    f64::NAN
                };
            }
            return;
        }
        let width = (max - base + 1) as usize;
        if self.totals.len() < width {
            self.totals.resize(width, 0.0);
            self.counts.resize(width, 0);
        }
        for (&v, &g) in row.iter().zip(groups) {
            if !v.is_nan() && g >= 0 {
                let g = (g - base) as usize;
                if self.counts[g] == 0 {
                    self.touched.push(g);
                }
                self.totals[g] += v;
                self.counts[g] += 1;
            }
        }
        for (v, &g) in row.iter_mut().zip(groups) {
            *v = if !v.is_nan() && g >= 0 {
                *v - self.totals[(g - base) as usize] / self.counts[(g - base) as usize] as f64
            } else {
                f64::NAN
            };
        }
        for g in self.touched.drain(..) {
            self.totals[g] = 0.0;
            self.counts[g] = 0;
        }
    }

    pub fn residual(
        &mut self,
        row: &mut [f64],
        vector: &[f64],
        dot: &impl Fn(&[f64], &[f64]) -> f64,
    ) {
        self.x.clear();
        self.y.clear();
        self.ids.clear();
        for (j, (&x, &y)) in row.iter().zip(vector).enumerate() {
            if x.is_finite() && y.is_finite() {
                self.x.push(x);
                self.y.push(y);
                self.ids.push(j);
            }
        }
        row.fill(f64::NAN);
        if self.x.len() < 2 {
            return;
        }
        let mx = sum(&self.x) / self.x.len() as f64;
        let my = sum(&self.y) / self.y.len() as f64;
        for (x, y) in self.x.iter_mut().zip(&mut self.y) {
            *x -= mx;
            *y -= my;
        }
        let den = dot(&self.y, &self.y);
        let beta = if den != 0.0 {
            dot(&self.x, &self.y) / den
        } else {
            0.0
        };
        for ((&j, &x), &y) in self.ids.iter().zip(&self.x).zip(&self.y) {
            let r = x - beta * y;
            row[j] = if r.is_finite() { r } else { f64::NAN };
        }
    }

    /// Stable ordinal grouping; false means too few distinct values.
    pub fn groups(
        &mut self,
        row: &[f64],
        groups: &mut [i64],
        n: usize,
        descending: bool,
        check_distinct: bool,
    ) -> bool {
        self.order.clear();
        groups.fill(-1);
        self.order.extend(
            row.iter()
                .enumerate()
                .filter(|(_, v)| !v.is_nan())
                .map(|(j, &v)| (super::order_key(if descending { -v } else { v }), j)),
        );
        self.order.sort_unstable(); // Secondary column key gives stable ties.
        let count = self.order.len();
        if count == 0 {
            return true;
        }
        if check_distinct
            && (count < n || 1 + self.order.windows(2).filter(|w| w[0].0 != w[1].0).count() < n)
        {
            return false;
        }
        for (rank, &(_, col)) in self.order.iter().enumerate() {
            groups[col] = (rank * n / count) as i64;
        }
        true
    }

    pub fn normalize(&mut self, row: &mut [f64]) {
        self.demean(row);
        self.temp.clear();
        let mut count = 0;
        for &v in row.iter() {
            self.temp.push(if v.is_nan() {
                0.0
            } else {
                count += 1;
                v.abs()
            });
        }
        let total = sum(&self.temp);
        if count >= 2 && total > 0.0 {
            for v in row {
                *v /= total;
            }
        } else {
            row.fill(f64::NAN);
        }
    }

    pub fn exposure(&mut self, held: &[f64], risk: &[f64], universe: &[bool]) -> f64 {
        self.temp.clear();
        let mut count = 0;
        for (&v, &u) in risk.iter().zip(universe) {
            self.temp.push(if u && v.is_finite() {
                count += 1;
                v
            } else {
                0.0
            });
        }
        let mean = if count > 0 {
            sum(&self.temp) / count as f64
        } else {
            0.0
        };
        for (j, v) in self.temp.iter_mut().enumerate() {
            *v = held[j]
                * if universe[j] && risk[j].is_finite() {
                    risk[j] - mean
                } else {
                    0.0
                };
        }
        sum(&self.temp)
    }
}
