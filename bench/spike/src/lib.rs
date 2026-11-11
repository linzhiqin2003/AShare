#![forbid(unsafe_code)]
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;

fn fee(v: f64) -> f64 {
    if v > 0.0 { (v * 0.00025).max(5.0) } else { 0.0 }
}
fn simulate(
    p: &[f64],
    close: &[f64],
    t: usize,
    n: usize,
    h: usize,
    interval: usize,
    variant: usize,
) -> Vec<f64> {
    let mut shares = vec![0.0; n];
    let mut weight = vec![0.0; n];
    let mut active: Vec<usize> = Vec::with_capacity(2 * h);
    let mut desired = Vec::with_capacity(2 * h);
    let mut orders = Vec::with_capacity(2 * h);
    let mut nav = Vec::with_capacity(t);
    let mut cash = 1e7;
    let exposure = 0.96 + 0.039 * (variant % 101) as f64 / 100.0;
    for day in 0..t {
        let base = day * n;
        if day % interval == 0 {
            let value = cash + active.iter().map(|&i| shares[i] * p[base + i]).sum::<f64>();
            for &i in &active {
                weight[i] = 0.0;
            }
            for j in 0..h {
                let i = ((day / interval) * 37 + j + variant * 11) % n;
                weight[i] = exposure / h as f64;
                active.push(i);
            }
            active.sort_unstable();
            active.dedup();
            desired.clear();
            orders.clear();
            for &i in &active {
                desired.push((weight[i] * value / p[base + i] / 100.0).floor() * 100.0);
            }
            for (&i, &target) in active.iter().zip(&desired) {
                let q = (shares[i] - target).max(0.0);
                if q > 0.0 {
                    let v = q * p[base + i] * (1.0 - 0.0005);
                    cash += v - fee(v) - v * 0.001;
                    shares[i] -= q;
                }
                orders.push((target - shares[i]).max(0.0));
            }
            let total: f64 = active
                .iter()
                .zip(&orders)
                .map(|(&i, &q)| {
                    let v = q * p[base + i] * (1.0 + 0.0005);
                    v + fee(v)
                })
                .sum();
            let scale = if total > cash { cash / total } else { 1.0 };
            for (&i, &q) in active.iter().zip(&orders) {
                let price = p[base + i] * (1.0 + 0.0005);
                let mut q = if scale < 1.0 {
                    (q * scale / 100.0).floor() * 100.0
                } else {
                    q
                };
                if q * price + fee(q * price) > cash {
                    q = (cash / (price * 1.00025))
                        .min((cash - 5.0) / price)
                        .max(0.0);
                    q = (q / 100.0).floor() * 100.0;
                }
                if q > 0.0 {
                    let v = q * price;
                    cash -= v + fee(v);
                    shares[i] += q;
                }
            }
            active.retain(|&i| shares[i] > 0.0);
        }
        nav.push(
            (cash
                + active
                    .iter()
                    .map(|&i| shares[i] * close[base + i])
                    .sum::<f64>())
                / 1e7,
        );
    }
    nav
}

#[pyfunction]
fn run(
    py: Python<'_>,
    p: PyReadonlyArray2<'_, f64>,
    close: PyReadonlyArray2<'_, f64>,
    h: usize,
    interval: usize,
    variants: PyReadonlyArray1<'_, i64>,
    threads: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let shape = p.shape();
    let (t, n) = (shape[0], shape[1]);
    let ps = p.as_slice()?;
    let cs = close.as_slice()?;
    let vs = variants.as_slice()?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(py.detach(|| {
        pool.install(|| {
            vs.par_iter()
                .map(|&v| simulate(ps, cs, t, n, h, interval, v as usize))
                .collect()
        })
    }))
}
#[pymodule]
fn panelbt_spike(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run, m)?)?;
    Ok(())
}
