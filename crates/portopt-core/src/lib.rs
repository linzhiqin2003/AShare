#![forbid(unsafe_code)]
//! Independent numerical kernels. All array inputs use row-major layout.

use clarabel::algebra::CscMatrix;
pub type SparseMatrix = CscMatrix<f64>;
use clarabel::solver::*;
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

static NUM_THREADS: AtomicUsize = AtomicUsize::new(0);
static POOLS: OnceLock<Mutex<HashMap<usize, Arc<rayon::ThreadPool>>>> = OnceLock::new();

pub fn set_num_threads(threads: usize) {
    NUM_THREADS.store(threads, Ordering::Relaxed);
}
pub fn get_num_threads() -> usize {
    match NUM_THREADS.load(Ordering::Relaxed) {
        0 => available_threads(),
        n => n,
    }
}

pub fn available_threads() -> usize {
    std::thread::available_parallelism().map_or(1, usize::from)
}

pub fn pool(threads: usize) -> Result<Arc<rayon::ThreadPool>, String> {
    let count = if threads == 0 {
        get_num_threads()
    } else {
        threads
    };
    let mut pools = POOLS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map_err(|e| e.to_string())?;
    if let Some(pool) = pools.get(&count) {
        return Ok(Arc::clone(pool));
    }
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(count)
            .build()
            .map_err(|e| e.to_string())?,
    );
    pools.insert(count, Arc::clone(&pool));
    Ok(pool)
}

pub fn ew_variance(
    x: &[f64],
    t: usize,
    n: usize,
    half_life: f64,
    lags: usize,
    threads: usize,
) -> Result<Vec<f64>, String> {
    let weights: Vec<f64> = (0..t)
        .map(|i| (-((t - 1 - i) as f64) / half_life).exp2())
        .collect();
    Ok(pool(threads)?.install(|| {
        (0..n)
            .into_par_iter()
            .map(|j| {
                let mut sum = 0.;
                let mut mean = 0.;
                for i in 0..t {
                    let v = x[i * n + j];
                    if v.is_finite() {
                        sum += weights[i];
                        mean += weights[i] * v;
                    }
                }
                if sum == 0. {
                    return 0.;
                }
                mean /= sum;
                let z: Vec<f64> = (0..t)
                    .map(|i| {
                        let v = x[i * n + j];
                        if v.is_finite() {
                            (v - mean) * (weights[i] / sum).sqrt()
                        } else {
                            0.
                        }
                    })
                    .collect();
                let mut variance = z.iter().map(|v| v * v).sum::<f64>();
                for lag in 1..=lags {
                    let cross = (lag..t).map(|i| z[i] * z[i - lag]).sum::<f64>();
                    variance += 2. * (1. - lag as f64 / (lags + 1) as f64) * cross;
                }
                variance.max(0.)
            })
            .collect()
    }))
}

pub fn rolling_bias(
    r: &[f64],
    p: &[f64],
    t: usize,
    n: usize,
    window: usize,
    threads: usize,
) -> Result<Vec<f64>, String> {
    let columns: Vec<Vec<f64>> = pool(threads)?.install(|| {
        (0..n)
            .into_par_iter()
            .map(|j| {
                let mut sum = 0.;
                let mut square = 0.;
                let mut count = 0usize;
                let mut ring = vec![f64::NAN; window];
                let mut out = Vec::with_capacity(t - window + 1);
                for i in 0..t {
                    let idx = i % window;
                    let old = ring[idx];
                    if old.is_finite() {
                        sum -= old;
                        square -= old * old;
                        count -= 1;
                    }
                    let v = if p[i * n + j] > 0. {
                        r[i * n + j] / p[i * n + j]
                    } else {
                        f64::NAN
                    };
                    ring[idx] = v;
                    if v.is_finite() {
                        sum += v;
                        square += v * v;
                        count += 1;
                    }
                    if i + 1 >= window {
                        out.push(if count == window {
                            ((square - sum * sum / window as f64) / (window - 1) as f64)
                                .max(0.)
                                .sqrt()
                        } else {
                            f64::NAN
                        });
                    }
                }
                out
            })
            .collect()
    });
    let mut out = vec![0.; (t - window + 1) * n];
    for j in 0..n {
        for i in 0..t - window + 1 {
            out[i * n + j] = columns[j][i];
        }
    }
    Ok(out)
}

pub struct RegressionInput<'a> {
    pub returns: &'a [f64],
    pub exposures: &'a [f64],
    pub cap: &'a [f64],
    pub estimation: &'a [bool],
    pub t: usize,
    pub n: usize,
    pub k: usize,
    pub varying: bool,
    pub industries: &'a [usize],
}

pub fn regress(
    input: &RegressionInput<'_>,
    threads: usize,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let results: Vec<(Vec<f64>, Vec<f64>)> = pool(threads)?.install(|| {
        (0..input.t)
            .into_par_iter()
            .map(|day| {
                let (n, k) = (input.n, input.k);
                let r = &input.returns[day * n..(day + 1) * n];
                let cap = &input.cap[day * n..(day + 1) * n];
                let est = &input.estimation[day * n..(day + 1) * n];
                let offset = if input.varying { day * n * k } else { 0 };
                let b = &input.exposures[offset..offset + n * k];
                let mut design = DMatrix::zeros(n, k);
                let mut response = DVector::zeros(n);
                let mut constraint = DVector::<f64>::zeros(k);
                let mut weight_sum = 0.;
                let mut valid_count = 0;
                for i in 0..n {
                    if est[i]
                        && r[i].is_finite()
                        && cap[i].is_finite()
                        && cap[i] > 0.
                        && b[i * k..(i + 1) * k].iter().all(|v| v.is_finite())
                    {
                        weight_sum += cap[i].sqrt();
                        valid_count += 1;
                    }
                }
                let mean = weight_sum / (valid_count as f64).max(1.);
                for i in 0..n {
                    if est[i]
                        && r[i].is_finite()
                        && cap[i].is_finite()
                        && cap[i] > 0.
                        && b[i * k..(i + 1) * k].iter().all(|v| v.is_finite())
                    {
                        let root = (cap[i].sqrt() / mean).sqrt();
                        response[i] = r[i] * root;
                        for j in 0..k {
                            design[(i, j)] = b[i * k + j] * root;
                        }
                        for &j in input.industries {
                            constraint[j] += cap[i] * b[i * k + j];
                        }
                    }
                }
                // Remove empty columns before checking the constrained normal system.
                let active: Vec<usize> = (0..k)
                    .filter(|&j| design.column(j).iter().any(|x| *x != 0.))
                    .collect();
                let q = active.len();
                let x = DMatrix::from_fn(n, q, |i, j| design[(i, active[j])]);
                let c = DVector::from_fn(q, |j, _| constraint[active[j]]);
                let norm = c.norm();
                let size = q + usize::from(norm > 0.);
                let gram = x.transpose() * &x;
                let rhs = x.transpose() * &response;
                let mut system = DMatrix::zeros(size, size);
                let mut target = DVector::zeros(size);
                system.view_mut((0, 0), (q, q)).copy_from(&gram);
                target.rows_mut(0, q).copy_from(&rhs);
                if norm > 0. {
                    for j in 0..q {
                        system[(j, q)] = c[j] / norm;
                        system[(q, j)] = c[j] / norm;
                    }
                }
                let well_conditioned = if q == 0 {
                    false
                } else {
                    let singular = system.clone().svd(false, false).singular_values;
                    singular.min() > singular.max() * 1e-12
                };
                let normal = if well_conditioned {
                    system.full_piv_lu().solve(&target)
                } else {
                    None
                };
                let beta = if let Some(solution) = normal {
                    solution.rows(0, q).into_owned()
                } else if q == 0 {
                    DVector::zeros(0)
                } else {
                    // Orthogonal Householder null space preserves the minimum-norm solution.
                    let transform = if norm > 0. {
                        let pivot = (0..q)
                            .max_by(|&i, &j| c[i].abs().total_cmp(&c[j].abs()))
                            .unwrap();
                        let mut v = &c / norm;
                        v[pivot] += if c[pivot] >= 0. { 1. } else { -1. };
                        v /= v.norm();
                        let h = DMatrix::identity(q, q) - 2. * &v * v.transpose();
                        let free: Vec<usize> = (0..q).filter(|&j| j != pivot).collect();
                        DMatrix::from_fn(q, q - 1, |i, j| h[(i, free[j])])
                    } else {
                        DMatrix::identity(q, q)
                    };
                    if transform.ncols() == 0 {
                        DVector::zeros(q)
                    } else {
                        let svd = (&x * &transform).svd(true, true);
                        let threshold = svd.singular_values.max() * 1e-12;
                        &transform
                            * svd
                                .solve(&response, threshold)
                                .expect("SVD vectors requested")
                    }
                };
                let mut coef = vec![0.; k];
                for (j, &column) in active.iter().enumerate() {
                    coef[column] = beta[j];
                }
                let residual = (0..n)
                    .map(|i| r[i] - (0..k).map(|j| b[i * k + j] * coef[j]).sum::<f64>())
                    .collect();
                (coef, residual)
            })
            .collect()
    });
    let mut factors = Vec::with_capacity(input.t * input.k);
    let mut residuals = Vec::with_capacity(input.t * input.n);
    for (f, r) in results {
        factors.extend(f);
        residuals.extend(r);
    }
    Ok((factors, residuals))
}

#[derive(Clone)]
pub struct ConicProblem {
    pub p: CscMatrix<f64>,
    pub q: Vec<f64>,
    pub a: CscMatrix<f64>,
    pub b: Vec<f64>,
    pub cones: Vec<(u8, usize)>,
}
pub type SolveResult = (Vec<f64>, String, f64, f64, f64, f64, u32, Vec<String>);

pub fn solve(problem: &ConicProblem, tolerance: f64) -> Result<SolveResult, String> {
    let cones: Vec<SupportedConeT<f64>> = problem
        .cones
        .iter()
        .map(|&(kind, size)| match kind {
            0 => ZeroConeT(size),
            1 => NonnegativeConeT(size),
            _ => SecondOrderConeT(size),
        })
        .collect();
    let mut statuses = Vec::new();
    let mut result = None;
    for (equilibrate, regularization) in
        [(true, 1e-10), (false, 1e-10), (true, 1e-12), (false, 1e-12)]
    {
        let settings = DefaultSettings {
            verbose: false,
            max_iter: 200,
            tol_gap_abs: tolerance,
            tol_gap_rel: tolerance,
            tol_feas: tolerance,
            max_threads: 1,
            static_regularization_constant: regularization,
            equilibrate_enable: equilibrate,
            ..DefaultSettings::default()
        };
        let mut solver = DefaultSolver::new(
            &problem.p, &problem.q, &problem.a, &problem.b, &cones, settings,
        )
        .map_err(|e| e.to_string())?;
        solver.solve();
        let solution = solver.solution;
        let status = solution.status;
        statuses.push(format!("{status:?}"));
        result = Some(solution);
        if matches!(
            status,
            SolverStatus::Solved | SolverStatus::PrimalInfeasible | SolverStatus::DualInfeasible
        ) {
            break;
        }
    }
    let s = result.expect("The numerical configuration list is nonempty");
    Ok((
        s.x,
        format!("{:?}", s.status),
        s.obj_val,
        s.obj_val_dual,
        s.r_prim,
        s.r_dual,
        s.iterations,
        statuses,
    ))
}

pub fn solve_many(
    template: &ConicProblem,
    qs: &[Vec<f64>],
    bs: &[Vec<f64>],
    tolerance: f64,
    threads: usize,
) -> Result<Vec<SolveResult>, String> {
    pool(threads)?.install(|| {
        qs.par_iter()
            .zip(bs.par_iter())
            .map(|(q, b)| {
                let mut problem = template.clone();
                problem.q = q.clone();
                problem.b = b.clone();
                solve(&problem, tolerance)
            })
            .collect()
    })
}

pub fn csc(
    rows: usize,
    cols: usize,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<f64>,
) -> CscMatrix<f64> {
    CscMatrix::new(rows, cols, indptr, indices, data)
}

/// Weighted Gram covariance with Bartlett lag taper. Missing positions stay in time.
pub fn ew_covariance(
    x: &[f64],
    t: usize,
    k: usize,
    half_life: f64,
    lags: usize,
    threads: usize,
) -> Result<Vec<f64>, String> {
    let weights: Vec<f64> = (0..t)
        .map(|i| (-((t - 1 - i) as f64) / half_life).exp2())
        .collect();
    let thread_pool = pool(threads)?;
    let centered: Vec<Vec<f64>> = thread_pool.install(|| {
        (0..k)
            .into_par_iter()
            .map(|j| {
                let mut sum = 0.;
                let mut mean = 0.;
                for i in 0..t {
                    let v = x[i * k + j];
                    if v.is_finite() {
                        sum += weights[i];
                        mean += weights[i] * v;
                    }
                }
                if sum == 0. {
                    return vec![0.; t];
                }
                mean /= sum;
                (0..t)
                    .map(|i| {
                        let v = x[i * k + j];
                        if v.is_finite() {
                            (v - mean) * (weights[i] / sum).sqrt()
                        } else {
                            0.
                        }
                    })
                    .collect()
            })
            .collect()
    });
    let rows: Vec<Vec<f64>> = thread_pool.install(|| {
        (0..k)
            .into_par_iter()
            .map(|i| {
                let a = &centered[i];
                (0..=i)
                    .map(|j| {
                        let b = &centered[j];
                        let mut result = a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
                        for lag in 1..=lags {
                            let cross = (lag..t)
                                .map(|h| a[h] * b[h - lag] + b[h] * a[h - lag])
                                .sum::<f64>();
                            result += (1. - lag as f64 / (lags + 1) as f64) * cross;
                        }
                        result
                    })
                    .collect()
            })
            .collect()
    });
    let mut result = vec![0.; k * k];
    for i in 0..k {
        for j in 0..=i {
            result[i * k + j] = rows[i][j];
            result[j * k + i] = rows[i][j];
        }
    }
    Ok(result)
}

/// Number of moments kept per column by the exponentially weighted window moments.
pub fn ew_moment_count(lags: usize) -> usize {
    3 + 4 * lags
}

/// Exponentially weighted moments of rows `first..=last` of a time ordered `t` by `n`
/// window, built from empty one row at a time (`state = decay * state + terms`), so a
/// segment's value refers to its last row. Values are shifted by `shift` per column.
///
/// Per column, column-major in the result: weight sum, weighted shifted sum and squares;
/// then per lag (1-based) the pair weight, the two pair sums and the cross products,
/// each pair `(i, i - lag)` weighted by its later row and kept only if `i - lag >= floor`.
/// Every term is added with a positive weight: no value is ever subtracted, so window
/// sums assembled from segments carry no cancellation drift.
#[allow(clippy::too_many_arguments)]
pub fn ew_moments_segment(
    window: &[f64],
    t: usize,
    n: usize,
    first: usize,
    last: usize,
    floor: usize,
    shift: &[f64],
    half_life: f64,
    lags: usize,
    threads: usize,
) -> Result<Vec<f64>, String> {
    if window.len() != t * n || shift.len() != n || first > last || last >= t {
        return Err("window, shift or segment bounds have the wrong shape".into());
    }
    let m = ew_moment_count(lags);
    let decay = (-1. / half_life).exp2();
    let mut state = vec![0.; n * m];
    pool(threads)?.install(|| {
        state.par_chunks_mut(m).enumerate().for_each(|(j, s)| {
            let offset = shift[j];
            for i in first..=last {
                for value in s.iter_mut() {
                    *value *= decay;
                }
                let newest = window[i * n + j];
                if !newest.is_finite() {
                    continue;
                }
                let x = newest - offset;
                s[0] += 1.;
                s[1] += x;
                s[2] += x * x;
                for lag in 1..=lags {
                    if i < lag || i - lag < floor {
                        break;
                    }
                    let earlier = window[(i - lag) * n + j];
                    if earlier.is_finite() {
                        let y = earlier - offset;
                        let base = 3 + 4 * (lag - 1);
                        s[base] += 1.;
                        s[base + 1] += x;
                        s[base + 2] += y;
                        s[base + 3] += x * y;
                    }
                }
            }
        })
    });
    Ok(state)
}

/// Newey-West variance from window moments with `used` lags (Bartlett weights over
/// `used`), equal in exact arithmetic to `ew_variance` on the same window. Also returns
/// the magnitude of the summed terms, a scale for the cancellation in the result.
pub fn ew_moments_variance(
    state: &[f64],
    n: usize,
    lags: usize,
    used: usize,
    half_life: f64,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let m = ew_moment_count(lags);
    if state.len() != n * m || used > lags {
        return Err("moment state has the wrong length or too many lags".into());
    }
    let mut variance = vec![0.; n];
    let mut magnitude = vec![0.; n];
    for j in 0..n {
        let s = &state[j * m..(j + 1) * m];
        if s[0] == 0. {
            continue;
        }
        let mean = s[1] / s[0];
        let second = s[2] / s[0];
        let mut v = second - mean * mean;
        let mut size = second + mean * mean;
        for lag in 1..=used {
            let base = 3 + 4 * (lag - 1);
            let (count, later, earlier, cross) = (s[base], s[base + 1], s[base + 2], s[base + 3]);
            let scale = 2.
                * (1. - lag as f64 / (used + 1) as f64)
                * (-(lag as f64) / (2. * half_life)).exp2()
                / s[0];
            v += scale * (cross - mean * (later + earlier) + mean * mean * count);
            size += scale
                * (cross.abs()
                    + mean.abs() * (later.abs() + earlier.abs())
                    + mean * mean * count.abs());
        }
        variance[j] = v;
        magnitude[j] = size;
    }
    Ok((variance, magnitude))
}
