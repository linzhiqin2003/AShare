//! Borrow read-only arrays while releasing the GIL for all native computation.
#![forbid(unsafe_code)]
use numpy::{
    IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods, ndarray::Array2,
};
use panelops_core::{CrossOptions, Options, Panel};
use pyo3::{exceptions::PyValueError, prelude::*};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};

fn pool(threads: usize) -> PyResult<Arc<rayon::ThreadPool>> {
    static POOLS: OnceLock<Mutex<HashMap<usize, Arc<rayon::ThreadPool>>>> = OnceLock::new();
    let count = if threads == 0 {
        std::thread::available_parallelism().map_or(1, usize::from)
    } else {
        threads
    };
    let mut pools = POOLS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if let Some(p) = pools.get(&count) {
        return Ok(p.clone());
    }
    let p = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(count)
            .build()
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
    );
    pools.insert(count, p.clone());
    Ok(p)
}
fn check(shape: &[usize], other: &[usize], empty: bool) -> PyResult<()> {
    if !empty && shape != other {
        Err(PyValueError::new_err("panel shape mismatch"))
    } else {
        Ok(())
    }
}
fn matrix<'py>(
    py: Python<'py>,
    values: Vec<f64>,
    t: usize,
    n: usize,
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    Ok(Array2::from_shape_vec((t, n), values)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .into_pyarray(py))
}
#[pyfunction]
fn actual_threads(threads: usize) -> PyResult<usize> {
    Ok(pool(threads)?.current_num_threads())
}
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn ts<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
    mask: PyReadonlyArray2<'py, bool>,
    op: &str,
    window: usize,
    min_periods: usize,
    ddof: usize,
    q: f64,
    normalize: bool,
    alpha: f64,
    periods: usize,
    threads: usize,
    layout: usize,
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    if panelops_core::TimeOp::parse(op).is_none()
        || window == 0
        || min_periods == 0
        || min_periods > window
        || !q.is_finite()
        || !(0.0..=1.0).contains(&q)
        || !(0.0..=1.0).contains(&alpha)
        || alpha == 0.
        || layout > 2
    {
        return Err(PyValueError::new_err(
            "invalid native time-series parameters",
        ));
    }
    check(x.shape(), y.shape(), y.is_empty())?;
    if !y.is_c_contiguous() {
        return Err(PyValueError::new_err("y must be C-contiguous"));
    }
    check(x.shape(), mask.shape(), mask.is_empty())?;
    if !x.is_c_contiguous() || !mask.is_c_contiguous() {
        return Err(PyValueError::new_err("panels must be C-contiguous"));
    }

    if [
        "corr",
        "cov",
        "regression_slope",
        "regression_residual",
        "sumif",
    ]
    .contains(&op)
        && y.shape() != x.shape()
    {
        return Err(PyValueError::new_err("paired operator requires y"));
    }
    let (t, n) = (x.shape()[0], x.shape()[1]);
    let pan = Panel {
        x: x.as_slice()?,
        y: y.as_slice()?,
        mask: mask.as_slice()?,
        t,
        n,
    };
    let options = Options {
        window,
        min_periods,
        ddof,
        q,
        normalize,
        alpha,
        periods,
    };
    let pool = pool(threads)?;
    let output = py.detach(|| pool.install(|| panelops_core::ts(pan, op, options, layout)));
    matrix(py, output, t, n)
}
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn cs<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    mask: PyReadonlyArray2<'py, bool>,
    op: &str,
    groups: PyReadonlyArray2<'py, i64>,
    exposures: Vec<PyReadonlyArray2<'py, f64>>,
    weights: PyReadonlyArray2<'py, f64>,
    normalize: bool,
    ddof: usize,
    lower: f64,
    upper: f64,
    scale: f64,
    rcond: f64,
    threads: usize,
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    if ![
        "rank",
        "zscore",
        "winsor_quantile",
        "winsor_mad",
        "group_demean",
        "neutralize",
        "scale",
        "demean",
    ]
    .contains(&op)
        || !(0.0..=1.0).contains(&lower)
        || !(lower..=1.0).contains(&upper)
        || !scale.is_finite()
        || scale < 0.
        || !rcond.is_finite()
        || rcond <= 0.
    {
        return Err(PyValueError::new_err(
            "invalid native cross-sectional parameters",
        ));
    }
    for (shape, empty) in [
        (mask.shape(), mask.is_empty()),
        (groups.shape(), groups.is_empty()),
        (weights.shape(), weights.is_empty()),
    ] {
        check(x.shape(), shape, empty)?;
    }
    if !x.is_c_contiguous()
        || !mask.is_c_contiguous()
        || !groups.is_c_contiguous()
        || !weights.is_c_contiguous()
    {
        return Err(PyValueError::new_err("panels must be C-contiguous"));
    }
    if op == "group_demean" && groups.shape() != x.shape() {
        return Err(PyValueError::new_err("group_demean requires groups"));
    }
    for e in &exposures {
        check(x.shape(), e.shape(), false)?;
        if !e.is_c_contiguous() {
            return Err(PyValueError::new_err("exposures must be C-contiguous"));
        }
    }
    let (t, n) = (x.shape()[0], x.shape()[1]);
    let pan = Panel {
        x: x.as_slice()?,
        y: &[],
        mask: mask.as_slice()?,
        t,
        n,
    };
    let options = CrossOptions {
        normalize,
        ddof,
        lower,
        upper,
        scale,
        rcond,
        groups: groups.as_slice()?,
        weights: weights.as_slice()?,
        exposures: exposures
            .iter()
            .map(|e| e.as_slice())
            .collect::<Result<_, _>>()?,
    };
    let pool = pool(threads)?;
    let output = py.detach(|| pool.install(|| panelops_core::cs(pan, op, &options)));
    matrix(py, output, t, n)
}
#[pyfunction]
fn ic<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
    mask: PyReadonlyArray2<'py, bool>,
    rank: bool,
    threads: usize,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    check(x.shape(), y.shape(), false)?;
    if !y.is_c_contiguous() {
        return Err(PyValueError::new_err("y must be C-contiguous"));
    }
    check(x.shape(), mask.shape(), mask.is_empty())?;
    if !x.is_c_contiguous() || !mask.is_c_contiguous() {
        return Err(PyValueError::new_err("panels must be C-contiguous"));
    }

    let pan = Panel {
        x: x.as_slice()?,
        y: y.as_slice()?,
        mask: mask.as_slice()?,
        t: x.shape()[0],
        n: x.shape()[1],
    };
    let pool = pool(threads)?;
    Ok(py
        .detach(|| pool.install(|| panelops_core::ic(pan, rank)))
        .into_pyarray(py))
}
#[pyfunction]
fn autocorr<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    mask: PyReadonlyArray2<'py, bool>,
    periods: usize,
    rank: bool,
    threads: usize,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    check(x.shape(), mask.shape(), mask.is_empty())?;
    if !x.is_c_contiguous() || !mask.is_c_contiguous() {
        return Err(PyValueError::new_err("panels must be C-contiguous"));
    }
    let pan = Panel {
        x: x.as_slice()?,
        y: &[],
        mask: mask.as_slice()?,
        t: x.shape()[0],
        n: x.shape()[1],
    };
    let pool = pool(threads)?;
    Ok(py
        .detach(|| pool.install(|| panelops_core::evaluation::autocorr(pan, periods, rank)))
        .into_pyarray(py))
}
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn grouped<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
    mask: PyReadonlyArray2<'py, bool>,
    weights: PyReadonlyArray2<'py, f64>,
    q: usize,
    periods: usize,
    turnover: bool,
    threads: usize,
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    if q == 0 {
        return Err(PyValueError::new_err("q must be positive"));
    }
    check(x.shape(), y.shape(), turnover)?;
    if !y.is_c_contiguous() {
        return Err(PyValueError::new_err("y must be C-contiguous"));
    }
    check(x.shape(), mask.shape(), mask.is_empty())?;
    if !x.is_c_contiguous() || !mask.is_c_contiguous() {
        return Err(PyValueError::new_err("panels must be C-contiguous"));
    }

    check(x.shape(), weights.shape(), weights.is_empty())?;
    if !weights.is_c_contiguous() {
        return Err(PyValueError::new_err("weights must be C-contiguous"));
    }
    let (t, n) = (x.shape()[0], x.shape()[1]);
    let pan = Panel {
        x: x.as_slice()?,
        y: y.as_slice()?,
        mask: mask.as_slice()?,
        t,
        n,
    };
    let w = weights.as_slice()?;
    let pool = pool(threads)?;
    matrix(
        py,
        py.detach(|| pool.install(|| panelops_core::grouped(pan, q, w, periods, turnover))),
        t,
        q,
    )
}
type SparseArrays<'py> = (
    Bound<'py, numpy::PyArray1<i64>>,
    Bound<'py, numpy::PyArray1<i64>>,
    Bound<'py, numpy::PyArray1<f64>>,
);
#[pyfunction]
fn targets<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    mask: PyReadonlyArray2<'py, bool>,
    days: PyReadonlyArray1<'py, i64>,
    k: usize,
    fraction: f64,
    threads: usize,
) -> PyResult<SparseArrays<'py>> {
    check(x.shape(), mask.shape(), mask.is_empty())?;
    if !x.is_c_contiguous() || !mask.is_c_contiguous() {
        return Err(PyValueError::new_err("panels must be C-contiguous"));
    }

    let d = days.as_slice()?;
    if !fraction.is_finite()
        || fraction <= 0.
        || fraction > 1.
        || d.iter().any(|&v| v < 0 || v as usize >= x.shape()[0])
        || d.windows(2).any(|v| v[0] >= v[1])
    {
        return Err(PyValueError::new_err("invalid target selection parameters"));
    }
    let pan = Panel {
        x: x.as_slice()?,
        y: &[],
        mask: mask.as_slice()?,
        t: x.shape()[0],
        n: x.shape()[1],
    };
    let pool = pool(threads)?;
    let (ptr, cols, weights) =
        py.detach(|| pool.install(|| panelops_core::targets(pan, d, k, fraction)));
    Ok((
        ptr.into_pyarray(py),
        cols.into_pyarray(py),
        weights.into_pyarray(py),
    ))
}
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn elementwise<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
    z: PyReadonlyArray2<'py, f64>,
    mask: PyReadonlyArray2<'py, bool>,
    op: &str,
    scalars: Vec<f64>,
    scalar: Vec<bool>,
    t: usize,
    n: usize,
    lower: f64,
    upper: f64,
    threads: usize,
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    if !panelops_core::element::valid(op)
        || scalars.len() != 3
        || scalar.len() != 3
        || lower > upper
        || !lower.is_finite()
        || !upper.is_finite()
    {
        return Err(PyValueError::new_err("invalid elementwise parameters"));
    }
    for (i, a) in [&x, &y, &z].iter().enumerate() {
        if !a.is_c_contiguous() || (!scalar[i] && a.shape() != [t, n]) {
            return Err(PyValueError::new_err("invalid elementwise panel"));
        }
    }
    check(&[t, n], mask.shape(), mask.is_empty())?;
    if !mask.is_c_contiguous() {
        return Err(PyValueError::new_err("mask must be C-contiguous"));
    }
    let inputs = panelops_core::element::Inputs {
        panels: [x.as_slice()?, y.as_slice()?, z.as_slice()?],
        scalars: scalars.try_into().unwrap(),
        scalar: scalar.try_into().unwrap(),
        mask: mask.as_slice()?,
        length: t
            .checked_mul(n)
            .ok_or_else(|| PyValueError::new_err("shape overflow"))?,
        lower,
        upper,
    };
    let pool = pool(threads)?;
    matrix(
        py,
        py.detach(|| pool.install(|| panelops_core::element::run(inputs, op))),
        t,
        n,
    )
}
#[pymodule(name = "_core")]
fn panelops_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("VERSION", panelops_core::VERSION)?;
    m.add("BUILD_ID", env!("PANELOPS_BUILD_ID"))?;
    m.add_function(wrap_pyfunction!(elementwise, m)?)?;
    m.add_function(wrap_pyfunction!(ts, m)?)?;
    m.add_function(wrap_pyfunction!(cs, m)?)?;
    m.add_function(wrap_pyfunction!(ic, m)?)?;
    m.add_function(wrap_pyfunction!(actual_threads, m)?)?;
    m.add_function(wrap_pyfunction!(autocorr, m)?)?;
    m.add_function(wrap_pyfunction!(grouped, m)?)?;
    m.add_function(wrap_pyfunction!(targets, m)?)?;
    Ok(())
}
