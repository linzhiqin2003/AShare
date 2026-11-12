#![forbid(unsafe_code)]
use numpy::{
    IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn,
    PyUntypedArrayMethods,
};
use portopt_core::{ConicProblem, RegressionInput, SolveResult};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type ArrayPair<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);
type CscInput = (usize, usize, Vec<usize>, Vec<usize>, Vec<f64>);
fn matrix(v: CscInput) -> portopt_core::SparseMatrix {
    portopt_core::csc(v.0, v.1, v.2, v.3, v.4)
}

#[pyfunction]
fn ew_covariance<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    half_life: f64,
    lags: usize,
    threads: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let shape = x.shape();
    let (t, n) = (shape[0], shape[1]);
    let data = x.as_slice()?;
    let out = py
        .detach(|| portopt_core::ew_covariance(data, t, n, half_life, lags, threads))
        .map_err(PyValueError::new_err)?;
    Ok(out.into_pyarray(py))
}
#[pyfunction]
fn ew_variance<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    half_life: f64,
    lags: usize,
    threads: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let shape = x.shape();
    let (t, n) = (shape[0], shape[1]);
    let data = x.as_slice()?;
    let out = py
        .detach(|| portopt_core::ew_variance(data, t, n, half_life, lags, threads))
        .map_err(PyValueError::new_err)?;
    Ok(out.into_pyarray(py))
}
#[pyfunction]
fn rolling_bias<'py>(
    py: Python<'py>,
    r: PyReadonlyArray2<'py, f64>,
    p: PyReadonlyArray2<'py, f64>,
    window: usize,
    threads: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let shape = r.shape();
    let (t, n) = (shape[0], shape[1]);
    let rs = r.as_slice()?;
    let ps = p.as_slice()?;
    let out = py
        .detach(|| portopt_core::rolling_bias(rs, ps, t, n, window, threads))
        .map_err(PyValueError::new_err)?;
    Ok(out.into_pyarray(py))
}
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn regress<'py>(
    py: Python<'py>,
    r: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArrayDyn<'py, f64>,
    cap: PyReadonlyArray2<'py, f64>,
    est: PyReadonlyArray2<'py, bool>,
    industries: Vec<usize>,
    threads: usize,
) -> PyResult<ArrayPair<'py>> {
    let (t, n) = (r.shape()[0], r.shape()[1]);
    let k = *b.shape().last().unwrap();
    let input = RegressionInput {
        returns: r.as_slice()?,
        exposures: b.as_slice()?,
        cap: cap.as_slice()?,
        estimation: est.as_slice()?,
        t,
        n,
        k,
        varying: b.ndim() == 3,
        industries: &industries,
    };
    let (f, u) = py
        .detach(|| portopt_core::regress(&input, threads))
        .map_err(PyValueError::new_err)?;
    Ok((f.into_pyarray(py), u.into_pyarray(py)))
}
#[pyfunction]
fn solve(
    py: Python<'_>,
    p: CscInput,
    q: Vec<f64>,
    a: CscInput,
    b: Vec<f64>,
    cones: Vec<(u8, usize)>,
    tolerance: f64,
) -> PyResult<SolveResult> {
    let problem = ConicProblem {
        p: matrix(p),
        q,
        a: matrix(a),
        b,
        cones,
    };
    py.detach(|| portopt_core::solve(&problem, tolerance))
        .map_err(PyValueError::new_err)
}
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn solve_many(
    py: Python<'_>,
    p: CscInput,
    qs: Vec<Vec<f64>>,
    a: CscInput,
    bs: Vec<Vec<f64>>,
    cones: Vec<(u8, usize)>,
    tolerance: f64,
    threads: usize,
) -> PyResult<Vec<SolveResult>> {
    let problem = ConicProblem {
        p: matrix(p),
        q: qs[0].clone(),
        a: matrix(a),
        b: bs[0].clone(),
        cones,
    };
    py.detach(|| portopt_core::solve_many(&problem, &qs, &bs, tolerance, threads))
        .map_err(PyValueError::new_err)
}
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn ew_moments_segment<'py>(
    py: Python<'py>,
    window: PyReadonlyArray2<'py, f64>,
    first: usize,
    last: usize,
    floor: usize,
    shift: PyReadonlyArray1<'py, f64>,
    half_life: f64,
    lags: usize,
    threads: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (t, n) = (window.shape()[0], window.shape()[1]);
    let window = window.as_slice()?;
    let shift = shift.as_slice()?;
    let out = py
        .detach(|| {
            portopt_core::ew_moments_segment(
                window, t, n, first, last, floor, shift, half_life, lags, threads,
            )
        })
        .map_err(PyValueError::new_err)?;
    Ok(out.into_pyarray(py))
}
#[pyfunction]
fn ew_moments_variance<'py>(
    py: Python<'py>,
    state: PyReadonlyArray1<'py, f64>,
    n: usize,
    lags: usize,
    used: usize,
    half_life: f64,
) -> PyResult<ArrayPair<'py>> {
    let state = state.as_slice()?;
    let (v, m) = py
        .detach(|| portopt_core::ew_moments_variance(state, n, lags, used, half_life))
        .map_err(PyValueError::new_err)?;
    Ok((v.into_pyarray(py), m.into_pyarray(py)))
}
#[pyfunction]
fn available_threads() -> usize {
    portopt_core::available_threads()
}

#[pyfunction]
fn set_num_threads(threads: usize) {
    portopt_core::set_num_threads(threads);
}
#[pyfunction]
fn get_num_threads() -> usize {
    portopt_core::get_num_threads()
}

#[pymodule(name = "_core")]
fn portopt_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ew_variance, m)?)?;
    m.add_function(wrap_pyfunction!(ew_covariance, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_bias, m)?)?;
    m.add_function(wrap_pyfunction!(ew_moments_segment, m)?)?;
    m.add_function(wrap_pyfunction!(ew_moments_variance, m)?)?;
    m.add_function(wrap_pyfunction!(regress, m)?)?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add_function(wrap_pyfunction!(solve_many, m)?)?;
    m.add_function(wrap_pyfunction!(available_threads, m)?)?;
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(get_num_threads, m)?)?;
    Ok(())
}
