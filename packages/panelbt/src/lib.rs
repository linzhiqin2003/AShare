//! Read-only NumPy borrowing; release the GIL during independent simulations.
#![forbid(unsafe_code)]
use numpy::{
    IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods, ndarray::Array2,
};
use panelbt_core::{
    Futures, Market, Settings, Targets,
    validate::{self, Context, Domain},
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use rayon::prelude::*;
type Plan<'a> = (
    PyReadonlyArray1<'a, i64>,
    PyReadonlyArray1<'a, i64>,
    PyReadonlyArray1<'a, i64>,
    PyReadonlyArray1<'a, f64>,
    bool,
);
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn run<'py>(
    py: Python<'py>,
    floats: Vec<PyReadonlyArray2<'py, f64>>,
    masks: Vec<PyReadonlyArray2<'py, u8>>,
    plans: Vec<Plan<'py>>,
    vectors: Vec<PyReadonlyArray1<'py, f64>>,
    scalars: Vec<f64>,
    flags: Vec<bool>,
    first_delist: PyReadonlyArray1<'py, i64>,
    actions: (PyReadonlyArray1<'py, i64>, PyReadonlyArray1<'py, i64>),
    future: (
        PyReadonlyArray1<'py, u8>,
        PyReadonlyArray1<'py, f64>,
        PyReadonlyArray1<'py, i64>,
        Vec<PyReadonlyArray2<'py, f64>>,
    ),
    validated: Vec<bool>,
    context: (bool, bool),
    threads: usize,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    if floats.len() != 8
        || masks.len() != 4
        || vectors.len() != 16
        || scalars.len() != 10
        || flags.len() != 7
    {
        return Err(PyValueError::new_err("invalid internal argument lengths"));
    }
    let shape = floats[0].shape();
    let (t, n) = (shape[0], shape[1]);
    let fs: Vec<&[f64]> = floats
        .iter()
        .map(|x| x.as_slice())
        .collect::<Result<_, _>>()?;
    let ms: Vec<&[u8]> = masks
        .iter()
        .map(|x| x.as_slice())
        .collect::<Result<_, _>>()?;
    let vs: Vec<&[f64]> = vectors
        .iter()
        .map(|x| x.as_slice())
        .collect::<Result<_, _>>()?;
    for (i, a) in fs.iter().enumerate() {
        let valid = if i < 2 {
            a.len() == t * n
        } else if i < 5 {
            a.is_empty() || a.len() == t * n
        } else {
            a.len() == 1 || a.len() == t || a.len() == t * n || (i == 6 && a.is_empty())
        };
        if !valid {
            return Err(PyValueError::new_err("invalid panel length"));
        }
    }
    for (i, a) in ms.iter().enumerate() {
        if a.len() != t * n && (i < 2 || !a.is_empty()) {
            return Err(PyValueError::new_err("invalid mask length"));
        }
    }
    for (i, a) in vs.iter().enumerate() {
        if a.len() != if i == 6 || i == 7 { n } else { t } {
            return Err(PyValueError::new_err("invalid vector length"));
        }
    }
    let first = first_delist.as_slice()?;
    if first.len() != n {
        return Err(PyValueError::new_err("invalid delist length"));
    }
    let action_indptr = actions.0.as_slice()?;
    let action_cols = actions.1.as_slice()?;
    if action_indptr.len() != t + 1
        || action_indptr.first() != Some(&0)
        || action_indptr.last().copied() != Some(action_cols.len() as i64)
        || action_indptr.windows(2).any(|w| w[0] > w[1])
        || action_cols.iter().any(|&i| i < 0 || i as usize >= n)
    {
        return Err(PyValueError::new_err("invalid corporate-action index"));
    }
    let kind = future.0.as_slice()?;
    let multiplier = future.1.as_slice()?;
    let expiry = future.2.as_slice()?;
    let panels: Vec<&[f64]> = future
        .3
        .iter()
        .map(|x| x.as_slice())
        .collect::<Result<_, _>>()?;
    let futures = if kind.is_empty() {
        if !multiplier.is_empty() || !expiry.is_empty() || !panels.is_empty() {
            return Err(PyValueError::new_err("invalid empty futures arguments"));
        }
        None
    } else {
        if kind.len() != n
            || multiplier.len() != n
            || expiry.len() != n
            || panels.len() != 6
            || panels[0].len() != t * n
            || panels[1..]
                .iter()
                .any(|a| ![1, n, t * n].contains(&a.len()))
        {
            return Err(PyValueError::new_err("invalid futures argument lengths"));
        }
        Some(Futures {
            kind,
            multiplier,
            expiry,
            settlement: panels[0],
            margin_rate: panels[1],
            fee_rate: panels[2],
            fee_per_contract: panels[3],
            delivery_fee_rate: panels[4],
            delivery_fee_per_contract: panels[5],
        })
    };
    let m = Market {
        t,
        n,
        price: fs[0],
        close: fs[1],
        volume: fs[2],
        split: fs[3],
        dividend: fs[4],
        buy: ms[0],
        sell: ms[1],
        can_short: ms[3],
        first_delist: first,
        action_indptr,
        action_cols,
        futures,
    };
    let s = Settings {
        vectors: vs,
        scalars: &scalars,
        flags: &flags,
        slippage: fs[5],
        impact: fs[6],
        borrow: fs[7],
    };
    let targets: Vec<Targets<'_>> = plans
        .iter()
        .map(|p| {
            Ok(Targets {
                days: p.0.as_slice()?,
                indptr: p.1.as_slice()?,
                cols: p.2.as_slice()?,
                weights: p.3.as_slice()?,
                shares: p.4,
            })
        })
        .collect::<PyResult<_>>()?;
    if validated.len() != targets.len() {
        return Err(PyValueError::new_err("invalid internal argument lengths"));
    }
    // Context checks for every plan in parallel; the first failure in input
    // order is reported. Index bounds are checked even for unvalidated plans.
    let future_mask: Vec<bool> = kind.iter().map(|&x| x != 0).collect();
    let ctx = Context {
        t,
        n,
        future: &future_mask,
        allow_short: context.0,
        cash_weights: context.1,
    };
    let checked: Vec<Result<(), String>> = py.detach(|| {
        targets
            .par_iter()
            .zip(&validated)
            .map(|(p, &v)| {
                validate::check_context(p.days, p.indptr, p.cols, p.weights, p.shares, v, &ctx)
            })
            .collect()
    });
    if let Some(Err(message)) = checked.into_iter().find(|r| r.is_err()) {
        return Err(PyValueError::new_err(message));
    }
    for plan in &targets {
        if plan.indptr.len() != plan.days.len() + 1
            || plan.indptr.first() != Some(&0)
            || plan.indptr.last().copied() != Some(plan.cols.len() as i64)
            || plan.cols.len() != plan.weights.len()
            || plan.cols.iter().any(|&i| i < 0 || i as usize >= n)
            || plan.days.iter().any(|&d| d < 0 || d as usize >= t)
            || plan.indptr.windows(2).any(|w| w[0] > w[1])
        {
            return Err(PyValueError::new_err("invalid sparse plan"));
        }
    }
    let outputs = py
        .detach(|| panelbt_core::run_batch(&m, &targets, &s, threads))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    outputs
        .into_iter()
        .map(|o| {
            let d = PyDict::new(py);
            d.set_item(
                "daily",
                Array2::from_shape_vec((panelbt_core::DAILY, t), o.daily)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
                    .into_pyarray(py),
            )?;
            if let Some(p) = o.positions {
                d.set_item(
                    "positions",
                    Array2::from_shape_vec((t, n), p)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                        .into_pyarray(py),
                )?;
            } else {
                d.set_item("positions", py.None())?;
            }
            d.set_item("trades", o.trades)?;
            d.set_item("events", o.events)?;
            d.set_item("counts", o.counts.to_vec())?;
            d.set_item("contribution", o.contribution.into_pyarray(py))?;
            d.set_item("liquidations", o.liquidations)?;
            d.set_item("bankruptcies", o.bankruptcies)?;
            d.set_item("expiries", o.expiries)?;
            Ok(d)
        })
        .collect()
}
/// Market content checks (when `validate`) and preparation: first delist row
/// per column and the sparse corporate-action index.
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn prepare_market<'py>(
    py: Python<'py>,
    price: PyReadonlyArray2<'py, f64>,
    close: PyReadonlyArray2<'py, f64>,
    buy: PyReadonlyArray2<'py, u8>,
    sell: PyReadonlyArray2<'py, u8>,
    volume: PyReadonlyArray2<'py, f64>,
    split: PyReadonlyArray2<'py, f64>,
    dividend: PyReadonlyArray2<'py, f64>,
    delist: PyReadonlyArray2<'py, u8>,
    can_short: PyReadonlyArray2<'py, u8>,
    validate: bool,
) -> PyResult<(
    Bound<'py, numpy::PyArray1<i64>>,
    Bound<'py, numpy::PyArray1<i64>>,
    Bound<'py, numpy::PyArray1<i64>>,
)> {
    let shape = price.shape();
    let (t, n) = (shape[0], shape[1]);
    let floats = [
        price.as_slice()?,
        close.as_slice()?,
        volume.as_slice()?,
        split.as_slice()?,
        dividend.as_slice()?,
    ];
    let bytes = [
        buy.as_slice()?,
        sell.as_slice()?,
        delist.as_slice()?,
        can_short.as_slice()?,
    ];
    for (i, a) in floats.iter().enumerate() {
        if a.len() != t * n && (i < 2 || !a.is_empty()) {
            return Err(PyValueError::new_err("invalid panel length"));
        }
    }
    for (i, a) in bytes.iter().enumerate() {
        if a.len() != t * n && (i < 2 || !a.is_empty()) {
            return Err(PyValueError::new_err("invalid mask length"));
        }
    }
    let [price, close, volume, split, dividend] = floats;
    let [buy, sell, delist, can_short] = bytes;
    let result = py.detach(|| -> Result<_, &'static str> {
        if validate {
            // The original per-field order: the first invalid field is reported.
            if validate::any_bad(price, Domain::PositiveOrNan) {
                return Err("exec_price: expected finite positive values or NaN");
            }
            if validate::any_bad(close, Domain::PositiveOrNan) {
                return Err("close: expected finite positive values or NaN");
            }
            if validate::any_nonbool(buy) {
                return Err("can_buy: expected boolean values 0 or 1");
            }
            if validate::any_nonbool(sell) {
                return Err("can_sell: expected boolean values 0 or 1");
            }
            if validate::any_bad(volume, Domain::NonnegativeOrNan) {
                return Err("volume: expected finite nonnegative values or NaN");
            }
            if validate::any_bad(split, Domain::Positive) {
                return Err("split: expected finite positive values");
            }
            if validate::any_bad(dividend, Domain::Nonnegative) {
                return Err("cash_div: expected finite nonnegative values");
            }
            if validate::any_nonbool(delist) {
                return Err("delist: expected boolean values 0 or 1");
            }
            if validate::any_nonbool(can_short) {
                return Err("can_short: expected boolean values 0 or 1");
            }
        }
        let first = validate::first_delist(delist, t, n);
        let (indptr, cols) = validate::action_index(split, dividend, &first, t, n);
        Ok((first, indptr, cols))
    });
    let (first, indptr, cols) = result.map_err(PyValueError::new_err)?;
    Ok((
        first.into_pyarray(py),
        indptr.into_pyarray(py),
        cols.into_pyarray(py),
    ))
}

/// Futures content checks in the original order; shapes are checked in Python.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn validate_futures(
    py: Python<'_>,
    kind: PyReadonlyArray1<'_, u8>,
    multiplier: PyReadonlyArray1<'_, f64>,
    expiry: PyReadonlyArray1<'_, i64>,
    settlement: PyReadonlyArray2<'_, f64>,
    split: PyReadonlyArray2<'_, f64>,
    dividend: PyReadonlyArray2<'_, f64>,
    delist: PyReadonlyArray2<'_, u8>,
    rates: Vec<(String, PyReadonlyArray2<'_, f64>)>,
) -> PyResult<()> {
    let shape = settlement.shape();
    let (t, n) = (shape[0], shape[1]);
    let kind = kind.as_slice()?;
    let multiplier = multiplier.as_slice()?;
    let expiry = expiry.as_slice()?;
    let settlement = settlement.as_slice()?;
    let (split, dividend, delist) = (split.as_slice()?, dividend.as_slice()?, delist.as_slice()?);
    let rates: Vec<(String, &[f64])> = rates
        .iter()
        .map(|(k, a)| Ok((k.clone(), a.as_slice()?)))
        .collect::<PyResult<_>>()?;
    if kind.len() != n || multiplier.len() != n || expiry.len() != n {
        return Err(PyValueError::new_err("invalid futures argument lengths"));
    }
    for a in [split, dividend] {
        if !a.is_empty() && a.len() != t * n {
            return Err(PyValueError::new_err("invalid panel length"));
        }
    }
    if !delist.is_empty() && delist.len() != t * n {
        return Err(PyValueError::new_err("invalid mask length"));
    }
    let message = py.detach(|| -> Option<String> {
        if validate::any_nonbool(kind) {
            return Some("is_future: expected boolean values 0 or 1".into());
        }
        let columns: Vec<usize> = (0..n).filter(|&i| kind[i] != 0).collect();
        if columns
            .iter()
            .any(|&i| !multiplier[i].is_finite() || multiplier[i] <= 0.0)
        {
            return Some("multiplier: futures columns require finite positive values".into());
        }
        if columns
            .iter()
            .any(|&i| expiry[i] < 0 || expiry[i] > t as i64)
        {
            return Some("expiry: futures columns require row indices in [0, T]".into());
        }
        if validate::any_in_columns(settlement, n, &columns, |x: f64| !x.is_finite() || x <= 0.0) {
            return Some("settlement: futures columns require finite positive prices".into());
        }
        if !split.is_empty() && validate::any_in_columns(split, n, &columns, |x: f64| x != 1.0) {
            return Some("split: futures columns cannot have corporate actions".into());
        }
        if !dividend.is_empty()
            && validate::any_in_columns(dividend, n, &columns, |x: f64| x != 0.0)
        {
            return Some("cash_div: futures columns cannot have corporate actions".into());
        }
        if !delist.is_empty() && validate::any_in_columns(delist, n, &columns, |x: u8| x != 0) {
            return Some("delist: futures columns use expiry, not delisting".into());
        }
        for (name, a) in &rates {
            if validate::any_bad(a, Domain::Nonnegative) {
                return Some(format!("{name}: expected finite nonnegative values"));
            }
        }
        None
    });
    match message {
        Some(m) => Err(PyValueError::new_err(m)),
        None => Ok(()),
    }
}

/// Sparse-plan checks; returns (sorted, has_negative, max_row_sum).
#[pyfunction]
fn check_plan(
    py: Python<'_>,
    days: PyReadonlyArray1<'_, i64>,
    indptr: PyReadonlyArray1<'_, i64>,
    cols: PyReadonlyArray1<'_, i64>,
    weights: PyReadonlyArray1<'_, f64>,
    validate: bool,
) -> PyResult<(bool, bool, f64)> {
    let (days, indptr, cols, weights) = (
        days.as_slice()?,
        indptr.as_slice()?,
        cols.as_slice()?,
        weights.as_slice()?,
    );
    let summary = py
        .detach(|| validate::check_plan(days, indptr, cols, weights, validate))
        .map_err(PyValueError::new_err)?;
    Ok((summary.sorted, summary.has_negative, summary.max_row_sum))
}

#[pymodule(name = "_core")]
fn panelbt_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_market, m)?)?;
    m.add_function(wrap_pyfunction!(validate_futures, m)?)?;
    m.add_function(wrap_pyfunction!(check_plan, m)?)?;
    m.add("VALIDATION_CHUNK", validate::CHUNK)?;
    m.add("BUILD_MARKER", panelbt_core::BUILD_MARKER)?;
    m.add("SOURCE_SHA256", env!("PANELBT_SOURCE_SHA256"))?;
    Ok(())
}
