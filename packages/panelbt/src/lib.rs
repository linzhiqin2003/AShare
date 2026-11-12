//! Read-only NumPy borrowing; release the GIL during independent simulations.
#![forbid(unsafe_code)]
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods, ndarray::Array2,
};
use panelbt_core::{
    Futures, Market, Output, Row, Settings, Stepper, Targets, Vectors,
    rules::Rules,
    validate::{self, Context, Domain},
};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyBytes, PyDict},
};
use rayon::prelude::*;
type Plan<'a> = (
    PyReadonlyArray1<'a, i64>,
    PyReadonlyArray1<'a, i64>,
    PyReadonlyArray1<'a, i64>,
    PyReadonlyArray1<'a, f64>,
    bool,
    PyReadonlyArray1<'a, f64>,
);
/// Rule terms: (max_hold, timing, both_target, cooldown, dividend_adjusts,
/// sell_lock, until_next, limit_touch).
type RuleTerms = (i64, u8, bool, i64, bool, bool, bool, bool);

/// Rules from six flattened arrays (open, high, low: T×N; stop, take, trail:
/// empty, 1, N or T×N) and the scalar terms.
fn rules<'a>(
    t: usize,
    n: usize,
    arrays: &[&'a [f64]],
    terms: RuleTerms,
    futures: bool,
) -> PyResult<Rules<'a>> {
    if futures {
        return Err(PyValueError::new_err(
            "rules: futures columns are not supported with exit rules or limit targets",
        ));
    }
    if arrays.len() != 6
        || arrays[..3].iter().any(|a| a.len() != t * n)
        || arrays[3..]
            .iter()
            .any(|a| ![0, 1, n, t * n].contains(&a.len()))
        || terms.1 > 2
    {
        return Err(PyValueError::new_err("invalid rule argument lengths"));
    }
    Ok(Rules {
        open: arrays[0],
        high: arrays[1],
        low: arrays[2],
        stop: arrays[3],
        take: arrays[4],
        trail: arrays[5],
        max_hold: terms.0,
        timing: terms.1,
        both_target: terms.2,
        cooldown: terms.3,
        div_adjust: terms.4,
        sell_lock: terms.5,
        until_next: terms.6,
        limit_touch: terms.7,
    })
}
/// Length and index checks protecting every slice the kernel reads, then the
/// borrowed market and settings. Shared by `run` and `Simulator`.
#[allow(clippy::too_many_arguments)]
fn assemble<'a>(
    t: usize,
    n: usize,
    fs: &[&'a [f64]],
    ms: &[&'a [u8]],
    vs: Vec<&'a [f64]>,
    scalars: &'a [f64],
    flags: &'a [bool],
    first: &'a [i64],
    action_indptr: &'a [i64],
    action_cols: &'a [i64],
    kind: &'a [u8],
    multiplier: &'a [f64],
    expiry: &'a [i64],
    panels: &[&'a [f64]],
) -> PyResult<(Market<'a>, Settings<'a>)> {
    if fs.len() != 8 || ms.len() != 4 || vs.len() != 16 || scalars.len() != 10 || flags.len() != 7 {
        return Err(PyValueError::new_err("invalid internal argument lengths"));
    }
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
    if first.len() != n {
        return Err(PyValueError::new_err("invalid delist length"));
    }
    if action_indptr.len() != t + 1
        || action_indptr.first() != Some(&0)
        || action_indptr.last().copied() != Some(action_cols.len() as i64)
        || action_indptr.windows(2).any(|w| w[0] > w[1])
        || action_cols.iter().any(|&i| i < 0 || i as usize >= n)
    {
        return Err(PyValueError::new_err("invalid corporate-action index"));
    }
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
        scalars,
        flags,
        slippage: fs[5],
        impact: fs[6],
        borrow: fs[7],
        rules: None,
    };
    Ok((m, s))
}
fn output_dict<'py>(
    py: Python<'py>,
    o: Output,
    t: usize,
    n: usize,
) -> PyResult<Bound<'py, PyDict>> {
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
    d.set_item("reasons", o.reasons)?;
    d.set_item("rule_counts", o.rule_counts)?;
    Ok(d)
}
#[pyfunction]
#[pyo3(signature = (floats, masks, plans, vectors, scalars, flags, first_delist, actions, future, validated, context, threads, rule_spec=None))]
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
    rule_spec: Option<(Vec<PyReadonlyArray1<'py, f64>>, RuleTerms)>,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    if floats.is_empty() {
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
    let panels: Vec<&[f64]> = future
        .3
        .iter()
        .map(|x| x.as_slice())
        .collect::<Result<_, _>>()?;
    let kind = future.0.as_slice()?;
    let rule_arrays: Vec<&[f64]> = match &rule_spec {
        Some((a, _)) => a.iter().map(|x| x.as_slice()).collect::<Result<_, _>>()?,
        None => Vec::new(),
    };
    let (m, mut s) = assemble(
        t,
        n,
        &fs,
        &ms,
        vs,
        &scalars,
        &flags,
        first_delist.as_slice()?,
        actions.0.as_slice()?,
        actions.1.as_slice()?,
        kind,
        future.1.as_slice()?,
        future.2.as_slice()?,
        &panels,
    )?;
    if let Some((_, terms)) = rule_spec {
        s.rules = Some(rules(t, n, &rule_arrays, terms, m.futures.is_some())?);
    }
    let targets: Vec<Targets<'_>> = plans
        .iter()
        .map(|p| {
            Ok(Targets {
                days: p.0.as_slice()?,
                indptr: p.1.as_slice()?,
                cols: p.2.as_slice()?,
                weights: p.3.as_slice()?,
                limits: p.5.as_slice()?,
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
            || (!plan.limits.is_empty() && plan.limits.len() != plan.weights.len())
            || (!plan.limits.is_empty() && s.rules.is_none())
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
        .map(|o| output_dict(py, o, t, n))
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

/// Caller-visible per-security state arrays of a stepped simulation.
struct Arrays {
    shares: Py<PyArray1<f64>>,
    last: Py<PyArray1<f64>>,
    basis: Py<PyArray1<f64>>,
    cost: Py<PyArray1<f64>>,
    peak: Py<PyArray1<f64>>,
    entry: Py<PyArray1<i64>>,
}

/// One simulation advanced a row at a time; identical arithmetic to `run`.
///
/// Market panels are borrowed again on every step (no copy); the small per-day
/// and per-security vectors are copied once at construction. `step` releases
/// the GIL, so different simulators can advance in different threads.
#[pyclass(module = "panelbt._core")]
struct Simulator {
    floats: Vec<Py<PyArray2<f64>>>,
    masks: Vec<Py<PyArray2<u8>>>,
    vectors: Vec<Vec<f64>>,
    scalars: Vec<f64>,
    flags: Vec<bool>,
    first: Vec<i64>,
    indptr: Vec<i64>,
    cols: Vec<i64>,
    kind: Vec<u8>,
    multiplier: Vec<f64>,
    expiry: Vec<i64>,
    panels: Vec<Py<PyArray2<f64>>>,
    t: usize,
    n: usize,
    shares_kind: bool,
    validate: bool,
    context: (bool, bool),
    future_mask: Vec<bool>,
    rule_arrays: Vec<Py<PyArray1<f64>>>,
    rule_terms: Option<RuleTerms>,
    stepper: Stepper,
    arrays: Arrays,
}

type FutureSpec<'py> = (
    PyReadonlyArray1<'py, u8>,
    PyReadonlyArray1<'py, f64>,
    PyReadonlyArray1<'py, i64>,
    Vec<Bound<'py, PyArray2<f64>>>,
);

/// Borrow every market panel of `sim` read-only and call `f` with the market,
/// settings and the remaining mutable parts.
macro_rules! with_market {
    ($py:expr, $sim:expr, |$m:ident, $s:ident, $stepper:ident, $arrays:ident| $body:expr) => {{
        let Simulator {
            floats,
            masks,
            vectors,
            scalars,
            flags,
            first,
            indptr,
            cols,
            kind,
            multiplier,
            expiry,
            panels,
            t,
            n,
            rule_arrays,
            rule_terms,
            stepper: $stepper,
            arrays: $arrays,
            ..
        } = $sim;
        let fr: Vec<PyReadonlyArray2<'_, f64>> = floats
            .iter()
            .map(|a| a.bind($py).try_readonly())
            .collect::<Result<_, _>>()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mr: Vec<PyReadonlyArray2<'_, u8>> = masks
            .iter()
            .map(|a| a.bind($py).try_readonly())
            .collect::<Result<_, _>>()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let pr: Vec<PyReadonlyArray2<'_, f64>> = panels
            .iter()
            .map(|a| a.bind($py).try_readonly())
            .collect::<Result<_, _>>()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let fs: Vec<&[f64]> = fr.iter().map(|x| x.as_slice()).collect::<Result<_, _>>()?;
        let ms: Vec<&[u8]> = mr.iter().map(|x| x.as_slice()).collect::<Result<_, _>>()?;
        let ps: Vec<&[f64]> = pr.iter().map(|x| x.as_slice()).collect::<Result<_, _>>()?;
        let vs: Vec<&[f64]> = vectors.iter().map(|v| v.as_slice()).collect();
        let rr: Vec<PyReadonlyArray1<'_, f64>> = rule_arrays
            .iter()
            .map(|a| a.bind($py).try_readonly())
            .collect::<Result<_, _>>()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let rs: Vec<&[f64]> = rr.iter().map(|x| x.as_slice()).collect::<Result<_, _>>()?;
        #[allow(unused_mut)]
        let ($m, mut $s) = assemble(
            *t, *n, &fs, &ms, vs, scalars, flags, first, indptr, cols, kind, multiplier, expiry,
            &ps,
        )?;
        if let Some(terms) = *rule_terms {
            $s.rules = Some(rules(*t, *n, &rs, terms, $m.futures.is_some())?);
        }
        $body
    }};
}

macro_rules! with_vectors {
    ($py:expr, $arrays:expr, |$v:ident| $body:expr) => {{
        let mut a = $arrays
            .shares
            .bind($py)
            .try_readwrite()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut b = $arrays
            .last
            .bind($py)
            .try_readwrite()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut c = $arrays
            .basis
            .bind($py)
            .try_readwrite()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut d = $arrays
            .cost
            .bind($py)
            .try_readwrite()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut e = $arrays
            .peak
            .bind($py)
            .try_readwrite()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut f = $arrays
            .entry
            .bind($py)
            .try_readwrite()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let $v = Vectors {
            shares: a.as_slice_mut()?,
            last: b.as_slice_mut()?,
            basis: c.as_slice_mut()?,
            cost: d.as_slice_mut()?,
            peak: e.as_slice_mut()?,
            entry: f.as_slice_mut()?,
        };
        $body
    }};
}

impl Simulator {
    /// Fingerprint of the inputs a snapshot must be restored against.
    fn header(&self, py: Python<'_>) -> PyResult<Vec<u64>> {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut mix = |x: u64| {
            for b in x.to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x100_0000_01b3);
            }
        };
        for v in &self.vectors {
            for x in v {
                mix(x.to_bits());
            }
        }
        for x in &self.scalars {
            mix(x.to_bits());
        }
        for &f in &self.flags {
            mix(u64::from(f));
        }
        if let Some(r) = self.rule_terms {
            for x in [
                r.0 as u64,
                u64::from(r.1),
                u64::from(r.2),
                r.3 as u64,
                u64::from(r.4),
                u64::from(r.5),
                u64::from(r.6),
                u64::from(r.7),
            ] {
                mix(x);
            }
            // Every rule percentage cell (stop, take, trail), with its length;
            // the open/high/low panels are market data and not fingerprinted.
            for a in self.rule_arrays.iter().skip(3) {
                let a = a
                    .bind(py)
                    .try_readonly()
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let a = a.as_slice()?;
                mix(a.len() as u64);
                for x in a {
                    mix(x.to_bits());
                }
            }
        } else {
            mix(u64::MAX);
        }
        Ok(vec![
            2,
            self.t as u64,
            self.n as u64,
            u64::from(!self.kind.is_empty()),
            u64::from(self.shares_kind),
            h,
        ])
    }
}

#[pymethods]
impl Simulator {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new<'py>(
        py: Python<'py>,
        floats: Vec<Bound<'py, PyArray2<f64>>>,
        masks: Vec<Bound<'py, PyArray2<u8>>>,
        vectors: Vec<PyReadonlyArray1<'py, f64>>,
        scalars: Vec<f64>,
        flags: Vec<bool>,
        first_delist: PyReadonlyArray1<'py, i64>,
        actions: (PyReadonlyArray1<'py, i64>, PyReadonlyArray1<'py, i64>),
        future: FutureSpec<'py>,
        shares: bool,
        context: (bool, bool),
        validate: bool,
        rule_spec: Option<(Vec<Bound<'py, PyArray1<f64>>>, RuleTerms)>,
    ) -> PyResult<Self> {
        if floats.is_empty() {
            return Err(PyValueError::new_err("invalid internal argument lengths"));
        }
        let shape = floats[0].shape();
        let (t, n) = (shape[0], shape[1]);
        let kind = future.0.as_slice()?.to_vec();
        let mut sim = Simulator {
            floats: floats.iter().map(|a| a.clone().unbind()).collect(),
            masks: masks.iter().map(|a| a.clone().unbind()).collect(),
            vectors: vectors
                .iter()
                .map(|v| Ok(v.as_slice()?.to_vec()))
                .collect::<PyResult<_>>()?,
            scalars,
            flags,
            first: first_delist.as_slice()?.to_vec(),
            indptr: actions.0.as_slice()?.to_vec(),
            cols: actions.1.as_slice()?.to_vec(),
            future_mask: kind.iter().map(|&x| x != 0).collect(),
            kind,
            multiplier: future.1.as_slice()?.to_vec(),
            expiry: future.2.as_slice()?.to_vec(),
            panels: future.3.iter().map(|a| a.clone().unbind()).collect(),
            t,
            n,
            shares_kind: shares,
            validate,
            context,
            rule_arrays: rule_spec
                .as_ref()
                .map(|(a, _)| a.iter().map(|x| x.clone().unbind()).collect())
                .unwrap_or_default(),
            rule_terms: rule_spec.as_ref().map(|x| x.1),
            stepper: Stepper::new_empty(),
            arrays: Arrays {
                shares: PyArray1::zeros(py, n, false).unbind(),
                last: PyArray1::zeros(py, n, false).unbind(),
                basis: PyArray1::zeros(py, n, false).unbind(),
                cost: PyArray1::zeros(py, n, false).unbind(),
                peak: PyArray1::zeros(py, n, false).unbind(),
                entry: PyArray1::from_vec(py, vec![-1i64; n]).unbind(),
            },
        };
        with_market!(py, &mut sim, |m, s, stepper, arrays| {
            *stepper = Stepper::new(&m, &s);
            with_vectors!(py, arrays, |v| stepper.begin(v, &m, &s));
        });
        Ok(sim)
    }

    /// Trade and close the current row with an optional target row, then
    /// open the next row. Returns that day's daily metric values.
    #[pyo3(signature = (cols=None, weights=None, limits=None))]
    fn step(
        &mut self,
        py: Python<'_>,
        cols: Option<PyReadonlyArray1<'_, i64>>,
        weights: Option<PyReadonlyArray1<'_, f64>>,
        limits: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<Vec<f64>> {
        let day = self.stepper.day;
        if day >= self.t {
            return Err(PyValueError::new_err(
                "simulation finished: every row has been stepped",
            ));
        }
        let (c, w) = match (&cols, &weights) {
            (Some(c), Some(w)) => (Some(c.as_slice()?), Some(w.as_slice()?)),
            (None, None) => (None, None),
            _ => {
                return Err(PyValueError::new_err(
                    "cols and weights must be given together",
                ));
            }
        };
        if let (Some(c), Some(w)) = (c, w) {
            if c.len() != w.len() {
                return Err(PyValueError::new_err(
                    "cols/weights: expected equal lengths",
                ));
            }
            if c.iter().any(|&x| x < 0 || x as usize >= self.n) {
                return Err(PyValueError::new_err(format!(
                    "cols: expected index < N={}",
                    self.n
                )));
            }
            if c.windows(2).any(|x| x[0] >= x[1]) {
                return Err(PyValueError::new_err(
                    "cols: expected sorted unique columns",
                ));
            }
            if self.validate && validate::any_bad(w, Domain::Finite) {
                return Err(PyValueError::new_err("weights: expected finite targets"));
            }
            let ctx = Context {
                t: self.t,
                n: self.n,
                future: &self.future_mask,
                allow_short: self.context.0,
                cash_weights: self.context.1,
            };
            validate::check_context(
                &[day as i64],
                &[0, c.len() as i64],
                c,
                w,
                self.shares_kind,
                self.validate,
                &ctx,
            )
            .map_err(PyValueError::new_err)?;
        }
        let lim: &[f64] = match &limits {
            Some(x) => x.as_slice()?,
            None => &[],
        };
        if !lim.is_empty() {
            if self.rule_terms.is_none() {
                return Err(PyValueError::new_err(
                    "limits: limit targets require Config(rules=Rules(...))",
                ));
            }
            if c.is_none_or(|c| c.len() != lim.len()) {
                return Err(PyValueError::new_err(
                    "limits: expected one limit per target entry",
                ));
            }
            if lim.iter().any(|&x| x.is_infinite() || x <= 0.0) {
                return Err(PyValueError::new_err(
                    "limits: expected positive limit prices or NaN",
                ));
            }
        }
        let row = c.zip(w).map(|(cols, weights)| Row {
            cols,
            weights,
            limits: lim,
        });
        let shares = self.shares_kind;
        let t = self.t;
        with_market!(py, self, |m, s, stepper, arrays| {
            with_vectors!(py, arrays, |v| py
                .detach(|| stepper.step(v, row, shares, &m, &s)));
            let d = &stepper.core.out.daily;
            Ok((0..panelbt_core::DAILY).map(|k| d[k * t + day]).collect())
        })
    }

    /// (next row, cash, last-close equity, margin call due, bankrupt).
    fn scalars(&self) -> (usize, f64, f64, bool, bool) {
        let c = &self.stepper.core;
        (
            self.stepper.day,
            self.stepper.cash,
            c.equity(),
            c.margin_call(),
            c.bankrupt(),
        )
    }

    /// Carried blocked orders as (columns, targets) arrays.
    fn pending<'py>(
        &self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>) {
        let p = self.stepper.core.pending();
        (
            p.iter()
                .map(|x| x.0 as i64)
                .collect::<Vec<_>>()
                .into_pyarray(py),
            p.iter().map(|x| x.1).collect::<Vec<_>>().into_pyarray(py),
        )
    }

    /// The live state arrays: shares, last, futures basis, cost, peak, entry.
    #[allow(clippy::type_complexity)]
    fn arrays<'py>(
        &self,
        py: Python<'py>,
    ) -> (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<i64>>,
    ) {
        let a = &self.arrays;
        (
            a.shares.bind(py).clone(),
            a.last.bind(py).clone(),
            a.basis.bind(py).clone(),
            a.cost.bind(py).clone(),
            a.peak.bind(py).clone(),
            a.entry.bind(py).clone(),
        )
    }

    fn result<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        if self.stepper.day < self.t {
            return Err(PyValueError::new_err(format!(
                "simulation unfinished: {} of {} rows stepped",
                self.stepper.day, self.t
            )));
        }
        let shares = self
            .arrays
            .shares
            .bind(py)
            .try_readonly()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let last = self
            .arrays
            .last
            .bind(py)
            .try_readonly()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let out = self.stepper.result(
            shares.as_slice()?,
            last.as_slice()?,
            self.rule_terms.is_some(),
        );
        output_dict(py, out, self.t, self.n)
    }

    fn snapshot<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let header = self.header(py)?;
        let stepper = &self.stepper;
        let blob = with_vectors!(py, self.arrays, |v| stepper.encode(&header, &v));
        Ok(PyBytes::new(py, &blob))
    }

    fn load(&mut self, py: Python<'_>, blob: &[u8]) -> PyResult<()> {
        let header = self.header(py)?;
        let mut fresh = self.stepper.clone();
        with_vectors!(py, self.arrays, |v| fresh.decode(blob, &header, v))
            .map_err(PyValueError::new_err)?;
        self.stepper = fresh;
        Ok(())
    }
}

#[pymodule(name = "_core")]
fn panelbt_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_market, m)?)?;
    m.add_function(wrap_pyfunction!(validate_futures, m)?)?;
    m.add_function(wrap_pyfunction!(check_plan, m)?)?;
    m.add_class::<Simulator>()?;
    m.add("VALIDATION_CHUNK", validate::CHUNK)?;
    m.add("BUILD_MARKER", panelbt_core::BUILD_MARKER)?;
    m.add("SOURCE_SHA256", env!("PANELBT_SOURCE_SHA256"))?;
    Ok(())
}
