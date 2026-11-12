//! Read-only NumPy borrowing; release the GIL during independent simulations.
#![forbid(unsafe_code)]
use numpy::{
    IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods, ndarray::Array2,
};
use panelbt_core::{Futures, Market, Settings, Targets};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
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
    masks: Vec<PyReadonlyArray2<'py, bool>>,
    plans: Vec<Plan<'py>>,
    vectors: Vec<PyReadonlyArray1<'py, f64>>,
    scalars: Vec<f64>,
    flags: Vec<bool>,
    first_delist: PyReadonlyArray1<'py, i64>,
    actions: (PyReadonlyArray1<'py, i64>, PyReadonlyArray1<'py, i64>),
    future: (
        PyReadonlyArray1<'py, bool>,
        PyReadonlyArray1<'py, f64>,
        PyReadonlyArray1<'py, i64>,
        Vec<PyReadonlyArray2<'py, f64>>,
    ),
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
    let ms: Vec<&[bool]> = masks
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
#[pymodule(name = "_core")]
fn panelbt_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add("BUILD_MARKER", panelbt_core::BUILD_MARKER)?;
    m.add("SOURCE_SHA256", env!("PANELBT_SOURCE_SHA256"))?;
    Ok(())
}
