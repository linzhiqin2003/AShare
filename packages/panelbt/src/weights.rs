//! Checked array-only interface for pure weight accounting.
use numpy::{IntoPyArray, PyReadonlyArray2, PyUntypedArrayMethods, ndarray::Array2};
use panelbt_core::weights::{Market, Plan};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

type Target<'py> = (
    PyReadonlyArray2<'py, f64>,
    PyReadonlyArray2<'py, i64>,
    i64,
    i64,
    f64,
    f64,
    bool,
);

#[pyfunction]
pub fn run_weights_batch<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
    buy: PyReadonlyArray2<'py, bool>,
    sell: PyReadonlyArray2<'py, bool>,
    targets: Vec<Target<'py>>,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let shape = returns.shape();
    for flags in [&buy, &sell] {
        if !flags.is_empty() && flags.shape() != shape {
            return Err(PyValueError::new_err("permission shape mismatch"));
        }
    }
    let market = Market {
        rows: shape[0],
        cols: shape[1],
        returns: returns.as_slice()?,
        buy: buy.as_slice()?,
        sell: sell.as_slice()?,
    };
    let mut plans = Vec::with_capacity(targets.len());
    for (dense, groups, long, short, lb, sb, keep) in &targets {
        let plan = if *long < 0 {
            if dense.shape() != shape {
                return Err(PyValueError::new_err("target shape mismatch"));
            }
            Plan::Dense(dense.as_slice()?)
        } else {
            if groups.shape() != shape
                || *short < -1
                || !lb.is_finite()
                || !sb.is_finite()
                || *lb < 0.0
                || *sb < 0.0
            {
                return Err(PyValueError::new_err("invalid grouped weight targets"));
            }
            Plan::Groups {
                labels: groups.as_slice()?,
                long: *long,
                short: *short,
                long_budget: *lb,
                short_budget: *sb,
            }
        };
        plans.push((plan, *keep));
    }
    let dot = crate::numpy_dot::NumpyDot::load(py)?;
    let outputs =
        py.detach(|| panelbt_core::weights::run_batch(&market, &plans, |x, y| dot.dot(x, y)));
    outputs
        .into_iter()
        .map(|out| {
            let result = PyDict::new(py);
            result.set_item("pnl", out.pnl.into_pyarray(py))?;
            result.set_item("turnover", out.turnover.into_pyarray(py))?;
            if let Some(weights) = out.weights {
                result.set_item(
                    "weights",
                    Array2::from_shape_vec((shape[0], shape[1]), weights)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                        .into_pyarray(py),
                )?;
            } else {
                result.set_item("weights", py.None())?;
            }
            Ok(result)
        })
        .collect()
}
