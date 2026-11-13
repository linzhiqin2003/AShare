//! Borrowed mixed-precision price inputs for streaming grouped weights.
use numpy::{IntoPyArray, PyReadonlyArray2, PyUntypedArrayMethods, ndarray::Array2};
use panelbt_core::group_weights::{Input, Labels, Numbers};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

enum Prices<'py> {
    F32(PyReadonlyArray2<'py, f32>),
    F64(PyReadonlyArray2<'py, f64>),
}
enum Groups<'py> {
    U32(PyReadonlyArray2<'py, u32>),
    I64(PyReadonlyArray2<'py, i64>),
}
impl<'py> Groups<'py> {
    fn extract(x: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(x) = x.extract::<PyReadonlyArray2<'py, u32>>() {
            Ok(Self::U32(x))
        } else {
            x.extract::<PyReadonlyArray2<'py, i64>>().map(Self::I64)
        }
    }
    fn shape(&self) -> &[usize] {
        match self {
            Self::U32(x) => x.shape(),
            Self::I64(x) => x.shape(),
        }
    }
    fn values(&self) -> PyResult<Labels<'_>> {
        Ok(match self {
            Self::U32(x) => Labels::U32(x.as_slice()?),
            Self::I64(x) => Labels::I64(x.as_slice()?),
        })
    }
}
impl<'py> Prices<'py> {
    fn extract(x: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(x) = x.extract::<PyReadonlyArray2<'py, f64>>() {
            Ok(Self::F64(x))
        } else {
            x.extract::<PyReadonlyArray2<'py, f32>>().map(Self::F32)
        }
    }
    fn shape(&self) -> &[usize] {
        match self {
            Self::F32(x) => x.shape(),
            Self::F64(x) => x.shape(),
        }
    }
    fn values(&self) -> PyResult<Numbers<'_>> {
        Ok(match self {
            Self::F32(x) => Numbers::F32(x.as_slice()?),
            Self::F64(x) => Numbers::F64(x.as_slice()?),
        })
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn run_group_weights<'py>(
    py: Python<'py>,
    factor: PyReadonlyArray2<'py, f64>,
    logcap: PyReadonlyArray2<'py, f64>,
    industry: Bound<'py, PyAny>,
    vectors: Vec<PyReadonlyArray2<'py, f64>>,
    universes: Vec<PyReadonlyArray2<'py, bool>>,
    price: Bound<'py, PyAny>,
    close: Bound<'py, PyAny>,
    adjustment: Bound<'py, PyAny>,
    buy: PyReadonlyArray2<'py, bool>,
    sell: PyReadonlyArray2<'py, bool>,
    options: (
        usize,
        usize,
        usize,
        usize,
        usize,
        bool,
        bool,
        bool,
        usize,
        bool,
    ),
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let (
        signal_start,
        execution_start,
        groups,
        neutralize,
        decay,
        weighted,
        profile,
        industry_zero_missing,
        rebalance,
        record_holdings,
    ) = options;
    let industry = Groups::extract(&industry)?;
    let (price, close, adjustment) = (
        Prices::extract(&price)?,
        Prices::extract(&close)?,
        Prices::extract(&adjustment)?,
    );
    let shape = price.shape();
    let (rows, cols) = (shape[0], shape[1]);
    if signal_start > rows
        || execution_start < signal_start
        || groups == 0
        || neutralize > 3
        || rebalance == 0
        || cols > u32::MAX as usize
        || close.shape() != shape
        || adjustment.shape() != shape
    {
        return Err(PyValueError::new_err(
            "invalid grouped price inputs or options",
        ));
    }
    let active = [rows - signal_start, cols];
    if factor.shape() != active
        || logcap.shape() != active
        || ((neutralize == 1 || neutralize == 3) && industry.shape() != active)
        || vectors.iter().any(|v| v.shape() != active)
        || universes.iter().any(|v| v.shape() != active)
        || (!buy.is_empty() && buy.shape() != active)
        || (!sell.is_empty() && sell.shape() != active)
    {
        return Err(PyValueError::new_err("grouped array shape mismatch"));
    }
    let input = Input {
        rows,
        cols,
        signal_start,
        execution_start,
        factor: factor.as_slice()?,
        logcap: logcap.as_slice()?,
        industry: industry.values()?,
        industry_zero_missing,
        vectors: vectors
            .iter()
            .map(|v| v.as_slice())
            .collect::<Result<_, _>>()?,
        universes: universes
            .iter()
            .map(|v| v.as_slice())
            .collect::<Result<_, _>>()?,
        price: price.values()?,
        close: close.values()?,
        adjustment: adjustment.values()?,
        buy: buy.as_slice()?,
        sell: sell.as_slice()?,
        groups,
        neutralize,
        decay,
        weighted,
        profile,
        rebalance,
        record_holdings,
    };
    let dot = crate::numpy_dot::NumpyDot::load(py)?;
    let output = py.detach(|| panelbt_core::group_weights::run(&input, |x, y| dot.dot(x, y)));
    let books = groups + 1 + usize::from(weighted);
    output
        .into_iter()
        .map(|o| {
            let result = PyDict::new(py);
            result.set_item(
                "pnl",
                Array2::from_shape_vec((books, active[0]), o.pnl)
                    .unwrap()
                    .into_pyarray(py),
            )?;
            result.set_item(
                "turnover",
                Array2::from_shape_vec((books, active[0]), o.turnover)
                    .unwrap()
                    .into_pyarray(py),
            )?;
            result.set_item(
                "exposure",
                Array2::from_shape_vec((4, active[0]), o.exposure)
                    .unwrap()
                    .into_pyarray(py),
            )?;
            result.set_item("first_live", o.first_live)?;
            result.set_item("bad_rows", o.bad_rows)?;
            result.set_item("stages", o.stages)?;
            if let Some(snapshot) = o.holdings {
                let held = PyDict::new(py);
                held.set_item("days", snapshot.days.into_pyarray(py))?;
                held.set_item("offsets", snapshot.offsets.into_pyarray(py))?;
                held.set_item("columns", snapshot.columns.into_pyarray(py))?;
                held.set_item("weights", snapshot.weights.into_pyarray(py))?;
                result.set_item("holdings", held)?;
            }
            Ok(result)
        })
        .collect()
}
