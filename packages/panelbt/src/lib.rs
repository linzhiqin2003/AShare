//! Python 数组保持只读借用；模拟期间释放 GIL。
#![forbid(unsafe_code)]
use numpy::{
    IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods, ndarray::Array2,
};
use panelbt_core::{Market, Settings, Targets};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

type Plan<'a> = (
    PyReadonlyArray1<'a, i64>,
    PyReadonlyArray1<'a, i64>,
    PyReadonlyArray1<'a, i64>,
    PyReadonlyArray1<'a, f64>,
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
    threads: usize,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    if floats.len() != 5
        || masks.len() != 3
        || vectors.len() != 4
        || scalars.len() != 8
        || flags.len() != 2
    {
        return Err(PyValueError::new_err("内部绑定参数长度不正确"));
    }
    let shape = floats[0].shape();
    let (t, n) = (shape[0], shape[1]);
    let fs: Vec<&[f64]> = floats
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<_, _>>()?;
    let ms: Vec<&[bool]> = masks
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<_, _>>()?;
    let vs: Vec<&[f64]> = vectors
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<_, _>>()?;
    for (i, a) in fs.iter().enumerate() {
        if a.len() != t * n && (i < 2 || !a.is_empty()) {
            return Err(PyValueError::new_err("行情面板形状不匹配"));
        }
    }
    for (i, a) in ms.iter().enumerate() {
        if a.len() != t * n && (i < 2 || !a.is_empty()) {
            return Err(PyValueError::new_err("掩码面板形状不匹配"));
        }
    }
    if vs[0].len() != t || vs[1].len() != t || vs[2].len() != n || vs[3].len() != n {
        return Err(PyValueError::new_err("费率/申报单位长度不匹配"));
    }
    let m = Market {
        t,
        n,
        price: fs[0],
        close: fs[1],
        buy: ms[0],
        sell: ms[1],
        volume: (!fs[2].is_empty()).then_some(fs[2]),
        split: (!fs[3].is_empty()).then_some(fs[3]),
        dividend: (!fs[4].is_empty()).then_some(fs[4]),
        delist: (!ms[2].is_empty()).then_some(ms[2]),
    };
    let s = Settings {
        commission: scalars[0],
        min_commission: scalars[1],
        transfer: scalars[2],
        slippage: scalars[3],
        initial: scalars[4],
        participation: scalars[5],
        recovery: scalars[6],
        buffer: scalars[7],
        tax_sell: vs[0],
        tax_buy: vs[1],
        min_qty: vs[2],
        step: vs[3],
        carry: flags[0],
        record: flags[1],
    };
    let targets: Vec<Targets<'_>> = plans
        .iter()
        .map(|p| {
            Ok(Targets {
                days: p.0.as_slice()?,
                indptr: p.1.as_slice()?,
                cols: p.2.as_slice()?,
                weights: p.3.as_slice()?,
            })
        })
        .collect::<PyResult<_>>()?;
    let outputs = py
        .detach(|| panelbt_core::run_batch(&m, &targets, &s, threads))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    outputs
        .into_iter()
        .map(|o| {
            let d = PyDict::new(py);
            d.set_item(
                "daily",
                Array2::from_shape_vec((9, t), o.daily)
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
            Ok(d)
        })
        .collect()
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add("BUILD_MARKER", panelbt_core::BUILD_MARKER)?;
    Ok(())
}
