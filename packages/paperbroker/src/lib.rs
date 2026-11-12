//! Borrow contiguous NumPy inputs without copying; keep the GIL during borrowing.
#![forbid(unsafe_code)]
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use paperbroker_core::{Config, Engine, Input, SimpleCosts, SimpleLots, WIDTH};
use pyo3::{
    exceptions::{PyOverflowError, PyValueError},
    prelude::*,
};

#[pyclass]
struct BrokerCore {
    engine: Engine,
}
#[pymethods]
impl BrokerCore {
    #[new]
    fn new(config: &str) -> PyResult<Self> {
        let config: Config = serde_json::from_str(config)
            .map_err(|_| PyValueError::new_err("invalid config JSON"))?;
        Ok(Self {
            engine: Engine::new(config).map_err(PyValueError::new_err)?,
        })
    }
    fn apply(&mut self, event: &str) -> PyResult<()> {
        let input: Input =
            serde_json::from_str(event).map_err(|_| PyValueError::new_err("invalid event JSON"))?;
        let before = self.engine.clone();
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.engine.apply(input))) {
            Ok(()) => Ok(()),
            Err(_) => {
                self.engine = before;
                Err(PyOverflowError::new_err(
                    "accounting overflow; input not applied",
                ))
            }
        }
    }
    #[pyo3(signature = (quotes, received_at=None))]
    fn feed(
        &mut self,
        quotes: PyReadonlyArray2<'_, i64>,
        received_at: Option<i64>,
    ) -> PyResult<()> {
        if quotes.shape()[1] != WIDTH {
            return Err(PyValueError::new_err("quotes must have 18 columns"));
        }
        let rows = quotes.as_slice()?;
        // No accounting is performed without open orders; avoid cloning a whole
        // market universe on the common observation-only path.
        let before = self
            .engine
            .state
            .orders
            .iter()
            .any(|o| o.status == "open")
            .then(|| self.engine.clone());
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.engine.feed_at(rows, received_at)
        })) {
            Ok(result) => result.map_err(PyValueError::new_err),
            Err(_) => {
                if let Some(before) = before {
                    self.engine = before;
                }
                Err(PyOverflowError::new_err(
                    "accounting overflow; input not applied",
                ))
            }
        }
    }
    #[staticmethod]
    fn validate_quotes(quotes: PyReadonlyArray2<'_, i64>, symbols: usize) -> PyResult<()> {
        if quotes.shape()[1] != WIDTH {
            return Err(PyValueError::new_err("quotes must have 18 columns"));
        }
        Engine::<SimpleCosts, SimpleLots>::validate_quotes(quotes.as_slice()?, symbols)
            .map_err(PyValueError::new_err)
    }
    fn dump(&self) -> String {
        self.engine.dump()
    }
    fn account(&self) -> String {
        self.engine.account().to_string()
    }
    fn events(&mut self) -> String {
        serde_json::to_string(&self.engine.drain()).expect("integer events")
    }
    #[staticmethod]
    fn restore(state: &str) -> PyResult<Self> {
        Ok(Self {
            engine: Engine::restore(state).map_err(PyValueError::new_err)?,
        })
    }
}
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BrokerCore>()?;
    Ok(())
}
