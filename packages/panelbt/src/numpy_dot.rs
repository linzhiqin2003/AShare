//! Small audited bridge to the installed NumPy float64 dot kernel.
//! NumPy >= 2.1 is a package requirement. No user-provided function pointers.
use numpy::{PyArrayDescrMethods, dtype, npyffi};
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyCapsule};
use std::ffi::c_void;

type Dot =
    unsafe extern "C" fn(*mut c_void, isize, *mut c_void, isize, *mut c_void, isize, *mut c_void);

pub struct NumpyDot(Dot);

impl NumpyDot {
    pub fn load(py: Python<'_>) -> PyResult<Self> {
        let capsule = py
            .import("numpy._core._multiarray_umath")?
            .getattr("_ARRAY_API")?
            .cast_into::<PyCapsule>()?;
        let pointer = capsule.pointer_checked(None)?;
        let dt = dtype::<f64>(py);
        // SAFETY: the NumPy 2.x array API defines slot 365 with this signature
        // (numpy/npy_2_compat.h: PyDataType_GetArrFuncs). Check the runtime
        // feature version before indexing. dt is NumPy's built-in native f64
        // descriptor, kept alive throughout the call under the GIL.
        let dot = unsafe {
            if npyffi::PY_ARRAY_API.PyArray_GetNDArrayCFeatureVersion(py) < 0x00000012 {
                return Err(PyRuntimeError::new_err("weight mode requires NumPy >= 2.0"));
            }
            let get: unsafe extern "C" fn(
                *const npyffi::PyArray_Descr,
            ) -> *mut npyffi::PyArray_ArrFuncs =
                std::mem::transmute(*pointer.as_ptr().cast::<*const c_void>().add(365));
            let funcs = get(dt.as_dtype_ptr());
            if funcs.is_null() {
                return Err(PyRuntimeError::new_err("missing NumPy float64 functions"));
            }
            (*funcs)
                .dotfunc
                .ok_or_else(|| PyRuntimeError::new_err("missing NumPy float64 dot"))?
        };
        Ok(Self(dot))
    }

    pub fn dot(&self, x: &[f64], y: &[f64]) -> f64 {
        assert_eq!(x.len(), y.len());
        // np.dot's BLAS dispatcher treats a one-element vector as scalar
        // multiplication, without the +0 identity in DOUBLE_dot (-0 matters).
        if x.len() == 1 {
            return x[0] * y[0];
        }
        let mut out = 0.0;
        // SAFETY: equally sized contiguous, aligned f64 slices are readable for
        // n elements, output is one writable f64, stride is sizeof(f64).
        // DOUBLE_dot does not mutate inputs, dereference its unused context,
        // allocate Python objects, or require the GIL. NumPy itself invokes it
        // with NPY_BEGIN_ALLOW_THREADS. The built-in function lives for the
        // duration of the imported NumPy module, including this detached call.
        unsafe {
            (self.0)(
                x.as_ptr().cast_mut().cast(),
                8,
                y.as_ptr().cast_mut().cast(),
                8,
                (&mut out as *mut f64).cast(),
                x.len() as isize,
                std::ptr::null_mut(),
            );
        }
        out
    }
}
