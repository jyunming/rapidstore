use pyo3::prelude::*;

pub mod linalg;
pub mod python;
pub mod quantizer;
pub mod storage;

#[pymodule]
fn turboquantdb(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register(py, m)?;
    Ok(())
}
