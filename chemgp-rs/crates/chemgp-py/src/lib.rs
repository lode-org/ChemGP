use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn chemgp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;
    Ok(())
}
