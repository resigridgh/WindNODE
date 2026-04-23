use pyo3::prelude::*;

/// Returns a greeting from the WindNODE Rust core.
#[pyfunction]
fn hello_from_windnode() -> String {
    "Hello from WindNODE!".to_string()
}

/// A Python module implemented in Rust.
///
/// The module name here must match the `lib.name` field in `Cargo.toml`.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_from_windnode, m)?)?;
    Ok(())
}
