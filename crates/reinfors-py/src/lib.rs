//! PyO3 bindings for reinfors. The compiled module is `reinfors._reinfors`; the ergonomic Python
//! API in `python/reinfors/` wraps it. Phase 1 exposes only enough to differential-test the snake
//! core against `CleanSnakeEnv`.

use pyo3::prelude::*;

#[pyfunction]
fn core_version() -> &'static str {
    reinfors_core::version()
}

#[pymodule]
fn _reinfors(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    Ok(())
}
