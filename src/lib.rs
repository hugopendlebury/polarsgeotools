mod nearest;
mod expressions;
use pyo3::types::PyModule;
use pyo3::{pymodule, PyResult, Python};

#[cfg(target_os = "linux")]
use jemallocator::Jemalloc;

#[global_allocator]
#[cfg(target_os = "linux")]
static ALLOC: Jemalloc = Jemalloc;

#[pymodule]
fn _internal(_py: Python, m: &PyModule) -> PyResult<()> {
    // A good place to install the Rust -> Python logger.
    print!("YOYOYO");
    pyo3_log::init();
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}