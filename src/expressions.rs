use crate::nearest::*;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use lazy_static::lazy_static;


lazy_static! {
    static ref LOG_INIT: bool =  false;
}

#[derive(Deserialize)]
pub struct FromDatetimeKwargs {
    from_tz: String
}

fn init_log() {
    let x = *LOG_INIT;
    if x {
        println!("init log");
        pyo3_log::init();
        //x = true;
    }
}

#[polars_expr(output_type_func=knn_full_output)]
fn find_nearest(inputs: &[Series]) -> Result<Series, PolarsError> {
    //init_log();
    //pyo3_log::init();
    impl_find_nearest(inputs)
}

#[polars_expr(output_type_func=nearest_output)]
fn find_nearest_cache(inputs: &[Series]) -> Result<Series, PolarsError> {
    //init_log();
   // pyo3_log::init();
    impl_find_nearest_cache(inputs)
}