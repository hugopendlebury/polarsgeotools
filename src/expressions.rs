use crate::nearest::*;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;


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