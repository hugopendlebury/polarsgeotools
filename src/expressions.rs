use crate::dateconversions::*;
use crate::nearest::*;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

#[polars_expr(output_type_func=knn_full_output)]
fn find_nearest_knn_tree(inputs: &[Series]) -> Result<Series, PolarsError> {
    //init_log();
    //pyo3_log::init();
    impl_find_nearest_knn_tree(inputs)
}

#[polars_expr(output_type_func=nearest_output)]
fn find_nearest(inputs: &[Series]) -> Result<Series, PolarsError> {
    //init_log();
    // pyo3_log::init();
    impl_find_nearest(inputs)
}

#[polars_expr(output_type_func=nearest_output)]
fn find_nearest_multiple(
    inputs: &[Series],
    kwargs: NumberOfPointKwargs,
) -> Result<Series, PolarsError> {
    //init_log();
    // pyo3_log::init();
    impl_find_nearest_multiple(inputs, kwargs)
}

#[polars_expr(output_type_func=nearest_output_with_value)]
fn find_nearest_none_null(
    inputs: &[Series],
    kwargs: MaxDistanceKwargs,
) -> Result<Series, PolarsError> {
    impl_find_nearest_none_null(inputs, kwargs)
}

#[polars_expr(output_type=Utf8)]
fn lookup_timezone(inputs: &[Series]) -> PolarsResult<Series> {
    let lats = &inputs[0];
    let lons = &inputs[1];
    impl_lookup_timezone(lats, lons)
}

#[polars_expr(output_type_func=from_local_datetime)]
fn to_local_in_new_timezone(
    inputs: &[Series],
    kwargs: DateConversionKwargs,
) -> PolarsResult<Series> {
    let dates = &inputs[0];
    let lats = &inputs[1];
    let lons = &inputs[2];
    impl_to_local_in_new_timezone(dates, lats, lons, kwargs)
}

#[polars_expr(output_type_func=from_local_datetime)]
fn to_local_in_new_timezone_using_timezone(
    inputs: &[Series],
    kwargs: DateConversionKwargs,
) -> PolarsResult<Series> {
    let dates = &inputs[0];
    let timezones = &inputs[1];

    impl_to_local_in_new_timezone_using_timezone(dates, timezones, kwargs)
}

#[polars_expr(output_type_func=from_local_datetime)]
fn to_local_in_new_timezone_cache_timezone_string(
    inputs: &[Series],
    kwargs: DateConversionKwargs,
) -> PolarsResult<Series> {
    let dates = &inputs[0];
    let lats = &inputs[1];
    let lons = &inputs[2];
    impl_utc_to_local_in_new_timezone_using_timezone_cache(dates, lats, lons, kwargs)
}

pub fn from_local_datetime(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = input_fields[0].clone();
    let dtype = match field.dtype {
        DataType::Datetime(_, _) => field.dtype,
        _ => polars_bail!(InvalidOperation:
            "dtype '{}' not supported", field.dtype
        ),
    };

    Ok(Field::new(&field.name, dtype))
}
