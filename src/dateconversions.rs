use std::collections::HashMap;
use std::io::Error;

use chrono::{LocalResult, NaiveDateTime, TimeZone};
use lazy_static::lazy_static;
use polars::prelude::*;

use chrono_tz::Tz;
use pyo3_polars::export::polars_core::error::PolarsError;

use ordered_float::NotNan;
use pyo3::pyclass;
use serde::Deserialize;

#[pyclass]
#[derive(Deserialize)]
pub enum Ambiguous {
    Earliest = 0,
    Latest,
    Raise,
}

#[derive(Deserialize)]
pub struct DateConversionKwargs {
    ambiguous: Ambiguous,
}

//use polars_core::utils::{_set_partition_size, slice_offsets, split_ca};
use pyo3_polars::export::polars_core::utils::arrow::temporal_conversions::{
    timestamp_ms_to_datetime, timestamp_ns_to_datetime, timestamp_us_to_datetime,
};

use tzf_rs::DefaultFinder;

lazy_static! {
    static ref FINDER: DefaultFinder = DefaultFinder::new();
}

//Rust doesn't impletement eq for f64 so split the f64 into parts
fn integer_decode(val: f64) -> (u64, i16, i8) {
    let bits: u64 = val.to_bits();
    let sign: i8 = if bits >> 63 == 0 { 1 } else { -1 };
    let mut exponent: i16 = ((bits >> 52) & 0x7ff) as i16;
    let mantissa = if exponent == 0 {
        (bits & 0xfffffffffffff) << 1
    } else {
        (bits & 0xfffffffffffff) | 0x10000000000000
    };

    exponent -= 1023 + 52;
    (mantissa, exponent, sign)
}

#[derive(Hash, Eq, PartialEq, Clone, Copy)]
struct Distance((u64, i16, i8));

impl Distance {
    fn new(val: f64) -> Distance {
        Distance(integer_decode(val))
    }
}

#[derive(Eq, Hash, PartialEq, Clone, Copy)]
struct Coordinates {
    lat: Distance,
    lon: Distance,
}

#[allow(dead_code)]
#[derive(Eq, Hash, PartialEq, Clone, Copy)]
struct CoodinateTime {
    location: Coordinates,
    dt: i64,
}

pub(crate) fn impl_lookup_timezone(lat: &Series, lons: &Series) -> PolarsResult<Series> {
    let mut cache = HashMap::<Coordinates, &str>::new();
    let lats_iter = lat.f64()?.into_iter();
    let lons_iter = lons.f64()?.into_iter();
    cache.get(&Coordinates {
        lat: Distance::new(1.1),
        lon: Distance::new(1.2),
    });
    let results = lats_iter.zip(lons_iter).map(|coords| {
        let lat = coords.0.map_or(0.0, |f| f);
        let lng = coords.1.map_or(0.0, |f| f);
        let lkp_key = Coordinates {
            lat: Distance::new(lat),
            lon: Distance::new(lng),
        };
        let cache_key = cache.get(&lkp_key);

        match cache_key {
            Some(key) => key,
            None => {
                let timezone_names = FINDER.get_tz_names(lng, lat);
                let time_zone = timezone_names.last().map_or("UNKNOWN", |f| f);
                cache.insert(lkp_key, time_zone);
                time_zone
            }
        }
    });

    Ok(Series::from_iter(results))
}

fn parse_time_zone(time_zone: &str) -> Result<Tz, PolarsError> {
    let x: Result<Tz, chrono_tz::ParseError> = time_zone.parse::<Tz>();
    match x {
        Ok(r) => Ok(r),
        Err(_) => Err(PolarsError::Io(Error::new(
            std::io::ErrorKind::Unsupported,
            format!("Unable to convert timezone {}", time_zone),
        ))),
    }
}

fn naive_local_to_naive_local_in_new_time_zone(
    from_tz: &Tz,
    to_tz: &Tz,
    ndt: NaiveDateTime,
    ambiguous: &Ambiguous,
) -> PolarsResult<NaiveDateTime> {
    match from_tz.from_local_datetime(&ndt) {
        LocalResult::Single(dt) => Ok(dt.with_timezone(to_tz).naive_local()),
        LocalResult::Ambiguous(dt_earliest, dt_latest) => match ambiguous {
            Ambiguous::Earliest => Ok(dt_earliest.with_timezone(to_tz).naive_local()),
            Ambiguous::Latest => Ok(dt_latest.with_timezone(to_tz).naive_local()),
            Ambiguous::Raise => {
                polars_bail!(ComputeError: "datetime '{}' is ambiguous in time zone '{}'. Please use `ambiguous` to tell how it should be localized.", ndt, to_tz)
            }
        },
        LocalResult::None => polars_bail!(ComputeError:
            "datetime '{}' is non-existent in time zone '{}'. Non-existent datetimes are not yet supported",
            ndt, to_tz
        ),
    }
}

#[derive(Eq, Hash, PartialEq, Copy, Clone)]
struct GeoPoint {
    lat: NotNan<f64>,
    lon: NotNan<f64>,
}

#[derive(Eq, Hash, PartialEq)]
struct GeoPointTime {
    location: GeoPoint,
    dt: i64,
}

#[derive(Eq, Hash, PartialEq)]
struct DateTimeZone {
    timezone: String,
    dt: i64,
}

pub(crate) fn impl_to_local_in_new_timezone_using_timezone(
    dates: &Series,
    timezones: &Series,
    kwargs: DateConversionKwargs,
) -> PolarsResult<Series> {
    let dtype = dates.dtype();

    let from_time_zone = "UTC";
    let from_tz = parse_time_zone(from_time_zone)?;

    let dates_iter = dates.datetime()?.into_iter();
    let timezone_iter = timezones.utf8()?.into_iter();

    let timestamp_to_datetime: fn(i64) -> NaiveDateTime = match dtype {
        DataType::Datetime(TimeUnit::Microseconds, _) => timestamp_us_to_datetime,
        DataType::Datetime(TimeUnit::Milliseconds, _) => timestamp_ms_to_datetime,
        DataType::Datetime(TimeUnit::Nanoseconds, _) => timestamp_ns_to_datetime,
        _ => panic!("Unsupported dtype {}", dtype),
    };

    let results = dates_iter.zip(timezone_iter).map(|fields| {
        match fields.0 {
            Some(dt) => {
                //ok we have  a date
                match fields.1 {
                    Some(timezone) => {
                        //get the converted date.
                        let ndt = timestamp_to_datetime(dt);
                        let to_tz = parse_time_zone(timezone)?;
                        let result = naive_local_to_naive_local_in_new_time_zone(
                            &from_tz,
                            &to_tz,
                            ndt,
                            &kwargs.ambiguous,
                        )?;
                        Ok::<Option<NaiveDateTime>, PolarsError>(Some(result))
                    }
                    None => {
                        //No timezone
                        Ok::<Option<NaiveDateTime>, PolarsError>(None)
                    }
                }
            }
            None => Ok::<Option<NaiveDateTime>, PolarsError>(None),
        }
    });

    let data = results.map(|r| r.unwrap_or_default());

    let s = Series::new("ts", data.collect::<Vec<_>>());
    Ok(s)
}

pub(crate) fn impl_to_local_in_new_timezone(
    dates: &Series,
    lat: &Series,
    lons: &Series,
    kwargs: DateConversionKwargs,
) -> PolarsResult<Series> {
    let dtype = dates.dtype();

    let from_time_zone = "UTC";
    let from_tz = parse_time_zone(from_time_zone)?;

    let mut coordinates_cache = HashMap::<GeoPoint, &str>::new();
    let mut dates_cache = HashMap::<GeoPointTime, NaiveDateTime>::new();

    let dates_iter = dates.datetime()?.into_iter();
    let lats_iter = lat.f64()?.into_iter();
    let lons_iter = lons.f64()?.into_iter();

    let timestamp_to_datetime: fn(i64) -> NaiveDateTime = match dtype {
        DataType::Datetime(TimeUnit::Microseconds, _) => timestamp_us_to_datetime,
        DataType::Datetime(TimeUnit::Milliseconds, _) => timestamp_ms_to_datetime,
        DataType::Datetime(TimeUnit::Nanoseconds, _) => timestamp_ns_to_datetime,
        _ => panic!("Unsupported dtype {}", dtype),
    };

    let results = lats_iter.zip(lons_iter).zip(dates_iter).map(|coords| {
        let lat = coords.0 .0.expect("latitude should be float64");
        let lng = coords.0 .1.expect("longitude should be float64");
        let coordinates = &GeoPoint {
            lat: NotNan::new(lat).expect("latitude should not be nan / null"),
            lon: NotNan::new(lng).expect("longitude should not be nan / null"),
        };
        if let Some(timestamp) = coords.1 {
            let location_time = GeoPointTime {
                location: *coordinates,
                dt: timestamp,
            };
            let cached_date = dates_cache.get(&location_time);
            match cached_date {
                //Ok we have already come across this specific date in the same lat / lon
                //return the result
                Some(dt) => Ok::<Option<NaiveDateTime>, PolarsError>(Some(*dt)),
                None => {
                    //Check if we have already looked up the timezone for this lat / long
                    let cache_key = coordinates_cache.get(coordinates);

                    let time_zone = match cache_key {
                        Some(key) => key,
                        None => {
                            let timezone_names = FINDER.get_tz_names(lng, lat);
                            let time_zone = timezone_names.last().map_or("UNKNOWN", |f| f);
                            coordinates_cache.insert(*coordinates, time_zone);
                            time_zone
                        }
                    };

                    //We not have the timezone either from the cache or from a function call
                    //now get the local time and cache it
                    let ndt = timestamp_to_datetime(timestamp);
                    let to_tz = parse_time_zone(time_zone)?;
                    let result = naive_local_to_naive_local_in_new_time_zone(
                        &from_tz,
                        &to_tz,
                        ndt,
                        &kwargs.ambiguous,
                    )?;
                    dates_cache.insert(location_time, result);
                    Ok::<Option<NaiveDateTime>, PolarsError>(Some(result))
                }
            }
        } else {
            polars_bail!(ComputeError:"Is not struct type")
        }
    });

    let data = results.map(|r| r.unwrap_or_default());

    let s = Series::new("ts", data.collect::<Vec<_>>());
    Ok(s)
}

pub(crate) fn impl_utc_to_local_in_new_timezone_using_timezone_cache(
    dates: &Series,
    lat: &Series,
    lons: &Series,
    kwargs: DateConversionKwargs,
) -> PolarsResult<Series> {
    let dtype = dates.dtype();

    let from_time_zone = "UTC";
    let from_tz = parse_time_zone(from_time_zone)?;

    let mut coordinates_cache = HashMap::<GeoPoint, &str>::new();
    let mut dates_cache = HashMap::<DateTimeZone, NaiveDateTime>::new();

    let dates_iter = dates.datetime()?.into_iter();
    let lats_iter = lat.f64()?.into_iter();
    let lons_iter = lons.f64()?.into_iter();

    let timestamp_to_datetime: fn(i64) -> NaiveDateTime = match dtype {
        DataType::Datetime(TimeUnit::Microseconds, _) => timestamp_us_to_datetime,
        DataType::Datetime(TimeUnit::Milliseconds, _) => timestamp_ms_to_datetime,
        DataType::Datetime(TimeUnit::Nanoseconds, _) => timestamp_ns_to_datetime,
        _ => panic!("Unsupported dtype {}", dtype),
    };

    let results = lats_iter.zip(lons_iter).zip(dates_iter).map(|coords| {
        let lat = coords.0 .0.expect("latitude should be float64");
        let lng = coords.0 .1.expect("longitude should be float64");
        let coordinates = &GeoPoint {
            lat: NotNan::new(lat).expect("latitude should not be nan / null"),
            lon: NotNan::new(lng).expect("longitude should not be nan / null"),
        };

        if let Some(timestamp) = coords.1 {
            let cached_timezone = coordinates_cache.get(coordinates);
            match cached_timezone {
                Some(timezone) => {
                    //Ok we have a timezone
                    let check_date_location = DateTimeZone {
                        timezone: String::from(*timezone),
                        dt: timestamp,
                    };
                    let converted_date_in_cache = dates_cache.get(&check_date_location);
                    match converted_date_in_cache {
                        Some(dt) => Ok::<Option<NaiveDateTime>, PolarsError>(Some(*dt)),
                        None => {
                            let ndt = timestamp_to_datetime(timestamp);
                            let to_tz = parse_time_zone(timezone)?;
                            let result = naive_local_to_naive_local_in_new_time_zone(
                                &from_tz,
                                &to_tz,
                                ndt,
                                &kwargs.ambiguous,
                            )?;
                            dates_cache.insert(check_date_location, result);
                            Ok::<Option<NaiveDateTime>, PolarsError>(Some(result))
                        }
                    }
                }
                None => {
                    //Ok we don't have a timezone
                    let timezone_names = FINDER.get_tz_names(lng, lat);
                    let timezone = timezone_names.last().map_or("UNKNOWN", |f| f);
                    coordinates_cache.insert(*coordinates, timezone);
                    //Now we have a timezone get the datetime
                    let ndt = timestamp_to_datetime(timestamp);
                    let to_tz = parse_time_zone(timezone)?;
                    let result = naive_local_to_naive_local_in_new_time_zone(
                        &from_tz,
                        &to_tz,
                        ndt,
                        &kwargs.ambiguous,
                    )?;
                    let check_date_location = DateTimeZone {
                        timezone: String::from(timezone),
                        dt: timestamp,
                    };
                    dates_cache.insert(check_date_location, result);
                    Ok::<Option<NaiveDateTime>, PolarsError>(Some(result))
                }
            }
        } else {
            polars_bail!(ComputeError:"Is not struct type")
        }
    });

    let data = results.map(|r| r.unwrap_or_default());

    let s = Series::new("ts", data.collect::<Vec<_>>());
    Ok(s)
}
