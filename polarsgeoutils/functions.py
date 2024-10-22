from __future__ import annotations

import re
import sys
from datetime import date
from typing import TYPE_CHECKING, Literal, Sequence

import polars as pl
from polars.utils.udfs import _get_shared_lib_location

from polarsgeoutils.utils import parse_into_expr


if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr

lib = _get_shared_lib_location(__file__)

if TYPE_CHECKING:
    from polars import Expr
    # The following have no equivalent on the Rust side
    Ambiguous: TypeAlias = Literal["Earliest", "Latest", "Raise"]

def find_nearest_knn_tree(latitude: str | pl.Expr, 
                 longitude: IntoExpr, 
                 find_latitudes : IntoExpr,
                 find_longitudes: IntoExpr,
                 identifier: IntoExpr) -> pl.Expr:
    expr = parse_into_expr(latitude)
    return expr.register_plugin(
        lib=lib,
        symbol="find_nearest_knn_tree",
        args=[longitude, find_latitudes, find_longitudes, identifier],
        is_elementwise=True,
    )

def find_nearest(latitude: str | pl.Expr, 
                 longitude: IntoExpr, 
                 find_latitudes : IntoExpr,
                 find_longitudes: IntoExpr,
                 identifier: IntoExpr) -> pl.Expr:
    expr = parse_into_expr(latitude)
    return expr.register_plugin(
        lib=lib,
        symbol="find_nearest",
        args=[longitude, find_latitudes, find_longitudes, identifier],
        is_elementwise=True,
    )

def find_nearest_none_null(latitude: str | pl.Expr, 
                 longitude: IntoExpr, 
                 values: IntoExpr,
                 find_latitudes : IntoExpr,
                 find_longitudes: IntoExpr,
                 identifier: IntoExpr,
                 max_distance=75) -> pl.Expr:
    max_distance = float(max_distance)
    expr = parse_into_expr(latitude)
    return expr.register_plugin(
        lib=lib,
        symbol="find_nearest_none_null",
        args=[longitude, values, find_latitudes, find_longitudes, identifier],
        kwargs={"max_distance": max_distance},
        is_elementwise=True,
    )

def find_nearest_multiple(latitude: str | pl.Expr, 
                 longitude: IntoExpr, 
                 find_latitudes : IntoExpr,
                 find_longitudes: IntoExpr,
                 identifier: IntoExpr,
                 number_of_points=4,
                 max_distance=100) -> pl.Expr:
    expr = parse_into_expr(latitude)
    return expr.register_plugin(
        lib=lib,
        symbol="find_nearest_multiple",
        args=[longitude, find_latitudes, find_longitudes, identifier],
        kwargs={"number_of_points" : number_of_points, "max_distance": max_distance},
        is_elementwise=True,
    )

def lookup_timezone(expr: str | pl.Expr, other: IntoExpr) -> pl.Expr:
    """
    Return the Timezone as a string based on the latitude and longitude of a point

    Examples
    --------
    >>> import polars as pl
    >>> import polarsgeoutils as geo
    >>> from datetime import datetime
    >>> df = (
    ...     pl.DataFrame(data = {'lat' : [51.5054, 51.5054, 40.7128, 40.7128, 40.7128], 
    ...                         'lon' : [0.0235, 0.0235, -74.0060, -74.0060, -74.0060],
    ...                         'dt': [datetime(2024,1,24,9,0,0),
    ...                                 datetime(2024,1,24,9,0,0),
    ...                                 datetime(2024,1,24,9,0,0),
    ...                                 datetime(2024,1,24,3,0,0),
    ...                                 datetime(2024,1,24,3,0,0)],
    ...                         'values' : [5,5,5,10,10]}
    ...                 )
    ...     .with_columns(
    ...          timezone = geo.lookup_timezone(pl.col("lat"), pl.col("lon"))
    ...     )
    ... ) 

    shape: (5, 5)
    ┌─────────┬─────────┬─────────────────────┬────────┬──────────────────┐
    │ lat     ┆ lon     ┆ dt                  ┆ values ┆ timezone         │
    │ ---     ┆ ---     ┆ ---                 ┆ ---    ┆ ---              │
    │ f64     ┆ f64     ┆ datetime[μs]        ┆ i64    ┆ str              │
    ╞═════════╪═════════╪═════════════════════╪════════╪══════════════════╡
    │ 51.5054 ┆ 0.0235  ┆ 2024-01-24 09:00:00 ┆ 5      ┆ Europe/London    │
    │ 51.5054 ┆ 0.0235  ┆ 2024-01-24 09:00:00 ┆ 5      ┆ Europe/London    │
    │ 40.7128 ┆ -74.006 ┆ 2024-01-24 09:00:00 ┆ 5      ┆ America/New_York │
    │ 40.7128 ┆ -74.006 ┆ 2024-01-24 03:00:00 ┆ 10     ┆ America/New_York │
    │ 40.7128 ┆ -74.006 ┆ 2024-01-24 03:00:00 ┆ 10     ┆ America/New_York │
    └─────────┴─────────┴─────────────────────┴────────┴──────────────────┘
    """
    expr = parse_into_expr(expr)
    return expr.register_plugin(
        lib=lib,
        symbol="lookup_timezone",
        args=[other],
        is_elementwise=True,
    )


def convert_date_using_timezone(expr: str | pl.Expr, other: IntoExpr, ambiguous: Ambiguous = "Earliest") -> pl.Expr:
    """
    Return the Timezone as a string based on the latitude and longitude of a point

    Examples
    --------
    >>> import polars as pl
    >>> import polarsgeoutils as geo
    >>> from datetime import datetime
    >>> df = (
    ...     pl.DataFrame(data = {'lat' : [51.5054, 51.5054, 40.7128, 40.7128, 40.7128], 
    ...                         'lon' : [0.0235, 0.0235, -74.0060, -74.0060, -74.0060],
    ...                         'dt': [datetime(2024,1,24,9,0,0),
    ...                                 datetime(2024,1,24,9,0,0),
    ...                                 datetime(2024,1,24,9,0,0),
    ...                                 datetime(2024,1,24,3,0,0),
    ...                                 datetime(2024,1,24,3,0,0)],
    ...                         'values' : [5,5,5,10,10]}
    ...                 )
    ...     .with_columns(
    ...          timezone = geo.lookup_timezone(pl.col("lat"), pl.col("lon"))
    ...     ).with_columns(
    ...          converted = pl.col("dt").geoutils.convert_date_using_timezone(pl.col("timezone"))
    ...     )
    ... ) 

    shape: (5, 5)
    ┌─────────┬─────────┬─────────────────────┬────────┬──────────────────┐
    │ lat     ┆ lon     ┆ dt                  ┆ values ┆ timezone         │
    │ ---     ┆ ---     ┆ ---                 ┆ ---    ┆ ---              │
    │ f64     ┆ f64     ┆ datetime[μs]        ┆ i64    ┆ str              │
    ╞═════════╪═════════╪═════════════════════╪════════╪══════════════════╡
    │ 51.5054 ┆ 0.0235  ┆ 2024-01-24 09:00:00 ┆ 5      ┆ Europe/London    │
    │ 51.5054 ┆ 0.0235  ┆ 2024-01-24 09:00:00 ┆ 5      ┆ Europe/London    │
    │ 40.7128 ┆ -74.006 ┆ 2024-01-24 09:00:00 ┆ 5      ┆ America/New_York │
    │ 40.7128 ┆ -74.006 ┆ 2024-01-24 03:00:00 ┆ 10     ┆ America/New_York │
    │ 40.7128 ┆ -74.006 ┆ 2024-01-24 03:00:00 ┆ 10     ┆ America/New_York │
    └─────────┴─────────┴─────────────────────┴────────┴──────────────────┘
    """
    expr = parse_into_expr(expr)
    return expr.register_plugin(
        lib=lib,
        symbol="to_local_in_new_timezone_using_timezone",
        args=[other],
        is_elementwise=True,
        kwargs={"ambiguous" : ambiguous}
    )

def to_local_in_new_timezone_struct(expr: str | pl.Expr, lat_lon: IntoExpr, ambiguous: Ambiguous = "Earliest") -> pl.Expr:
    """
    Uses the latitude and longitude to find the time zone then returns a date / time which is 
    in the local time.

    Examples
    --------
    >>> from datetime import datetime
    >>> df = (
    ...     pl.DataFrame(data = {'lat' : [51.5054, 51.5054, 40.7128, 40.7128, 40.7128], 
    ...                         'lon' : [0.0235, 0.0235, -74.0060, -74.0060, -74.0060],
    ...                         'dt': [datetime(2024,1,24,9,0,0),
    ...                                 datetime(2024,1,24,9,0,0),
    ...                                 datetime(2024,1,24,9,0,0),
    ...                                 datetime(2024,1,24,3,0,0),
    ...                                 datetime(2024,1,24,3,0,0)],
    ...                         'values' : [5,5,5,10,10]}
    ...                 )
    ...     .with_columns(
    ...          timezone_conv = pl.col("dt").geoutils.to_local_in_new_timezone(pl.col("lat"), pl.col("lon"))
    ...     )
    ... )  
    shape: (5, 5)
    ┌─────────┬─────────┬─────────────────────┬────────┬─────────────────────┐
    │ lat     ┆ lon     ┆ dt                  ┆ values ┆ timezone_conv       │
    │ ---     ┆ ---     ┆ ---                 ┆ ---    ┆ ---                 │
    │ f64     ┆ f64     ┆ datetime[μs]        ┆ i64    ┆ datetime[ms]        │
    ╞═════════╪═════════╪═════════════════════╪════════╪═════════════════════╡
    │ 51.5054 ┆ 0.0235  ┆ 2024-01-24 09:00:00 ┆ 5      ┆ 2024-01-24 09:00:00 │
    │ 51.5054 ┆ 0.0235  ┆ 2024-01-24 09:00:00 ┆ 5      ┆ 2024-01-24 09:00:00 │
    │ 40.7128 ┆ -74.006 ┆ 2024-01-24 09:00:00 ┆ 5      ┆ 2024-01-24 04:00:00 │
    │ 40.7128 ┆ -74.006 ┆ 2024-01-24 03:00:00 ┆ 10     ┆ 2024-01-23 22:00:00 │
    │ 40.7128 ┆ -74.006 ┆ 2024-01-24 03:00:00 ┆ 10     ┆ 2024-01-23 22:00:00 │
    └─────────┴─────────┴─────────────────────┴────────┴─────────────────────┘
    """

    expr = parse_into_expr(expr)
    return expr.register_plugin(
        lib=lib,
        symbol="to_local_in_new_timezone_struct",
        args=[lat_lon],
        is_elementwise=True,
        kwargs={"ambiguous" : ambiguous}
    )

def to_local_in_new_timezone(expr: str | pl.Expr, lat: IntoExpr, lon: IntoExpr, ambiguous: Ambiguous = "Earliest") -> pl.Expr:
    """
    Uses the latitude and longitude to find the time zone then returns a date / time which is 
    in the local time.

    Examples
    --------
    >>> from datetime import datetime
    >>> df = (
    ...     pl.DataFrame(data = {'lat' : [51.5054, 51.5054, 40.7128, 40.7128, 40.7128], 
    ...                         'lon' : [0.0235, 0.0235, -74.0060, -74.0060, -74.0060],
    ...                         'dt': [datetime(2024,1,24,9,0,0),
    ...                                 datetime(2024,1,24,9,0,0),
    ...                                 datetime(2024,1,24,9,0,0),
    ...                                 datetime(2024,1,24,3,0,0),
    ...                                 datetime(2024,1,24,3,0,0)],
    ...                         'values' : [5,5,5,10,10]}
    ...                 )
    ...     .with_columns(
    ...          timezone_conv = pl.col("dt").geoutils.to_local_in_new_timezone(pl.col("lat"), pl.col("lon"))
    ...     )
    ... )  
    shape: (5, 5)
    ┌─────────┬─────────┬─────────────────────┬────────┬─────────────────────┐
    │ lat     ┆ lon     ┆ dt                  ┆ values ┆ timezone_conv       │
    │ ---     ┆ ---     ┆ ---                 ┆ ---    ┆ ---                 │
    │ f64     ┆ f64     ┆ datetime[μs]        ┆ i64    ┆ datetime[ms]        │
    ╞═════════╪═════════╪═════════════════════╪════════╪═════════════════════╡
    │ 51.5054 ┆ 0.0235  ┆ 2024-01-24 09:00:00 ┆ 5      ┆ 2024-01-24 09:00:00 │
    │ 51.5054 ┆ 0.0235  ┆ 2024-01-24 09:00:00 ┆ 5      ┆ 2024-01-24 09:00:00 │
    │ 40.7128 ┆ -74.006 ┆ 2024-01-24 09:00:00 ┆ 5      ┆ 2024-01-24 04:00:00 │
    │ 40.7128 ┆ -74.006 ┆ 2024-01-24 03:00:00 ┆ 10     ┆ 2024-01-23 22:00:00 │
    │ 40.7128 ┆ -74.006 ┆ 2024-01-24 03:00:00 ┆ 10     ┆ 2024-01-23 22:00:00 │
    └─────────┴─────────┴─────────────────────┴────────┴─────────────────────┘
    """
    expr = parse_into_expr(expr)
    return expr.register_plugin(
        lib=lib,
        symbol="to_local_in_new_timezone",
        args=[lat, lon],
        is_elementwise=True,
        kwargs={"ambiguous" : ambiguous}
    )

def to_local_in_new_timezone_cache_timezone_string(expr: str | pl.Expr, lat: IntoExpr, lon: IntoExpr, ambiguous: Ambiguous = "Earliest") -> pl.Expr:
    """
    Uses the latitude and longitude to find the time zone then returns a date / time which is 
    in the local time.

    Works the same way as to_local_in_new_timezone but caches the timezone string for a given lat/lon

    If the same date / timezone string are encountered again the preconverted date will be returned.
    Since the number of timezones is smaller than the number of lat/lons this should be a quicker.

    Examples
    --------
    >>> from datetime import datetime
    >>> df = (
    ...     pl.DataFrame(data = {'lat' : [51.5054, 51.5054, 40.7128, 40.7128, 40.7128], 
    ...                         'lon' : [0.0235, 0.0235, -74.0060, -74.0060, -74.0060],
    ...                         'dt': [datetime(2024,1,24,9,0,0),
    ...                                 datetime(2024,1,24,9,0,0),
    ...                                 datetime(2024,1,24,9,0,0),
    ...                                 datetime(2024,1,24,3,0,0),
    ...                                 datetime(2024,1,24,3,0,0)],
    ...                         'values' : [5,5,5,10,10]}
    ...                 )
    ...     .with_columns(
    ...          timezone_conv = pl.col("dt").dateconversions.to_local_in_new_timezone_cache_timezone_string(pl.col("lat"), pl.col("lon"))
    ...     )
    ... )  
    shape: (5, 5)
    ┌─────────┬─────────┬─────────────────────┬────────┬─────────────────────┐
    │ lat     ┆ lon     ┆ dt                  ┆ values ┆ timezone_conv       │
    │ ---     ┆ ---     ┆ ---                 ┆ ---    ┆ ---                 │
    │ f64     ┆ f64     ┆ datetime[μs]        ┆ i64    ┆ datetime[ms]        │
    ╞═════════╪═════════╪═════════════════════╪════════╪═════════════════════╡
    │ 51.5054 ┆ 0.0235  ┆ 2024-01-24 09:00:00 ┆ 5      ┆ 2024-01-24 09:00:00 │
    │ 51.5054 ┆ 0.0235  ┆ 2024-01-24 09:00:00 ┆ 5      ┆ 2024-01-24 09:00:00 │
    │ 40.7128 ┆ -74.006 ┆ 2024-01-24 09:00:00 ┆ 5      ┆ 2024-01-24 04:00:00 │
    │ 40.7128 ┆ -74.006 ┆ 2024-01-24 03:00:00 ┆ 10     ┆ 2024-01-23 22:00:00 │
    │ 40.7128 ┆ -74.006 ┆ 2024-01-24 03:00:00 ┆ 10     ┆ 2024-01-23 22:00:00 │
    └─────────┴─────────┴─────────────────────┴────────┴─────────────────────┘
    """
    expr = parse_into_expr(expr)
    return expr.register_plugin(
        lib=lib,
        symbol="to_local_in_new_timezone_cache_timezone_string",
        args=[lat, lon],
        is_elementwise=True,
        kwargs={"ambiguous" : ambiguous}
    )