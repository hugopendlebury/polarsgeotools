use haversine::Location;
use itertools::{izip, Itertools};
use kdtree::distance::squared_euclidean;
use kdtree::KdTree;
use log::info;
use polars::prelude::*;
use serde::Deserialize;
use rust_decimal::Decimal;
use rust_decimal::prelude::FromPrimitive;
use std::collections::HashMap;


#[derive(Deserialize)]
pub struct NumberOfPointKwargs {
    number_of_points: u32,
    max_distance: u32,
}


#[derive(Deserialize)]
pub struct MaxDistanceKwargs {
    max_distance: f64,
}

pub struct NearestDetails<'a> {
    latitude: f64,
    longitude: f64,
    nearest_latitude: f64,
    nearest_longitude: f64,
    location: Option<&'a str>,
    distance: f64,
}

/*
pub struct NearestDetailsWithValue<'a> {
    latitude: f64,
    longitude: f64,
    nearest_latitude: f64,
    nearest_longitude: f64,
    location: Option<&'a str>,
    distance: f64,
    value: f64,
}
*/


#[derive(Debug, Eq, Hash, PartialEq, Copy, Clone)]
pub struct LocationKey{
    pub lat: Decimal,
    pub lon: Decimal,
}

impl LocationKey {
    pub fn from_f64(lat: f64, lon: f64) -> Option<LocationKey> {
        let lat_value = Decimal::from_f64(lat)?;
        let lon_value = Decimal::from_f64(lon)?;

        Some(LocationKey { lat: lat_value , lon: lon_value})
    }
}

macro_rules! struct_to_dataframe {
    ($input:expr, [$($field:ident),+]) => {
        {
            let len = $input.len().to_owned();

            // Extract the field values into separate vectors
            $(let mut $field = Vec::with_capacity(len);)*

            for e in $input.into_iter() {
                $($field.push(e.$field);)*
            }
            df! {
                $(stringify!($field) => $field,)*
            }
        }
    };
}

pub fn knn_full_output(_: &[Field]) -> PolarsResult<Field> {
    let v = vec![
        Field::new("located", DataType::List(Box::new(DataType::Boolean))),
        Field::new("identifier", DataType::List(Box::new(DataType::Utf8))),
        Field::new(
            "resolved_longitude",
            DataType::List(Box::new(DataType::Float64)),
        ),
        Field::new(
            "resolved_latitude",
            DataType::List(Box::new(DataType::Float64)),
        ),
        Field::new("distance", DataType::List(Box::new(DataType::Float64))),
    ];

    Ok(Field::new("knn_dist", DataType::Struct(v)))
}

pub fn nearest_output(_: &[Field]) -> PolarsResult<Field> {
    let v = vec![
        Field::new("latitude", DataType::List(Box::new(DataType::Float64))),
        Field::new("longitude", DataType::List(Box::new(DataType::Float64))),
        Field::new(
            "nearest_latitude",
            DataType::List(Box::new(DataType::Float64)),
        ),
        Field::new(
            "nearest_longitude",
            DataType::List(Box::new(DataType::Float64)),
        ),
        Field::new("location", DataType::List(Box::new(DataType::Utf8))),
        Field::new("distance", DataType::List(Box::new(DataType::Float64))),
    ];

    Ok(Field::new("locatedresults", DataType::Struct(v)))
}

pub fn nearest_output_with_value(_: &[Field]) -> PolarsResult<Field> {
    let v = vec![
        Field::new("latitude", DataType::List(Box::new(DataType::Float64))),
        Field::new("longitude", DataType::List(Box::new(DataType::Float64))),
        Field::new(
            "nearest_latitude",
            DataType::List(Box::new(DataType::Float64)),
        ),
        Field::new(
            "nearest_longitude",
            DataType::List(Box::new(DataType::Float64)),
        ),
        Field::new("location", DataType::List(Box::new(DataType::Utf8))),
        Field::new("distance", DataType::List(Box::new(DataType::Float64))),
        Field::new("value", DataType::List(Box::new(DataType::Float64))),
    ];

    Ok(Field::new("locatedresults", DataType::Struct(v)))
}

pub(crate) fn impl_find_nearest_knn_tree(coordinates: &[Series]) -> Result<Series, PolarsError> {
    info!("Started got args {:?}", coordinates);
    println!("Started");

    let lats = &coordinates[0];
    let lons = &coordinates[1];
    let combined = get_coordinate_iter(lats, lons);

    let values: Vec<_> = combined?
        .map(|c| {
            let lat = c.0.map_or_else(|| 0.0f64, |f| f);
            let lon = c.1.map_or_else(|| 0.0f64, |f| f);
            [lat, lon]
        })
        .collect();

    let to_find_lats = &coordinates[2];
    let to_find_lons = &coordinates[3];
    let identifiers = &coordinates[4];

    //find the points

    let to_find_points = izip!(
        to_find_lats.f64()?.into_iter(),
        to_find_lons.f64()?.into_iter(),
        identifiers.utf8()?.into_iter()
    );

    //Check if we have already located this point if so return it

    let dimensions = 2;
    let mut kdtree = KdTree::new(dimensions);

    //let mut results: Vec::<nearest_coordinates> = Vec::with_capacity(values.len());
    let mut located_vec: Vec<bool> = Vec::with_capacity(values.len());
    let mut identifier_vec: Vec<Option<&str>> = Vec::with_capacity(values.len());
    let mut resolved_latitude_vec: Vec<Option<f64>> = Vec::with_capacity(values.len());
    let mut resolved_longitude_vec: Vec<Option<f64>> = Vec::with_capacity(values.len());
    let mut distance_vec: Vec<Option<f64>> = Vec::with_capacity(values.len());

    values.iter().enumerate().for_each(|f| {
        let point = [f.1[0], f.1[1]];
        //owned_array.copy_from_slice(owned_array);
        //let p = &owned_array;
        //info!("Adding point {:?} to tree", point);
        kdtree.add(point, f.0).unwrap();
        //results.push(default_value);
        located_vec.push(false);
        identifier_vec.push(None);
        resolved_latitude_vec.push(None);
        resolved_longitude_vec.push(None);
        distance_vec.push(None);
    });

    to_find_points.for_each(|find_coords| {
        let lat = find_coords.0.map_or_else(|| 0.0f64, |f| f);
        let lon = find_coords.1.map_or_else(|| 0.0f64, |f| f);
        let location = find_coords.2;
        let point = [lat, lon];
        //info!("Looking for point {:?}", point);
        let found = kdtree.nearest(&point, 1, &squared_euclidean);
        let x = found.unwrap();
        //info!("Got x {:?}", x);
        //assume everything found
        let first = *x[0].1;
        //info!("first {:?}", first);
        let nearest_point = values[first];
        resolved_latitude_vec[first] = Some(lat);
        resolved_longitude_vec[first] = Some(lon);
        located_vec[first] = true;
        identifier_vec[first] = location;
        //get the distance
        let location1 = Location {
            latitude: nearest_point[0],
            longitude: nearest_point[1],
        };
        let location2 = Location {
            latitude: lat,
            longitude: lon,
        };
        let dist = haversine::distance(location1, location2, haversine::Units::Kilometers);
        distance_vec[first] = Some(dist);
    });

    let struct_out = df!(
        "located" => located_vec,
        "resolved_latitude" => resolved_latitude_vec,
        "resolved_longitude" => resolved_longitude_vec,
        "distance" => distance_vec,
        "identifer" => identifier_vec
    )
    .unwrap()
    .into_struct("nearest");

    Ok(struct_out.into_series())

    // Ok(*lats)
}

type LatLonIterator<'a> = Box<dyn PolarsIterator<Item = std::option::Option<f64>> + 'a>;

fn get_coordinate_iter<'a>(
    lats: &'a Series,
    lons: &'a Series,
) -> Result<std::iter::Zip<LatLonIterator<'a>, LatLonIterator<'a>>, PolarsError> {
    let lats_iter = lats.f64()?.into_iter();
    let lons_iter = lons.f64()?.into_iter();

    Ok(lats_iter.zip(lons_iter))
}

pub(crate) fn impl_find_nearest(coordinates: &[Series]) -> Result<Series, PolarsError> {
    let incomming_lats = &coordinates[0];
    let incomming_lons = &coordinates[1];

    let lats = &incomming_lats.unique()?.sort(false);
    let lons = &incomming_lons.unique()?.sort(false);

    let to_find_lats = &coordinates[2];
    let to_find_lons = &coordinates[3];
    let identifiers = &coordinates[4];

    let to_find_points = izip!(
        to_find_lats.f64()?.into_iter(),
        to_find_lons.f64()?.into_iter(),
        identifiers.utf8()?.into_iter()
    );

    let nearest_details: Vec<_> = to_find_points
        .map(|point_to_find| {
            let lat = point_to_find.0.map_or_else(|| 0.0f64, |f| f);
            let lon = point_to_find.1.map_or_else(|| 0.0f64, |f| f);
            let location = point_to_find.2;

            //TODO handle this better
            let lat_index = (lats - lat).abs().unwrap().arg_min().unwrap();
            let lon_index = (lons - lon).abs().unwrap().arg_min().unwrap();

            let nearest_lat = lats
                .f64()
                .expect("latitudes not f64")
                .get(lat_index)
                .expect("latitude was null");
            let nearest_lon = lons
                .f64()
                .expect("longitudes not f64")
                .get(lon_index)
                .expect("longitude was null");

            //get the distance
            let location1 = Location {
                latitude: nearest_lat,
                longitude: nearest_lon,
            };
            let location2 = Location {
                latitude: lat,
                longitude: lon,
            };
            let dist = haversine::distance(location1, location2, haversine::Units::Kilometers);

            NearestDetails {
                latitude: lat,
                longitude: lon,
                nearest_latitude: nearest_lat,
                nearest_longitude: nearest_lon,
                location,
                distance: dist,
            }
        })
        .collect();

    let out_df = struct_to_dataframe!(
        nearest_details,
        [
            latitude,
            longitude,
            nearest_longitude,
            nearest_latitude,
            location,
            distance
        ]
    );

    Ok(out_df?.into_struct("nearest").into_series())
}

pub(crate) fn impl_find_nearest_none_null(
    coordinates: &[Series],
    max_distance_kwargs: MaxDistanceKwargs,
) -> Result<Series, PolarsError> {

    let incomming_lats = &coordinates[0];
    let incomming_lons = &coordinates[1];
    let values = &coordinates[2];
    /* 
    let values_df = DataFrame::new(vec![incomming_lats.clone(), 
                                                                    incomming_lons.clone(), values.clone()]);
                                                                    */


    let mut cache = HashMap::<LocationKey, Option<Decimal>>::new();
    let lats_iter = incomming_lats.f64()?.into_iter();
    let lons_iter = incomming_lons.f64()?.into_iter();
    let value_vec : Vec<_> = values.f64()?.into_iter().collect();

    lats_iter.zip(lons_iter).enumerate().for_each(|(i, v)| {
        let lat_lon = LocationKey::from_f64(v.0.unwrap(), v.1.unwrap()).unwrap();
        cache.insert(lat_lon, Decimal::from_f64_retain(value_vec[i].unwrap()));
    });


    let lats = &incomming_lats.unique()?.sort(false);
    let lons = &incomming_lons.unique()?.sort(false);

    let to_find_lats = &coordinates[3];
    let to_find_lons = &coordinates[4];
    let identifiers = &coordinates[5];
    

    let to_find_points = izip!(
        to_find_lats.f64()?.into_iter(),
        to_find_lons.f64()?.into_iter(),
        identifiers.utf8()?.into_iter()
    );

    let nearest_details: Vec<_> = to_find_points
        .map_while(|point_to_find| {
            let latitude = point_to_find.0.map_or_else(|| 0.0f64, |f| f);
            let longitude = point_to_find.1.map_or_else(|| 0.0f64, |f| f);
            let location = point_to_find.2;

            //TODO handle this better
            let lat_binding = (lats - latitude).abs().unwrap();
            let lat_iter = lat_binding.f64().unwrap().into_no_null_iter();
            let lon_binding = (lons - longitude).abs().unwrap();
            let lon_iter = lon_binding.f64().unwrap().into_no_null_iter();

            let sorted_lat = lat_iter.enumerate().sorted_by(|a, b| a.1.total_cmp(&b.1));
            let sorted_lon = lon_iter.enumerate().sorted_by(|a, b| a.1.total_cmp(&b.1));

            let nearest_indexes = sorted_lat.cartesian_product(sorted_lon);
            let mut found_all_points = false;

            let results: Vec<NearestDetails> = nearest_indexes
                .map_while(|indexes| {
                    if !found_all_points {
                        let lat_index = indexes.0 .0;
                        let lon_index = indexes.1 .0;
                        let nearest_latitude = lats
                            .f64()
                            .expect("latitudes not f64")
                            .get(lat_index)
                            .expect("latitude was null");
                        let nearest_longitude = lons
                            .f64()
                            .expect("longitudes not f64")
                            .get(lon_index)
                            .expect("longitude was null");

                        //get the distance
                        let location1 = Location {
                            latitude: nearest_latitude,
                            longitude: nearest_longitude,
                        };
                        let location2 = Location {
                            latitude,
                            longitude,
                        };

                        let distance =
                            haversine::distance(location1, location2, haversine::Units::Kilometers);

                        //let lkp_key = LocationKey::from_f64(nearest_latitude, nearest_longitude).unwrap();

                        //let value = cache.get(&lkp_key)?;

                        if distance < max_distance_kwargs.max_distance  {
                            Some(NearestDetails {
                                latitude,
                                longitude,
                                nearest_latitude,
                                nearest_longitude,
                                location,
                                distance,
                                //value : value.unwrap().into()
                            })
                        } else { 
                            found_all_points = true;
                            None 
                        }
                    } else { None }
                }).collect();
            Some(results)
        }).flatten().collect();


    let out_df = struct_to_dataframe!(
        nearest_details,
        [
            latitude,
            longitude,
            nearest_longitude,
            nearest_latitude,
            location,
            distance
        ]
    )?;

    println!("out_df = {:?}", out_df);

    /* 
    let joined = values_df?.join(&out_df, 
    [incomming_lats.name(), incomming_lons.name()], 
    ["nearest_latitude", "nearest_longitude"], JoinArgs::new(JoinType::Inner));

    
        joined?.sort(["location", "distance"] , false, true)
        ?.lazy().group_by(["location", "latitude_right", "longitude_right"])
        .agg([
             col("value").is_not_null().alias("value_flags")
            ,col("value")
            ,col("distance")
            ,col("latitude")
            ,col("longitude")
        ]).with_columns([
            col("value_flags")
        ]);

        .agg(
            col("value").is_not_null().alias("value_flags"),
            col("value"),
            col("distance"),
            col("latitude"),
            col("longitude")
        ).with_columns(
            index = pl.col("value_flags").list.eval((pl.element() == True).cast(pl.UInt8).arg_max()).list.get(0)
        ).with_columns(
            distance = pl.col("distance").list.get(pl.col("index"))
            ,value = pl.col("value").list.get(pl.col("index"))
            ,latitude = pl.col("latitude").list.get(pl.col("index"))
            ,longitude = pl.col("longitude").list.get(pl.col("index"))
        ).select(pl.exclude("value_flags", "index"));
    */

    //let _ =joined?.lazy().sink_parquet("/var/tmp/joined.parquet".into(), ParquetWriteOptions::default());

    Ok(out_df.into_struct("nearest").into_series())
}


pub(crate) fn impl_find_nearest_multiple(
    coordinates: &[Series],
    number_of_points: NumberOfPointKwargs,
) -> Result<Series, PolarsError> {
    let coordinate_slice_size = match number_of_points.number_of_points {
        0..=1 => 1,
        2..=4 => 2,
        5..=9 => 3,
        10..=16 => 4,
        17..=25 => 5,
        26..=36 => 6,
        37..=49 => 7,
        50..=64 => 8,
        65..=81 => 9,
        82..=100 => 10,
        101..=121 => 11,
        122..=144 => 12,
        145_u32..=u32::MAX => todo!(),
    };

    let incomming_lats = &coordinates[0];
    let incomming_lons = &coordinates[1];

    let lats = &incomming_lats.unique()?.sort(false);
    let lons = &incomming_lons.unique()?.sort(false);

    let to_find_lats = &coordinates[2];
    let to_find_lons = &coordinates[3];
    let identifiers = &coordinates[4];

    let to_find_points = izip!(
        to_find_lats.f64()?.into_iter(),
        to_find_lons.f64()?.into_iter(),
        identifiers.utf8()?.into_iter()
    );

    let nearest_details: Vec<_> = to_find_points
        .map(|point_to_find| {
            let latitude = point_to_find.0.map_or_else(|| 0.0f64, |f| f);
            let longitude = point_to_find.1.map_or_else(|| 0.0f64, |f| f);
            let location = point_to_find.2;

            //TODO handle this better
            let lat_binding = (lats - latitude).abs().unwrap();
            let lat_iter = lat_binding.f64().unwrap().into_no_null_iter();
            let lon_binding = (lons - longitude).abs().unwrap();
            let lon_iter = lon_binding.f64().unwrap().into_no_null_iter();

            let sorted_lat = lat_iter.enumerate().sorted_by(|a, b| a.1.total_cmp(&b.1));
            let sorted_lon = lon_iter.enumerate().sorted_by(|a, b| a.1.total_cmp(&b.1));

            let nearest_indexes = sorted_lat
                .take(coordinate_slice_size)
                .cartesian_product(sorted_lon.take(coordinate_slice_size));

            let results: Vec<NearestDetails> = nearest_indexes
                .map(|indexes| {
                    let lat_index = indexes.0 .0;
                    let lon_index = indexes.1 .0;
                    let nearest_latitude = lats
                        .f64()
                        .expect("latitudes not f64")
                        .get(lat_index)
                        .expect("latitude was null");
                    let nearest_longitude = lons
                        .f64()
                        .expect("longitudes not f64")
                        .get(lon_index)
                        .expect("longitude was null");

                    //get the distance
                    let location1 = Location {
                        latitude: nearest_latitude,
                        longitude: nearest_longitude,
                    };
                    let location2 = Location {
                        latitude,
                        longitude,
                    };

                    let distance =
                        haversine::distance(location1, location2, haversine::Units::Kilometers);

                    NearestDetails {
                        latitude,
                        longitude,
                        nearest_latitude,
                        nearest_longitude,
                        location,
                        distance,
                    }
                })
                .collect();
            results
        })
        .collect();

    let results_iter = nearest_details.into_iter();

    let ranked_distances = &results_iter
        .flat_map(|f| {
            f.into_iter()
                .sorted_by(|a, b| a.distance.total_cmp(&b.distance))
                .take(number_of_points.number_of_points.try_into().unwrap())
        })
        .collect::<Vec<_>>();

    let out_df = struct_to_dataframe!(
        ranked_distances,
        [
            latitude,
            longitude,
            nearest_longitude,
            nearest_latitude,
            location,
            distance
        ]
    )?
    .lazy()
    .filter(col("distance").lt_eq(lit(number_of_points.max_distance)))
    .collect();

    Ok(out_df?.into_struct("nearest").into_series())
}


#[cfg(test)]
mod test {

    use polars::prelude::*;
    use crate::nearest::*;

    #[test]
    fn test_nearest_none_null() {

        let locations = df!(
            "place" => &["Canary Wharf", "Manchester", "Lewes"],
            "lats" => &[51.5054, 53.4808, 52.5227],
            "lons" => &[-0.027176, 2.2426, 0.0027],
  
        ).unwrap();

        let test_data_path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_data/swh.parquet");
        let df = LazyFrame::scan_parquet(test_data_path, Default::default()).unwrap().collect().unwrap();

        let argss = &[
            df["latitude"].clone(),
            df["longitude"].clone(),
            df["value"].clone(),
            locations["lats"].clone(),
            locations["lons"].clone(),
            locations["place"].clone()
        ];


        let results = impl_find_nearest_none_null(argss, MaxDistanceKwargs{max_distance:75.0});

        println!("results = {:?}", results)

    }
}