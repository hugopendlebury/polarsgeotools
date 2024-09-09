use kdtree::KdTree;
use kdtree::distance::squared_euclidean;
use polars::prelude::*;
use log::info;
use itertools::izip;
use haversine::Location;



pub fn knn_full_output(_: &[Field]) -> PolarsResult<Field> {
    let v = vec! [

        Field::new("located", DataType::List(Box::new(DataType::Boolean))),
        Field::new("identifier", DataType::List(Box::new(DataType::Utf8))),
        Field::new("resolved_longitude", DataType::List(Box::new(DataType::Float64))),
        Field::new("resolved_latitude", DataType::List(Box::new(DataType::Float64))),
        Field::new("distance", DataType::List(Box::new(DataType::Float64))),
    ];

    Ok(Field::new("knn_dist", DataType::Struct(v)))
}


pub fn nearest_output(_: &[Field]) -> PolarsResult<Field> {
    let v = vec! [
        Field::new("latitude", DataType::List(Box::new(DataType::Float64))),
        Field::new("longitude", DataType::List(Box::new(DataType::Float64))),
        Field::new("nearest_latitude", DataType::List(Box::new(DataType::Float64))),
        Field::new("nearest_longitude", DataType::List(Box::new(DataType::Float64))),
        Field::new("location", DataType::List(Box::new(DataType::Utf8))),
        Field::new("distance", DataType::List(Box::new(DataType::Float64))),
    ];

    Ok(Field::new("locatedresults", DataType::Struct(v)))
}

pub(crate) fn impl_find_nearest_knn_tree(
    coordinates: &[Series]
)  -> Result<Series, PolarsError> {

    info!("Started got args {:?}", coordinates);
    println!("Started");

    let lats = &coordinates[0];
    let lons = &coordinates[1];
    let combined = get_coordinate_iter(lats, lons);

    let values : Vec<_> = combined?.map(|c|{
        let lat = c.0.map_or_else(|| 0.0f64, |f| f);
        let lon = c.1.map_or_else(|| 0.0f64, |f| f);
        [lat, lon]
    }).collect();


    let to_find_lats = &coordinates[2];
    let to_find_lons = &coordinates[3];
    let identifiers = &coordinates[4];

    //find the points

    let to_find_points = izip!(to_find_lats.f64()?.into_iter(), 
                                                                            to_find_lons.f64()?.into_iter(),
                                                                            identifiers.utf8()?.into_iter());

    //Check if we have already located this point if so return it


    let dimensions = 2;
    let mut kdtree = KdTree::new(dimensions);



    //let mut results: Vec::<nearest_coordinates> = Vec::with_capacity(values.len());
    let mut located_vec: Vec::<bool> = Vec::with_capacity(values.len());
    let mut identifier_vec: Vec::<Option<&str>> = Vec::with_capacity(values.len());
    let mut resolved_latitude_vec: Vec::<Option<f64>> = Vec::with_capacity(values.len());
    let mut resolved_longitude_vec: Vec::<Option<f64>> = Vec::with_capacity(values.len());
    let mut distance_vec: Vec::<Option<f64>> = Vec::with_capacity(values.len());

    values.iter().enumerate().for_each(|f| {
        let point = [f.1[0], f.1[1]];
        //owned_array.copy_from_slice(owned_array);
        //let p = &owned_array;
        //info!("Adding point {:?} to tree", point);
        kdtree.add(point , f.0).unwrap();
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
        let location1 = Location{latitude: nearest_point[0] , longitude: nearest_point[1]};
        let location2 = Location{latitude:lat, longitude: lon};
        let dist = haversine::distance(location1, location2, haversine::Units::Kilometers);
        distance_vec[first] = Some(dist);


    });

    let struct_out = df!(
        "located" => located_vec,
        "resolved_latitude" => resolved_latitude_vec,
        "resolved_longitude" => resolved_longitude_vec,
        "distance" => distance_vec,
        "identifer" => identifier_vec
    ).unwrap().into_struct("nearest");

    Ok(struct_out.into_series())

   // Ok(*lats)


}

fn get_coordinate_iter<'a>(lats: &'a Series, lons: &'a Series) -> Result<std::iter::Zip<Box<dyn PolarsIterator<Item = std::option::Option<f64>> + 'a>, 
                                                                    Box<dyn PolarsIterator<Item = std::option::Option<f64>> + 'a >>, PolarsError>
{

    let lats_iter = lats.f64()?.into_iter();
    let lons_iter = lons.f64()?.into_iter();

    Ok(lats_iter.zip(lons_iter))

}




pub(crate) fn impl_find_nearest(
    coordinates: &[Series]
)  -> Result<Series, PolarsError> {


    let incomming_lats = &coordinates[0];
    let incomming_lons = &coordinates[1];
  

    let df = df!(
         "lats" => incomming_lats
        ,"lons" => incomming_lons
    )?.sort(["lats"], false, true);

    let binding = df?;
    let columns = binding.get_columns();

    let lats = &columns[0];
    let lons = &columns[1];
    

    let to_find_lats = &coordinates[2];
    let to_find_lons = &coordinates[3];
    let identifiers = &coordinates[4];

    let to_find_points = izip!(to_find_lats.f64()?.into_iter(), 
                                                                    to_find_lons.f64()?.into_iter(),
                                                                    identifiers.utf8()?.into_iter());


    let nearest_details : Vec<_> = to_find_points.map(|point_to_find| {
        let lat = point_to_find.0.map_or_else(|| 0.0f64, |f| f);
        let lon = point_to_find.1.map_or_else(|| 0.0f64, |f| f);
        let location = point_to_find.2;

        //TODO handle this better
        let lat_index = (lats - lat).abs().unwrap().arg_min().unwrap();
        let lon_index = (lons - lon).abs().unwrap().arg_min().unwrap();


        let nearest_lat = lats.f64().expect("latitudes not f64").get(lat_index).expect("latitude was null");
        let nearest_lon = lons.f64().expect("longitudes not f64").get(lon_index).expect("longitude was null");

        //get the distance
        let location1 = Location{latitude: nearest_lat , longitude: nearest_lon};
        let location2 = Location{latitude:lat, longitude: lon};
        let dist = haversine::distance(location1, location2, haversine::Units::Kilometers);

        (lat, lon, nearest_lat, nearest_lon, location, dist)

    }).collect();

    let results_iter = nearest_details.into_iter();

    let source_lats = results_iter.clone().map(|c| c.0).collect::<Vec<_>>();
    let source_lons = results_iter.clone().map(|c| c.1).collect::<Vec<_>>();
    let nearest_lats = results_iter.clone().map(|c| c.2).collect::<Vec<_>>();
    let nearest_lons = results_iter.clone().map(|c| c.3).collect::<Vec<_>>();
    let nearest_location = results_iter.clone().map(|c| c.4).collect::<Vec<_>>();
    let nearest_distances = results_iter.map(|c| c.5).collect::<Vec<_>>();

    let out_df = df!(
         "latitude" => source_lats
        ,"longitude" => source_lons
        ,"nearest_latitude" => nearest_lats
        ,"nearest_longitude" => nearest_lons
        ,"location" => nearest_location
        ,"distance" => nearest_distances
    );

    Ok(out_df?.into_struct("nearest").into_series())
   
}

