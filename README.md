# Polars-GeoTools

Geo tools for [Polars](https://www.pola.rs/).

- ✅ blazingly fast, written in Rust!
- ✅ seamless Polars integration!
- ✅ Determine timezone string from latitude and longitudes!
- ✅ Get localised date times based on the latitude and longitudes!
- ✅ lookup timezones based on latitude / longitude
- ✅ find nearest locations based on latitude / longitude





Installation
------------

First, you need to [install Polars](https://pola-rs.github.io/polars/user-guide/installation/).

Then, you'll need to install `polarsgeotools`:
```console
pip install polarsgeotools
```

Usage
-------------
The module creates a custom namespace which is attached to a polars expression

What does this mean. If you are using polars regularly you will be aware of the .str and .arr 
(now renamed .list) namespaces which have special functions related to string and arrays / lists

This module creates a custom namespace 

