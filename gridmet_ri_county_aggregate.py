#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download gridMET (METDATA) daily data and area-weight-aggregate it to counties.

Variables : tmmx, tmmn, vpd, vs
Period     : 2006-01-01 .. 2025-12-31  (20 years, daily)
Geography  : the 5 Rhode Island counties in your shapefile

Two data-access modes (pick one with MODE below):
  "opendap"  -> stream a small lat/lon window straight from the THREDDS
               OPeNDAP server. Downloads only a few MB. RECOMMENDED.
  "download" -> download the full per-year CONUS .nc files to LOCAL_DIR first
               (tens of GB total), then read from disk.

Aggregation: exact area weighting. For every grid cell the fraction of its
area that falls inside each county is computed (in an equal-area projection)
and used as the weight in a per-day weighted mean. Cells that are NaN on a
given day (e.g. ocean / fill) are dropped and the weights renormalised.

Output: one CSV per county in OUT_DIR, e.g.
  gridmet_Providence_44007_2006_2025.csv
each with columns:
  date, FIPS, county, tmmx, tmmn, vpd, vs
  units: tmmx/tmmn = Kelvin, vpd = kPa, vs = m/s   (raw gridMET units)

Requirements:
  pip install xarray netCDF4 geopandas shapely pandas numpy requests
  (OPeNDAP mode needs netCDF4 built with DAP support, which the standard
   pip / conda netCDF4 wheels have.)
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import box

# -----------------------------------------------------------------------------
# CONFIG  -- edit these
# -----------------------------------------------------------------------------
MODE        = "opendap"          # "opendap"  or  "download"
SHP         = "rhode_island_counties_shapefile/rhode_island_counties.shp"  # your shapefile
VARIABLES   = ["tmmx", "tmmn", "vpd", "vs"]    # vpd replaces rmax/rmin (kPa)
YEARS       = list(range(1986, 2026))            # 2006..2025 inclusive (20 yrs)
START, END  = "1986-01-01", "2025-12-31"
NAME_FIELD  = "NAME"             # county-name column in the shapefile
FIPS_FIELD  = "FIPS"             # FIPS column (falls back to "id" / GEO_ID)
BUFFER_DEG  = 0.15               # padding around county bbox when subsetting grid
EQUAL_AREA  = 5070               # EPSG:5070 CONUS Albers, for area weighting
LOCAL_DIR   = "gridmet_raw"      # where "download" mode stores .nc files
OUT_DIR     = "gridmet_csv"      # one CSV per county is written here

OPENDAP_TMPL  = ("http://thredds.northwestknowledge.net:8080/thredds/dodsC/"
                 "agg_met_{var}_1979_CurrentYear_CONUS.nc#fillmismatch")
DOWNLOAD_TMPL = "https://www.northwestknowledge.net/metdata/data/{var}_{year}.nc"


# -----------------------------------------------------------------------------
# 1. counties
# -----------------------------------------------------------------------------
def load_counties(shp_path):
    g = gpd.read_file(shp_path).to_crs(4326)
    if FIPS_FIELD in g.columns:
        fips = FIPS_FIELD
    elif "id" in g.columns:
        fips = "id"
    else:
        fips = "GEO_ID"
    g = g.rename(columns={fips: "FIPS", NAME_FIELD: "NAME"})
    g["FIPS"] = g["FIPS"].astype(str)
    return g[["FIPS", "NAME", "geometry"]].reset_index(drop=True)


# -----------------------------------------------------------------------------
# 2. open one variable as a (time, lat, lon) DataArray, subset to the bbox
# -----------------------------------------------------------------------------
def _standardise(ds):
    """Return the single 3-D data variable, with dims (time, lat, lon)."""
    data_var = [v for v in ds.data_vars if ds[v].ndim == 3][0]
    da = ds[data_var]
    # identify the time dim (the one that is not lat/lon)
    tdim = [d for d in da.dims if d not in ("lat", "lon")][0]
    da = da.rename({tdim: "time"})
    da = da.sortby("lat").sortby("lon")          # force ascending for slicing
    if float(da.lon.max()) > 180:                # guard against 0..360 grids
        da = da.assign_coords(lon=(((da.lon + 180) % 360) - 180)).sortby("lon")
    return da, data_var


def open_opendap(var, lat_bounds, lon_bounds):
    url = OPENDAP_TMPL.format(var=var)
    ds = xr.open_dataset(url)                     # decodes time, scale/offset
    da, name = _standardise(ds)
    da = da.sel(lat=slice(*lat_bounds), lon=slice(*lon_bounds),
                time=slice(START, END))
    return da.load(), name


def open_download(var, lat_bounds, lon_bounds):
    paths = []
    for y in YEARS:
        p = os.path.join(LOCAL_DIR, f"{var}_{y}.nc")
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found. Run download_files() first, or set MODE='opendap'.")
        paths.append(p)
    ds = xr.open_mfdataset(paths, combine="by_coords")
    da, name = _standardise(ds)
    da = da.sel(lat=slice(*lat_bounds), lon=slice(*lon_bounds),
                time=slice(START, END))
    return da.load(), name


def download_files():
    """Optional helper for MODE='download': pull the full CONUS .nc files."""
    import requests
    os.makedirs(LOCAL_DIR, exist_ok=True)
    for var in VARIABLES:
        for y in YEARS:
            url = DOWNLOAD_TMPL.format(var=var, year=y)
            dst = os.path.join(LOCAL_DIR, f"{var}_{y}.nc")
            if os.path.exists(dst):
                continue
            print("downloading", url)
            with requests.get(url, stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(dst, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)


# -----------------------------------------------------------------------------
# 3. exact area-weighting: weight[county] -> 2-D array over the (lat, lon) window
# -----------------------------------------------------------------------------
def build_weights(lats, lons, counties):
    """For each county return a (nlat, nlon) array of intersection AREA per cell."""
    lats = np.asarray(lats); lons = np.asarray(lons)
    dlat = float(np.median(np.diff(lats)))
    dlon = float(np.median(np.diff(lons)))

    # build a GeoDataFrame of every cell rectangle in the window
    cells = []
    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            cells.append({"i": i, "j": j,
                          "geometry": box(lo - dlon / 2, la - dlat / 2,
                                          lo + dlon / 2, la + dlat / 2)})
    cells = gpd.GeoDataFrame(cells, crs=4326).to_crs(EQUAL_AREA)
    cty_ea = counties.to_crs(EQUAL_AREA)

    weights = {}
    for _, c in cty_ea.iterrows():
        inter = cells.geometry.intersection(c.geometry)
        w = np.zeros((len(lats), len(lons)))
        area = inter.area.values
        w[cells["i"].values, cells["j"].values] = area
        weights[c["FIPS"]] = w
    return weights


# -----------------------------------------------------------------------------
# 4. per-day area-weighted mean for every county
# -----------------------------------------------------------------------------
def aggregate(da, weights, counties):
    out = {}
    for fips, w2d in weights.items():
        w = xr.DataArray(w2d, dims=("lat", "lon"),
                         coords={"lat": da.lat, "lon": da.lon})
        valid = da.notnull()
        num = (da * w).sum(dim=("lat", "lon"), skipna=True)
        den = (w * valid).sum(dim=("lat", "lon"))           # drop NaN cells/day
        out[fips] = (num / den.where(den > 0)).to_series()
    return out


# -----------------------------------------------------------------------------
# 5. driver
# -----------------------------------------------------------------------------
def main():
    counties = load_counties(SHP)
    minx, miny, maxx, maxy = counties.total_bounds
    lat_bounds = (miny - BUFFER_DEG, maxy + BUFFER_DEG)
    lon_bounds = (minx - BUFFER_DEG, maxx + BUFFER_DEG)
    opener = open_opendap if MODE == "opendap" else open_download

    weights = None
    per_var = {}
    for var in VARIABLES:
        print(f"[{var}] opening ({MODE}) ...")
        da, internal = opener(var, lat_bounds, lon_bounds)
        print(f"[{var}] internal var = '{internal}', "
              f"grid {da.sizes.get('lat')}x{da.sizes.get('lon')}, "
              f"{da.sizes['time']} days")
        if weights is None:                       # same grid for all vars
            weights = build_weights(da.lat.values, da.lon.values, counties)
        per_var[var] = aggregate(da, weights, counties)

    # one CSV per county: columns date, FIPS, county, tmmx, tmmn, vpd, vs
    os.makedirs(OUT_DIR, exist_ok=True)
    name_of = dict(zip(counties["FIPS"], counties["NAME"]))
    for fips in counties["FIPS"]:
        cols = {var: per_var[var][fips] for var in VARIABLES}   # one Series each
        df = pd.DataFrame(cols)
        df.index.name = "date"
        df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df.insert(1, "FIPS", fips)
        df.insert(2, "county", name_of[fips])
        df = df[["date", "FIPS", "county"] + VARIABLES].sort_values("date")
        safe = "".join(ch if ch.isalnum() else "_" for ch in name_of[fips])
        path = os.path.join(OUT_DIR, f"gridmet_{safe}_{fips}_2006_2025.csv")
        df.to_csv(path, index=False)
        print(f"wrote {path}  {df.shape}")


if __name__ == "__main__":
    # For MODE='download', uncomment the next line to fetch the raw files first:
    # download_files()
    main()
