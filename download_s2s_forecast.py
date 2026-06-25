#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download the CFSv2-METDATA 90-day subseasonal forecast, subset to Rhode Island,
keep ALL ensemble members and the first 28 lead days, and area-weight-aggregate
to the 5 counties (same exact-area weighting used for the gridMET historical data).

Dataset (THREDDS):
  https://thredds.northwestknowledge.net/thredds/catalog/
      NWCSC_INTEGRATED_SCENARIOS_ALL_CLIMATE/cfsv2_metdata_90day/catalog.html
File naming:
  cfsv2_metdata_forecast_<var>_daily_<run>_<i>_<j>.nc   (one ensemble member each)
Variables: tmmx, tmmn, vpd, vs

The ensemble list is NOT hard-coded: the script reads catalog.xml at run time and
discovers every member file for each variable. Each member is opened over OPeNDAP,
subset to the RI bounding box and the first 28 days (a few KB per request), so the
total transfer is small even though there are many members.

Output: one tidy CSV per county
  s2s_forecast_<county>_<FIPS>.csv
  columns: ensemble, lead_day, valid_date, tmmx, tmmn, vpd, vs
  units: tmmx/tmmn = K, vpd = kPa, vs = m/s   (raw forecast units)

Requirements:
  pip install xarray netCDF4 geopandas shapely pandas numpy requests
"""

import os
import re
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import box
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CATALOG_URL = ("https://thredds.northwestknowledge.net/thredds/catalog/"
               "NWCSC_INTEGRATED_SCENARIOS_ALL_CLIMATE/cfsv2_metdata_90day/catalog.xml")
SHP        = "rhode_island_counties_shapefile/rhode_island_counties.shp"
VARIABLES  = ["tmmx", "tmmn", "vpd", "vs"]    # vpd replaces rmax/rmin (kPa)
N_LEAD     = 28                  # number of forecast lead days to keep
BUFFER_DEG = 0.15                # padding around county bbox when subsetting grid
EQUAL_AREA = 5070                # EPSG:5070 CONUS Albers, for area weighting
NAME_FIELD = "NAME"
FIPS_FIELD = "FIPS"
OUT_DIR    = "s2s_forecast_csv"
FILL       = "#fillmismatch"     # NKN-recommended suffix for their netCDF over DAP
# ---------------------------------------------------------------------------


# ---------- 1. counties + exact-area weights (same as the gridMET workflow) ----------
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


def build_weights(lats, lons, counties):
    lats = np.asarray(lats); lons = np.asarray(lons)
    dlat = float(np.median(np.diff(lats))); dlon = float(np.median(np.diff(lons)))
    cells = [{"i": i, "j": j,
              "geometry": box(lo - dlon / 2, la - dlat / 2, lo + dlon / 2, la + dlat / 2)}
             for i, la in enumerate(lats) for j, lo in enumerate(lons)]
    cells = gpd.GeoDataFrame(cells, crs=4326).to_crs(EQUAL_AREA)
    cty = counties.to_crs(EQUAL_AREA)
    weights = {}
    for _, c in cty.iterrows():
        w = np.zeros((len(lats), len(lons)))
        inter = cells.geometry.intersection(c.geometry)
        w[cells["i"].values, cells["j"].values] = inter.area.values
        weights[c["FIPS"]] = w
    return weights


def weighted_mean(da, w2d):
    """Area-weighted mean over (lat, lon) per step of the remaining dim."""
    w = xr.DataArray(w2d, dims=("lat", "lon"), coords={"lat": da.lat, "lon": da.lon})
    num = (da * w).sum(dim=("lat", "lon"), skipna=True)
    den = (w * da.notnull()).sum(dim=("lat", "lon"))
    return (num / den.where(den > 0))


# ---------- 2. discover ensemble files from the THREDDS catalog ----------
def list_members(catalog_url):
    """Return {var: [(ensemble_label, opendap_url), ...]} by parsing catalog.xml."""
    r = requests.get(catalog_url, timeout=120); r.raise_for_status()
    root = ET.fromstring(r.content)
    ns = {"t": "http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0"}

    # OPeNDAP service base (default /thredds/dodsC/)
    dods_base = "/thredds/dodsC/"
    for s in root.iter("{%s}service" % ns["t"]):
        if (s.get("serviceType") or "").upper() == "OPENDAP":
            dods_base = s.get("base") or dods_base
    sp = urlsplit(catalog_url)
    host_root = f"{sp.scheme}://{sp.netloc}"

    out = {v: [] for v in VARIABLES}
    for ds in root.iter("{%s}dataset" % ns["t"]):
        name = ds.get("name") or ""
        url_path = ds.get("urlPath")
        if not url_path or not name.endswith(".nc"):
            continue
        for v in VARIABLES:
            prefix = f"cfsv2_metdata_forecast_{v}_daily_"
            if name.startswith(prefix):
                label = name[len(prefix):-3]            # e.g. "00_2_1"
                opendap = host_root + dods_base + url_path + FILL
                out[v].append((label, opendap))
    for v in VARIABLES:
        out[v].sort()
    return out


# ---------- 3. open one member, subset, area-aggregate ----------
def _data_var(ds):
    cands = [v for v in ds.data_vars if {"lat", "lon"}.issubset(ds[v].dims)]
    return max(cands, key=lambda v: ds[v].ndim)        # the gridded field


def member_series(url, lat_bounds, lon_bounds, weights):
    ds = xr.open_dataset(url)
    da = ds[_data_var(ds)]
    tdim = [d for d in da.dims if d not in ("lat", "lon")][0]
    da = da.rename({tdim: "day"}).sortby("lat").sortby("lon")
    if float(da.lon.max()) > 180:
        da = da.assign_coords(lon=(((da.lon + 180) % 360) - 180)).sortby("lon")
    da = da.sel(lat=slice(*lat_bounds), lon=slice(*lon_bounds)).isel(day=slice(0, N_LEAD))
    da = da.load()
    valid = pd.to_datetime(da["day"].values) if "day" in da.coords else \
        pd.Index(range(da.sizes["day"]))
    series = {f: weighted_mean(da, w).values for f, w in weights.items()}  # {fips: [N_LEAD]}
    return series, valid


# ---------- 4. driver ----------
def main():
    counties = load_counties(SHP)
    minx, miny, maxx, maxy = counties.total_bounds
    lat_bounds = (miny - BUFFER_DEG, maxy + BUFFER_DEG)
    lon_bounds = (minx - BUFFER_DEG, maxx + BUFFER_DEG)
    name_of = dict(zip(counties["FIPS"], counties["NAME"]))

    members = list_members(CATALOG_URL)
    for v in VARIABLES:
        print(f"{v}: {len(members[v])} ensemble member files")

    weights = None
    # records[fips] -> list of row dicts
    records = {f: {} for f in counties["FIPS"]}      # records[fips][(ens,lead)] = rowdict
    for v in VARIABLES:
        for k, (label, url) in enumerate(members[v]):
            try:
                if weights is None:
                    ds = xr.open_dataset(url)
                    da = ds[_data_var(ds)]
                    tdim = [d for d in da.dims if d not in ("lat", "lon")][0]
                    da = da.rename({tdim: "day"}).sortby("lat").sortby("lon")
                    da = da.sel(lat=slice(*lat_bounds), lon=slice(*lon_bounds))
                    weights = build_weights(da.lat.values, da.lon.values, counties)
                series, valid = member_series(url, lat_bounds, lon_bounds, weights)
            except Exception as e:
                print(f"  [skip] {v} {label}: {e}")
                continue
            for fips in counties["FIPS"]:
                vals = series[fips]
                for d in range(len(vals)):
                    key = (label, d + 1)
                    row = records[fips].setdefault(
                        key, {"ensemble": label, "lead_day": d + 1,
                              "valid_date": valid[d] if d < len(valid) else None})
                    row[v] = vals[d]
            print(f"  {v} member {k+1}/{len(members[v])} ({label}) ok")

    os.makedirs(OUT_DIR, exist_ok=True)
    for fips in counties["FIPS"]:
        df = pd.DataFrame(list(records[fips].values()))
        if df.empty:
            continue
        df["county"] = name_of[fips]; df["FIPS"] = fips
        cols = ["ensemble", "lead_day", "valid_date", "FIPS", "county"] + VARIABLES
        df = df.reindex(columns=cols).sort_values(["ensemble", "lead_day"])
        if "valid_date" in df:
            df["valid_date"] = pd.to_datetime(df["valid_date"]).dt.date
        safe = "".join(ch if ch.isalnum() else "_" for ch in name_of[fips])
        path = os.path.join(OUT_DIR, f"s2s_forecast_{safe}_{fips}.csv")
        df.to_csv(path, index=False)
        print(f"wrote {path}  {df.shape}")


if __name__ == "__main__":
    main()
