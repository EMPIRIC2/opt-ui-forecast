#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-process the per-county gridMET CSVs (now: date, FIPS, county, tmmx, tmmn, vpd, vs):

  tmmx, tmmn (K)   -> tmmx_C, tmmn_C   (degrees Celsius;  C = K - 273.15)
  vpd  (kPa) + T   -> rh_mean (%)      (relative humidity derived from VPD + temp)
  vs   (10 m, m/s) -> vs_2m  (2 m, m/s) FAO-56 log wind profile, Eq. 47

Why derive RH instead of using VPD directly:
  VPD = es(T) - ea is dominated by the exponential temperature term es(T), so it is
  strongly collinear with temperature. Using VPD as the "wet/dry" axis in a compound
  hot/cold extreme would make the two thresholds non-independent. RH = 100*ea/es(T)
  removes the leading temperature dependence and is the appropriate humidity axis.

  From VPD = es(T)-ea and RH = 100*ea/es(T):   RH = 100*(1 - VPD/es(T))
  Saturation vapour pressure (Tetens, kPa):    es(T) = 0.6108*exp(17.27*T/(T+237.3))
  gridMET vpd is a DAILY-MEAN built from Tmax/Tmin, so we use the mean saturation
  vapour pressure es_mean = [es(Tmax)+es(Tmin)]/2  (FAO-56 convention):
       rh_mean = 100*(1 - vpd / es_mean)     (clipped to [0, 100])

Reads every CSV in IN_DIR and writes a converted copy (same filename) to OUT_DIR.

Requirements: pip install pandas numpy
"""

import os
import glob
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
IN_DIR    = "gridmet_csv"             # folder produced by the aggregation script
OUT_DIR   = "gridmet_csv_converted"   # new folder for the converted CSVs
WIND_Z    = 10.0                      # measurement height of gridMET wind (m)
ROUND     = 4                         # decimals to round outputs (None = no rounding)
KEEP_VPD  = False                     # set True to also keep the raw vpd column
# ---------------------------------------------------------------------------

WIND_FACTOR = 4.87 / np.log(67.8 * WIND_Z - 5.42)   # FAO-56 Eq.47; z=10 -> 0.74797


def es_kpa(T_C):
    """Saturation vapour pressure (kPa) from temperature in Celsius (Tetens)."""
    return 0.6108 * np.exp(17.27 * T_C / (T_C + 237.3))


def to_celsius(s):
    """Convert a temperature series to Celsius; auto-detect Kelvin vs Celsius."""
    s = pd.to_numeric(s, errors="coerce")
    return s - 273.15 if s.median() > 100 else s


def convert(df):
    tmax_c = to_celsius(df["tmmx"])
    tmin_c = to_celsius(df["tmmn"])
    es_mean = 0.5 * (es_kpa(tmax_c) + es_kpa(tmin_c))
    rh_mean = 100.0 * (1.0 - df["vpd"] / es_mean)        # VPD must be in kPa

    out = pd.DataFrame()
    out["date"]   = df["date"]
    out["FIPS"]   = df["FIPS"]
    out["county"] = df["county"]
    out["tmmx_C"] = tmax_c
    out["tmmn_C"] = tmin_c
    out["rh_mean"] = rh_mean.clip(0.0, 100.0)            # RH derived from VPD + T
    if KEEP_VPD:
        out["vpd"] = df["vpd"]
    out["vs_2m"]  = df["vs"] * WIND_FACTOR               # 10 m -> 2 m wind
    if ROUND is not None:
        num = out.select_dtypes("number").columns
        out[num] = out[num].round(ROUND)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(IN_DIR, "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files found in '{IN_DIR}'. Set IN_DIR correctly.")
    print(f"wind 10m->2m factor = {WIND_FACTOR:.5f}")
    for fp in files:
        df = pd.read_csv(fp)
        out = convert(df)
        dst = os.path.join(OUT_DIR, os.path.basename(fp))
        out.to_csv(dst, index=False)
        print(f"wrote {dst}  {out.shape}")


if __name__ == "__main__":
    main()
