#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert and standardize the per-county S2S forecast CSVs.

Input  (from download_s2s_forecast.py), per county:
    ensemble, lead_day, valid_date, FIPS, county, tmmx, tmmn, vpd, vs
    units: tmmx/tmmn = K, vpd = kPa, vs = m/s (10 m)

Step 1 - CONVERT (identical units/definitions to the historical convert_units.py):
    tmmx, tmmn (K)  -> tmmx_C, tmmn_C   (Celsius)
    vpd (kPa) + T   -> rh_mean (%)      RH = 100*(1 - vpd/es_mean),
                                        es(T)=0.6108*exp(17.27T/(T+237.3)),
                                        es_mean=[es(Tmax)+es(Tmin)]/2
    vs (10 m)       -> vs_2m (2 m)      FAO-56 Eq.47

Step 2 - STANDARDIZE (z-score) each converted variable against the HISTORICAL
    climatology, per county, per variable:   z = (x - mu) / sigma
    mu/sigma come from the converted gridMET baseline (BASELINE_DIR). By default
    they are SEASONAL: for each forecast valid-date, mu/sigma are taken from
    historical days within +/- DOY_WINDOW days-of-year (so a July forecast is
    standardized against July climatology, not the annual mean). Set
    DOY_WINDOW=None to use whole-record statistics. If no baseline is found for a
    county, it falls back to standardizing against the forecast's own statistics
    (less meaningful - no climatology - and a warning is printed).

Output: one CSV per county in OUT_DIR with the converted columns plus *_z columns:
    ensemble, lead_day, valid_date, FIPS, county,
    tmmx_C, tmmn_C, rh_mean, vs_2m, tmmx_C_z, tmmn_C_z, rh_mean_z, vs_2m_z

Requirements: pip install pandas numpy
"""

import os
import glob
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
IN_DIR       = "s2s_forecast_csv"          # per-county S2S forecast CSVs
OUT_DIR      = "s2s_forecast_processed"    # output folder
BASELINE_DIR = "gridmet_csv_converted"     # historical converted CSVs; None -> self
DOY_WINDOW   = 15                          # +/- days-of-year for seasonal climatology; None=whole record
WIND_Z       = 10.0
ROUND        = 4
CONV_VARS    = ["tmmx_C", "tmmn_C", "rh_mean", "vs_2m"]   # variables to standardize
# ---------------------------------------------------------------------------

WIND_FACTOR = 4.87 / np.log(67.8 * WIND_Z - 5.42)         # FAO-56; z=10 -> 0.74797


def es_kpa(T_C):
    return 0.6108 * np.exp(17.27 * T_C / (T_C + 237.3))


def to_celsius(s):
    s = pd.to_numeric(s, errors="coerce")
    return s - 273.15 if s.median() > 100 else s


def convert(df):
    """Add converted columns to a forecast frame (any extra columns preserved)."""
    tmax_c = to_celsius(df["tmmx"]); tmin_c = to_celsius(df["tmmn"])
    es_mean = 0.5 * (es_kpa(tmax_c) + es_kpa(tmin_c))
    out = df.copy()
    out["tmmx_C"] = tmax_c
    out["tmmn_C"] = tmin_c
    out["rh_mean"] = (100.0 * (1.0 - df["vpd"] / es_mean)).clip(0.0, 100.0)
    out["vs_2m"] = df["vs"] * WIND_FACTOR
    return out


def _circ_dist(doy, t, period=366):
    d = np.abs(doy - t)
    return np.minimum(d, period - d)


def baseline_stats(baseline_dir, doy_window):
    """Return {FIPS: {var: (mu366, sd366)}} with arrays indexed by day-of-year 1..366.
    For whole-record stats (doy_window=None) every doy entry holds the same value."""
    stats = {}
    for fp in glob.glob(os.path.join(baseline_dir, "*.csv")):
        h = pd.read_csv(fp)
        if "FIPS" not in h.columns or "date" not in h.columns:
            continue
        fips = str(h["FIPS"].iloc[0])
        doy = pd.to_datetime(h["date"]).dt.dayofyear.values
        per_var = {}
        for v in CONV_VARS:
            if v not in h.columns:
                continue
            vals = pd.to_numeric(h[v], errors="coerce").values
            mu = np.full(367, np.nan); sd = np.full(367, np.nan)
            if doy_window is None:
                m = np.nanmean(vals); s = np.nanstd(vals)
                mu[:] = m; sd[:] = s
            else:
                for t in range(1, 367):
                    sel = vals[_circ_dist(doy, t) <= doy_window]
                    sel = sel[~np.isnan(sel)]
                    if sel.size:
                        mu[t] = sel.mean(); sd[t] = sel.std()
            per_var[v] = (mu, sd)
        stats[fips] = per_var
    return stats


def standardize(df, fips, stats):
    """Add *_z columns using historical stats; fall back to self-stats if absent."""
    doy = pd.to_datetime(df["valid_date"], errors="coerce").dt.dayofyear.fillna(1).astype(int).values
    have = fips in stats
    for v in CONV_VARS:
        if have and v in stats[fips]:
            mu_arr, sd_arr = stats[fips][v]
            mu = mu_arr[doy]; sd = sd_arr[doy]
        else:                                   # fallback: forecast's own stats
            x = df[v].values
            mu = np.full(len(df), np.nanmean(x)); sd = np.full(len(df), np.nanstd(x))
        sd = np.where(sd > 0, sd, np.nan)
        df[v + "_z"] = (df[v].values - mu) / sd
    return df, have


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    stats = baseline_stats(BASELINE_DIR, DOY_WINDOW) if BASELINE_DIR else {}
    if BASELINE_DIR:
        print(f"baseline climatology loaded for FIPS: {sorted(stats)} "
              f"(DOY_WINDOW={DOY_WINDOW})")
    files = sorted(glob.glob(os.path.join(IN_DIR, "*.csv")))
    if not files:
        raise SystemExit(f"No CSVs in '{IN_DIR}'.")

    base_cols = ["ensemble", "lead_day", "valid_date", "FIPS", "county"]
    z_cols = [v + "_z" for v in CONV_VARS]
    for fp in files:
        df = pd.read_csv(fp)
        fips = str(df["FIPS"].iloc[0]) if "FIPS" in df.columns else ""
        df = convert(df)
        df, used_baseline = standardize(df, fips, stats)
        if not used_baseline:
            print(f"  [warn] {os.path.basename(fp)}: no baseline for FIPS {fips}; "
                  f"standardized against the forecast's own statistics.")
        keep = [c for c in base_cols if c in df.columns] + CONV_VARS + z_cols
        out = df[keep]
        if ROUND is not None:
            num = out.select_dtypes("number").columns
            out[num] = out[num].round(ROUND)
        dst = os.path.join(OUT_DIR, os.path.basename(fp))
        out.to_csv(dst, index=False)
        print(f"wrote {dst}  {out.shape}")


if __name__ == "__main__":
    main()
