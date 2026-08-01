#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimal Path Threshold (OPT) method  --  Zhao, Horvat & Gao (2025), ERL 20 024048
Applied to compound extremes per Rhode Island county, for several event types.

Each event = two variables that must be simultaneously extreme (AND), e.g.:
  hot_wet    : hot (tmmx, high tail) AND wet (rh,  high tail)
  cold_windy : cold (tmmn, LOW tail) AND windy (vs_2m, high tail)
  cold_wet   : cold (tmmn, LOW tail) AND wet  (rh,    high tail)

Tail direction:
  high tail  -> extreme = value >= the k-th percentile  (large values extreme)
  low  tail  -> extreme = value <= the (100-k)-th pct   (SMALL values extreme; cold)

In both cases a higher grid level k = stricter = fewer days qualify, so the
occurrence-rate matrix P stays monotone non-increasing in both indices and the
OPT dynamic program (paper Eq. 2: max-sum monotone staircase) applies unchanged.

  p = sum(D_i)/N  (Eq.1); with MIN_DURATION=1 this is the joint-exceedance fraction.

Output:
  opt_compound_thresholds.csv          (one row per county x event x target rate)
  opt_paths/<county>_<event>.csv       (full optimal path; suffixed per event)

Requirements: pip install pandas numpy
"""

import os
import glob
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- 
# CONFIG
# --------------------------------------------------------------------------- 
IN_DIR = "gridmet_csv_converted"          # folder with the per-county CSVs

# column candidates (first found is used) for each physical variable
HOT   = ["tmmx", "tmmx_C"]
COLD  = ["tmmn", "tmmn_C"]
WET   = ["rh", "rh_mean"]
WINDY = ["vs_2m", "vs"]

# event types: (role, column-candidates, tail).  tail in {"high","low"}.
EVENTS = [
    {"name": "hot_wet",    "v1": ("hot",  HOT,  "high"), "v2": ("wet",   WET,   "high")},
    {"name": "cold_windy", "v1": ("cold", COLD, "low"),  "v2": ("windy", WINDY, "high")},
    {"name": "cold_wet",   "v1": ("cold", COLD, "low"),  "v2": ("wet",   WET,   "high")},
]

NC_EVENTS = [
    {"name": "hot",   "v1": ("hot",   HOT,   "high")},
    {"name": "cold",  "v1": ("cold",  COLD,  "low")},
    {"name": "wet",   "v1": ("wet",   WET,   "high")},
    {"name": "windy", "v1": ("windy", WINDY, "high")},
]

# Target occurrence rates. You wrote 0.5% / 1% / 5% -> 0.005 / 0.01 / 0.05.
# The PAPER uses per-mille (‰): for that use [0.0005, 0.001, 0.005] instead.
TARGET_RATES = [0.025, 0.05, 0.075]

PCT_MIN, PCT_MAX, STEP = 0.0, 100.0, 1.0  # grid level range for both variables
MIN_DURATION = 1                          # min consecutive days per event
OUT_SUMMARY  = "opt_compound_thresholds.csv"
OUT_PATH_DIR = "opt_paths"                # None to skip per-county path files
# --------------------------------------------------------------------------- 


def detect_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    if candidates is WET and {"rmax", "rmin"}.issubset(df.columns):
        df["rh"] = (df["rmax"] + df["rmin"]) / 2.0
        return "rh"
    raise KeyError(f"none of {candidates} in {list(df.columns)}")


def var_masks(x, tail, levels):
    """Return (mask[L,N], threshold[L], data_pct[L]) aligned to grid index.
    Higher grid level = stricter for BOTH tails."""
    x = np.asarray(x, float)
    if tail == "high":
        data_pct = levels                       # threshold sits at the k-th pct
        thr = np.percentile(x, levels)
        mask = x[None, :] >= thr[:, None]
    elif tail == "low":
        data_pct = 100.0 - levels               # threshold sits at the (100-k)-th pct
        thr = np.percentile(x, data_pct)
        mask = x[None, :] <= thr[:, None]
    else:
        raise ValueError(tail)
    return mask, thr, data_pct


def min_duration_filter(mask, min_dur):
    if min_dur <= 1:
        return mask
    m = mask.copy(); n = len(m); i = 0
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]:
                j += 1
            if (j - i) < min_dur:
                m[i:j] = False
            i = j
        else:
            i += 1
    return m


def occurrence_matrix(x1, tail1, x2, tail2, levels, min_dur=1):
    N = len(x1)
    m1, thr1, p1 = var_masks(x1, tail1, levels)
    m2, thr2, p2 = var_masks(x2, tail2, levels)
    L = len(levels)
    if min_dur <= 1:
        P = (m1.astype(np.float64) @ m2.astype(np.float64).T) / N
    else:
        P = np.empty((L, L))
        for i in range(L):
            for j in range(L):
                P[i, j] = min_duration_filter(m1[i] & m2[j], min_dur).sum() / N
    return P, thr1, p1, thr2, p2


def optimal_path(P):
    """DP (Eq.2): max-sum monotone staircase from (0,0) to (L-1,L-1)."""
    L = P.shape[0]
    dp = np.full((L, L), -np.inf)
    back = np.empty((L, L), dtype="U1")
    dp[0, 0] = P[0, 0]
    for i in range(L):
        for j in range(L):
            if i == 0 and j == 0:
                continue
            best, b = -np.inf, ""
            if i > 0 and dp[i - 1, j] > best:
                best, b = dp[i - 1, j], "i"
            if j > 0 and dp[i, j - 1] > best:
                best, b = dp[i, j - 1], "j"
            dp[i, j] = best + P[i, j]
            back[i, j] = b
    path = []; i = j = L - 1
    while not (i == 0 and j == 0):
        path.append((i, j))
        if back[i, j] == "i":
            i -= 1
        else:
            j -= 1
    path.append((0, 0)); path.reverse()
    return path, dp[L - 1, L - 1]


def run_pair(df, v1, v2, levels, targets, min_dur):
    r1, col1, t1 = v1
    r2, col2, t2 = v2
    d = df[[col1, col2]].dropna()
    P, thr1, dp1, thr2, dp2 = occurrence_matrix(
        d[col1].values, t1, d[col2].values, t2, levels, min_dur)
    path, _ = optimal_path(P)
    rows = [{f"{r1}_pct": dp1[i], f"{r1}_threshold": thr1[i],
             f"{r2}_pct": dp2[j], f"{r2}_threshold": thr2[j],
             "p": P[i, j]} for (i, j) in path]
    path_df = pd.DataFrame(rows)
    picks = []
    pmax = path_df["p"].max()
    for pt in targets:
        r = path_df.loc[(path_df["p"] - pt).abs().idxmin()]
        picks.append({"target_rate": pt, "achieved_p": r["p"],
                      "v1_role": r1, "v1_col": col1, "v1_tail": t1,
                      "v1_pct": r[f"{r1}_pct"], "v1_threshold": r[f"{r1}_threshold"],
                      "v2_role": r2, "v2_col": col2, "v2_tail": t2,
                      "v2_pct": r[f"{r2}_pct"], "v2_threshold": r[f"{r2}_threshold"],
                      "reachable": bool(pmax >= pt)})
    return pd.DataFrame(picks), path_df

# For non-compound events
def run_single(df, v1, targets):
    r1, col1, t1 = v1
    x = df[col1].dropna().values
    rows = []

    for pt in targets:
        # High-tail 2.5% event -> 97.5th percentile.
        # Low-tail  2.5% event -> 2.5th percentile.
        pct = 100.0 - pt * 100.0 if t1 == "high" else pt * 100.0

        thr = np.percentile(x, pct)

        rows.append({
            "target_rate": pt,
            "achieved_p": pt,
            "v1_role": r1,
            "v1_col": col1,
            "v1_tail": t1,
            "v1_pct": pct,
            "v1_threshold": thr,
            "v2_role": "",
            "v2_col": "",
            "v2_tail": "",
            "v2_pct": np.nan,
            "v2_threshold": np.nan,
            "reachable": True,
        })

    return pd.DataFrame(rows)

def main():
    levels = np.round(np.arange(PCT_MIN, PCT_MAX + 1e-9, STEP), 6)
    files = sorted(glob.glob(os.path.join(IN_DIR, "*.csv")))
    if not files:
        raise SystemExit(f"No CSVs in '{IN_DIR}'.")
    if OUT_PATH_DIR:
        os.makedirs(OUT_PATH_DIR, exist_ok=True)

    summary = []
    for fp in files:
        df = pd.read_csv(fp)
        county = df["county"].iloc[0] if "county" in df.columns else \
            os.path.splitext(os.path.basename(fp))[0]
        safe = "".join(ch if ch.isalnum() else "_" for ch in str(county))
        for ev in EVENTS:
            v1 = (ev["v1"][0], detect_col(df, ev["v1"][1]), ev["v1"][2])
            v2 = (ev["v2"][0], detect_col(df, ev["v2"][1]), ev["v2"][2])
            picks, path_df = run_pair(df, v1, v2, levels, TARGET_RATES, MIN_DURATION)
            picks.insert(0, "event", ev["name"])
            picks.insert(0, "county", county)
            summary.append(picks)
            if OUT_PATH_DIR:                      # suffix avoids overwrite
                path_df.to_csv(os.path.join(OUT_PATH_DIR,
                               f"{safe}_{ev['name']}.csv"), index=False)
            print(f"[{county}/{ev['name']}] {v1[0]}({v1[1]},{v1[2]}) "
                  f"& {v2[0]}({v2[1]},{v2[2]})")
            for _, r in picks.iterrows():
                print(f"    {r['target_rate']*100:>5.2f}% -> "
                      f"{r['v1_role']} p{r['v1_pct']:g}={r['v1_threshold']:.2f}, "
                      f"{r['v2_role']} p{r['v2_pct']:g}={r['v2_threshold']:.2f} "
                      f"(got {r['achieved_p']*100:.3f}%)")

        for ev in NC_EVENTS:
            v1 = (ev["v1"][0], detect_col(df, ev["v1"][1]), ev["v1"][2])
            picks = run_single(df, v1, TARGET_RATES)
            picks.insert(0, "event", ev["name"])
            picks.insert(0, "county", county)
            summary.append(picks)
            print(f"[{county}/{ev['name']}] {v1[0]}({v1[1]},{v1[2]})")
            for _, r in picks.iterrows():
                print(f"    {r['target_rate']*100:>5.2f}% -> "
                      f"{r['v1_role']} p{r['v1_pct']:g}={r['v1_threshold']:.2f}")

    out = pd.concat(summary, ignore_index=True)
    out.to_csv(OUT_SUMMARY, index=False)
    print("\nwrote", OUT_SUMMARY, out.shape)


if __name__ == "__main__":
    main()
