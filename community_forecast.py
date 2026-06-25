#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Community compound-event forecast (simple version)
===================================================
Plain question it answers, per county and per day for the next few weeks:

    "What share of the forecast scenarios show an extreme <event> day?"

For each future day we count the share of ensemble members in which BOTH
conditions of an event cross the historical "extreme" line, and turn that share
into an easy risk label (Low / Moderate / High / Very High).

Severity levels: your OPT thresholds come at three occurrence rates. We keep ALL
THREE and give each a plain label so the public can choose how rare "extreme"
should be:
    5%   -> "Notable"  (uncommon)
    1%   -> "Severe"   (rare)
    0.5% -> "Extreme"  (very rare)

Events (using your historical OPT thresholds):
    Hot & Humid  = hot day  AND humid day
    Cold & Windy = cold day AND windy day
    Cold & Wet   = cold day AND humid day

Inputs:
    THRESH_CSV : opt_compound_thresholds.csv   (historical thresholds; all severities)
    FCST_DIR   : per-county forecast CSVs        (the ensemble forecast)

Outputs:
    community_forecast.csv   county, event, level, date, chance_pct, risk
    forecast_data.js         data the web page reads (all counties/events/levels)
    forecast_<event>.png     static calendar for the PNG_SEVERITY level
    community_summary.txt     one plain sentence per county & event (PNG_SEVERITY)

Requirements: pip install pandas numpy matplotlib
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# ---------------------------------------------------------------------------
THRESH_CSV = "opt_compound_thresholds.csv"
FCST_DIR   = "s2s_forecast_processed"
OUT_CSV    = "community_forecast.csv"
OUT_JS     = "forecast_data.js"
OUT_DIR    = "."

# occurrence rate -> (plain label, plain note). Order = most frequent first.
SEVERITY_LEVELS = [
    (0.05,  "Notable", "uncommon · about 18 days a year historically"),
    (0.01,  "Severe",  "rare · about 4 days a year"),
    (0.005, "Extreme", "very rare · about 2 days a year"),
]
PNG_SEVERITY = 0.05          # which level the static PNG/summary use

EVENT_LABELS = {"hot_wet": "Hot & Humid",
                "cold_windy": "Cold & Windy",
                "cold_wet": "Cold & Wet"}

# risk bins: (upper bound %, label, color)
RISK_BINS  = [(10, "Low", "#2f8f46"),
              (30, "Moderate", "#f1c33b"),
              (60, "High", "#ec7a2a"),
              (101, "Very High", "#d23b3b")]
# ---------------------------------------------------------------------------


def risk_label(pct):
    for hi, label, _ in RISK_BINS:
        if pct < hi:
            return label
    return RISK_BINS[-1][1]


def level_for_rate(rate):
    for r, lab, _ in SEVERITY_LEVELS:
        if np.isclose(rate, r):
            return lab
    return None


def exceed(x, thr, tail):
    return (x >= thr) if tail == "high" else (x <= thr)


def compute():
    th = pd.read_csv(THRESH_CSV)
    rows = []
    for fp in sorted(glob.glob(os.path.join(FCST_DIR, "*.csv"))):
        f = pd.read_csv(fp)
        county = str(f["county"].iloc[0])
        f["valid_date"] = pd.to_datetime(f["valid_date"]).dt.date
        for _, ev in th[th["county"] == county].iterrows():
            level = level_for_rate(ev["target_rate"])
            if level is None:          # threshold at a rate we don't expose
                continue
            for date, g in f.groupby("valid_date"):
                hit = (exceed(g[ev["v1_col"]].values, ev["v1_threshold"], ev["v1_tail"]) &
                       exceed(g[ev["v2_col"]].values, ev["v2_threshold"], ev["v2_tail"]))
                chance = round(100 * hit.mean(), 1)
                rows.append(dict(county=county,
                                 event=EVENT_LABELS.get(ev["event"], ev["event"]),
                                 level=level, date=date,
                                 chance_pct=chance, risk=risk_label(chance)))
    return pd.DataFrame(rows).sort_values(["event", "level", "county", "date"])


def calendar_png(df, event, level):
    d = df[(df["event"] == event) & (df["level"] == level)]
    if d.empty:
        return
    counties = sorted(d["county"].unique())
    piv = d.pivot_table(index="county", columns="date", values="chance_pct").reindex(counties)
    bounds = [0] + [b[0] for b in RISK_BINS]
    cmap = ListedColormap([b[2] for b in RISK_BINS]); norm = BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots(figsize=(min(16, 0.45 * piv.shape[1] + 3), 0.6 * piv.shape[0] + 2))
    ax.imshow(piv.values, aspect="auto", cmap=cmap, norm=norm)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7)
    ax.set_yticks(range(len(counties))); ax.set_yticklabels(counties, fontsize=9)
    cols = list(piv.columns); ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([pd.Timestamp(c).strftime("%b %d") for c in cols], rotation=90, fontsize=7)
    ax.set_title(f"Chance of an extreme {event} day · {level} level (next {piv.shape[1]} days)\n"
                 f"numbers = % of forecast scenarios", fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in RISK_BINS]
    ax.legend(handles, ["Low (<10%)", "Moderate (10-30%)", "High (30-60%)", "Very High (>60%)"],
              loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, title="Risk")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"forecast_{event.replace(' & ', '_').replace(' ', '')}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def summary_txt(df, level):
    d = df[df["level"] == level]
    lines = [f"Severity level shown: {level}", ""]
    for (county, event), g in d.groupby(["county", "event"]):
        peak = g.loc[g["chance_pct"].idxmax()]
        elevated = g[g["chance_pct"] >= 30]["date"].tolist()
        msg = (f"{county} — {event}: highest chance {peak['chance_pct']:.0f}% "
               f"on {pd.Timestamp(peak['date']).strftime('%b %d')} ({peak['risk']}).")
        msg += (" Days to watch (High+): " +
                ", ".join(pd.Timestamp(x).strftime("%b %d") for x in elevated) + "."
                if elevated else " No high-risk days in this period.")
        lines.append(msg)
    text = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "community_summary.txt"), "w") as fh:
        fh.write(text + "\n")
    print("\n" + text)


def write_js(df):
    rows = [dict(county=r["county"], event=r["event"], level=r["level"],
                 date=pd.Timestamp(r["date"]).strftime("%Y-%m-%d"),
                 chance=float(r["chance_pct"]), risk=r["risk"])
            for _, r in df.iterrows()]
    present = list(df["level"].unique())
    levels = [dict(label=lab, note=note) for r, lab, note in SEVERITY_LEVELS if lab in present]
    meta = dict(updated=pd.Timestamp.today().strftime("%Y-%m-%d"),
                horizon_days=int(df["date"].nunique()),
                levels=levels,
                source="gridMET history + CFSv2-METDATA ensemble forecast")
    js = ("// Auto-generated by community_forecast.py — re-run daily to refresh.\n"
          "window.FORECAST_META = " + json.dumps(meta) + ";\n"
          "window.FORECAST_DATA = " + json.dumps(rows) + ";\n")
    with open(os.path.join(OUT_DIR, OUT_JS), "w") as fh:
        fh.write(js)
    print("wrote", os.path.join(OUT_DIR, OUT_JS))


def main():
    df = compute()
    df.to_csv(OUT_CSV, index=False)
    print("wrote", OUT_CSV, df.shape)
    png_level = level_for_rate(PNG_SEVERITY)
    for event in df["event"].unique():
        calendar_png(df, event, png_level)
    summary_txt(df, png_level)
    write_js(df)


if __name__ == "__main__":
    main()
