#!/usr/bin/env python3
"""Calibrates the DESI matched-filter screen (desi_04_ew_screen.py) on
the 34 DESI DR1 DLAs with known Rafelski+2012 metallicity
(out/desi_zdla_vs_metal_velocity_offset.csv) - same role as
07_calibrate_screen.py for SDSS. All 34 are metal-rich/metal-poor by
literature measurement (not a "clean" system among them), so this reports
the false-negative rate only (fraction the screen wrongly calls
candidate_metalpoor)."""
import os
import pandas as pd
from importlib import import_module

ew4 = import_module("desi_04_ew_screen")

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "out")

def main():
    """Runs the DESI screen on the Rafelski-matched set and reports the
    false-negative rate with a Wilson CI."""
    matches = pd.read_csv(os.path.join(OUT, "desi_zdla_vs_metal_velocity_offset.csv"))
    targets = pd.read_csv(os.path.join(OUT, "desi_targets.csv"))
    cal = targets[targets["TARGETID"].isin(matches["desi_targetid"])].drop_duplicates("TARGETID")
    print(f"Калибровочная выборка: {len(cal)} систем (все с известной [M/H])")

    client = ew4.get_client()
    specs = ew4.fetch_batch(client, cal["TARGETID"].tolist())
    rows = []
    for row in cal.itertuples():
        rec = ew4.process_one(row, specs.get(row.TARGETID, []))
        m = matches[matches["desi_targetid"] == row.TARGETID].iloc[0]
        rec["MH"] = m["MH"]
        rows.append(rec)
        print(f"  ID={row.TARGETID} MH={m['MH']:.2f} -> {rec.get('status')}, "
              f"cand={rec.get('candidate_metalpoor')}")

    res = pd.DataFrame(rows)
    out_path = os.path.join(OUT, "desi_ew_screen_calibration.csv")
    res.to_csv(out_path, index=False)

    ok = res[res["status"] == "ok"]
    fn = int(ok["candidate_metalpoor"].sum())
    n = len(ok)
    from math import sqrt
    def wilson(k, n, z=1.96):
        if n == 0:
            return (float("nan"), float("nan"))
        p = k / n
        denom = 1 + z**2 / n
        center = p + z**2 / (2 * n)
        half = z * sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
        return ((center - half) / denom, (center + half) / denom)
    lo, hi = wilson(fn, n)
    print(f"\n=== КАЛИБРОВКА DESI EW-СКРИНА (n={n}) ===")
    print(f"False-negative (ложно 'чистые'): {fn}/{n} = {fn/n*100:.1f}% "
          f"[95% Wilson CI {lo*100:.1f}-{hi*100:.1f}%]")
    print(f"-> {out_path}")

if __name__ == "__main__":
    main()
