#!/usr/bin/env python3
"""v4-trigger candidate, edge-of-window re-scan (author go-ahead
2026-07-11). out/highres_purity.csv's Stage 2b run used find_peak()'s
default SEARCH_KMS=500 (inherited unchanged from the SDSS/DESI protocol),
and one target (39627762889133674) had all 4 detected lines sitting
exactly at dv=+-500 - the scan boundary, not necessarily the true peak.
Checked first, not assumed: out of all 10 usable targets' 29 detections,
this is the ONLY target with any line at |dv|==500 (verified against the
CSV directly) - the re-scan below is scoped to exactly the targets that
actually need it, not applied blanket.

Re-runs find_peak() UNCHANGED except search_kms=1500 (3x the original
window) on the same best-per-line products already selected in Stage 2b,
then re-classifies. Does NOT silently overwrite: the pre-rescan sigma/dv/
ew/status are preserved in *_orig columns, and rescan_window_kms marks
which rows were touched (NaN = untouched, 1500 = rescanned) - a reader of
the CSV can always recover what Stage 2b originally reported."""
import os
import warnings
from importlib import import_module

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ew = import_module("05_ew_screen")
h3 = import_module("hires_03_measure_purity")

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "out")
RESCAN_SEARCH_KMS = 1500.0


def rescan_target(target_id, z_abs, provenance, purity_rows):
    """Re-measures all 4 lines for one target at the wider search window,
    reusing the exact same product per line that Stage 2b already picked."""
    peaks = {}
    line_meta = {}
    for _, row in purity_rows.iterrows():
        name = row["line"]
        if pd.isna(row["archive"]):  # was unavailable in Stage 2b - nothing to re-scan
            peaks[name] = None
            line_meta[name] = None
            continue
        wave, flux, ivar, and_mask = h3.load_spectrum(target_id, row["archive"], row["product_id"])
        p = ew.find_peak(wave, flux, ivar, and_mask, z_abs, ew.LINES[name], search_kms=RESCAN_SEARCH_KMS)
        peaks[name] = p
        line_meta[name] = {"archive": row["archive"], "instrument": row["instrument"],
                            "product_id": row["product_id"]}
    n_ok, detected, candidate, solo_hits, pair_hit = ew.classify(peaks)
    target_cls = h3.survey_class(provenance, n_ok, detected)

    out = {}
    for name in ew.LINES:
        p = peaks[name]
        if p is None:
            continue  # unavailable lines stay untouched - nothing to rescan
        line_detected = (name in solo_hits) or h3.line_is_pair_member(name, peaks, pair_hit)
        ew_rest = p["sigma"] * p["ew_err_rest"]
        out[name] = {
            "status": "detected" if line_detected else "upper_limit",
            "sigma": p["sigma"],
            "ew_ang": ew_rest if line_detected else 3 * p["ew_err_rest"],
            "dv_kms": p["dv"] if line_detected else np.nan,
            "n_lines_ok": n_ok, "solo_hits": ",".join(solo_hits), "pair_hit": pair_hit,
            "target_class": target_cls,
        }
    return out


def main():
    pur = pd.read_csv(os.path.join(OUT, "highres_purity.csv"))
    pur["ID"] = pur["ID"].astype(str)

    det = pur[pur["status"] == "detected"]
    edge_ids = sorted(det[det["dv_kms"].abs() == 500.0]["ID"].unique())
    print(f"Целей на границе окна (|dv|==500): {len(edge_ids)} -> {edge_ids}")
    if not edge_ids:
        print("Перепрогон не нужен.")
        return

    for col in ("sigma_orig", "dv_kms_orig", "ew_ang_orig", "status_orig"):
        if col not in pur.columns:
            pur[col] = np.nan
    if "rescan_window_kms" not in pur.columns:
        pur["rescan_window_kms"] = np.nan

    targets = pd.read_csv(os.path.join(OUT, "merged_candidates_clean.csv"))
    targets["ID"] = targets["ID"].astype(str)

    n_changed = 0
    for tid in edge_ids:
        trow = targets[targets["ID"] == tid].iloc[0]
        target_rows = pur[pur["ID"] == tid]
        results = rescan_target(tid, trow["z_abs"], trow["provenance"], target_rows)
        print(f"\n=== {tid} (z_abs={trow['z_abs']:.4f}, {trow['provenance']}) ===")
        for name, new in results.items():
            idx = pur[(pur["ID"] == tid) & (pur["line"] == name)].index[0]
            old_sigma, old_dv = pur.at[idx, "sigma"], pur.at[idx, "dv_kms"]
            old_status = pur.at[idx, "status"]
            print(f"  {name}: sigma {old_sigma:.2f}->{new['sigma']:.2f}, "
                  f"dv {old_dv:.0f}->{new['dv_kms'] if pd.notna(new['dv_kms']) else float('nan'):.0f} km/s, "
                  f"status {old_status}->{new['status']}")
            pur.at[idx, "sigma_orig"] = old_sigma
            pur.at[idx, "dv_kms_orig"] = old_dv
            pur.at[idx, "ew_ang_orig"] = pur.at[idx, "ew_ang"]
            pur.at[idx, "status_orig"] = old_status
            pur.at[idx, "rescan_window_kms"] = RESCAN_SEARCH_KMS
            pur.at[idx, "status"] = new["status"]
            pur.at[idx, "sigma"] = new["sigma"]
            pur.at[idx, "ew_ang"] = new["ew_ang"]
            pur.at[idx, "dv_kms"] = new["dv_kms"]
            pur.at[idx, "n_lines_ok"] = new["n_lines_ok"]
            pur.at[idx, "solo_hits"] = new["solo_hits"]
            pur.at[idx, "pair_hit"] = new["pair_hit"]
            pur.at[idx, "target_class"] = new["target_class"]
            n_changed += 1

    out_path = os.path.join(OUT, "highres_purity.csv")
    pur.to_csv(out_path, index=False)
    print(f"\nИзменено строк: {n_changed}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
