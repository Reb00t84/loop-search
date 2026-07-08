#!/usr/bin/env python3
"""Reproduces the RETRACTED Stage-2 box-car screen (v1, fixed +-300 km/s
integration window; see CLAUDE.md for why it was retracted) purely to
regenerate a public, citable primary artifact backing the "most box-car
'candidates' actually contain a >4sigma line the box-car missed" claim
(CLAUDE.md project rule 5: numbers cited elsewhere need a regeneratable
source, not a paragraph restating them). The box-car method itself is not
used for anything in this repository's actual results
(out/final_candidates.csv comes from the matched-filter screen,
05_ew_screen.py).

Восстановление ОТОЗВАННОГО box-car скрина (v1) исключительно для
регенерации первичного артефакта под соответствующую цифру (правило 5
в CLAUDE.md) — сам box-car нигде в реальных результатах репозитория
не используется."""
import os
import time
import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from astropy.io import fits
from io import BytesIO
from importlib import import_module

ew = import_module("05_ew_screen")  # reuse fetch_spec/LINES/C_KMS

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "out")

EW_MAX_SANE = 5.0
EW_UL_INFORMATIVE = 0.3
BOXCAR_WINDOW_KMS = 300
CONT_KMS = (600, 4000)
RECHECK_WINDOW_KMS = 500
RECHECK_SIGMA = 4.0

def boxcar_measure(wave, flux, ivar, and_mask, z_abs):
    """Original v1 box-car EW measurement (retracted method)."""
    out = {}
    for name, lam0 in ew.LINES.items():
        lam_obs = lam0 * (1 + z_abs)
        dv = (wave - lam_obs) / lam_obs * ew.C_KMS
        good = (ivar > 0) & (and_mask == 0)
        cont_sel = good & (np.abs(dv) > CONT_KMS[0]) & (np.abs(dv) < CONT_KMS[1])
        line_sel = good & (np.abs(dv) < BOXCAR_WINDOW_KMS)
        if cont_sel.sum() < 15 or line_sel.sum() < 2:
            out[name] = (np.nan, np.nan, False, False)
            continue
        cont = np.median(flux[cont_sel])
        if not np.isfinite(cont) or cont <= 0:
            out[name] = (np.nan, np.nan, False, False)
            continue
        dlam = np.gradient(wave)
        f, iv, dl = flux[line_sel], ivar[line_sel], dlam[line_sel]
        e = np.sum((1 - f / cont) * dl) / (1 + z_abs)
        sigma_f = 1.0 / np.sqrt(iv)
        e_err = np.sqrt(np.sum((sigma_f / cont * dl) ** 2)) / (1 + z_abs)
        ok = abs(e) < EW_MAX_SANE and e_err < EW_MAX_SANE
        out[name] = (e, e_err, bool(e > 3 * e_err) if ok else False, ok)
    return out

def single_pixel_recheck(wave, flux, ivar, and_mask, z_abs):
    """Objective recheck: max single-pixel depth-significance in +-500 km/s
    per line (the check that exposed the box-car's blind spots)."""
    best_sigma, best_line, best_dv = 0.0, None, None
    for name, lam0 in ew.LINES.items():
        lam_obs = lam0 * (1 + z_abs)
        dv = (wave - lam_obs) / lam_obs * ew.C_KMS
        good = (ivar > 0) & (and_mask == 0)
        cont_sel = good & (np.abs(dv) > CONT_KMS[0]) & (np.abs(dv) < CONT_KMS[1])
        search = good & (np.abs(dv) < RECHECK_WINDOW_KMS)
        if search.sum() < 3 or cont_sel.sum() < 10:
            continue
        cont = np.median(flux[cont_sel])
        if not np.isfinite(cont) or cont <= 0:
            continue
        imin = np.argmin(flux[search])
        depth = 1 - flux[search][imin] / cont
        local_err = 1 / np.sqrt(ivar[search][imin]) / cont
        if local_err <= 0:
            continue
        sig = depth / local_err
        if sig > best_sigma:
            best_sigma, best_line, best_dv = sig, name, float(dv[search][imin])
    return best_sigma, best_line, best_dv

def process_one(row):
    content = ew.fetch_spec(int(row.Plate), int(row.MJD), int(row.Fiber))
    if content is None:
        return {"ID": row.ID, "status": "fetch_failed"}
    try:
        with fits.open(BytesIO(content)) as d:
            t = d[1].data
            wave, flux, ivar, mask = 10 ** t["loglam"], t["flux"], t["ivar"], t["and_mask"]
    except Exception as e:
        return {"ID": row.ID, "status": f"read_failed:{e}"}

    res = boxcar_measure(wave, flux, ivar, mask, row.zCNN)
    n_ok, n_det, max_ul = 0, 0, 0.0
    for name in ew.LINES:
        e, e_err, det, ok = res[name]
        if ok:
            n_ok += 1
            if det:
                n_det += 1
            max_ul = max(max_ul, 3 * e_err)
    is_boxcar_candidate = bool(n_ok >= 3 and n_det == 0 and max_ul < EW_UL_INFORMATIVE)

    rec = {"ID": row.ID, "zCNN": row.zCNN, "SNR": row.SNR, "status": "ok",
           "n_lines_ok": n_ok, "max_3sig_UL_boxcar": max_ul,
           "boxcar_candidate": is_boxcar_candidate}
    if is_boxcar_candidate:
        sigma, line, dv = single_pixel_recheck(wave, flux, ivar, mask, row.zCNN)
        rec["recheck_max_sigma"] = sigma
        rec["recheck_line"] = line
        rec["recheck_dv_kms"] = dv
        rec["missed_gt_4sigma"] = bool(sigma > RECHECK_SIGMA)
    return rec

def main():
    targets = pd.read_csv(os.path.join(OUT, "dla_targets.csv"))
    known = pd.read_csv(os.path.join(OUT, "dla_known_metallicity_matches.csv"))
    targets = targets[~targets["ID"].isin(known["target_ID"])].reset_index(drop=True)
    print(f"Rescreening {len(targets)} targets with the retracted box-car method...")

    t0 = time.time()
    rows = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(process_one, row): row.ID for row in targets.itertuples()}
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % 500 == 0 or i == len(targets):
                print(f"  [{i}/{len(targets)}] {time.time()-t0:.0f}s", flush=True)

    res = pd.DataFrame(rows)
    ok = res[res["status"] == "ok"]
    cand = ok[ok["boxcar_candidate"] == True].drop_duplicates("ID")
    checked = cand.dropna(subset=["recheck_max_sigma"])
    n_missed = int(checked["missed_gt_4sigma"].sum())

    print(f"\nBox-car candidates (0/4 detected, UL<{EW_UL_INFORMATIVE}A): "
          f"{ok['boxcar_candidate'].sum()} rows, {len(cand)} unique IDs")
    print(f"Recheck completed on: {len(checked)}")
    print(f">4sigma single-pixel miss: {n_missed}/{len(checked)} = "
          f"{n_missed/len(checked)*100:.0f}%")

    out_path = os.path.join(OUT, "boxcar_recheck.csv")
    cand.to_csv(out_path, index=False)
    print(f"-> {out_path}")

if __name__ == "__main__":
    main()
