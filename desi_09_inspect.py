#!/usr/bin/env python3
"""v3-trigger Stage 3.5, second half (author's request 2026-07-09):
mirrors the SDSS due-diligence pass (08_inspect_top.py PNG panels + the
single-pixel objective recheck that originally caught the box-car
problem) for the 38 DESI candidates, which had NOT been through either
check before this. Two functions:
  1. plot_one(): saves a PNG panel (full spectrum + 4 line windows) for
     eyeballing, same layout as 08_inspect_top.py.
  2. recheck(): objective single-pixel max-significance check in
     +-500 km/s per line, independent of the matched-filter's own
     narrow-window statistic - the same class of check that exposed the
     retracted box-car method's blind spots."""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from importlib import import_module

ew = import_module("05_ew_screen")
desi4 = import_module("desi_04_ew_screen")

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "out")
INSPECT = os.path.join(OUT, "inspect_desi")
os.makedirs(INSPECT, exist_ok=True)

def plot_one(row, wave, flux, ivar, mask, outpath):
    """Saves a PNG panel: full spectrum + one zoom per diagnostic line."""
    z = row.z_abs
    fig, axes = plt.subplots(1, 5, figsize=(22, 3.2))
    good = ivar > 0
    axes[0].plot(wave, flux, lw=0.4, color="k")
    axes[0].set_title(f"ID={row.ID} z={z:.4f} SNR_FOREST={row.SNR_native:.1f}")
    axes[0].set_xlabel("obs wave (A)")
    for lam0, name, ax in zip(ew.LINES.values(), ew.LINES.keys(), axes[1:]):
        lam_obs = lam0 * (1 + z)
        dv = (wave - lam_obs) / lam_obs * ew.C_KMS
        sel = np.abs(dv) < 1500
        if sel.sum() < 5:
            ax.set_title(f"{name}: вне покрытия")
            continue
        ax.plot(wave[sel], flux[sel], lw=0.6, color="k")
        ax.axvspan(lam_obs * (1 - 300 / ew.C_KMS), lam_obs * (1 + 300 / ew.C_KMS),
                   color="C1", alpha=0.2)
        cont_sel = good & sel & (np.abs(dv) > 600) & (np.abs(dv) < 4000)
        cont = np.median(flux[cont_sel]) if cont_sel.sum() else np.nan
        ax.axhline(cont, color="C0", ls="--", lw=0.8)
        masked = mask[sel & (np.abs(dv) < 300)]
        n_masked = int((masked != 0).sum())
        ax.set_title(f"{name}\nmasked={n_masked}, cont={cont:.2f}", fontsize=8)
        ax.set_xlim(lam_obs - 20, lam_obs + 20)
    plt.tight_layout()
    plt.savefig(outpath, dpi=100)
    plt.close(fig)

def single_pixel_max(wave, flux, ivar, mask, z_abs, window_kms=500, cont_kms=(600, 4000)):
    """Max single-pixel depth-significance per line - the statistic that
    exposed the box-car method's blind spots (see CLAUDE.md)."""
    best_sigma, best_line, best_dv = 0.0, None, None
    for name, lam0 in ew.LINES.items():
        lam_obs = lam0 * (1 + z_abs)
        dv = (wave - lam_obs) / lam_obs * ew.C_KMS
        good = (ivar > 0) & (mask == 0)
        cont_sel = good & (np.abs(dv) > cont_kms[0]) & (np.abs(dv) < cont_kms[1])
        search = good & (np.abs(dv) < window_kms)
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

def main(n_plot=15):
    cand = pd.read_csv(os.path.join(OUT, "merged_candidates_clean.csv"))
    cand = cand[cand["survey"] == "DESI"].copy()
    cand["ID"] = cand["ID"].astype("int64")
    print(f"DESI candidates to recheck: {len(cand)}")

    client = desi4.get_client()
    specs = desi4.fetch_batch(client, cand["ID"].tolist())

    plot_targets = cand.sort_values("brightness", ascending=False).head(n_plot)
    recheck_rows = []
    for i, row in enumerate(cand.itertuples(), 1):
        recs = specs.get(row.ID, [])
        if not recs:
            recheck_rows.append({"ID": row.ID, "status": "fetch_failed"})
            continue
        rec = recs[0]
        wave, flux = np.array(rec["wavelength"]), np.array(rec["flux"])
        ivar, mask = np.array(rec["ivar"]), np.array(rec["mask"])

        sigma, line, dv = single_pixel_max(wave, flux, ivar, mask, row.z_abs)
        recheck_rows.append({"ID": row.ID, "status": "ok", "max_sigma": sigma,
                              "line": line, "dv_kms": dv})

        if row.ID in plot_targets["ID"].values:
            rank = list(plot_targets["ID"].values).index(row.ID) + 1
            outpath = os.path.join(INSPECT, f"{rank:02d}_ID{row.ID}.png")
            plot_one(row, wave, flux, ivar, mask, outpath)
            print(f"  [plotted {rank}/{len(plot_targets)}] ID={row.ID}")

    res = pd.DataFrame(recheck_rows)
    out_path = os.path.join(OUT, "desi_candidates_recheck.csv")
    res.to_csv(out_path, index=False)

    ok = res[res["status"] == "ok"]
    for thr in (3, 4, 5):
        n = (ok["max_sigma"] > thr).sum()
        print(f">{thr}sigma: {n}/{len(ok)} = {n/len(ok)*100:.0f}%")
    print(f"-> {out_path}")
    print(f"PNG panels (top {n_plot} by brightness) -> {INSPECT}/")

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    main(n)
