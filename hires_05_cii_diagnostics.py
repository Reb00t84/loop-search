#!/usr/bin/env python3
"""v4-trigger candidate, CII1334 diagnostics on 39627762889133674 (author
request 2026-07-11, after the edge-of-window re-scan left this line's
dv=-1500 unresolved while SiII/OI/AlII converged cleanly on dv=+650).

Two checks, both against real data/catalogs, not assumption:

1. Local significance of CII1334 AT dv=+650 (the velocity the other 3
   lines converged on), not the scan's global maximum. find_peak() only
   returns the best point over its whole scan range; this re-implements
   its single-point inner loop (same continuum, same window, same EW/
   sigma formula) to query one exact trial center.

2. Identify the dv=-1500 feature: converted to observed wavelength and
   checked against the DESI DLA Toolkit catalog entries for this same
   sightline (out/desi_targets.csv has TWO DLAs here, DLAID .../000 and
   .../001) - is some other line at some other real, catalogued redshift
   sitting there? Also checked whether this sightline has SDSS coverage
   (the v3 cross-survey overlap) for independent corroboration.

Result written back to out/highres_purity.csv WITHOUT overwriting the
Stage-2b (*_orig) or edge-of-window-rescan history: this row's ±1500
scan-max values move to scan_max_* columns (flagged, not citable), and
the local peak at the consensus dv is recorded in local_peak_* columns.
status becomes "blend_or_artifact_at_scan_max" so a reader sees
immediately that the naive scan-max number needed manual disambiguation."""
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

TARGET_ID = "39627762889133674"
CONSENSUS_DV_KMS = 650.0     # where SiII/OI/AlII converged in the +-1500 rescan
SCAN_MAX_DV_KMS = -1500.0    # CII1334's +-1500 scan maximum (the value under question)


def sigma_at(wave, flux, ivar, and_mask, z_abs, lam0, center_kms,
             integ_kms=ew.INTEG_KMS, cont_kms=ew.CONT_KMS):
    """Single-point version of find_peak()'s inner loop: same continuum,
    same narrow window, same EW/sigma definition, but evaluated at one
    exact trial center instead of scanning for the maximum."""
    lam_obs = lam0 * (1 + z_abs)
    dv = (wave - lam_obs) / lam_obs * ew.C_KMS
    good = (ivar > 0) & (and_mask == 0)
    cont_sel = good & (np.abs(dv) > cont_kms[0]) & (np.abs(dv) < cont_kms[1])
    if cont_sel.sum() < 10:
        return None
    cont = np.median(flux[cont_sel])
    if not np.isfinite(cont) or cont <= 0:
        return None
    dlam = np.gradient(wave)
    sel = good & (np.abs(dv - center_kms) < integ_kms)
    if sel.sum() < 2:
        return None
    f, iv, dl = flux[sel], ivar[sel], dlam[sel]
    ew_val = np.sum((1 - f / cont) * dl)
    ew_err = np.sqrt(np.sum((1.0 / np.sqrt(iv) / cont * dl) ** 2))
    if not np.isfinite(ew_err) or ew_err <= 0:
        return None
    sigma = ew_val / ew_err
    return {"sigma": sigma, "dv": center_kms, "ew_err_rest": ew_err / (1 + z_abs)}


def identify_feature(z_abs, dv_kms, lam0):
    """Converts a (z_abs, dv) pair to observed wavelength and checks it
    against every OTHER DLA catalogued on the same sightline."""
    lam_obs = lam0 * (1 + z_abs) * (1 + dv_kms / ew.C_KMS)
    desi = pd.read_csv(os.path.join(OUT, "desi_targets.csv"))
    desi["TARGETID"] = desi["TARGETID"].astype("int64").astype(str)
    sightline = desi[desi["TARGETID"] == TARGET_ID]
    print(f"Наблюдаемая длина волны исследуемой точки: {lam_obs:.3f} Å")
    print(f"DLA на этом сайтлайне (DESI DLA Toolkit): {len(sightline)}")
    best = None
    for _, row in sightline.iterrows():
        for name, rest in ew.LINES.items():
            pred = rest * (1 + row["zDLA"])
            dv_off = (pred - lam_obs) / lam_obs * ew.C_KMS
            if abs(dv_off) < 500:
                print(f"  candidate match: {name} @ zDLA={row['zDLA']:.6f} "
                      f"(NHI={row['NHI']:.2f}, DLAID={row['DLAID']}) -> "
                      f"predicted {pred:.3f} Å, offset {dv_off:+.1f} km/s")
                if best is None or abs(dv_off) < abs(best[2]):
                    best = (name, row["zDLA"], dv_off, row["NHI"], row["DLAID"])
    return lam_obs, best


def check_sdss_overlap(ra, dec):
    sdss = pd.read_csv(os.path.join(OUT, "dla_targets.csv"))
    r1, d1 = np.radians(ra), np.radians(dec)
    r2, d2 = np.radians(sdss["ra"].values), np.radians(sdss["dec"].values)
    sep = np.degrees(np.arccos(np.clip(
        np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(r1 - r2), -1, 1))) * 3600
    return int((sep < 3.0).sum())


def main():
    pur = pd.read_csv(os.path.join(OUT, "highres_purity.csv"), dtype={"ID": str})
    row_idx = pur[(pur["ID"] == TARGET_ID) & (pur["line"] == "CII1334")].index[0]
    row = pur.loc[row_idx]
    z_abs = row["z_abs"]

    wave, flux, ivar, and_mask = h3.load_spectrum(TARGET_ID, row["archive"], row["product_id"])
    lam0 = ew.LINES["CII1334"]

    local = sigma_at(wave, flux, ivar, and_mask, z_abs, lam0, CONSENSUS_DV_KMS)
    print(f"Локальная значимость CII1334 на dv={CONSENSUS_DV_KMS:+.0f} км/с "
          f"(куда сошлись SiII/OI/AlII): sigma={local['sigma']:.2f}")
    print(f"(для сравнения, значение на scan-максимуме dv={SCAN_MAX_DV_KMS:.0f}: "
          f"sigma={row['sigma']:.2f}, было записано как основное после rescan)")
    print()

    m = pd.read_csv(os.path.join(OUT, "merged_candidates_clean.csv"), dtype={"ID": str})
    trow = m[m["ID"] == TARGET_ID].iloc[0]
    lam_obs, best = identify_feature(z_abs, SCAN_MAX_DV_KMS, lam0)
    n_sdss = check_sdss_overlap(trow["ra"], trow["dec"])
    print(f"\nSDSS-покрытие того же сайтлайна (3\", для кросс-обзорной сверки v3): {n_sdss}")

    if best:
        name, z_other, dv_off, nhi, dlaid = best
        note = (f"{name} @ z={z_other:.6f} of a SECOND catalogued DLA on the same "
                f"sightline (NHI={nhi:.2f}, DLAID={dlaid}), predicted {dv_off:+.1f} km/s "
                f"from the scan-max point - a real, independently catalogued absorber, "
                f"not noise/telluric/order-stitching")
    else:
        note = "no catalogued-DLA line identified within 500 km/s - cause still unresolved"
    print(f"\nВывод: {note}")

    # write back without destroying history: the not-citable scan-max
    # (-1500, from the +-1500 rescan) moves to scan_max_* (flagged,
    # archived); the primary sigma/dv_kms/ew_ang columns - what anyone
    # reading the CSV without checking `status` first would grab - get
    # the actual citable number (the local peak at the consensus dv),
    # not the blend. status names the situation explicitly either way.
    local_ew = local["sigma"] * local["ew_err_rest"]
    pur.at[row_idx, "scan_max_sigma"] = row["sigma"]
    pur.at[row_idx, "scan_max_dv_kms"] = row["dv_kms"]
    pur.at[row_idx, "scan_max_ew_ang"] = row["ew_ang"]
    pur.at[row_idx, "scan_max_note"] = note
    pur.at[row_idx, "local_peak_sigma"] = local["sigma"]
    pur.at[row_idx, "local_peak_dv_kms"] = CONSENSUS_DV_KMS
    pur.at[row_idx, "local_peak_ew_ang"] = local_ew
    pur.at[row_idx, "sigma"] = local["sigma"]
    pur.at[row_idx, "dv_kms"] = CONSENSUS_DV_KMS
    pur.at[row_idx, "ew_ang"] = local_ew
    pur.at[row_idx, "status"] = "blend_or_artifact_at_scan_max"

    pur.to_csv(os.path.join(OUT, "highres_purity.csv"), index=False)
    print(f"\n-> out/highres_purity.csv обновлён (1 строка: {TARGET_ID}/CII1334)")


if __name__ == "__main__":
    main()
