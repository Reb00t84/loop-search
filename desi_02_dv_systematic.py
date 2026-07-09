#!/usr/bin/env python3
"""v3-trigger Stage 2: velocity offset between DESI DR1 Z_DLA (combined
GP+CNN+template classifier, Brodzeller et al. 2025) and the metal-line
redshift from Rafelski et al. 2012, for the sightlines both catalogs see.
Compares against the SDSS/Chabanier zCNN-vs-metal systematic (median
79 km/s, 10% beyond 300 km/s, max 749 km/s; out/zcnn_vs_metal_velocity_offset.csv)
to test whether DESI's multi-classifier redshift is better tied to the
metal-line velocity than Chabanier's CNN-only H I redshift, worse, or the
same order — i.e. whether the SDSS-side dv systematic is a property of
one pipeline or of the DLA-redshift problem in general.

Answers on n=30-34 have a noisy tail (see CLAUDE.md): quantiles are
reported with percentile-bootstrap 95% CIs, not point estimates alone."""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

C_KMS = 299792.458
TOL_POS_ARCSEC = 3.0
TOL_Z = 0.02
N_BOOT = 20000
SEED = 42

def angsep_arcsec(ra1, dec1, ra2, dec2):
    """Angular separation in arcsec (vectorized)."""
    r1, d1, r2, d2 = map(np.radians, (ra1, dec1, ra2, dec2))
    c = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(r1 - r2)
    return np.degrees(np.arccos(np.clip(c, -1, 1))) * 3600

def load_desi_dla():
    """Loads the DESI DR1 DLA Toolkit catalog, dropping the multidimensional
    COEFF column that blocks a direct to-pandas conversion."""
    from astropy.table import Table
    t = Table.read(os.path.join(DATA, "desi_dr1_dlacat_v2.0.fits"))
    names = [n for n in t.colnames if len(t[n].shape) <= 1]
    return t[names].to_pandas()

def crossmatch(desi, rafelski, tol_pos=TOL_POS_ARCSEC, tol_z=TOL_Z):
    """Position+z_abs match, one row per matched DESI absorber (a QSO with
    N matched Rafelski absorbers contributes N rows, same convention as
    the SDSS zCNN-vs-metal systematic file)."""
    rows = []
    for _, m in rafelski.iterrows():
        d = angsep_arcsec(desi["RA"].values, desi["DEC"].values, m["ra"], m["dec"])
        for j in np.where(d < tol_pos)[0]:
            if abs(desi.iloc[j]["Z_DLA"] - m["z_abs"]) < tol_z:
                rows.append({
                    "desi_targetid": desi.iloc[j]["TARGETID"],
                    "rafelski_qso": m["qso"], "sep_arcsec": round(d[j], 2),
                    "z_dla": desi.iloc[j]["Z_DLA"], "z_metal": m["z_abs"],
                    "MH": m["MH"], "NHI": desi.iloc[j]["NHI"],
                    "SNR_FOREST": desi.iloc[j]["SNR_FOREST"],
                })
    return pd.DataFrame(rows)

def bootstrap_ci(x, stat_fn, n_boot=N_BOOT, seed=SEED):
    """Percentile bootstrap for a continuous statistic (e.g. the median).
    NOT valid for a proportion with a 0 (or n) count — resampling a
    zero-success sample can never produce a nonzero fraction, so it
    always returns [0,0] regardless of the true uncertainty; use
    wilson_ci for that case instead."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    boots = np.array([stat_fn(rng.choice(x, size=len(x), replace=True))
                       for _ in range(n_boot)])
    return np.percentile(boots, [2.5, 97.5])

def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion k/n — well-behaved
    at k=0 or k=n, unlike a data-resampling bootstrap on the same case
    (see 07_calibrate_screen.py, used there for the same reason)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return ((center - half) / denom, (center + half) / denom)

def main():
    """Cross-matches DESI DR1 x Rafelski, computes the dv systematic with
    honest CIs, and prints a direct SDSS-vs-DESI comparison recomputed
    from both primary files."""
    desi = load_desi_dla()
    rafelski = pd.read_csv(os.path.join(DATA, "dla_metallicity_compilation.csv"))
    print(f"DESI DR1 DLA: {len(desi)}, Rafelski: {len(rafelski)}")

    m = crossmatch(desi, rafelski)
    m["dv_kms"] = C_KMS * (m["z_dla"] - m["z_metal"]) / (1 + m["z_metal"])
    m = m.sort_values("dv_kms").reset_index(drop=True)
    out_path = os.path.join(OUT, "desi_zdla_vs_metal_velocity_offset.csv")
    m.to_csv(out_path, index=False)
    print(f"Matches: {len(m)} absorbers, {m['rafelski_qso'].nunique()} unique QSOs -> {out_path}")

    dv = m["dv_kms"].values
    adv = np.abs(dv)
    median = np.median(adv)
    med_lo, med_hi = bootstrap_ci(adv, np.median)
    n_over300 = int(np.sum(adv > 300))
    frac300 = n_over300 / len(adv)
    frac300_lo, frac300_hi = wilson_ci(n_over300, len(adv))

    print(f"\n=== DESI DR1 x Rafelski: |dv(Z_DLA - z_metal)| (n={len(m)}) ===")
    print(f"median |dv|: {median:.1f} km/s [95% CI {med_lo:.1f}-{med_hi:.1f}]")
    print(f"max |dv|: {adv.max():.1f} km/s")
    print(f"|dv|>300 km/s: {n_over300}/{len(m)} = {frac300*100:.1f}% "
          f"[95% Wilson CI {frac300_lo*100:.1f}-{frac300_hi*100:.1f}%]")

    # Пересчитываем SDSS-статистику из первички тем же кодом (правило 5:
    # число из файла, а не из абзаца), чтобы сравнение было честным.
    sdss_path = os.path.join(OUT, "zcnn_vs_metal_velocity_offset.csv")
    sdss = pd.read_csv(sdss_path)
    sdv = np.abs(sdss["dv_kms"].values)
    s_median = np.median(sdv)
    s_med_lo, s_med_hi = bootstrap_ci(sdv, np.median)
    s_n_over300 = int(np.sum(sdv > 300))
    s_frac300 = s_n_over300 / len(sdv)
    s_frac300_lo, s_frac300_hi = wilson_ci(s_n_over300, len(sdv))

    print(f"\n=== SDSS/Chabanier zCNN x Rafelski (пересчитано из {sdss_path}) ===")
    print(f"median |dv|: {s_median:.1f} km/s [95% CI {s_med_lo:.1f}-{s_med_hi:.1f}], n={len(sdv)}")
    print(f"max |dv|: {sdv.max():.1f} km/s")
    print(f"|dv|>300 km/s: {s_n_over300}/{len(sdv)} = {s_frac300*100:.1f}% "
          f"[95% Wilson CI {s_frac300_lo*100:.1f}-{s_frac300_hi*100:.1f}%]")

    print(f"\n=== SDSS vs DESI, honest CIs ===")
    print(f"median |dv|:   SDSS {s_median:.1f} [{s_med_lo:.1f}-{s_med_hi:.1f}]  "
          f"vs  DESI {median:.1f} [{med_lo:.1f}-{med_hi:.1f}] km/s")
    print(f"|dv|>300km/s:  SDSS {s_frac300*100:.1f}% [{s_frac300_lo*100:.1f}-{s_frac300_hi*100:.1f}%]  "
          f"vs  DESI {frac300*100:.1f}% [{frac300_lo*100:.1f}-{frac300_hi*100:.1f}%]")
    overlap = not (frac300_hi < s_frac300_lo or s_frac300_hi < frac300_lo)
    print(f"Tail-fraction 95% CIs {'OVERLAP' if overlap else 'DO NOT OVERLAP'} "
          f"-> {'not' if overlap else ''} distinguishable at current n.")

if __name__ == "__main__":
    main()
