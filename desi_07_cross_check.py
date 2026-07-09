#!/usr/bin/env python3
"""v3-trigger Stage 3.5 (author's request 2026-07-09): converts the "0
cross-survey candidates" result into an asset. 1110 sightlines sit in
BOTH the SDSS and DESI shortlists by position+z, but a sightline only
counts as a formal cross-survey confirmation if it independently passed
BOTH surveys' candidate screen - a much rarer coincidence than simply
having spectra from both instruments. For every one of the 77 merged
candidates, this checks whether the OTHER survey also has a spectrum at
the same position+z (regardless of whether that spectrum made the other
survey's own candidate cut) and, where it does, measures that spectrum
with the SAME find_peak/classify code at the candidate's z_abs - mutual
corroboration (or contradiction) from a second instrument, for free, no
new telescope time."""
import os
import numpy as np
import pandas as pd
from importlib import import_module

ew = import_module("05_ew_screen")     # find_peak, classify, LINES (SDSS spectra)
desi4 = import_module("desi_04_ew_screen")  # get_client, fetch_batch (DESI spectra)

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "out")
TOL_POS_ARCSEC = 3.0
TOL_Z = 0.02

def angsep_arcsec(ra1, dec1, ra2, dec2):
    """Angular separation in arcsec (vectorized)."""
    r1, d1, r2, d2 = map(np.radians, (ra1, dec1, ra2, dec2))
    c = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(r1 - r2)
    return np.degrees(np.arccos(np.clip(c, -1, 1))) * 3600

def check_sdss_candidate_against_desi(row, desi_targets, client):
    """SDSS candidate: is this sightline also in the DESI shortlist? If
    so, fetch the DESI spectrum via SPARCL and measure at the SDSS z_abs."""
    sep = angsep_arcsec(row.ra, row.dec, desi_targets["ra"].values, desi_targets["dec"].values)
    dz = np.abs(desi_targets["zDLA"].values - row.z_abs)
    hit = np.where((sep < TOL_POS_ARCSEC) & (dz < TOL_Z))[0]
    if len(hit) == 0:
        return {"other_survey_target": False}
    d_row = desi_targets.iloc[hit[0]]
    specs = desi4.fetch_batch(client, [d_row["TARGETID"]])
    recs = specs.get(d_row["TARGETID"], [])
    if not recs:
        return {"other_survey_target": True, "other_survey_status": "fetch_failed"}
    rec = recs[0]
    wave, flux = np.array(rec["wavelength"]), np.array(rec["flux"])
    ivar, mask = np.array(rec["ivar"]), np.array(rec["mask"])
    peaks = {name: ew.find_peak(wave, flux, ivar, mask, row.z_abs, lam0)
             for name, lam0 in ew.LINES.items()}
    n_ok, detected, candidate, solo_hits, pair_hit = ew.classify(peaks)
    return {"other_survey_target": True, "other_survey_status": "ok",
            "other_survey_targetid": d_row["TARGETID"], "other_survey_sep_arcsec": round(sep[hit[0]], 2),
            "other_survey_n_lines_ok": n_ok, "other_survey_detected": detected,
            "other_survey_corroborates": (not detected) and n_ok >= 3}

def check_desi_candidate_against_sdss(row, sdss_targets):
    """DESI candidate: is this sightline also in the SDSS shortlist? If
    so, fetch the SDSS spectrum over HTTP and measure at the DESI z_abs."""
    sep = angsep_arcsec(row.ra, row.dec, sdss_targets["ra"].values, sdss_targets["dec"].values)
    dz = np.abs(sdss_targets["zCNN"].values - row.z_abs)
    hit = np.where((sep < TOL_POS_ARCSEC) & (dz < TOL_Z))[0]
    if len(hit) == 0:
        return {"other_survey_target": False}
    s_row = sdss_targets.iloc[hit[0]]
    content = ew.fetch_spec(int(s_row["Plate"]), int(s_row["MJD"]), int(s_row["Fiber"]))
    if content is None:
        return {"other_survey_target": True, "other_survey_status": "fetch_failed"}
    from astropy.io import fits
    from io import BytesIO
    with fits.open(BytesIO(content)) as d:
        t = d[1].data
        wave, flux, ivar, mask = 10 ** t["loglam"], t["flux"], t["ivar"], t["and_mask"]
    peaks = {name: ew.find_peak(wave, flux, ivar, mask, row.z_abs, lam0)
             for name, lam0 in ew.LINES.items()}
    n_ok, detected, candidate, solo_hits, pair_hit = ew.classify(peaks)
    return {"other_survey_target": True, "other_survey_status": "ok",
            "other_survey_targetid": f"{int(s_row['Plate'])}-{int(s_row['MJD'])}-{int(s_row['Fiber'])}",
            "other_survey_sep_arcsec": round(sep[hit[0]], 2),
            "other_survey_n_lines_ok": n_ok, "other_survey_detected": detected,
            "other_survey_corroborates": (not detected) and n_ok >= 3}

def main():
    """Checks every merged candidate against the other survey's spectrum
    where one exists and reports corroboration/contradiction."""
    final = pd.read_csv(os.path.join(OUT, "merged_candidates_sdss_desi.csv"))
    desi_targets = pd.read_csv(os.path.join(OUT, "desi_targets.csv"))
    sdss_targets = pd.read_csv(os.path.join(OUT, "dla_targets.csv"))
    client = desi4.get_client()

    rows = []
    for row in final.itertuples():
        if row.survey == "SDSS":
            res = check_sdss_candidate_against_desi(row, desi_targets, client)
        else:  # DESI
            res = check_desi_candidate_against_sdss(row, sdss_targets)
        res["ID"] = row.ID
        res["survey"] = row.survey
        rows.append(res)
        flag = res.get("other_survey_corroborates")
        print(f"  {row.survey} ID={row.ID}: other_survey_target="
              f"{res['other_survey_target']}, corroborates={flag}")

    res_df = pd.DataFrame(rows)
    out_path = os.path.join(OUT, "cross_survey_corroboration.csv")
    res_df.to_csv(out_path, index=False)

    n_in_overlap = int(res_df["other_survey_target"].sum())
    n_checked = int((res_df.get("other_survey_status") == "ok").sum())
    n_corr = int(res_df.get("other_survey_corroborates", pd.Series(dtype=bool)).sum())
    n_contra = n_checked - n_corr
    print(f"\n=== ИТОГ ===")
    print(f"Из {len(final)} кандидатов: {n_in_overlap} лежат в зоне пересечения "
          f"(есть спектр другого обзора на той же позиции+z)")
    print(f"Измерено независимо: {n_checked}")
    print(f"Подтверждено (другой инструмент тоже не видит линий): {n_corr}")
    print(f"Противоречие (другой инструмент ВИДИТ линии): {n_contra}")
    print(f"-> {out_path}")

if __name__ == "__main__":
    main()
