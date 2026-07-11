#!/usr/bin/env python3
"""v4-trigger candidate, Stage 2a: download public archival spectra for the
15 out/archival_coverage.csv targets and build a real (not metadata-level)
inventory - actual wavelength coverage and SNR measured from the spectrum
itself, per metal line (SiII1526/CII1334/OI1302/AlII1670 x (1+z_abs)).

Product selection, per archive:
  - ESO (UVES/ESPRESSO/XSHOOTER): phase3 (calib_level>=2, dp_id=ADP.*)
    reduced 1D spectra - ivoa.ObsCore already only surfaces these for our
    instrument set (verified in Stage 1). Downloaded via the ObsCore
    access_url -> datalink #this link -> direct FITS GET, no auth needed
    for public data (verified live).
  - KOA HIRES: level-1 (lev1) products - verified live to exist and be a
    real per-echelle-order reduction (wave/Flux/Error/Sig_to_Noise columns
    per pixel, KOA's own MAKEE-derived pipeline), NOT a single merged
    spectrum. Downloaded via Koa.download(lev1file=1); only the `flux`
    sub-product is kept (has wave+flux+err in one table), hdr/arcids/
    trace/profile deleted immediately after download - calibration
    intermediates we don't need, and they roughly double the volume.
  - KOA ESI: **no level-1 product exists at all** - verified live
    (Koa.download returned "Instrument [ESI] does not have level1 data.",
    a direct system statement, not an assumption). Only raw 2D
    echellograms are available; turning those into a wavelength-
    calibrated 1D spectrum needs order-tracing + wavelength calibration +
    extraction, i.e. writing a reduction pipeline - out of scope for an
    "download + inventory" stage, and not something to attempt uncalibrated
    (project rule 2: no screen ships without a known-ground-truth check,
    and there is no time/data budget for that here). Affected: 3 targets
    whose ONLY archival coverage is ESI (39628019890914673, 400352403,
    438508218), plus the ESI fallback for AlII1670 at 531686697. These are
    recorded as rows with suitability="raw_only_no_reduced_pipeline" and
    NaN measured SNR/coverage - an honest null, not a silent drop, per the
    project's "even a null is citable" stance from Stage 1.

Two more real findings caught by re-checking suspicious "zero coverage"
results against the actual downloaded files, not trusting the summary:
  - Some HIRES exposures ship level-1 data in an older layout
    ("makee/ccdN/fits" etc: separate Flux/Var/Sky/Arc/profile multi-
    extension FITS images per order, no wave+flux+err bintable) instead
    of the "binaryfits/ccdN/flux" bintable this script reads. Same
    archive, same instrument, even the same NIGHT as a working exposure
    (407352752's two HI.20100404.* exposures are makee-format, while
    387205842's HI.20100404.* exposures from the SAME night are
    binaryfits) - not a date cutoff, just which reduction succeeded for
    that specific exposure. Recorded as suitability=
    "unsupported_lev1_format" rather than silently returning zero rows -
    same "honest null, not a silent drop" principle as the ESI case, and
    genuinely out of scope to add a second-format reader for under this
    stage's time budget (would need its own wavelength-solution handling,
    unverified, same rule-2/rule-4 concern as ESI raw extraction).
  - Even within the binaryfits/flux format, some exposures have a real,
    valid, correctly-shaped bintable where the `wave` column is filled
    entirely with the sentinel -1.0 (real Flux values alongside it,
    e.g. -436 to +4755 - not a corrupt or truncated download, verified
    by re-fetching a fresh copy of one such file byte-for-byte and
    getting the same result) - MAKEE extracted the order but the
    wavelength calibration itself failed for that exposure (531686697,
    HI.20080329.39341). Recorded as suitability="no_wavelength_solution",
    distinct from "wavelength solution exists but doesn't cover our
    lines" - conflating the two would misreport WHY a target has no
    usable HIRES coverage.

SNR measurement window: median(flux/err) over pixels within +-50 km/s of
each line's observed wavelength (half the Stage-2 EW integration window,
appropriate here since this is a per-line SNR sanity number for Stage 2b
to threshold on, not an EW measurement).

Resume: same discipline as Stage 1 (rule established there after the
environment killed a mid-run process) - a marker file records fully-
processed target IDs; a target with e.g. zero usable ESO products still
needs to be marked "processed", same "absent != not yet done" trap."""
import os
import time
import warnings
from importlib import import_module

import numpy as np
import pandas as pd
import requests
from astropy.io import fits

warnings.filterwarnings("ignore")

ew = import_module("05_ew_screen")           # LINES, C_KMS
h1 = import_module("hires_01_archival_coverage")  # metal_wavelengths, KOA/ESO instrument maps

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "out")
DATA_DIR = os.path.join(HERE, "data", "highres_spectra")  # gitignored (data/), re-fetchable
DONE_IDS_PATH = os.path.join(OUT, "highres_inventory_done_ids.txt")

RETRIES = 5
BACKOFF_S = 4
SNR_WINDOW_KMS = 50.0


# ---------------------------------------------------------------- ESO ----

def eso_products(tap, ra, dec, radius_deg):
    q = f"""
    SELECT target_name, instrument_name, calib_level, t_exptime, em_min, em_max,
           dp_id, obs_release_date, proposal_id, snr, access_url
    FROM ivoa.ObsCore
    WHERE INTERSECTS(s_region, CIRCLE('ICRS', {ra}, {dec}, {radius_deg})) = 1
    AND instrument_name IN ('UVES','ESPRESSO','XSHOOTER')
    """
    for attempt in range(RETRIES):
        try:
            res = tap.search(q)
            return res.to_table().to_pandas()
        except Exception:
            time.sleep(BACKOFF_S)
    return None


def eso_resolve_file_url(dp_id):
    from pyvo.dal.adhoc import DatalinkResults
    url = f"http://archive.eso.org/datalink/links?ID=ivo://eso.org/ID?{dp_id}"
    for attempt in range(RETRIES):
        try:
            dl = DatalinkResults.from_result_url(url)
            t = dl.to_table().to_pandas()
            this = t[t["semantics"] == "#this"]
            if len(this) and this.iloc[0]["access_url"]:
                return this.iloc[0]["access_url"]
            return None
        except Exception:
            time.sleep(BACKOFF_S)
    return None


def download_file(url, dest_path):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return True  # already downloaded (resume within a target)
    for attempt in range(RETRIES):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 200 and len(r.content) > 0:
                with open(dest_path, "wb") as f:
                    f.write(r.content)
                return True
        except Exception:
            pass
        time.sleep(BACKOFF_S)
    return False


FLUX_COL_CANDIDATES = ["FLUX", "FLUX_REDUCED"]
ERR_COL_CANDIDATES = ["ERR", "ERR_FLUX", "ERR_REDUCED"]


def read_eso_spectrum(path):
    """ESO phase3 spectrum products are NOT a single fixed schema across
    collections/pipeline versions - verified on real downloaded files, not
    assumed: WAVE can be 'nm' or 'angstrom' (TUNIT1, checked per file, not
    hardcoded - one real product had wave_min/max come out as 533-1020
    Angstrom, i.e. below the Lyman limit, before this fix: it was actually
    533.66-1020.0 NM, a perfectly normal X-Shooter VIS-arm range read with
    the wrong assumed unit), and the flux/error columns can be named
    FLUX/ERR, FLUX/ERR_FLUX, or FLUX_REDUCED/ERR_REDUCED depending on the
    product. Column names and the wave unit are resolved per file."""
    with fits.open(path) as h:
        d = h[1].data
        cols = h[1].data.columns.names
        hdr = h[1].header
        wave = np.array(d["WAVE"][0], dtype=float)
        unit = str(hdr.get("TUNIT1", "angstrom")).strip().lower()
        if unit in ("nm", "nanometer", "nanometers"):
            wave = wave * 10.0
        elif unit not in ("angstrom", "angstroms", "a"):
            raise ValueError(f"unrecognized WAVE unit '{unit}' in {path}")
        flux_col = next((c for c in FLUX_COL_CANDIDATES if c in cols), None)
        err_col = next((c for c in ERR_COL_CANDIDATES if c in cols), None)
        if flux_col is None or err_col is None:
            raise ValueError(f"no recognized flux/err columns in {path}: {cols}")
        flux = np.array(d[flux_col][0], dtype=float)
        err = np.array(d[err_col][0], dtype=float)
    return wave, flux, err


# ---------------------------------------------------------------- KOA ----

def koa_exposure_table(instrument, ra, dec):
    return h1.koa_query(instrument, ra, dec)


def koa_lev1_flux_files(koaid, filehand):
    """Direct replacement for Koa.download(lev1file=1): that call fetches
    ALL lev1 sub-products (flux/hdr/arcids/trace/profile, ~233 files for
    one real exposure, verified live) with one HTTP GET per file and no
    filter option in the public API - we only ever use `flux` (has
    wave+Flux+Error+Sig_to_Noise per pixel, everything needed). Verified
    live against pykoa's own source (core.py __download_lev1files) that
    this hits the identical nph-getL1list/nph-dnloadL1data endpoints it
    uses internally, byte-identical files - just skipping the ~93% of
    requests for sub-products we always delete anyway. Returns a list of
    (subdir, filename, download_url)."""
    from pykoa.koa import Koa
    url = Koa.lev1list_url + f"instrument=hires&koaid={koaid}&filehand={filehand}"
    for attempt in range(RETRIES):
        try:
            r = requests.get(url, timeout=60)
            data = r.json()
            break
        except Exception:
            time.sleep(BACKOFF_S)
    else:
        return None
    prefix = data.get("result", {}).get("lev1subdir_prefix")
    entries = data.get("result", {}).get("data", [])
    if prefix is None or not entries:
        return []  # genuinely no lev1 data for this exposure
    flux_entries = [e for e in entries if e["subdir"].startswith("binaryfits") and e["subdir"].endswith("/flux")]
    if not flux_entries:
        # lev1 data exists but not in the layout this script reads (e.g.
        # older "makee/ccdN/fits" multi-extension format) - raise rather
        # than silently returning [] indistinguishable from "no lev1 data"
        seen = sorted({e["subdir"].split("/")[0] for e in entries})
        raise UnsupportedLev1Format(seen)
    out = []
    for entry in flux_entries:
        for fname in entry["lev1files"]:
            filehand_lev1 = f"{prefix}/{entry['subdir']}/{fname}"
            dl_url = (Koa.baseurl + "cgi-bin/KoaAPI/nph-dnloadL1data?"
                      f"instrument=hires&koaid={koaid}&filehand={filehand_lev1}")
            out.append((entry["subdir"], fname, dl_url))
    return out


class UnsupportedLev1Format(Exception):
    pass


def read_hires_order(path):
    with fits.open(path) as h:
        d = h[1].data
        wave = np.array(d["wave"], dtype=float)
        flux = np.array(d["Flux"], dtype=float)
        err = np.array(d["Error"], dtype=float)
    return wave, flux, err


# ------------------------------------------------------------ shared -----

def measure_snr_near_line(wave, flux, err, line_obs_ang):
    dv = (wave - line_obs_ang) / line_obs_ang * ew.C_KMS
    mask = (np.abs(dv) <= SNR_WINDOW_KMS) & (err > 0) & np.isfinite(flux) & np.isfinite(err)
    if mask.sum() < 3:
        return np.nan, int(mask.sum())
    return float(np.median(flux[mask] / err[mask])), int(mask.sum())


def lines_in_range(wave_min, wave_max, z_abs):
    obs = h1.metal_wavelengths(z_abs)
    return {name: (wave_min <= w <= wave_max) for name, w in obs.items()}


def inventory_row(target_id, archive, instrument, product_id, wave, flux, err, z_abs, extra):
    """suitability distinguishes "covers a line nominally" from "actually
    measurable" - real case hit on the first sanity target (387205842,
    HIRES order ccd3/07): AlII1670 formally sits inside [wave_min,
    wave_max], but the line falls right at the order edge where the
    pipeline flags nearly all pixels bad (err=-1) - covers_AlII1670=True,
    npix_AlII1670=0. Exactly the "echelle inter-order gap" caveat flagged
    in Stage 1's module docstring, now caught concretely by actually
    opening the spectrum rather than trusting the nominal range."""
    wave_min, wave_max = float(np.nanmin(wave)), float(np.nanmax(wave))
    if not (wave_max > wave_min > 0):
        # sentinel wavelength column (verified real case: 531686697,
        # HI.20080329.39341 - wave is -1.0 for every pixel while Flux
        # holds real varying values; MAKEE extracted the order but the
        # wavelength calibration failed for that exposure) - distinct
        # from "solved but doesn't cover our lines", not the same finding
        return {"ID": target_id, "archive": archive, "instrument": instrument,
                "product_id": product_id, "wave_min_ang": wave_min, "wave_max_ang": wave_max,
                "suitability": "no_wavelength_solution", **extra}
    covered = lines_in_range(wave_min, wave_max, z_abs)
    obs = h1.metal_wavelengths(z_abs)
    row = {
        "ID": target_id, "archive": archive, "instrument": instrument, "product_id": product_id,
        "wave_min_ang": wave_min, "wave_max_ang": wave_max,
    }
    any_measurable = False
    for name in ew.LINES:
        row[f"covers_{name}"] = covered[name]
        if covered[name]:
            snr, npix = measure_snr_near_line(wave, flux, err, obs[name])
            row[f"snr_{name}"] = snr
            row[f"npix_{name}"] = npix
            any_measurable = any_measurable or npix >= 3
        else:
            row[f"snr_{name}"] = np.nan
            row[f"npix_{name}"] = 0
    if any_measurable:
        row["suitability"] = "usable"
    elif any(covered.values()):
        row["suitability"] = "covered_but_unmeasurable"
    else:
        row["suitability"] = "no_line_in_range"
    row.update(extra)
    return row


def process_eso_target(tap, target_id, ra, dec, z_abs, radius_deg, workdir):
    df = eso_products(tap, ra, dec, radius_deg)
    if df is None:
        return [{"ID": target_id, "archive": "ESO", "instrument": "ANY", "product_id": None,
                  "suitability": "query_failed"}]
    rows = []
    for _, p in df.iterrows():
        # pre-filter cheaply on metadata range before spending a download
        wmin_ang, wmax_ang = float(p["em_min"]) * 1e10, float(p["em_max"]) * 1e10
        covered = lines_in_range(wmin_ang, wmax_ang, z_abs)
        if not any(covered.values()):
            continue
        rel = pd.to_datetime(p["obs_release_date"], utc=True, errors="coerce")
        if pd.isna(rel) or rel > pd.Timestamp.now(tz="UTC"):
            continue  # proprietary - not ours to download
        dest = os.path.join(workdir, f"{p['dp_id']}.fits")
        url = eso_resolve_file_url(p["dp_id"])
        if url is None or not download_file(url, dest):
            rows.append({"ID": target_id, "archive": "ESO", "instrument": p["instrument_name"],
                          "product_id": p["dp_id"], "suitability": "download_failed"})
            continue
        try:
            wave, flux, err = read_eso_spectrum(dest)
        except Exception:
            rows.append({"ID": target_id, "archive": "ESO", "instrument": p["instrument_name"],
                          "product_id": p["dp_id"], "suitability": "read_failed"})
            continue
        rows.append(inventory_row(target_id, "ESO", p["instrument_name"], p["dp_id"],
                                   wave, flux, err, z_abs,
                                   {"calib_level": int(p["calib_level"]),
                                    "snr_metadata": float(p["snr"]) if not pd.isna(p["snr"]) else np.nan,
                                    "t_exptime": float(p["t_exptime"]), "proposal_id": p["proposal_id"]}))
    return rows


def process_hires_target(target_id, ra, dec, z_abs, workdir):
    exp_df = koa_exposure_table("hires", ra, dec)
    if exp_df is None:
        return [{"ID": target_id, "archive": "KOA", "instrument": "HIRES", "product_id": None,
                  "suitability": "query_failed"}]
    if len(exp_df) == 0:
        return []
    lev1_dir = os.path.join(workdir, "lev1")
    os.makedirs(lev1_dir, exist_ok=True)
    rows = []
    for exp in exp_df.itertuples():
        try:
            flux_specs = koa_lev1_flux_files(exp.koaid, exp.filehand)
        except UnsupportedLev1Format as e:
            rows.append({"ID": target_id, "archive": "KOA", "instrument": "HIRES",
                          "product_id": exp.koaid, "suitability": "unsupported_lev1_format",
                          "note": f"lev1 top-level subdirs seen: {e.args[0]}"})
            continue
        if flux_specs is None:
            rows.append({"ID": target_id, "archive": "KOA", "instrument": "HIRES",
                          "product_id": exp.koaid, "suitability": "download_failed"})
            continue
        for subdir, fname, dl_url in flux_specs:
            dest_dir = os.path.join(lev1_dir, subdir)
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, fname)
            if not download_file(dl_url, dest):
                rows.append({"ID": target_id, "archive": "KOA", "instrument": "HIRES",
                              "product_id": f"{subdir}/{fname}", "suitability": "download_failed"})
                continue
            product_id = f"{os.path.basename(exp.koaid)}/{subdir}/{fname}"
            try:
                wave, flux, err = read_hires_order(dest)
            except Exception:
                rows.append({"ID": target_id, "archive": "KOA", "instrument": "HIRES",
                              "product_id": product_id, "suitability": "read_failed"})
                continue
            row = inventory_row(target_id, "KOA", "HIRES", product_id, wave, flux, err, z_abs, {})
            if row["suitability"] != "no_line_in_range":  # keep "covered_but_unmeasurable" too - informative
                rows.append(row)
    return rows


def process_esi_target(target_id, ra, dec, z_abs):
    """ESI has no level-1 product (verified live) - record an honest null,
    do not download raw echellograms we cannot turn into a measurement."""
    exp_df = koa_exposure_table("esi", ra, dec)
    if exp_df is None or len(exp_df) == 0:
        return []
    return [{"ID": target_id, "archive": "KOA", "instrument": "ESI", "product_id": None,
              "n_raw_exposures": len(exp_df), "suitability": "raw_only_no_reduced_pipeline"}]


def main():
    import pyvo
    cov = pd.read_csv(os.path.join(OUT, "archival_coverage.csv"))
    cov["ID"] = cov["ID"].astype(str)
    targets = pd.read_csv(os.path.join(OUT, "merged_candidates_clean.csv"))
    targets["ID"] = targets["ID"].astype(str)
    target_list = targets[targets["ID"].isin(cov["ID"].unique())][["ID", "ra", "dec", "z_abs"]]

    tap = pyvo.dal.TAPService("https://archive.eso.org/tap_obs")
    os.makedirs(DATA_DIR, exist_ok=True)

    done_ids = set()
    if os.path.exists(DONE_IDS_PATH):
        with open(DONE_IDS_PATH) as f:
            done_ids = {line.strip() for line in f if line.strip()}
    out_path = os.path.join(OUT, "highres_inventory.csv")
    rows = pd.read_csv(out_path).to_dict("records") if os.path.exists(out_path) else []
    if done_ids:
        print(f"Резюме: {len(done_ids)} целей уже обработаны, пропускаем")

    for i, t in enumerate(target_list.itertuples()):
        if t.ID in done_ids:
            continue
        print(f"[{i+1}/{len(target_list)}] {t.ID} z_abs={t.z_abs:.3f}", flush=True)
        workdir = os.path.join(DATA_DIR, t.ID)
        os.makedirs(workdir, exist_ok=True)

        tset = set(cov[cov["ID"] == t.ID]["instrument"])
        if tset & {"UVES", "ESPRESSO", "XSHOOTER"}:
            r = process_eso_target(tap, t.ID, t.ra, t.dec, t.z_abs, h1.RADIUS_DEG, workdir)
            rows += r
            print(f"    ESO: {len(r)} usable/attempted products")
        if "HIRES" in tset:
            r = process_hires_target(t.ID, t.ra, t.dec, t.z_abs, workdir)
            rows += r
            print(f"    KOA/HIRES: {len(r)} usable orders")
        if "ESI" in tset:
            r = process_esi_target(t.ID, t.ra, t.dec, t.z_abs)
            rows += r
            print(f"    KOA/ESI: {len(r)} (raw only, not reduced - see module docstring)")

        pd.DataFrame(rows).to_csv(out_path, index=False)
        with open(DONE_IDS_PATH, "a") as f:
            f.write(t.ID + "\n")

    inv = pd.DataFrame(rows)
    inv.to_csv(out_path, index=False)

    usable = inv[inv["suitability"] == "usable"]
    print("\n=== ИТОГ Stage 2a ===")
    print(f"Целей на входе: {len(target_list)}")
    print(f"Продуктов usable (реальная линия в реальном диапазоне): {len(usable)}")
    print(f"Целей с хотя бы одним usable-продуктом: {usable['ID'].nunique()}")
    print(f"\nПо suitability:")
    print(inv["suitability"].value_counts())
    print(f"\nПо инструментам (usable продуктов):")
    print(usable.groupby("instrument")["ID"].count())
    print(f"\nПокрытие по линиям (usable продуктов, где линия в диапазоне):")
    for name in ew.LINES:
        print(f"  {name}: {int(inv.get(f'covers_{name}', pd.Series(dtype=bool)).sum())} продуктов, "
              f"{inv[inv.get(f'covers_{name}', False) == True]['ID'].nunique()} целей")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
