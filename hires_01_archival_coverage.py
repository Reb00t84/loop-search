#!/usr/bin/env python3
"""v4-trigger candidate, Stage 1 only: archival high-/medium-resolution
spectroscopy coverage recon for the 75 out/merged_candidates_clean.csv
targets. Cone-searches KOA (Keck, via the official PyKOA client from
NExScI) and the ESO archive (TAP ObsCore service at archive.eso.org) for
already-taken PUBLIC spectra at each target's position+z - a way to test
the metal non-detections without a telescope proposal, and even a null
result is a citable argument for future observers. Metadata only, no
spectra downloaded.

Rule 4 (CLAUDE.md): a new interface is a new source of surprises, sanity-
check on a handful of real targets before the batch. Both clients were
verified here with live queries first (see report). Two real findings
from that check, both handled below:
  1. This environment's outbound proxy intermittently 502s on both KOA
     and ESO endpoints (~30-40% of calls on the first try) - a plain
     retry-with-backoff, same pattern as the SPARCL precedent in
     desi_04_ew_screen.py.
  2. PyKOA's query_position can swallow a failed request silently: on
     proxy failure it sometimes prints an error and returns normally
     WITHOUT raising and WITHOUT writing the output file, rather than
     raising an exception. A bare try/except retry loop would treat that
     as "zero coverage" instead of "the query never actually ran" - the
     retry wrapper below checks os.path.exists+size on the output file
     explicitly, not just the absence of an exception (another instance
     of "правильное окно вокруг неправильного центра": the window here
     is "no exception raised", wrongly treated as proof the query ran).

Line-coverage caveat (documented, not hidden): HIRES/UVES/ESPRESSO are
echelle spectrographs. The wavelength range reported here is each
exposure's overall blue-red span, not its actual per-order coverage -
inter-order gaps mean a line can sit inside the reported [min,max] range
and still be unobserved. "covers_metal_lines" is therefore a NECESSARY,
not sufficient, condition - a real answer needs the reduced wavelength
solution, deferred to a Stage 2 if this recon turns up candidates worth
pursuing.

Public/proprietary caveat: ESO's ObsCore has an explicit obs_release_date
column - used directly. KOA has no equivalent column in the position-
query metadata; public status is approximated from semester (parsed from
semid, e.g. "2000a") + propint (proprietary period in months), using
nominal semester-end dates (A: Jul 31, B: Jan 31 of year+1). This is an
approximation, not an authoritative release date - flagged as such in
the output.

SNR estimate: added where the archive's own metadata provides one, not
computed here. ESO's ivoa.ObsCore has a per-product `snr` column, phase3
pipeline output - verified live to hold real varying values (not a
placeholder). KOA's HIRES metadata has `sig2nois`, also verified to vary
exposure-to-exposure within one program (not a fixed proposal-nominal
number). KOA's ESI metadata has no S/N-equivalent column at all -
snr_source records "not available" rather than silently leaving 0/NaN
indistinguishable from "measured and zero". No EXPTIME filtering at this
stage: total_exptime_s is recorded as-is (including any short
acquisition/test frames the archives return) for Stage 2 to threshold -
cutting here would silently throw away information at the wrong layer."""
import os
import re
import time
import warnings
import calendar
from datetime import date, datetime, timezone
from importlib import import_module

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")  # NotOpenSSLWarning (urllib3/LibreSSL) - environment noise, not a signal

ew = import_module("05_ew_screen")  # reuse LINES: rest-frame metal-line wavelengths (project convention)

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "out")
TMP_KOA_PATH = "/tmp/hires_01_koa_query.csv"

RADIUS_ARCSEC = 5.0
RADIUS_DEG = RADIUS_ARCSEC / 3600.0

KOA_INSTRUMENTS = {"hires": "high", "esi": "medium"}
ESO_INSTRUMENTS = {"UVES": "high", "ESPRESSO": "high", "XSHOOTER": "medium"}
RETRIES = 5
BACKOFF_S = 4

TODAY = datetime.now(timezone.utc)


def koa_query(instrument, ra, dec):
    """Cone search one KOA instrument. Returns a DataFrame (possibly
    empty - real zero coverage) or None (query never successfully ran
    after all retries - a genuine failure, must not be read as zero)."""
    from pykoa.koa import Koa
    pos = f"circle {ra} {dec} {RADIUS_DEG}"
    for attempt in range(RETRIES):
        if os.path.exists(TMP_KOA_PATH):
            os.remove(TMP_KOA_PATH)
        try:
            Koa.query_position(instrument, pos, TMP_KOA_PATH, format="csv", overwrite=True)
        except Exception:
            pass
        if os.path.exists(TMP_KOA_PATH) and os.path.getsize(TMP_KOA_PATH) > 0:
            return pd.read_csv(TMP_KOA_PATH)
        time.sleep(BACKOFF_S)
    return None


def eso_query(tap, ra, dec):
    """Cone search ESO ObsCore for UVES/ESPRESSO/XSHOOTER. Returns a
    DataFrame (possibly empty) or None (query never ran).

    Uses the TAP service directly (verified live below), not
    astroquery.eso: ivoa.ObsCore already exposes a real per-product `snr`
    column (confirmed populated with varying measured values, e.g.
    270.6/327.1/349.5 across three XSHOOTER products of the same target -
    not a constant placeholder), so there was nothing astroquery.eso's
    higher-level wrapper would add here."""
    q = f"""
    SELECT target_name, s_ra, s_dec, instrument_name, t_exptime, em_min, em_max,
           dp_id, obs_release_date, proposal_id, snr
    FROM ivoa.ObsCore
    WHERE INTERSECTS(s_region, CIRCLE('ICRS', {ra}, {dec}, {RADIUS_DEG})) = 1
    AND instrument_name IN ('UVES','ESPRESSO','XSHOOTER')
    """
    for attempt in range(RETRIES):
        try:
            res = tap.search(q)
            return res.to_table().to_pandas()
        except Exception:
            time.sleep(BACKOFF_S)
    return None


def koa_release_date_approx(semid, propint_months):
    """Nominal release date from KOA semester + proprietary period.
    Approximation (see module docstring) - not an authoritative date."""
    if pd.isna(semid) or pd.isna(propint_months):
        return None
    m = re.match(r"(\d{4})([ab])", str(semid).strip().lower())
    if not m:
        return None
    year, half = int(m.group(1)), m.group(2)
    end = date(year, 7, 31) if half == "a" else date(year + 1, 1, 31)
    total_month = end.month - 1 + int(propint_months)
    y = end.year + total_month // 12
    mo = total_month % 12 + 1
    day = min(end.day, calendar.monthrange(y, mo)[1])
    return date(y, mo, day)


def metal_wavelengths(z_abs):
    return {name: lam0 * (1 + z_abs) for name, lam0 in ew.LINES.items()}


def covers_all_lines(wave_min_ang, wave_max_ang, z_abs):
    if wave_min_ang is None or wave_max_ang is None or pd.isna(wave_min_ang) or pd.isna(wave_max_ang):
        return False
    obs = metal_wavelengths(z_abs)
    return all(wave_min_ang <= w <= wave_max_ang for w in obs.values())


def process_koa_hits(df, instrument, res_class, target_id, z_abs):
    """Aggregates raw KOA exposure rows for one instrument into one
    coverage record (or none, if df is empty).

    snr_estimate: KOA's `sig2nois` header keyword, HIRES only - ESI's
    position-query metadata has no S/N-equivalent column at all (verified
    live, confirmed absent, not just unpopulated). Checked this is a real
    per-exposure estimate, not a fixed proposal-nominal number: sig2nois
    varies exposure-to-exposure within the SAME program/night (e.g.
    target 438888113, program K293Hb: 6-9 across 9 exposures) - it tracks
    actual conditions, not just a cover-sheet target."""
    if df is None or len(df) == 0:
        return None
    wave_min = df["waveblue"].min()
    wave_max = df["wavered"].max()
    release_dates = [koa_release_date_approx(sid, pi) for sid, pi in
                      zip(df.get("semid", []), df.get("propint", []))]
    release_dates = [d for d in release_dates if d is not None]
    n_public = sum(1 for d in release_dates if d <= TODAY.date())
    if "sig2nois" in df.columns:
        snr_vals = pd.to_numeric(df["sig2nois"], errors="coerce").dropna()
    else:
        snr_vals = pd.Series(dtype=float)
    return {
        "ID": target_id, "archive": "KOA", "instrument": instrument.upper(),
        "res_class": res_class, "n_exposures": len(df),
        "total_exptime_s": float(df["elaptime"].sum()) if "elaptime" in df else np.nan,
        "wave_min_ang": float(wave_min), "wave_max_ang": float(wave_max),
        "covers_metal_lines": covers_all_lines(wave_min, wave_max, z_abs),
        "n_exposures_public": n_public,
        "is_public": n_public > 0,
        "public_flag_method": "approx: semester-end + propint months",
        "snr_median": float(snr_vals.median()) if len(snr_vals) else np.nan,
        "snr_values": ";".join(f"{v:g}" for v in snr_vals) if len(snr_vals) else "",
        "snr_source": "sig2nois (KOA header keyword)" if len(snr_vals) else "not available for this instrument",
        "programs": ";".join(sorted(set(df.get("progid", pd.Series(dtype=str)).astype(str)))),
        "query_status": "ok",
    }


def process_eso_hits(df, target_id, z_abs):
    """Aggregates raw ESO ObsCore rows into one coverage record per
    instrument (or none, if df is empty).

    snr_estimate: ObsCore's own `snr` column - confirmed live to be a
    real per-product measured value (varies product-to-product on the
    same target, e.g. 5.7-26.0 across 7 XSHOOTER products of one quasar),
    not a placeholder. Only present for phase3 (reduced, dp_id=ADP.*)
    products, which is all `ivoa.ObsCore` returns for these instruments
    here anyway (see module docstring)."""
    if df is None or len(df) == 0:
        return []
    out = []
    for instrument, res_class in ESO_INSTRUMENTS.items():
        sub = df[df["instrument_name"] == instrument]
        if len(sub) == 0:
            continue
        wave_min_ang = float(sub["em_min"].min()) * 1e10  # ObsCore em_min/em_max are in metres
        wave_max_ang = float(sub["em_max"].max()) * 1e10
        rel = pd.to_datetime(sub["obs_release_date"], utc=True, errors="coerce")
        n_public = int((rel <= TODAY).sum())
        snr_vals = pd.to_numeric(sub["snr"], errors="coerce").dropna()
        out.append({
            "ID": target_id, "archive": "ESO", "instrument": instrument,
            "res_class": res_class, "n_exposures": len(sub),
            "total_exptime_s": float(sub["t_exptime"].sum()),
            "wave_min_ang": wave_min_ang, "wave_max_ang": wave_max_ang,
            "covers_metal_lines": covers_all_lines(wave_min_ang, wave_max_ang, z_abs),
            "n_exposures_public": n_public,
            "is_public": n_public > 0,
            "public_flag_method": "exact: obs_release_date",
            "snr_median": float(snr_vals.median()) if len(snr_vals) else np.nan,
            "snr_values": ";".join(f"{v:g}" for v in snr_vals) if len(snr_vals) else "",
            "snr_source": "ObsCore snr (phase3 pipeline)" if len(snr_vals) else "not available",
            "programs": ";".join(sorted(set(sub["proposal_id"].astype(str)))),
            "query_status": "ok",
        })
    return out


DONE_IDS_PATH = os.path.join(OUT, "archival_coverage_done_ids.txt")


def main():
    import pyvo
    targets = pd.read_csv(os.path.join(OUT, "merged_candidates_clean.csv"))
    targets["ID"] = targets["ID"].astype(str)
    tap = pyvo.dal.TAPService("https://archive.eso.org/tap_obs")

    # Resume support: a target with zero hits everywhere leaves NO row in
    # archival_coverage.csv (process_koa_hits/process_eso_hits return
    # None/[] on no match) - so "not in the CSV" is NOT a valid "not yet
    # processed" signal, it's indistinguishable from "processed, found
    # nothing". Track completed IDs in a separate marker file instead.
    done_ids = set()
    if os.path.exists(DONE_IDS_PATH):
        with open(DONE_IDS_PATH) as f:
            done_ids = {line.strip() for line in f if line.strip()}
    out_path = os.path.join(OUT, "archival_coverage.csv")
    rows = pd.read_csv(out_path).to_dict("records") if os.path.exists(out_path) else []
    if done_ids:
        print(f"Резюме: {len(done_ids)} целей уже обработаны в предыдущем запуске, пропускаем")

    for i, t in enumerate(targets.itertuples()):
        if t.ID in done_ids:
            continue
        print(f"[{i+1}/{len(targets)}] {t.ID} ({t.survey}, z_abs={t.z_abs:.3f})", flush=True)

        for instrument, res_class in KOA_INSTRUMENTS.items():
            df = koa_query(instrument, t.ra, t.dec)
            if df is None:
                rows.append({"ID": t.ID, "archive": "KOA", "instrument": instrument.upper(),
                             "res_class": res_class, "query_status": "query_failed"})
                print(f"    KOA/{instrument.upper()}: query failed after {RETRIES} retries")
                continue
            rec = process_koa_hits(df, instrument, res_class, t.ID, t.z_abs)
            if rec:
                rows.append(rec)
                print(f"    KOA/{instrument.upper()}: {rec['n_exposures']} exposures, "
                      f"public={rec['is_public']}, covers_lines={rec['covers_metal_lines']}")

        df = eso_query(tap, t.ra, t.dec)
        if df is None:
            rows.append({"ID": t.ID, "archive": "ESO", "instrument": "ANY",
                         "res_class": "n/a", "query_status": "query_failed"})
            print("    ESO: query failed after retries")
        else:
            recs = process_eso_hits(df, t.ID, t.z_abs)
            for rec in recs:
                rows.append(rec)
                print(f"    ESO/{rec['instrument']}: {rec['n_exposures']} exposures, "
                      f"public={rec['is_public']}, covers_lines={rec['covers_metal_lines']}")

        # incremental write - same discipline as 05_ew_screen.py, long-running network job
        pd.DataFrame(rows).to_csv(out_path, index=False)
        with open(DONE_IDS_PATH, "a") as f:
            f.write(t.ID + "\n")

    cov = pd.DataFrame(rows)
    cov.to_csv(out_path, index=False)

    hits = cov[cov["query_status"] == "ok"]
    failed = cov[cov["query_status"] == "query_failed"]
    n_targets_any = hits["ID"].nunique()
    n_targets_high = hits[hits["res_class"] == "high"]["ID"].nunique()
    n_targets_public = hits[hits["is_public"]]["ID"].nunique()

    print("\n=== ИТОГ ===")
    print(f"Целей всего: {len(targets)}")
    print(f"Целей с хотя бы одним архивным попаданием (любое разрешение): {n_targets_any}")
    print(f"Целей с high-res покрытием (HIRES/UVES/ESPRESSO): {n_targets_high}")
    print(f"Целей с публичным покрытием (хотя бы 1 публичная экспозиция): {n_targets_public}")
    print(f"\nПо инструментам (число целей с хотя бы 1 попаданием):")
    print(hits.groupby("instrument")["ID"].nunique())
    print(f"\nНеудачных запросов (query_failed, после {RETRIES} попыток каждый): {len(failed)}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
