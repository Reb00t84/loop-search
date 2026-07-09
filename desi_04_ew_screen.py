#!/usr/bin/env python3
"""v3-trigger Stage 3: matched-filter EW screen ported to DESI DR1, via
SPARCL for per-object spectrum retrieval (see CLAUDE.md Stage 1: healpix
coadds are ~219MB/file, bulk-unusable; SPARCL solves this). Reuses
find_peak/classify UNCHANGED from 05_ew_screen.py (SDSS) - both take
plain wave/flux/ivar/mask arrays and don't care about the data source;
DESI's mask convention (0=good, nonzero=bad) and ivar semantics were
verified identical to SDSS's during Stage 1 recon.

Same thresholds as SDSS (SOLO 3.36sigma Sidak n_eff=7, PAIR 2.0sigma
+-70km/s, scan +-500km/s / integrate +-100km/s): DESI's resolution
varies with wavelength (R~2000 blue to R~5500 red, unlike SDSS's more
uniform R~2000), so reusing the SDSS/blue-arm-equivalent threshold is
conservative (an upper bound on FWHM => fewer independent trials => an
easier-to-clear threshold than the true, wavelength-dependent one for
lines that land in DESI's redder, higher-R arms) - errs toward the cheap
failure mode (reject a clean sightline) rather than the expensive one
(call a real line clean), consistent with the project's asymmetric-cost
stance. The empirical calibration on the 34 Rafelski-matched systems
(desi_05_calibrate.py) is what actually validates this choice, not the
resolution argument alone."""
import os
import sys
import time
import numpy as np
import pandas as pd
from importlib import import_module

ew = import_module("05_ew_screen")  # find_peak, classify, LINES, C_KMS

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "out")

def get_client():
    """SparclClient.__init__ itself makes an unretried version-check GET
    (even with announcement=False) subject to the same 3.1s ceiling -
    construction itself can time out on a healthy connection and, unlike
    fetch_batch, this happens once at startup with nothing upstream to
    retry it; wrap it too (lost a full resume attempt to this once)."""
    from sparcl.client import SparclClient
    return _with_retry(lambda: SparclClient(
        connect_timeout=30, read_timeout=120, announcement=False))

def _with_retry(fn, tries=5, base_sleep=2.0):
    """sparclclient hardcodes MAX_CONNECT_TIMEOUT=3.1s (not overridable
    via the constructor's connect_timeout kwarg - it gets silently
    clamped), so a single slow TLS handshake can time out even on a
    healthy connection; also seen: mid-transfer disconnects on large
    retrieve() payloads, which sparclclient re-wraps as its own
    UnknownSparcl rather than passing through the underlying requests
    exception. Retry on the broad network-failure surface instead of
    failing the whole batch on a transient blip."""
    import time as _time
    from requests.exceptions import RequestException
    import sparcl.exceptions as _spex
    NETWORK_ERRORS = (RequestException, _spex.UnknownSparcl, _spex.ServerConnectionError)
    for i in range(tries):
        try:
            return fn()
        except NETWORK_ERRORS:
            if i == tries - 1:
                raise
            _time.sleep(base_sleep * (i + 1))

def fetch_batch(client, targetids, data_release="DESI-DR1"):
    """find() + retrieve() for a batch of targetids; returns
    dict[targetid] -> list of spectrum records (a target can have >1
    coadd/epoch)."""
    res = _with_retry(lambda: client.find(
        outfields=["sparcl_id", "targetid"],
        constraints={"targetid": [int(t) for t in targetids],
                     "data_release": [data_release]}))
    if not res.records:
        return {}
    uuids = [r["sparcl_id"] for r in res.records]
    uuid_to_tid = {r["sparcl_id"]: r["targetid"] for r in res.records}
    res2 = _with_retry(lambda: client.retrieve(
        uuid_list=uuids,
        include=["flux", "wavelength", "ivar", "mask", "targetid"]))
    by_target = {}
    for rec in res2.records:
        by_target.setdefault(rec["targetid"], []).append(rec)
    return by_target

def process_one(row, spec_records):
    """spec_records: list of SPARCL records for this target (>=1); uses
    the first one (DESI DR1 main-survey targets typically have a single
    coadd per data_release query at this constraint level)."""
    if not spec_records:
        return {"ID": row.TARGETID, "status": "fetch_failed"}
    rec = spec_records[0]
    try:
        wave = np.array(rec["wavelength"])
        flux = np.array(rec["flux"])
        ivar = np.array(rec["ivar"])
        mask = np.array(rec["mask"])
    except Exception as e:
        return {"ID": row.TARGETID, "status": f"read_failed:{e}"}

    z_abs = row.zDLA
    peaks = {name: ew.find_peak(wave, flux, ivar, mask, z_abs, lam0)
              for name, lam0 in ew.LINES.items()}
    n_ok, detected, candidate, solo_hits, pair_hit = ew.classify(peaks)

    out = {"ID": row.TARGETID, "ra": row.ra, "dec": row.dec, "zDLA": z_abs,
           "NHI": row.NHI, "SNR_FOREST": row.SNR_FOREST, "status": "ok"}
    max_3sig_UL = 0.0
    for name in ew.LINES:
        p = peaks[name]
        out[f"{name}_sigma"] = p["sigma"] if p else np.nan
        out[f"{name}_dv"] = p["dv"] if p else np.nan
        ul = 3 * p["ew_err_rest"] if p else np.nan
        out[f"{name}_3sigUL"] = ul
        if p:
            max_3sig_UL = max(max_3sig_UL, ul)
    out["n_lines_ok"] = n_ok
    out["solo_hits"] = ",".join(solo_hits)
    out["pair_hit"] = pair_hit
    out["detected"] = detected
    out["candidate_metalpoor"] = candidate
    out["max_3sig_UL"] = max_3sig_UL if n_ok else np.nan
    return out

def run(targets_df, out_path, batch_size=50, resume=True):
    """Resumes/dedupes on (TARGETID, zDLA), not TARGETID alone: 441 of
    4267 unique DESI targets have >1 DLA absorber in the shortlist (844
    rows total, see CLAUDE.md project rule 1 - the exact same duplicate-
    key class of bug as the SDSS Stage1+Stage2 92-vs-66 merge, caught here
    before it silently dropped absorbers on resume rather than after."""
    done_keys = set()
    if resume and os.path.exists(out_path):
        prev = pd.read_csv(out_path)
        done_keys = set(zip(prev["ID"], prev["zDLA"]))
        print(f"[resume] уже обработано {len(done_keys)}")
    todo_keys = list(zip(targets_df["TARGETID"], targets_df["zDLA"]))
    todo_mask = [k not in done_keys for k in todo_keys]
    todo = targets_df[todo_mask].reset_index(drop=True)
    print(f"К обработке: {len(todo)}")

    client = get_client()
    write_header = not (resume and os.path.exists(out_path))
    t0 = time.time()
    n_done = 0
    with open(out_path, "a") as fout:
        for start in range(0, len(todo), batch_size):
            chunk = todo.iloc[start:start + batch_size]
            try:
                specs = fetch_batch(client, chunk["TARGETID"].tolist())
            except Exception as e:
                # a batch that exhausts all retries must not kill the run
                # (lost ~35 min of progress to exactly this once already);
                # log it, mark the batch fetch_failed, move on.
                print(f"[batch FAILED after retries] {e}", flush=True)
                specs = {}
            rows = []
            for row in chunk.itertuples():
                rows.append(process_one(row, specs.get(row.TARGETID, [])))
            pd.DataFrame(rows).to_csv(fout, header=write_header, index=False)
            write_header = False
            fout.flush()
            n_done += len(chunk)
            dt = time.time() - t0
            print(f"[{n_done}/{len(todo)}] {dt:.0f}s, {dt/n_done:.2f}s/target, "
                  f"ETA {(len(todo)-n_done)*dt/n_done/60:.1f} min", flush=True)
    print(f"Готово -> {out_path}")

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    targets = pd.read_csv(os.path.join(OUT, "desi_targets.csv"))
    if n:
        targets = targets.sample(n, random_state=1).reset_index(drop=True)
        out_path = os.path.join(OUT, "desi_ew_screening_sample.csv")
    else:
        out_path = os.path.join(OUT, "desi_ew_screening.csv")
    run(targets, out_path)
