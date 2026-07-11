#!/usr/bin/env python3
"""v4-trigger candidate, Stage 2b: measurement. Detection/upper-limit EW
for SiII1526/CII1334/OI1302/AlII1670 on the 10 usable targets from
out/highres_inventory.csv, real per-pixel data (not metadata), one
product per line (highest measured SNR from Stage 2a among usable
products covering that line - picking the single best product rather
than co-adding multiple exposures; simpler and defensible, but not
maximally sensitive, noted here rather than silently assumed away).

Method: reuses 05_ew_screen.py's find_peak()/classify() UNCHANGED - the
exact matched-filter scan (narrow +-INTEG_KMS window slid across
+-SEARCH_KMS in STEP_KMS steps, max-significance point kept) and
SOLO/PAIR detection thresholds (Z_SOLO=3.36 Sidak, Z_PAIR=2.0 with
TOL_DV_PAIR=70km/s coherence) already used for SDSS Stage 2 and DESI
Stage 2. This is not a new method - it is the v3 protocol run on real
per-pixel error instead of SDSS/DESI pipeline ivar, so the numbers here
are directly comparable to the SDSS/DESI medians (same significance
definition, same window, same thresholds). HIRES/ESO products don't
carry SDSS's `ivar`/`and_mask` columns directly: ivar is reconstructed
as 1/err^2 (0 where err<=0, matching find_peak's own `ivar>0` masking),
and_mask is passed as all-zero (no per-pixel quality bitmask in this
data beyond err<=0, which ivar already encodes) - find_peak's own
`good = (ivar>0) & (and_mask==0)` selection is unchanged by this.

EW convention (reused, not reinvented): find_peak returns sigma and
ew_err_rest at its best-fit point; since sigma = ew/ew_err is frame-
invariant under the (1+z) rest-frame conversion, ew_rest = sigma *
ew_err_rest exactly - no need to modify find_peak to also return the raw
EW. Detected lines report this as the measured EW; non-detections report
the standard project convention 3*ew_err_rest (same as `3sigUL` in
05_ew_screen.py/out/final_candidates.csv). AlII1670 (or any line) with NO
usable product covering it is reported as status="unavailable" with a
reason, per the author's explicit instruction - not a spurious zero limit.

Survey-level comparison class depends on what the target's provenance
actually claims (CLAUDE.md precedent, desi_08_exclude_contradicted.py):
  - */Stage2 targets (SDSS or DESI matched-filter non-detections): the
    claim being tested is "0/4 lines". detected -> "contamination"
    (survey missed a real line, same finding class as the 2 confirmed
    DESI/SDSS cross-survey contaminants in v3 Stage 3.5); not detected
    with n_ok>=3 -> "confirmed_clean"; n_ok<3 -> "ambiguous" (too few
    usable lines to conclude).
  - */Stage1 targets (literature [M/H]<-2 from Rafelski+2012): the claim
    is "some weak metals", NOT zero - a detection here is CONSISTENT with
    the literature, not contamination (the same distinction already
    applied to the 2 Stage-1 entries excluded from desi_07's contaminant
    list for the identical reason). Classes: "consistent_with_literature"
    (detected), "unexpected_nondetection" (not detected, n_ok>=3),
    "ambiguous" (n_ok<3).

dv (v3 protocol): find_peak's own `dv` at the best-fit point, reported
ONLY for lines classified "detected" (SOLO or part of a PAIR hit) - for
an upper limit the scan's best-sigma point is just wherever the noise
happened to peak, not a real line centroid, so dv there isn't physically
meaningful and is left NaN rather than reported as if it were.

Resume: same marker-file discipline as Stages 1/2a."""
import os
import warnings
from importlib import import_module

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ew = import_module("05_ew_screen")        # LINES, C_KMS, find_peak, classify, Z_SOLO, Z_PAIR, TOL_DV_PAIR
h2 = import_module("hires_02_inventory")   # read_eso_spectrum, read_hires_order

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "out")
DATA_DIR = os.path.join(HERE, "data", "highres_spectra")
DONE_IDS_PATH = os.path.join(OUT, "highres_purity_done_ids.txt")


def resolve_product_path(target_id, archive, product_id):
    if archive == "ESO":
        return os.path.join(DATA_DIR, target_id, f"{product_id}.fits")
    if archive == "KOA":
        # product_id = "<koaid basename>/<subdir>/<fname>"; on-disk layout
        # is data/highres_spectra/<ID>/lev1/<subdir>/<fname> (no koaid dir)
        _, rest = product_id.split("/", 1)
        return os.path.join(DATA_DIR, target_id, "lev1", rest)
    raise ValueError(f"unknown archive {archive}")


def load_spectrum(target_id, archive, product_id):
    path = resolve_product_path(target_id, archive, product_id)
    if archive == "ESO":
        wave, flux, err = h2.read_eso_spectrum(path)
    else:
        wave, flux, err = h2.read_hires_order(path)
    ivar = np.where(err > 0, 1.0 / np.where(err > 0, err, 1.0) ** 2, 0.0)
    and_mask = np.zeros(wave.shape, dtype=int)
    return wave, flux, ivar, and_mask


def survey_class(provenance, n_ok, detected):
    if n_ok < 3:
        return "ambiguous"
    if str(provenance).endswith("Stage1"):
        return "consistent_with_literature" if detected else "unexpected_nondetection"
    return "contamination" if detected else "confirmed_clean"


def line_is_pair_member(name, peaks, pair_hit):
    if not pair_hit or peaks.get(name) is None:
        return False
    p = peaks[name]
    for other, q in peaks.items():
        if other == name or q is None:
            continue
        if (p["sigma"] > ew.Z_PAIR and q["sigma"] > ew.Z_PAIR
                and abs(p["dv"] - q["dv"]) < ew.TOL_DV_PAIR):
            return True
    return False


def process_target(target_id, z_abs, provenance, inv_rows):
    """Picks the best usable product per line, measures via find_peak,
    classifies, and returns the 4 per-line rows for this target."""
    peaks = {}
    meta = {}
    for name in ew.LINES:
        nominal = inv_rows[inv_rows[f"covers_{name}"] == True]
        if len(nominal) == 0:
            peaks[name] = None
            meta[name] = {"status": "unavailable", "reason": "no usable product covers this line"}
            continue
        cands = nominal[nominal[f"snr_{name}"].notna()]
        if len(cands) == 0:
            # nominally in range on every candidate product, but <3 valid
            # pixels near the line everywhere (Stage 2a's
            # "covered_but_unmeasurable", just at single-line granularity
            # within an otherwise-usable product) - distinct from no
            # coverage at all, so say so rather than reporting the same
            # generic reason for both.
            peaks[name] = None
            meta[name] = {"status": "unavailable",
                          "reason": "line nominally in range but <3 valid pixels near it "
                                    "in every usable product (covered_but_unmeasurable)"}
            continue
        best = cands.loc[cands[f"snr_{name}"].idxmax()]
        meta[name] = {"archive": best["archive"], "instrument": best["instrument"],
                       "product_id": best["product_id"]}
        try:
            wave, flux, ivar, and_mask = load_spectrum(target_id, best["archive"], best["product_id"])
            p = ew.find_peak(wave, flux, ivar, and_mask, z_abs, ew.LINES[name])
        except Exception as e:
            peaks[name] = None
            meta[name]["status"] = "read_failed"
            meta[name]["reason"] = str(e)
            continue
        peaks[name] = p
        if p is None:
            meta[name]["status"] = "measure_failed"
            meta[name]["reason"] = "find_peak returned None (insufficient continuum/window pixels)"

    n_ok, detected, candidate, solo_hits, pair_hit = ew.classify(peaks)
    target_cls = survey_class(provenance, n_ok, detected)

    rows = []
    for name in ew.LINES:
        m = meta[name]
        p = peaks[name]
        row = {"ID": target_id, "line": name, "z_abs": z_abs, "provenance": provenance,
               "archive": m.get("archive"), "instrument": m.get("instrument"),
               "product_id": m.get("product_id"),
               "n_lines_ok": n_ok, "solo_hits": ",".join(solo_hits), "pair_hit": pair_hit,
               "target_class": target_cls}
        if p is None:
            row.update({"status": m.get("status", "unavailable"), "reason": m.get("reason"),
                        "ew_ang": np.nan, "sigma": np.nan, "dv_kms": np.nan})
        else:
            line_detected = (name in solo_hits) or line_is_pair_member(name, peaks, pair_hit)
            ew_rest = p["sigma"] * p["ew_err_rest"]
            row.update({
                "status": "detected" if line_detected else "upper_limit", "reason": None,
                "sigma": p["sigma"],
                "ew_ang": ew_rest if line_detected else 3 * p["ew_err_rest"],
                "dv_kms": p["dv"] if line_detected else np.nan,
            })
        rows.append(row)
    return rows


def main():
    inv = pd.read_csv(os.path.join(OUT, "highres_inventory.csv"))
    inv["ID"] = inv["ID"].astype(str)
    usable = inv[inv["suitability"] == "usable"]
    targets = pd.read_csv(os.path.join(OUT, "merged_candidates_clean.csv"))
    targets["ID"] = targets["ID"].astype(str)
    target_ids = sorted(usable["ID"].unique())
    tinfo = targets[targets["ID"].isin(target_ids)][["ID", "z_abs", "provenance"]]

    done_ids = set()
    if os.path.exists(DONE_IDS_PATH):
        with open(DONE_IDS_PATH) as f:
            done_ids = {line.strip() for line in f if line.strip()}
    out_path = os.path.join(OUT, "highres_purity.csv")
    rows = pd.read_csv(out_path).to_dict("records") if os.path.exists(out_path) else []
    if done_ids:
        print(f"Резюме: {len(done_ids)} целей уже обработаны, пропускаем")

    for i, t in enumerate(tinfo.itertuples()):
        if t.ID in done_ids:
            continue
        print(f"[{i+1}/{len(tinfo)}] {t.ID} z_abs={t.z_abs:.3f} ({t.provenance})", flush=True)
        inv_rows = usable[usable["ID"] == t.ID]
        target_rows = process_target(t.ID, t.z_abs, t.provenance, inv_rows)
        rows += target_rows
        for r in target_rows:
            print(f"    {r['line']}: {r['status']}"
                  + (f" sigma={r['sigma']:.2f}" if pd.notna(r.get('sigma')) else "")
                  + (f" EW={r['ew_ang']*1000:.0f}mA" if pd.notna(r.get('ew_ang')) else "")
                  + (f" dv={r['dv_kms']:.0f}km/s" if pd.notna(r.get('dv_kms')) else ""))
        print(f"    -> target_class={target_rows[0]['target_class']}")

        pd.DataFrame(rows).to_csv(out_path, index=False)
        with open(DONE_IDS_PATH, "a") as f:
            f.write(t.ID + "\n")

    pur = pd.DataFrame(rows)
    pur.to_csv(out_path, index=False)

    print("\n=== ИТОГ Stage 2b ===")
    print(f"Целей: {len(tinfo)}, строк (цель x линия): {len(pur)}")
    print("\nПо статусу линий:")
    print(pur["status"].value_counts())
    tgt_cls = pur.drop_duplicates("ID")[["ID", "target_class"]]
    print("\nПо классу сверки (на цель):")
    print(tgt_cls["target_class"].value_counts())
    det = pur[pur["status"] == "detected"]
    if len(det):
        print(f"\ndv детектированных линий: median={det['dv_kms'].median():.1f} km/s, "
              f"|dv|>300km/s: {(det['dv_kms'].abs() > 300).sum()}/{len(det)}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
