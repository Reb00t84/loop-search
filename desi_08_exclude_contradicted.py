#!/usr/bin/env python3
"""v3-trigger Stage 3.5 follow-up: applies the cross-survey corroboration
check (desi_07_cross_check.py) as an exclusion filter, same precedent as
the SDSSJ1419+0829 exclusion in 06_merge_candidates.py — a candidate
independently contradicted by the OTHER survey's own spectrum at the same
position+z is a confirmed contaminant, not a borderline case.

Two of the 4 raw contradictions are Stage-1 (known Rafelski [M/H]) entries
and are NOT real contradictions: Stage-1 candidacy means "[M/H]<-2", not
"zero metals", so the other survey detecting (weak, expected) lines there
is consistent with the literature measurement, not evidence of a screen
failure. Excluded from consideration here.

The remaining 2 are genuine Stage-2 ("0/4 lines detected") candidates
contradicted by a real independent detection:
  - ID 83291431: DESI sees CII at 9.1sigma, OI at 11.7sigma (unambiguous,
    not noise) at dv=+250-300 km/s relative to DESI's OWN z_DLA, which
    itself differs from SDSS's z_abs for the same system by ~415 km/s -
    referenced to SDSS's z_abs the true line sits ~650-700 km/s away,
    outside SDSS's +-500 km/s scan entirely. Same failure mode as the
    original zCNN-vs-metal systematic, this time as an inter-survey
    z_DLA disagreement rather than a z_DLA-vs-z_metal one.
  - ID 374356741: marginal on both sides (SDSS CII 2.84sigma, OI 2.06sigma
    at a different dv than CII by chance; DESI CII 3.46sigma + OI 3.96sigma
    coincident at dv=0) - plausibly a real but modest line where SDSS's
    own noise realization put a competing peak elsewhere in the scan,
    beating the true line to the max() operation. Less clear-cut than
    83291431 but the coherent DESI detection is above the PAIR threshold
    on its own math, not asserted by eye.

Excludes both from the merged list; leaves out/final_candidates.csv (the
standalone SDSS result) untouched - this is a merge-stage finding on new
information (a DESI spectrum didn't exist as a check when that file was
built), not grounds to silently rewrite an already-committed, referenced
artifact. Documented instead."""
import os
import pandas as pd

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "out")

CONFIRMED_CONTAMINANTS = {
    "83291431": "DESI independently detects CII (9.1sigma)/OI (11.7sigma) - "
                 "outside SDSS's own +-500km/s window due to ~415km/s SDSS-DESI "
                 "z_DLA disagreement for this system",
    "374356741": "DESI independently detects a coherent CII(3.5sigma)+OI(4.0sigma) "
                  "pair at dv=0 that SDSS's own noise realization missed (SDSS's own "
                  "OI peak fell at a different, uncoincident dv by chance)",
}

def main():
    """Drops the confirmed cross-survey-contradicted candidates and
    writes the clean merged list."""
    final = pd.read_csv(os.path.join(OUT, "merged_candidates_sdss_desi.csv"))
    final["ID"] = final["ID"].astype(str)
    before = len(final)
    excluded = final[final["ID"].isin(CONFIRMED_CONTAMINANTS)]
    for _, row in excluded.iterrows():
        print(f"[EXCLUDE] {row.survey} ID={row.ID}: {CONFIRMED_CONTAMINANTS[row.ID]}")
    clean = final[~final["ID"].isin(CONFIRMED_CONTAMINANTS)].copy()
    out_path = os.path.join(OUT, "merged_candidates_clean.csv")
    clean.to_csv(out_path, index=False)
    print(f"\n{before} -> {len(clean)} after excluding {len(excluded)} "
          f"cross-survey-contradicted candidates")
    print(f"By survey: SDSS {(clean['survey']=='SDSS').sum()}, "
          f"DESI {(clean['survey']=='DESI').sum()}")
    print(f"-> {out_path}")

if __name__ == "__main__":
    main()
