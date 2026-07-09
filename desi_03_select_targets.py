#!/usr/bin/env python3
"""Builds the DESI DR1 DLA follow-up target shortlist. Threshold chosen
empirically (author's requirement 2026-07-09, see CLAUDE.md), not assigned:
NHI>=20.3 matches the SDSS-pipeline convention for "strong DLA"; SNR_FOREST
is DESI's closest analogue to Chabanier's SNR/Flux quality axis, and its
cut is set by checking where the 34 Rafelski+2012-confirmed absorbers in
this catalog actually sit (out/desi_zdla_vs_metal_velocity_offset.csv):
SNR_FOREST>=5 retains ~100% of them (SNR_FOREST>=7 already drops some) -
5 is the number written up, not guessed. DELTACHI2 was tested as a
confidence proxy and REJECTED as a hard gate: the confirmed sample's
minimum (0.023) sits below the full population's 25th percentile
(0.016-ish range), i.e. it does not cleanly separate real absorbers from
noise at the low end - a hard DELTACHI2 cut would risk excluding known-
real systems for no measured benefit. Recorded as a negative result, not
silently dropped."""
import os
import pandas as pd
from astropy.table import Table

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

NHI_MIN = 20.3
SNR_FOREST_MIN = 5.0

def load_desi_dla(path=None):
    """Loads the DESI DR1 DLA catalog, dropping the multidimensional
    COEFF column that blocks a direct to-pandas conversion."""
    if path is None:
        path = os.path.join(DATA, "desi_dr1_dlacat_v2.0.fits")
    t = Table.read(path)
    names = [n for n in t.colnames if len(t[n].shape) <= 1]
    return t[names].to_pandas()

def select_desi_targets(df, nhi_min=NHI_MIN, snr_forest_min=SNR_FOREST_MIN):
    """NHI + SNR_FOREST cut only - see module docstring for why DELTACHI2
    is not used as a hard gate."""
    sel = (df["NHI"] >= nhi_min) & (df["SNR_FOREST"] >= snr_forest_min)
    out = df[sel].copy()
    return out.rename(columns={"RA": "ra", "DEC": "dec", "Z_DLA": "zDLA"})

if __name__ == "__main__":
    desi = load_desi_dla()
    targets = select_desi_targets(desi)
    out_path = os.path.join(OUT, "desi_targets.csv")
    targets.to_csv(out_path, index=False)
    print(f"DESI DR1 DLA: {len(desi)} -> {len(targets)} targets "
          f"(NHI>={NHI_MIN}, SNR_FOREST>={SNR_FOREST_MIN}) -> {out_path}")
