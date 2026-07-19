#!/usr/bin/env python3
"""§8 candidate: 21cm radio channel feasibility for the 75-target list.
Stage 1: cross-match out/merged_candidates_clean.csv against NVSS (1.4 GHz,
VizieR VIII/65/nvss) by background-QSO position, 5" cone. All 75 targets
have dec > -40 deg (checked directly), so NVSS alone covers the full
sample - RACS-low is not queried (documented, not silently skipped)."""
import os
import time
import pandas as pd
from astroquery.vizier import Vizier
import astropy.units as u
from astropy.coordinates import SkyCoord

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
RADIUS_ARCSEC = 5.0

df = pd.read_csv(os.path.join(OUT, "merged_candidates_clean.csv"))
assert (df["dec"] > -40).all(), "some targets below NVSS dec limit -- RACS-low needed"

v = Vizier(columns=["*"], row_limit=1)

rows = []
for i, t in df.iterrows():
    coord = SkyCoord(ra=t["ra"] * u.deg, dec=t["dec"] * u.deg)
    for attempt in range(3):
        try:
            result = v.query_region(coord, radius=RADIUS_ARCSEC * u.arcsec, catalog="VIII/65/nvss")
            break
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(5)
    if len(result) > 0 and len(result[0]) > 0:
        row = result[0][0]
        s14 = float(row["S1.4"])
        e_s14 = float(row["e_S1.4"]) if row["e_S1.4"] is not None else float("nan")
        nvss_name = str(row["NVSS"])
        sep_arcsec = coord.separation(
            SkyCoord(ra=row["RAJ2000"], dec=row["DEJ2000"], unit=(u.hourangle, u.deg))
        ).arcsec
    else:
        s14, e_s14, nvss_name, sep_arcsec = float("nan"), float("nan"), None, float("nan")
    rows.append(dict(ID=t["ID"], ra=t["ra"], dec=t["dec"], z_abs=t["z_abs"], NHI=t["NHI"],
                      survey=t["survey"], NVSS_name=nvss_name, S1_4_mJy=s14, e_S1_4_mJy=e_s14,
                      sep_arcsec=sep_arcsec))
    print(f"{i+1}/{len(df)} ID={t['ID']} S1.4={s14}", flush=True)

out = pd.DataFrame(rows)
out_path = os.path.join(OUT, "radio_nvss_crossmatch.csv")
out.to_csv(out_path, index=False)
n_matched = out["S1_4_mJy"].notna().sum()
print(f"\nmatched {n_matched}/{len(out)} within {RADIUS_ARCSEC}\" -> {out_path}")
