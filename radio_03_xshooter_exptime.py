#!/usr/bin/env python3
"""VLT/X-Shooter exposure-time rough estimate for all VLT-reachable
(dec<=30) unresolved Stage-2 candidates. Uses the official ESO sensitivity curve (Fig 16,
VLT-MAN-ESO-14650-4942 P104v1) read off by eye at each target's arm/
wavelength, real PS1 photometry, and a Lyman-alpha forest transmission
correction (Faucher-Giguere et al. 2008 fit -- NOT independently
verified in this session, flagged) where the DLA's redshifted Lya
falls within the background QSO's own forest."""
import os
import time
import numpy as np
import pandas as pd
from astroquery.vizier import Vizier
from astroquery.exceptions import NoResultsWarning
import astropy.units as u
from astropy.coordinates import SkyCoord
from requests.exceptions import RequestException

OUT_CSV = 'out/xshooter_exptime_vlt_reachable.csv'


def with_retry(fn, *args, retries=3, backoff=8, **kwargs):
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(backoff)

LYA_NM = 121.567
DEC_LIMIT_VLT = 30.0  # comfortable airmass<1.85 cutoff, see chat derivation (ZD=|lat-dec|, lat=-24.6)

merged = pd.read_csv('out/merged_candidates_clean.csv')
purity = pd.read_csv('out/highres_purity.csv')
contaminated_ids = purity[(purity['target_class'] == 'contamination') &
                           (purity['status'].isin(['detected', 'blend_or_artifact_at_scan_max']))]['ID'].unique()
stage2 = merged[merged['provenance'].str.contains('Stage2')]
remaining = stage2[~stage2['ID'].isin(contaminated_ids)].copy()
vlt_ok = remaining[remaining['dec'] <= DEC_LIMIT_VLT].sort_values('brightness_percentile', ascending=False)
TARGETS = [(int(t.ID), t.ra, t.dec, t.z_abs, t.survey) for t in vlt_ok.itertuples()]
print(f"{len(TARGETS)} VLT-reachable (dec<={DEC_LIMIT_VLT}) unresolved Stage-2 candidates")

# eyeballed reading of Fig 16 (limiting AB mag, S/N=10, 1hr) -- coarse, by region
def mag_ref_fig16(lam_nm):
    if lam_nm < 560:  # UVB
        if lam_nm < 350:
            return 20.2
        elif lam_nm < 500:
            return 21.5   # near the deepest part of the UVB curve
        else:
            return 21.2
    else:  # VIS
        return 20.7

desi_qso_z = pd.read_csv('out/desi_targets.csv').set_index('TARGETID')['Z_QSO'] if True else None

v = Vizier(columns=['*'], row_limit=5)

def get_ps1_mag(ra, dec, lam_nm):
    coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    r = with_retry(v.query_region, coord, radius=3*u.arcsec, catalog='II/349/ps1')
    if len(r) == 0 or len(r[0]) == 0:
        return None, None
    row = r[0][0]
    # pick the PS1 band whose effective wavelength is closest to lam_nm
    bands = {'gmag': 481, 'rmag': 617, 'imag': 752, 'zmag': 866, 'ymag': 962}
    best = min(bands, key=lambda b: abs(bands[b] - lam_nm))
    val = row[best]
    return (float(val) if val is not None else None), best

def get_zqso(target_id, survey, ra, dec):
    if survey == 'DESI' and target_id in desi_qso_z.index:
        val = desi_qso_z.loc[target_id]
        return float(val.iloc[0] if hasattr(val, 'iloc') else val)
    # SDSS: try DR16Q via VizieR position match
    coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    for cat in ['V/154', 'VII/289']:
        r = with_retry(v.query_region, coord, radius=2*u.arcsec, catalog=cat)
        if len(r) > 0 and len(r[0]) > 0:
            for zcol in ['z', 'Z', 'zsp', 'Z_VI']:
                if zcol in r[0].colnames:
                    val = r[0][0][zcol]
                    if val is not None and not np.ma.is_masked(val):
                        return float(val)
    return None

done_ids = set()
if os.path.exists(OUT_CSV):
    done_ids = set(pd.read_csv(OUT_CSV)['ID'].tolist())
    print(f"resuming: {len(done_ids)} targets already done")

for i, (tid, ra, dec, z_abs, survey) in enumerate(TARGETS):
    if tid in done_ids:
        continue
    print(f"{i+1}/{len(TARGETS)} ID={tid}", flush=True)
    lam_obs = LYA_NM * (1 + z_abs)
    arm = 'UVB' if lam_obs < 559.5 else 'VIS'
    mag_cont, band_used = get_ps1_mag(ra, dec, lam_obs)
    z_qso = get_zqso(tid, survey, ra, dec)
    forest_applies = (z_qso is not None) and (lam_obs < LYA_NM * (1 + z_qso))
    if forest_applies:
        tau_eff = 0.0018 * (1 + z_abs) ** 3.92
        F_mean = np.exp(-tau_eff)
        mag_corr = -2.5 * np.log10(F_mean)
    else:
        mag_corr = 0.0
    mag_eff = (mag_cont + mag_corr) if mag_cont is not None else None
    mag_ref = mag_ref_fig16(lam_obs)
    if mag_eff is not None:
        flux_ratio = 10 ** (-0.4 * (mag_eff - mag_ref))
        sn_at_1hr = 10.0 * flux_ratio
        t_sn30 = 1.0 * (30.0 / sn_at_1hr) ** 2
        t_sn40 = 1.0 * (40.0 / sn_at_1hr) ** 2
    else:
        sn_at_1hr = t_sn30 = t_sn40 = None
    row = dict(ID=tid, survey=survey, z_abs=z_abs, lam_obs_nm=lam_obs, arm=arm,
               band_used=band_used, mag_cont=mag_cont, z_qso=z_qso,
               forest_applies=forest_applies, mag_corr=mag_corr, mag_eff=mag_eff,
               mag_ref_fig16=mag_ref, sn_at_1hr=sn_at_1hr,
               t_hr_sn30=t_sn30, t_hr_sn40=t_sn40)
    pd.DataFrame([row]).to_csv(OUT_CSV, mode='a', header=not os.path.exists(OUT_CSV), index=False)
    done_ids.add(tid)

out = pd.read_csv(OUT_CSV)
pd.set_option('display.width', 200)
print(out.to_string())
print(f"\ntotal S/N=30: {out['t_hr_sn30'].sum():.2f} hr, S/N=40: {out['t_hr_sn40'].sum():.2f} hr "
      f"({out['t_hr_sn30'].notna().sum()}/{len(out)} targets with a magnitude)")
