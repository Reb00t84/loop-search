#!/usr/bin/env python3
"""Full per-line EW_3sigma calculation: every one of the 4 diagnostic
metal lines (SiII1526, CII1334, OI1302, AlII1670), for every one of the
42 VLT-reachable candidates, gets its OWN observed wavelength, arm,
PS1 photometry, forest-transmission check, and Cayrel EW limit --
no reuse of the Lyman-alpha anchor's S/N for lines that land elsewhere.
Supersedes the anchor-only shortcut in radio_04 (kept for the D/H
point itself, which is a genuinely different measurement).

Same ingredients as radio_03/04, cited there: Fig 16 (ESO X-Shooter
User Manual P104v1) for the reference S/N=10,1hr curve; Table 10 for
R/sampling per arm; Cayrel (1988) for EW_3sigma; Faucher-Giguere et al.
2008 fit for forest transmission (not independently verified this
session, flagged as before)."""
import os
import time
import numpy as np
import pandas as pd
from astroquery.vizier import Vizier
import astropy.units as u
from astropy.coordinates import SkyCoord
from requests.exceptions import RequestException

OUT_CSV = 'out/xshooter_ew_limits_per_line.csv'
LYA_NM = 121.567
LINES_NM = {'SiII1526': 152.6, 'CII1334': 133.4, 'OI1302': 130.2, 'AlII1670': 167.0}
ARM_PARAMS = {
    'UVB': dict(R=5400, sampling=5.2, lo=300.0, hi=559.5),
    'VIS': dict(R=8900, sampling=5.8, lo=559.5, hi=1024.0),
}


def with_retry(fn, *args, retries=3, backoff=8, **kwargs):
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(backoff)


def arm_of(lam_nm):
    if ARM_PARAMS['UVB']['lo'] <= lam_nm < ARM_PARAMS['UVB']['hi']:
        return 'UVB'
    if ARM_PARAMS['VIS']['lo'] <= lam_nm <= ARM_PARAMS['VIS']['hi']:
        return 'VIS'
    return None


def mag_ref_fig16(lam_nm):
    if lam_nm < 560:
        if lam_nm < 350:
            return 20.2
        elif lam_nm < 500:
            return 21.5
        else:
            return 21.2
    else:
        return 20.7


def ew_3sigma_A(lam_obs_nm, arm, sn_pix):
    p = ARM_PARAMS[arm]
    lam_A = lam_obs_nm * 10
    fwhm_A = lam_A / p['R']
    pixel_scale_A = fwhm_A / p['sampling']
    sigma_ew = 1.5 * np.sqrt(fwhm_A * pixel_scale_A) / sn_pix
    return 3 * sigma_ew


v = Vizier(columns=['*'], row_limit=5)
PS1_BANDS = {'gmag': 481, 'rmag': 617, 'imag': 752, 'zmag': 866, 'ymag': 962}


def get_ps1_all(ra, dec):
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    r = with_retry(v.query_region, coord, radius=3 * u.arcsec, catalog='II/349/ps1')
    if len(r) == 0 or len(r[0]) == 0:
        return {}
    row = r[0][0]
    out = {}
    for b in PS1_BANDS:
        val = row[b]
        out[b] = float(val) if val is not None else None
    return out


def mag_at(ps1_row, lam_nm):
    best = min(PS1_BANDS, key=lambda b: abs(PS1_BANDS[b] - lam_nm))
    val = ps1_row.get(best)
    return val, best


desi_qso_z = pd.read_csv('out/desi_targets.csv').set_index('TARGETID')['Z_QSO']


def get_zqso(target_id, survey, ra, dec):
    if survey == 'DESI' and target_id in desi_qso_z.index:
        val = desi_qso_z.loc[target_id]
        return float(val.iloc[0] if hasattr(val, 'iloc') else val)
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    for cat in ['V/154', 'VII/289']:
        r = with_retry(v.query_region, coord, radius=2 * u.arcsec, catalog=cat)
        if len(r) > 0 and len(r[0]) > 0:
            for zcol in ['z', 'Z', 'zsp', 'Z_VI']:
                if zcol in r[0].colnames:
                    val = r[0][0][zcol]
                    if val is not None and not np.ma.is_masked(val):
                        return float(val)
    return None


exptime_df = pd.read_csv('out/xshooter_exptime_vlt_reachable.csv')
TARGETS = list(exptime_df[['ID', 'survey', 'z_abs']].itertuples(index=False))
merged = pd.read_csv('out/merged_candidates_clean.csv').set_index('ID')

done_pairs = set()
if os.path.exists(OUT_CSV):
    prev = pd.read_csv(OUT_CSV)
    done_pairs = set(zip(prev['ID'], prev['line']))
    print(f"resuming: {len(done_pairs)} (target,line) pairs already done")

for i, (tid, survey, z_abs) in enumerate(TARGETS):
    if all((tid, name) in done_pairs for name in LINES_NM):
        continue
    ra, dec = merged.loc[tid, 'ra'], merged.loc[tid, 'dec']
    print(f"{i+1}/{len(TARGETS)} ID={tid}", flush=True)
    ps1_row = get_ps1_all(ra, dec)
    z_qso = get_zqso(tid, survey, ra, dec)
    for name, rest_nm in LINES_NM.items():
        if (tid, name) in done_pairs:
            continue
        lam_nm = rest_nm * (1 + z_abs)
        arm = arm_of(lam_nm)
        row = dict(ID=tid, survey=survey, z_abs=z_abs, line=name, lam_obs_nm=lam_nm, arm=arm)
        if arm is None:
            row.update(mag_cont=None, forest_applies=None, sn_at_1hr=None,
                        EW3sig_1hr_A=None, EW3sig_2hr_A=None,
                        note='outside UVB/VIS coverage entirely')
        else:
            mag_cont, band_used = mag_at(ps1_row, lam_nm)
            forest_applies = (z_qso is not None) and (lam_nm < LYA_NM * (1 + z_qso))
            if forest_applies:
                tau_eff = 0.0018 * (1 + z_abs) ** 3.92
                mag_corr = -2.5 * np.log10(np.exp(-tau_eff))
            else:
                mag_corr = 0.0
            mag_eff = (mag_cont + mag_corr) if mag_cont is not None else None
            mag_ref = mag_ref_fig16(lam_nm)
            if mag_eff is not None:
                sn_1hr = 10.0 * 10 ** (-0.4 * (mag_eff - mag_ref))
                ew_1hr = ew_3sigma_A(lam_nm, arm, sn_1hr)
                ew_2hr = ew_3sigma_A(lam_nm, arm, sn_1hr * np.sqrt(2))
            else:
                sn_1hr = ew_1hr = ew_2hr = None
            row.update(band_used=band_used, mag_cont=mag_cont, z_qso=z_qso,
                       forest_applies=forest_applies, mag_corr=mag_corr, mag_eff=mag_eff,
                       sn_at_1hr=sn_1hr, EW3sig_1hr_A=ew_1hr, EW3sig_2hr_A=ew_2hr, note='')
        pd.DataFrame([row]).to_csv(OUT_CSV, mode='a', header=not os.path.exists(OUT_CSV), index=False)
        done_pairs.add((tid, name))

out = pd.read_csv(OUT_CSV)
pd.set_option('display.width', 220)
print(out.to_string())
print(f"\n{len(out)} (target,line) rows, {out['EW3sig_2hr_A'].notna().sum()} with a computed EW limit")
print(f"lines outside UVB/VIS coverage entirely: {(out['arm'].isna()).sum()}")
