#!/usr/bin/env python3
"""Cayrel (1988) EW_3sigma detection limits for X-Shooter, for all 42
VLT-reachable candidates (radio_03_xshooter_exptime.py output).

sigma(EW) = 1.5 * sqrt(FWHM * pixel_scale) / (S/N per pixel)
EW_3sigma = 3 * sigma(EW)

FWHM = lambda/R, pixel_scale = FWHM/sampling[pix/FWHM] -- both from
Table 10, X-Shooter User Manual VLT-MAN-ESO-14650-4942 P104v1 (real
values, not assumed):
  UVB, 1.0" slit: R=5400, sampling=5.2 pix/FWHM
  VIS, 0.9" slit: R=8900, sampling=5.8 pix/FWHM

S/N scaling reuses radio_03's per-target sn_at_1hr (S/N ~ sqrt(t)),
computed at the DLA's own redshifted Lyman-alpha (the D/H anchor
point). IMPORTANT CAVEAT, checked explicitly rather than assumed: the
four diagnostic metal lines (SiII1526, CII1334, OI1302, AlII1670) sit
at DIFFERENT observed wavelengths than Lyman-alpha for the same z_abs,
and could in principle fall in a different arm. This script computes
each line's observed wavelength per target and flags any that land in
an arm different from the Lyman-alpha anchor -- for those, the S/N
(and hence EW_3sigma) used here is not directly applicable and needs a
separate photometric point, not silently reused."""
import numpy as np
import pandas as pd

ARM_PARAMS = {
    'UVB': dict(R=5400, sampling=5.2, lo=300.0, hi=559.5),
    'VIS': dict(R=8900, sampling=5.8, lo=559.5, hi=1024.0),
}
LINES_NM = {'SiII1526': 152.6, 'CII1334': 133.4, 'OI1302': 130.2, 'AlII1670': 167.0}


def arm_of(lam_nm):
    if ARM_PARAMS['UVB']['lo'] <= lam_nm < ARM_PARAMS['UVB']['hi']:
        return 'UVB'
    if ARM_PARAMS['VIS']['lo'] <= lam_nm <= ARM_PARAMS['VIS']['hi']:
        return 'VIS'
    return None


def ew_3sigma_A(lam_obs_nm, arm, sn_pix):
    p = ARM_PARAMS[arm]
    lam_A = lam_obs_nm * 10
    fwhm_A = lam_A / p['R']
    pixel_scale_A = fwhm_A / p['sampling']
    sigma_ew = 1.5 * np.sqrt(fwhm_A * pixel_scale_A) / sn_pix
    return 3 * sigma_ew


df = pd.read_csv('out/xshooter_exptime_vlt_reachable.csv')

rows = []
for _, t in df.iterrows():
    if pd.isna(t['sn_at_1hr']):
        continue
    sn_1hr, arm_anchor, z_abs = t['sn_at_1hr'], t['arm'], t['z_abs']
    row = dict(ID=t['ID'], survey=t['survey'], z_abs=z_abs, arm_anchor=arm_anchor,
               lam_anchor_nm=t['lam_obs_nm'])
    for t_hr in [1.0, 2.0]:
        sn_t = sn_1hr * np.sqrt(t_hr)
        row[f'EW3sig_anchor_A_{t_hr:.0f}hr'] = ew_3sigma_A(t['lam_obs_nm'], arm_anchor, sn_t)
    # check each diagnostic line's actual arm for this z_abs -- don't assume
    mismatches = []
    for name, rest_nm in LINES_NM.items():
        lam_line_nm = rest_nm * (1 + z_abs)
        arm_line = arm_of(lam_line_nm)
        row[f'lam_{name}_nm'] = lam_line_nm
        row[f'arm_{name}'] = arm_line
        if arm_line != arm_anchor:
            mismatches.append(name)
    row['lines_outside_anchor_arm'] = ','.join(mismatches) if mismatches else ''
    rows.append(row)

out = pd.DataFrame(rows)
out.to_csv('out/xshooter_ew_limits.csv', index=False)
pd.set_option('display.width', 220)
print(out[['ID', 'survey', 'z_abs', 'arm_anchor', 'EW3sig_anchor_A_1hr', 'EW3sig_anchor_A_2hr',
           'lines_outside_anchor_arm']].to_string())

n_mismatch = (out['lines_outside_anchor_arm'] != '').sum()
print(f"\n{n_mismatch}/{len(out)} targets have >=1 diagnostic line landing outside the anchor arm "
      f"(EW limit for those specific lines needs its own S/N point, not reused from the anchor)")
print(f"\nEW3sig at 2hr: median={out['EW3sig_anchor_A_2hr'].median()*1000:.1f} mA, "
      f"range {out['EW3sig_anchor_A_2hr'].min()*1000:.1f}-{out['EW3sig_anchor_A_2hr'].max()*1000:.1f} mA")
