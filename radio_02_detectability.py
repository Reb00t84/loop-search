#!/usr/bin/env python3
"""§8 candidate: 21cm radio channel feasibility, stage 2.
For NVSS-matched (radio-loud) targets from radio_01: 21cm rest-frame
frequency + uGMRT band assignment, HI optical-depth grid (N_HI x T_s),
and integration time to 3sigma via the radiometer equation with real
uGMRT SEFD (Table 1, GMRT_specs.pdf, 2025-12-15). Writes
out/radio_21cm_detectability.csv; out/radio_21cm_feasibility.md written
separately once these numbers are in hand (protocol 2)."""
import os
import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(__file__), "out")

NU_21CM = 1420.405751768  # MHz, HI hyperfine rest frequency

# --- uGMRT Table 1 (GMRT_specs.pdf, 15 Dec 2025) ---
# band: (nu_lo, nu_hi MHz, gain K/Jy, Tsys at nu_lo K, Tsys at nu_hi K)
BANDS = {
    "Band-2": (125.0, 250.0, 0.33, 760.0, 240.0),
    "Band-3": (250.0, 500.0, 0.38, 165.0, 100.0),
}
NPOL = 2

def band_assign(nu_obs):
    for name, (lo, hi, *_rest) in BANDS.items():
        if lo <= nu_obs <= hi:
            return name
    return None

def sefd_at(band, nu_obs):
    lo, hi, gain, tsys_lo, tsys_hi = BANDS[band]
    tsys = np.interp(nu_obs, [lo, hi], [tsys_lo, tsys_hi])  # linear in freq between given endpoints
    return tsys / gain, tsys

# --- flux at observing frequency ---
# Target 531686697 has a DIRECT TGSS 150 MHz point (323.6 +- 32.8 mJy) vs NVSS
# 1.4 GHz (404.4 mJy) -> use the two real measured points (interpolate/extrapolate
# in log-log space using the actual local spectral index), not an assumed index.
# The other two targets have no TGSS match (not just "assume steep spectrum" --
# genuinely unmeasured at low freq) -> bracket with alpha = 0 (flat), -0.5, -0.8
# (S ~ nu^alpha), flagged explicitly as an assumption, not a measurement.
TGSS = {531686697: (150.0, 323.6)}
ALPHA_GRID = [0.0, -0.5, -0.8]

def flux_at_nu(target_id, nu_nvss, s_nvss, nu_obs):
    if target_id in TGSS:
        nu_lo, s_lo = TGSS[target_id]
        alpha_local = np.log(s_nvss / s_lo) / np.log(nu_nvss / nu_lo)
        s = s_lo * (nu_obs / nu_lo) ** alpha_local
        return {"alpha_used": alpha_local, "method": "TGSS+NVSS interpolation", "S_mJy": s}
    else:
        out = {}
        for a in ALPHA_GRID:
            s = s_nvss * (nu_obs / nu_nvss) ** a
            out[f"S_mJy_alpha{a:+.1f}"] = s
        out["method"] = "NVSS extrapolation (assumed alpha, no direct low-freq point)"
        return out

# --- HI optical depth: N_HI = 1.823e18 * Ts * integral(tau dv), single component
# of width DV_KMS -> tau = N_HI / (1.823e18 * Ts * DV_KMS). Kanekar & Chengalur
# convention (see literature anchors in the report).
DV_KMS = 20.0  # fiducial single-component velocity width
C_KMS = 2.99792458e5

def tau_of(NHI, Ts, dv_kms=DV_KMS):
    return NHI / (1.823e18 * Ts * dv_kms)

# --- integration time to 3sigma on tau, radiometer equation, matched-filter
# channel width = nu_obs * dv/c (single "channel" matched to the line width) ---
def t_3sigma_hours(sefd_jy, s_cont_mjy, tau, nu_obs_mhz, dv_kms=DV_KMS, nsigma=3.0):
    if s_cont_mjy <= 0 or tau <= 0:
        return np.inf
    d_nu_hz = nu_obs_mhz * 1e6 * (dv_kms / C_KMS)
    sefd_mjy = sefd_jy * 1e3
    # sigma_tau = SEFD/(S_cont*sqrt(npol*d_nu*t)) => t = (nsigma*SEFD/(S_cont*tau))^2/(npol*d_nu)
    t_sec = (nsigma * sefd_mjy / (s_cont_mjy * tau)) ** 2 / (NPOL * d_nu_hz)
    return t_sec / 3600.0

def main():
    df = pd.read_csv(os.path.join(OUT, "radio_nvss_crossmatch.csv"))
    cand = df[df["S1_4_mJy"].notna()].copy()
    print(f"{len(cand)} NVSS-matched candidates out of {len(df)}")

    NHI_GRID = [2e20, 1e21]
    TS_GRID = [100, 500, 1000]

    rows = []
    for _, t in cand.iterrows():
        nu_obs = NU_21CM / (1 + t["z_abs"])
        band = band_assign(nu_obs)
        flux_info = flux_at_nu(int(t["ID"]), 1400.0, t["S1_4_mJy"], nu_obs)
        base = dict(ID=t["ID"], z_abs=t["z_abs"], NHI_cat=t["NHI"], S1_4_mJy=t["S1_4_mJy"],
                    nu_obs_MHz=nu_obs, band=band)
        base.update(flux_info)
        if band is None:
            base["note"] = "21cm redshifted outside Band-2/Band-3 coverage"
            rows.append(base)
            continue
        sefd_jy, tsys = sefd_at(band, nu_obs)
        base["SEFD_Jy"] = sefd_jy
        base["Tsys_K"] = tsys

        s_keys = [k for k in flux_info if k.startswith("S_mJy")] or ["S_mJy"]
        for skey in s_keys:
            s_val = flux_info[skey]
            label = skey.replace("S_mJy", "").strip("_") or "measured"
            for NHI in NHI_GRID:
                for Ts in TS_GRID:
                    tau = tau_of(NHI, Ts)
                    t_hr = t_3sigma_hours(sefd_jy, s_val, tau, nu_obs)
                    row = dict(base)
                    row.update({"flux_case": label, "S_used_mJy": s_val, "NHI": NHI, "Ts": Ts,
                                "tau": tau, "t_3sigma_hr": t_hr, "feasible_le10hr": t_hr <= 10.0})
                    rows.append(row)

    out = pd.DataFrame(rows)
    out_path = os.path.join(OUT, "radio_21cm_detectability.csv")
    out.to_csv(out_path, index=False)
    print(f"-> {out_path}, {len(out)} rows")

    feas = out[out.get("feasible_le10hr", False) == True]
    print(f"\nrows with t_3sigma <= 10 hr: {len(feas)}")
    if len(feas):
        print(feas[["ID", "band", "flux_case", "NHI", "Ts", "tau", "t_3sigma_hr"]].to_string())
    n_qualifying_targets = feas["ID"].nunique() if len(feas) else 0
    print(f"\nunique targets with >=1 feasible (Ts,NHI,flux_case) combo: {n_qualifying_targets}")

if __name__ == "__main__":
    main()
