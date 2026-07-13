#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Порядковые оценки тёмного сектора для v5 (§3-4, «состав петли»).
Канон чисел отчёта out/dark_sector_v5_estimates.md — прогонять этот
скрипт, не пересказ отчёта, если нужно перепроверить/обновить числа.

Часть 1: доля ТМ под горизонтом при коллапсе (loss-cone, NFW минигало).
Часть 2: аккреция ТМ за чернодырную фазу vs бондиевская оценка §4 препринта.

Зависимости: numpy, scipy (integrate, optimize) — уже есть в .venv проекта.
Использование: python3 tools/dark_sector_v5_calc.py
"""
import numpy as np
from scipy import integrate
from scipy.optimize import brentq

# --- constants (cgs) ---
G = 6.674e-8
c = 2.998e10
Msun = 1.989e33
pc = 3.086e18
yr = 3.156e7
mp = 1.673e-24

# Planck 2018 VI (Planck Collaboration 2020, A&A 641, A6)
Om = 0.315
Obh2 = 0.0224
h = 0.674
Ob = Obh2 / h**2
ODM = Om - Ob
fDM = ODM / Om
print(f"Ob={Ob:.4f} ODM={ODM:.4f} fDM(DM/matter)={fDM:.4f}  ODM/Ob={ODM/Ob:.3f}")

rho_crit0 = 1.878e-29 * h**2
rho_m0 = Om * rho_crit0
print(f"rho_crit0={rho_crit0:.3e} g/cm3  rho_m0={rho_m0:.3e} g/cm3")


def rho_m(z):
    return rho_m0 * (1 + z)**3


# --- Part 1: loss-cone f_coll = M_DM,in/M_seed ---
# NOTE (2026-07-13): the first draft of this section used M_enc(r) (the halo's
# own enclosed mass at the particle's orbital radius r) inside J_lc^2, instead
# of M_seed (the mass that actually dominates the potential well AT the
# pericenter R_s). Caught by re-derivation + direct numerical check (a
# separate closed form for M_DM,in/M_seed that looked too clean: exact
# independence of M_seed). Fixed below: J_lc^2 = 2*G*Mseed*Rs always (Mseed
# sources the pericenter well by construction, Rs=2*G*Mseed/c^2); M_enc(r) is
# only used to set the AMBIENT halo velocity dispersion v(r)^2 ~ G*(Mseed+
# Menc(r))/r far from the seed. No more closed form (numerator/denominator no
# longer cancel on all r) -> integrate numerically. See
# out/dark_sector_v5_estimates.md, section "Уточнение метода", for the writeup.

Delta_c = 18 * np.pi**2  # Bryan & Norman 1998 (ApJ 495, 80), high-z (EdS) limit


def halo_params(sigma_kms, z, conc):
    sigma = sigma_kms * 1e5
    Rvir = sigma * np.sqrt(3 / (4 * np.pi * G * Delta_c * rho_m(z)))
    Mvir = sigma**2 * Rvir / G          # virial relation sigma^2 = G*Mvir/Rvir
    rs = Rvir / conc
    mc = np.log(1 + conc) - conc / (1 + conc)
    rho_s = Mvir / (4 * np.pi * rs**3 * mc)  # NFW normalization, TOTAL (DM+baryons) density
    return dict(Mvir=Mvir, Rvir=Rvir, rs=rs, rho_s=rho_s, conc=conc)


def f_coll(Mseed_Msun, sigma_kms=3, z=25, conc=4):
    """Numerically integrated loss-cone DM fraction captured under the horizon
    at collapse, f_coll = M_DM,in/M_seed (Newtonian pericenter condition;
    see report for the x4 relativistic-cone correction, applied on top)."""
    hp = halo_params(sigma_kms, z, conc)
    Mseed = Mseed_Msun * Msun
    R_s = 2 * G * Mseed / c**2
    rs, rho_s, Rvir = hp['rs'], hp['rho_s'], hp['Rvir']

    def Menc_halo(x):  # halo-only enclosed mass (DM+baryons), x=r/rs
        return 4 * np.pi * rho_s * rs**3 * (np.log(1 + x) - x / (1 + x))

    def rho_DM_of_x(x):
        return fDM * rho_s / (x * (1 + x)**2)

    def f_lc(x):
        r = x * rs
        Mtot_enc = Mseed + Menc_halo(x)
        return Mseed * R_s / (r * Mtot_enc)

    def integrand(x):
        r = x * rs
        dM_DM_dx = 4 * np.pi * r**2 * rho_DM_of_x(x) * rs
        return min(f_lc(x), 1.0) * dM_DM_dx  # cap loss-cone fraction at 1

    c_upper = Rvir / rs
    M_DM_in, _ = integrate.quad(integrand, 1e-12, c_upper, limit=200)
    return M_DM_in / Mseed


def r_infl_over_rs(Mseed_Msun, sigma_kms=3, z=25, conc=4):
    """Seed's sphere of influence: Mseed = Menc_halo(r_infl)."""
    hp = halo_params(sigma_kms, z, conc)
    rs, rho_s = hp['rs'], hp['rho_s']
    Mseed = Mseed_Msun * Msun

    def Menc_halo(x):
        return 4 * np.pi * rho_s * rs**3 * (np.log(1 + x) - x / (1 + x))

    return brentq(lambda x: Menc_halo(x) - Mseed, 1e-8, 1000)


print("f_coll(sigma), Mseed=1e4 Msun, z=25, c=4:")
for sigma_kms in [1, 3, 10]:
    print(f"  sigma={sigma_kms} km/s  ->  f_coll = {f_coll(1e4, sigma_kms):.3e}   "
          f"(x4 for relativistic cone: {4*f_coll(1e4, sigma_kms):.3e})")

print("\nf_coll(Mseed), sigma=3 km/s, z=25, c=4:")
for Mseed in [1e3, 1e4, 1e5]:
    print(f"  Mseed={Mseed:.0e} Msun  ->  f_coll = {f_coll(Mseed):.3e}   "
          f"x_infl=r_infl/rs={r_infl_over_rs(Mseed):.3f}")

print("\nf_coll(concentration), Mseed=1e4, sigma=3km/s, z=25:")
for conc in [2, 3, 4, 6]:
    print(f"  c={conc}  ->  f_coll = {f_coll(1e4, conc=conc):.3e}")

print("\nf_coll(z), Mseed=1e4, sigma=3km/s, c=4:")
for z in [20, 25, 30]:
    print(f"  z={z}  ->  f_coll = {f_coll(1e4, z=z):.3e}")

print("\nhalo Mvir,Rvir check (sigma=3km/s):")
for z in [20, 25, 30]:
    sigma = 3e5
    Rvir = sigma * np.sqrt(3 / (4 * np.pi * G * Delta_c * rho_m(z)))
    Mvir = sigma**2 * Rvir / G
    print(f"  z={z}: Rvir={Rvir/pc:.1f} pc   Mvir={Mvir/Msun:.3e} Msun")
for sigma_kms in [1, 10]:
    sigma = sigma_kms * 1e5
    Rvir = sigma * np.sqrt(3 / (4 * np.pi * G * Delta_c * rho_m(25)))
    Mvir = sigma**2 * Rvir / G
    print(f"  sigma={sigma_kms}km/s, z=25: Rvir={Rvir/pc:.1f} pc   Mvir={Mvir/Msun:.3e} Msun")

# --- Part 2: DM accretion during BH phase vs Bondi gas (paper's own number) ---
print("\n--- Part 2 ---")


def sigma_capture(M_Msun, vrel_cms):
    M = M_Msun * Msun
    return 16 * np.pi * G**2 * M**2 / (c**2 * vrel_cms**2)


def Mdot_DM(M_Msun, rho_DM, vrel_cms):
    M = M_Msun * Msun
    return 16 * np.pi * G**2 * M**2 * rho_DM / (c**2 * vrel_cms)  # g/s


# fiducial local rho_DM scaled from the paper's own halo gas density n=0.5 cm^-3
n_gas = 0.5
rho_gas_halo = n_gas * mp
rho_DM_fid = (ODM / Ob) * rho_gas_halo
print(f"rho_gas,halo(n=0.5/cc) = {rho_gas_halo:.3e} g/cm3   rho_DM,fid = {rho_DM_fid:.3e} g/cm3")

# enhanced/cuspy NFW central density estimate, z=25, c=4
z0 = 25
conc0 = 4
delta_c_nfw = (Delta_c / 3) * conc0**3 / (np.log(1 + conc0) - conc0 / (1 + conc0))
rho_s_total = delta_c_nfw * rho_m(z0)  # approx rho_crit(z)~rho_m(z) at high z
rho_s_DM = fDM * rho_s_total
print(f"delta_c(NFW,c=4)={delta_c_nfw:.1f}  rho_s(total)={rho_s_total:.3e} g/cm3  rho_s,DM={rho_s_DM:.3e} g/cm3")
print(f"enhancement factor rho_s,DM/rho_DM,fid = {rho_s_DM/rho_DM_fid:.1f}")

# reproduce the paper's own Bondi number: Mdot_gas = 3e3 Msun/1e8 yr (v4, section 4)
Mdot_gas_paper = 3e3 * Msun / (1e8 * yr)
print(f"\nMdot_gas (paper, 1e4 Msun seed) = {Mdot_gas_paper:.3e} g/s = {Mdot_gas_paper/Msun*yr:.3e} Msun/yr")

for vrel_kms in [1, 3, 10]:
    vrel = vrel_kms * 1e5
    for label, rhoDM in [("fiducial", rho_DM_fid), ("cuspy-enhanced", rho_s_DM)]:
        Md = Mdot_DM(1e4, rhoDM, vrel)
        Md_Msun_yr = Md / Msun * yr
        ratio = Md / Mdot_gas_paper
        print(f"  v_rel={vrel_kms}km/s {label:16s}: Mdot_DM={Md_Msun_yr:.3e} Msun/yr   ratio DM/gas={ratio:.3e}")

# mass-independence check: ratio at Mseed=1e3 and 1e5 (rate scales M^2 both channels -> ratio same)
print("\ncheck mass-independence of ratio (fiducial rho_DM, vrel=3km/s):")
for Mseed in [1e3, 1e4, 1e5]:
    Md = Mdot_DM(Mseed, rho_DM_fid, 3e5)
    # Bondi scales as M^2 too -> scale paper's anchor value accordingly
    Mdot_gas_scaled = Mdot_gas_paper * (Mseed / 1e4)**2
    print(f"  Mseed={Mseed:.0e}: Mdot_DM={Md/Msun*yr:.3e} Msun/yr, "
          f"Mdot_gas(scaled)={Mdot_gas_scaled/Msun*yr:.3e} Msun/yr, ratio={Md/Mdot_gas_scaled:.3e}")

# tau grid, dynamical floor and Hawking-like ceiling
print("\ntau window edges per seed mass (GM/c^3 floor, Hawking M^3 ceiling):")
t_evap_sun_yr = 2.1e67
for Mseed in [1e3, 1e4, 1e5]:
    tau_min = G * Mseed * Msun / c**3
    tau_max = t_evap_sun_yr * Mseed**3
    print(f"  Mseed={Mseed:.0e} Msun: tau_min={tau_min:.3e} s   tau_max={tau_max:.3e} yr")

# absolute DM accretion over a few representative tau (fiducial rho_DM, vrel=3km/s), M=1e4
print("\nabsolute Mdot_DM x tau, Mseed=1e4, fiducial rho_DM, vrel=3km/s:")
Md_rate = Mdot_DM(1e4, rho_DM_fid, 3e5) / Msun * yr  # Msun/yr
for tau_yr in [1, 1e3, 1e6, 1e8, 1.4e10]:
    print(f"  tau={tau_yr:.1e} yr -> Delta M_DM = {Md_rate*tau_yr:.3e} Msun")
