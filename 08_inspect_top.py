#!/usr/bin/env python3
"""Manual eyeball review: plots each of the 4 diagnostic lines + full
spectrum for the top-N candidates by flux, saved as PNGs in out/inspect/.

Ручной осмотр топ-N кандидатов (решение автора 2026-07-07, шаг 1 из
трёх после кейса SDSSJ1419+0829): скачивает спектр, рисует линии SiII
1526, CII 1334, OI 1302, AlII 1670 на z_abs + общий вид спектра, чтобы
глазами проверить — реальный чистый спектр или порог/маска/край съели
сигнал. PNG в out/inspect/, плюс печатает числовую сводку (SNR в окне,
маски, континуум) для каждой линии."""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from importlib import import_module

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "out")
INSPECT = os.path.join(OUT, "inspect")
os.makedirs(INSPECT, exist_ok=True)
ew = import_module("05_ew_screen")

def plot_one(row, outpath):
    """Fetches one target's spectrum and saves its diagnostic-line PNG panel."""
    content = ew.fetch_spec(int(row.Plate), int(row.MJD), int(row.Fiber))
    if content is None:
        print(f"  [FAIL] не удалось скачать ID={row.ID}")
        return None
    from astropy.io import fits
    from io import BytesIO
    with fits.open(BytesIO(content)) as d:
        t = d[1].data
        wave, flux, ivar, mask = 10 ** t["loglam"], t["flux"], t["ivar"], t["and_mask"]

    z = row.zCNN
    fig, axes = plt.subplots(1, 5, figsize=(22, 3.2))
    axes[0].plot(wave, flux, lw=0.4, color="k")
    good = ivar > 0
    axes[0].set_title(f"ID={int(row.ID)} z={z:.4f} SNR={row.SNR:.1f} Flux={row.Flux:.1f}")
    axes[0].set_xlabel("obs wave (A)")
    for lam0, name, ax in zip(ew.LINES.values(), ew.LINES.keys(), axes[1:]):
        lam_obs = lam0 * (1 + z)
        dv = (wave - lam_obs) / lam_obs * ew.C_KMS
        sel = np.abs(dv) < 1500
        if sel.sum() < 5:
            ax.set_title(f"{name}: вне покрытия")
            continue
        ax.plot(wave[sel], flux[sel], lw=0.6, color="k")
        m = (np.abs(dv) < 300) & sel
        ax.axvspan(lam_obs*(1-300/ew.C_KMS), lam_obs*(1+300/ew.C_KMS),
                   color="C1", alpha=0.2)
        cont_sel = good & sel & (np.abs(dv) > 600) & (np.abs(dv) < 4000)
        cont = np.median(flux[cont_sel]) if cont_sel.sum() else np.nan
        ax.axhline(cont, color="C0", ls="--", lw=0.8)
        masked = mask[sel & (np.abs(dv) < 300)]
        n_masked = int((masked != 0).sum())
        snr_local = np.nanmedian((flux[sel]*np.sqrt(np.clip(ivar[sel],0,None))))
        ax.set_title(f"{name}\nmasked={n_masked}, cont={cont:.2f}, "
                     f"SNR~{snr_local:.1f}", fontsize=8)
        ax.set_xlim(lam_obs - 20, lam_obs + 20)
    plt.tight_layout()
    plt.savefig(outpath, dpi=100)
    plt.close(fig)
    return outpath

def main(n=15):
    """Plots the top n final candidates by flux."""
    final = pd.read_csv(os.path.join(OUT, "final_candidates.csv"))
    top = final.drop_duplicates("ID").head(n)
    for i, row in enumerate(top.itertuples(), 1):
        outpath = os.path.join(INSPECT, f"{i:02d}_ID{int(row.ID)}.png")
        print(f"[{i}/{len(top)}] ID={row.ID} z={row.zCNN:.4f} SNR={row.SNR:.1f} "
              f"stage={row.stage}")
        plot_one(row, outpath)
    print(f"\nPNG сохранены в {INSPECT}/")

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    main(n)
