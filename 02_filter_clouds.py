#!/usr/bin/env python3
"""Main channel: selects candidate clouds by the jinn-gas D/H+He signature
and builds the DLA follow-up target shortlist (see select_dla_targets).

Главный канал: отбор облаков-кандидатов.
Сигнатура джинн-газа: металлы отсутствуют, D/H НИЖЕ стандартного BBN
(обычная химэволюция даёт обратное: сжигание D всегда с металлами),
He повышен. Наблюдаемое облако — смесь джинн-выброса с обычным газом,
кандидаты ложатся на линию смешивания в плоскости (D/H, Y_p)."""
import os
import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

# --- Стандартные значения (Planck-BBN) ---
DH_STD  = 2.53e-5      # (D/H)_p, Cooke et al. 2018
DH_ERR  = 0.03e-5
YP_STD  = 0.247        # первичный гелий

# --- Параметры джинн-компоненты (из препринта, §5: догорание нуклеосинтеза
#     при высокой барионной загрузке: D ~ 0, He усилен) ---
DH_JINN = 0.0
YP_JINN = 0.30

def mixing_line(f):
    """Jinn-gas fraction f in the mix -> (D/H, Y_p). Доля джинн-газа f в смеси -> (D/H, Y_p)."""
    dh = (1 - f) * DH_STD + f * DH_JINN
    yp = (1 - f) * YP_STD + f * YP_JINN
    return dh, yp

# --- Precision D/H системы ---
# Значения DH_1e5/DH_err/OH пересчитаны 2026-07-07 из log10(D/H) и [O/H] в
# Table 3, Cooke, Pettini & Steidel 2018, ApJ 855, 102 (arXiv:1710.11129) —
# это самосогласованная переанализированная выборка авторов (не смешивать
# со старыми значениями отдельных статей 2003/2006/2012/2014: у Q1243+307,
# например, Cooke2018 даёт новое измерение 2.39e-5, а не старое Kirkman+2003
# 2.42e-5, которое стояло здесь раньше). Ранее в таблице по ошибке не
# хватало SDSSJ1358+0349 (седьмая система выборки) — добавлена.
# ra/dec (J2000, deg, ICRS) — SIMBAD/LAMOST, сверены 2026-07-07; для
# Q1243+307 использована координата, явно указанная в тексте Cooke2018
# (иначе расхождение ~14" между старыми алиасами VV98/VV2010 в SIMBAD).
PRECISION_DH = pd.DataFrame([
    # qso,             z_abs,   DH_1e5, DH_err, OH,     ra_deg,      dec_deg,     precision_sample
    ("HS0105+1619",    2.53651, 2.58,   0.15,  -1.771,   17.026764,   16.597214,  True),
    ("Q0913+072",      2.61829, 2.53,   0.10,  -2.416,  139.058174,    7.040138,  True),
    ("Q1243+307",      2.52564, 2.39,   0.08,  -2.769,  191.545417,   30.525333,  True),
    ("SDSSJ1358+0349", 2.85305, 2.62,   0.07,  -2.804,  209.516562,    3.826670,  True),
    ("SDSSJ1358+6522", 3.06726, 2.58,   0.07,  -2.335,  209.678817,   65.376851,  True),
    ("SDSSJ1419+0829", 3.04973, 2.51,   0.05,  -1.922,  214.960636,    8.496749,  True),
    ("SDSSJ1558-0031", 2.70242, 2.40,   0.14,  -1.650,  239.542351,   -0.522232,  True),
    # Balashev+2016 (MNRAS 458, 2188), независимый пайплайн — Cooke2018 §4.1
    # прямо исключает его (и Riemer-Sørensen+2017, Zavarygin+2017) из
    # "Precision Sample" именно из-за разницы в анализе, не из-за качества
    # данных. Это и есть статус, который просили проверить.
    ("J1444+2919",     2.437,   1.97,   0.31,  -2.042,  221.223083,   29.318248, False),
], columns=["qso", "z_abs", "DH_1e5", "DH_err", "OH", "ra", "dec", "precision_sample"])

def flag_anomalies(df, sigma=3.0):
    """Flags D/H significantly below standard BBN at low metallicity. Флаг: D/H значимо ниже стандарта при низкой металличности."""
    dh = df["DH_1e5"].values * 1e-5
    err = df["DH_err"].values * 1e-5
    dev = (DH_STD - dh) / np.sqrt(err**2 + DH_ERR**2)
    df = df.copy()
    df["nsigma_low_DH"] = np.round(dev, 2)
    df["jinn_fraction"] = np.round((DH_STD - dh) / (DH_STD - DH_JINN), 4)
    df["Yp_predicted"]  = np.round(YP_STD + df["jinn_fraction"] * (YP_JINN - YP_STD), 4)
    df["candidate"] = (dev > sigma) & (df["OH"] < -1.5)
    return df

def load_sdss_dla(path, readme=None):
    """Loads the full DR16 DLA catalog (Chabanier+2022, J/ApJS/258/18).
    Загрузка полного DR16 DLA-каталога (Chabanier+2022, J/ApJS/258/18).
    Формат — CDS ASCII (не FITS), колонки описаны в ReadMe рядом с файлом;
    имя распакованного файла (dr16dla.dat) должно совпадать с указанным
    в File Summary ReadMe, иначе astropy не находит таблицу."""
    from astropy.io import ascii as asc
    if readme is None:
        readme = os.path.join(os.path.dirname(path), "sdss_dr16_dla_ReadMe")
    t = asc.read(path, readme=readme, format="cds")
    return t.to_pandas()

def select_dla_targets(df, nhi_min=20.3, conf_min=0.5, snr_min=3.0):
    """Builds a shortlist of DLA sightlines worth high-res follow-up
    spectroscopy (strong DLA + adequate SNR); NOT a metal-poor classification
    (see note below) — that requires the follow-up spectroscopy itself.
    ВАЖНО: каталог Chabanier+2022 НЕ содержит металличности вообще —
    только z, N(HI) (CNN и voigt-fit), поток и SNR сайтлайна. Отбор «нет
    металлов» по этому каталогу невозможен в принципе; функция строит
    только список ПЕРВИЧНЫХ ЦЕЛЕЙ для последующей спектроскопии высокого
    разрешения (сильный DLA + достаточный SNR, чтобы измерение металлов
    и D/H было вообще осуществимо) — метку 'кандидат петли' может дать
    только измерение [O/H] и D/H по данным цели, не этот отбор сам по себе.
    """
    nhi = df["NHIfit"].where(df["NHIfit"] > 0, df["NHIcnn"])
    sel = (nhi >= nhi_min) & (df["Conf"] >= conf_min) & (df["SNR"] >= snr_min)
    out = df[sel].copy()
    out["NHI_best"] = nhi[sel]
    return out.rename(columns={"RAdeg": "ra", "DEdeg": "dec"})

if __name__ == "__main__":
    res = flag_anomalies(PRECISION_DH)
    res.to_csv(os.path.join(OUT, "cloud_candidates.csv"), index=False)
    print(res.to_string(index=False))
    print("\nЛиния смешивания (f, D/H, Y_p):")
    for f in (0.0, 0.1, 0.25, 0.5):
        dh, yp = mixing_line(f)
        print(f"  f={f:.2f}: D/H={dh:.2e}, Y_p={yp:.4f}")
    n = int(res["candidate"].sum())
    print(f"\nКандидатов при пороге 3σ: {n}")
    print("Ключевая проверка кандидата: измерение He на той же системе "
          "должно дать Y_p ~ Yp_predicted; обычный газ даст 0.247.")

    DATA = os.path.join(os.path.dirname(__file__), "data")
    dla_path = os.path.join(DATA, "dr16dla.dat")
    if os.path.exists(dla_path):
        dla = load_sdss_dla(dla_path)
        targets = select_dla_targets(dla)
        targets.to_csv(os.path.join(OUT, "dla_targets.csv"), index=False)
        print(f"\nDR16 DLA: {len(dla)} записей, {len(targets)} целей для "
              f"спектроскопии высокого разрешения (N(HI)>=20.3, Conf>=0.5, "
              f"SNR>=3) -> out/dla_targets.csv. Металличности в каталоге "
              f"нет — это только шорт-лист сайтлайнов, годных для измерения "
              f"[O/H] и D/H, а не отбор «нет металлов» сам по себе.")
    else:
        print(f"\n[skip] {dla_path} не найден — запусти 01_fetch_catalogs.py")
