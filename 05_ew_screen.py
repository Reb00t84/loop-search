#!/usr/bin/env python3
"""Stage 2 metal-line screen: matched-filter scan for SiII/CII/OI/AlII
absorption at each DLA's z_abs, third iteration of the method (see CLAUDE.md
for why the box-car and single-pixel versions were retracted).

Stage 2 сужения DLA-целей, ВЕРСИЯ 3 (2026-07-07, matched-filter — см.
CLAUDE.md, "Правила проекта", п.3-4). Версия 1 (box-car EW-сумма по
фиксированному окну ±300 км/с) была отозвана: 58% из 55 проверенных
"кандидатов" при объективной перепроверке показали >4sigma провалы линий,
которые box-car не засчитал — либо zCNN смещён от истинной скорости
металлов на 300-500+ км/с (окно мимо), либо узкий реальный провал тонет
в шуме по всему широкому окну (систематика dv — см.
out/zcnn_vs_metal_velocity_offset.csv: median|dv|=79 км/с, но 10% систем
имеют |dv|>300 км/с). Версия 2 (одиночный пиксель) — тоже отозвана,
см. CLAUDE.md.

Новый метод (решение автора 2026-07-07) — matched-filter скан вместо
box-car по всему окну И вместо одиночного пикселя (промежуточная версия
на одном пикселе была ЕЩЁ ХУЖЕ старой: 868 "кандидатов" вместо 57 —
недобирала S/N у реальных линий, размазанных на 2-3 пикселя средней
значимости, единичный пиксель для этого слишком шумный статистик):
  1. Для каждой линии сканируем УЗКОЕ окно интегрирования (+-100 км/с)
     с шагом 50 км/с по всему широкому диапазону +-500 км/с; в каждой
     точке считаем EW-значимость как в box-car, но по узкому окну —
     это даёт правильное накопление S/N по нескольким пикселям реальной
     линии, не теряя её и не топя в шуме остального широкого диапазона.
     Берём максимум значимости по всем пробным точкам.
  2. SOLO-детекция: максимум по одной линии значим на исправленном
     (Sidak) пороге 3.36sigma — n_eff=7 независимых элементов разрешения
     BOSS (R~2000, FWHM~150 км/с) в диапазоне сканирования 1000 км/с,
     alpha_global=0.0027.
  3. PAIR-детекция: пики ДВУХ линий одновременно >2.0sigma И согласованы
     по скорости в допуске +-70 км/с (~1 элемент разрешения) — согласие
     по скорости само подавляет случайные совпадения (p~2e-4/систему на
     4 линиях, на порядок жёстче SOLO-цели), поэтому порог ниже.
  4. detected = SOLO или PAIR; candidate_metalpoor = НИ то, ни другое ни
     по одной линии, при >=3 пригодных линиях.
Асимметрия задачи (решение автора): дорогая ошибка — пропустить металлы
(ложно-чистое облако тратит ночь на Keck), дешёвая — забраковать чистое
(потеряли кандидата из тысяч). Пороги настроены агрессивно В СТОРОНУ
детекции металлов, а не в сторону чистых."""
import os
import sys
import time
import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from astropy.io import fits
from io import BytesIO

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "out")

LINES = {
    "SiII1526": 1526.7066,
    "CII1334":  1334.5323,
    "OI1302":   1302.1685,
    "AlII1670": 1670.7886,
}
C_KMS = 299792.458
RUN2D = "v5_13_0"  # DR16 eBOSS/BOSS; >99.9% наших целей (plate>=3000), см. TODO
N_WORKERS = 10

SEARCH_KMS = 500       # полуширина поиска (сканируем весь диапазон)
INTEG_KMS = 100        # полуширина УЗКОГО окна интегрирования вокруг пробной точки
STEP_KMS = 50          # шаг сканирования (INTEG_KMS перекрывается соседними шагами)
CONT_KMS = (600, 4000) # диапазон для континуума (медиана)
Z_SOLO = 3.36          # Sidak-порог, n_eff=7, alpha_global=0.0027 (см. докстринг)
Z_PAIR = 2.0           # порог на линию для парной/согласованной детекции
TOL_DV_PAIR = 70.0     # км/с, допуск согласованности пары по скорости

def spec_url(plate, mjd, fiber):
    """Builds the DR16 SDSS 'lite' spectrum SAS URL for plate/mjd/fiber."""
    return (f"https://data.sdss.org/sas/dr16/sdss/spectro/redux/{RUN2D}"
            f"/spectra/lite/{plate}/spec-{plate}-{mjd}-{fiber:04d}.fits")

def fetch_spec(plate, mjd, fiber, tries=2, timeout=25):
    """Downloads one spectrum's FITS bytes, or None on failure."""
    url = spec_url(plate, mjd, fiber)
    for _ in range(tries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200 and len(r.content) > 1000:
                return r.content
        except requests.RequestException:
            pass
    return None

def find_peak(wave, flux, ivar, and_mask, z_abs, lam0,
              search_kms=SEARCH_KMS, integ_kms=INTEG_KMS, step_kms=STEP_KMS,
              cont_kms=CONT_KMS):
    """Matched-filter scan: slides a narrow EW-integration window across the
    search range and returns the position/significance of the strongest dip.
    Скользящее (matched-filter) окно интегрирования EW шириной
    +-integ_kms, сканируемое с шагом step_kms по всему +-search_kms;
    возвращает позицию и значимость МАКСИМУМА. Узкое окно вокруг пробной
    точки (не весь широкий диапазон) даёт правильное накопление S/N по
    нескольким пикселям реальной линии, не проседая на одиночном шумном
    пикселе (см. CLAUDE.md: box-car по всему ±300 км/с топил сигнал в
    шуме, одиночный пиксель — наоборот, терял реальные линии на 2-3
    соседних пикселях средней значимости)."""
    lam_obs = lam0 * (1 + z_abs)
    dv = (wave - lam_obs) / lam_obs * C_KMS
    good = (ivar > 0) & (and_mask == 0)
    cont_sel = good & (np.abs(dv) > cont_kms[0]) & (np.abs(dv) < cont_kms[1])
    if cont_sel.sum() < 10:
        return None
    cont = np.median(flux[cont_sel])
    if not np.isfinite(cont) or cont <= 0:
        return None
    dlam = np.gradient(wave)
    best = None
    centers = np.arange(-search_kms, search_kms + step_kms, step_kms)
    for c in centers:
        sel = good & (np.abs(dv - c) < integ_kms)
        if sel.sum() < 2:
            continue
        f, iv, dl = flux[sel], ivar[sel], dlam[sel]
        ew = np.sum((1 - f / cont) * dl)
        ew_err = np.sqrt(np.sum((1.0 / np.sqrt(iv) / cont * dl) ** 2))
        if not np.isfinite(ew_err) or ew_err <= 0:
            continue
        sigma = ew / ew_err
        if best is None or sigma > best["sigma"]:
            best = {"sigma": sigma, "dv": float(c), "cont": cont,
                     "ew_err_rest": ew_err / (1 + z_abs)}
    return best

def classify(peaks):
    """Applies the SOLO/PAIR detection logic to the 4 lines' find_peak results.
    peaks: dict[line_name] -> find_peak() результат или None.
    Возвращает (n_ok, detected, candidate_metalpoor, solo_hits, pair_hit)."""
    valid = {k: v for k, v in peaks.items() if v is not None}
    n_ok = len(valid)
    solo_hits = [k for k, v in valid.items() if v["sigma"] > Z_SOLO]
    pair_hit = False
    names = list(valid.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = valid[names[i]], valid[names[j]]
            if (a["sigma"] > Z_PAIR and b["sigma"] > Z_PAIR
                    and abs(a["dv"] - b["dv"]) < TOL_DV_PAIR):
                pair_hit = True
    detected = bool(solo_hits) or pair_hit
    candidate = (n_ok >= 3) and not detected
    return n_ok, detected, candidate, solo_hits, pair_hit

def process_one(row):
    """Fetches one target's spectrum and returns its full screening record."""
    content = fetch_spec(int(row.Plate), int(row.MJD), int(row.Fiber))
    if content is None:
        return {"ID": row.ID, "status": "fetch_failed"}
    try:
        with fits.open(BytesIO(content)) as d:
            t = d[1].data
            wave, flux, ivar, mask = 10 ** t["loglam"], t["flux"], t["ivar"], t["and_mask"]
    except Exception as e:
        return {"ID": row.ID, "status": f"read_failed:{e}"}

    peaks = {name: find_peak(wave, flux, ivar, mask, row.zCNN, lam0)
              for name, lam0 in LINES.items()}
    n_ok, detected, candidate, solo_hits, pair_hit = classify(peaks)

    rec = {"ID": row.ID, "ra": row.ra, "dec": row.dec, "zCNN": row.zCNN,
           "NHI_best": row.NHI_best, "Flux": row.Flux, "SNR": row.SNR, "status": "ok"}
    max_3sig_UL = 0.0
    for name in LINES:
        p = peaks[name]
        rec[f"{name}_sigma"] = p["sigma"] if p else np.nan
        rec[f"{name}_dv"] = p["dv"] if p else np.nan
        rec[f"{name}_3sigUL"] = 3 * p["ew_err_rest"] if p else np.nan
        if p:
            max_3sig_UL = max(max_3sig_UL, 3 * p["ew_err_rest"])
    rec["n_lines_ok"] = n_ok
    rec["solo_hits"] = ",".join(solo_hits)
    rec["pair_hit"] = pair_hit
    rec["detected"] = detected
    rec["candidate_metalpoor"] = candidate
    # информативность предела: 3sigma UL самой слабо ограниченной из
    # пригодных линий (Å, rest-frame) -> для сортировки/фильтрации
    # кандидатов по тому, насколько предел вообще что-то говорит.
    rec["max_3sig_UL"] = max_3sig_UL if n_ok else np.nan
    return rec

def main(limit=None, out_name="dla_ew_screening.csv"):
    """Runs process_one over all not-yet-screened targets, 10 workers, resumable."""
    targets = pd.read_csv(os.path.join(OUT, "dla_targets.csv"))
    known = pd.read_csv(os.path.join(OUT, "dla_known_metallicity_matches.csv"))
    targets = targets[~targets["ID"].isin(known["target_ID"])].reset_index(drop=True)

    out_path = os.path.join(OUT, out_name)
    done_ids = set()
    if os.path.exists(out_path):
        done_ids = set(pd.read_csv(out_path)["ID"])
        print(f"[resume] уже обработано {len(done_ids)} целей")
    todo = targets[~targets["ID"].isin(done_ids)]
    if limit:
        todo = todo.iloc[:limit]
    print(f"К обработке: {len(todo)} из {len(targets)} (без известной металличности)")

    write_header = not os.path.exists(out_path)
    t0 = time.time()
    n_written = 0
    with open(out_path, "a") as fout, ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(process_one, row): row.ID for row in todo.itertuples()}
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            df1 = pd.DataFrame([rec])
            df1.to_csv(fout, header=write_header, index=False)
            write_header = False
            fout.flush()
            n_written += 1
            if i % 200 == 0 or i == len(todo):
                dt = time.time() - t0
                print(f"[{i}/{len(todo)}] {dt:.0f}s, {dt/i:.2f}s/target, "
                      f"ETA {(len(todo)-i)*dt/i/60:.1f} min", flush=True)
    print(f"Готово: {n_written} записей дописано в {out_path}")

if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=lim)
