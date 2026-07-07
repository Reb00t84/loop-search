#!/usr/bin/env python3
"""Background rate via RA-only scrambling (Dec fixed per GRB, preserves GBM's
declination-dependent exposure without building a poshist exposure map).

Фон случайных совпадений: скремблинг ТОЛЬКО по RA, Dec каждого GRB
фиксирован. Стандартный приём кросс-корреляций: экспозиция GBM неравномерна
по склонению (десятки процентов, вне маски Земли), но при фиксированных
масках инструмента почти не зависит от RA (вращение Земли усредняет за
сутки) — фиксация Dec сохраняет эту структуру без необходимости строить
экспозиционную карту по poshist-истории (десятки GB, см. CLAUDE.md: делать
только если появится кандидат с p~0.01 и дело пойдёт к публикации).
p-value = доля скремблов с числом совпадений >= наблюдённого."""
import os
import numpy as np
import pandas as pd
from importlib import import_module

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "out")
cm = import_module("03_crossmatch".replace("/", "."))  # angsep, загрузчики

def scramble_ra(n, rng):
    """Draws n RA values ~U(0,360); Dec is left untouched by the caller."""
    return rng.uniform(0, 360, n)

def run(clouds, grbs, n_trials=10000, seed=42):
    """Runs n_trials RA-scrambles and returns (obs, bg_mean, bg_std, p_value)."""
    rng = np.random.default_rng(seed)
    obs = len(cm.crossmatch(clouds, grbs))
    counts = np.empty(n_trials, int)
    g = grbs.copy()
    for i in range(n_trials):
        g["ra"] = scramble_ra(len(g), rng)
        counts[i] = len(cm.crossmatch(clouds, g))
    p = (counts >= obs).mean() if obs > 0 else 1.0
    return obs, counts.mean(), counts.std(), p

if __name__ == "__main__":
    clouds = pd.read_csv(os.path.join(OUT, "cloud_candidates.csv"))
    clouds = clouds[clouds["candidate"] == True]
    grbs = cm.select_short_grbs(
        cm.load_fermi_tdat(os.path.join(HERE, "data", "fermigbrst.tdat")))
    obs, mu, sd, p = run(clouds, grbs)
    msg = (f"наблюдено совпадений: {obs}\n"
           f"фон (MC): {mu:.2f} ± {sd:.2f}\n"
           f"p-value: {p:.4f}\n"
           "Интерпретация: p<~0.01 при наличии D/H-аномалии у совпавшего\n"
           "облака — кандидат петли; p~1 — позиционный канал пуст, что для\n"
           "GBM-ошибок локализации (градусы) ожидаемо и само по себе модель\n"
           "не ограничивает (ограничивает нуль по облакам, канал 02).")
    print(msg)
    with open(os.path.join(OUT, "background.txt"), "w") as f:
        f.write(msg + "\n")
