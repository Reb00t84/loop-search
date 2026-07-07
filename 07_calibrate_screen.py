#!/usr/bin/env python3
"""Calibrates the Stage-2 EW screen's false-negative/true-positive rate on
the 58 DR16 sightlines with known (Rafelski+2012) metallicity.

Калибровка EW-скрина (решение автора 2026-07-07, реакция на кейс
SDSSJ1419+0829): прогоняем 05_ew_screen.measure_ews на ВСЕХ 60 системах
Stage 1 (out/dla_known_metallicity_matches.csv) — той же выборке DR16
DLA-сайтлайнов (те же селекции Conf/SNR/NHI, что и у настоящих
кандидатов), но с уже ИЗВЕСТНОЙ металличностью из Rafelski+2012.
Это даёт честную, на тех же данных, оценку:
  - false-negative rate: доля metal-rich ([M/H]>=-2.0) систем, которые
    скрин ошибочно называет "нет линий" (candidate_metalpoor=True)
  - true-positive rate: доля metal-poor ([M/H]<-2.0) систем, которые
    скрин правильно ловит.
n=60 (50 metal-rich + 10 metal-poor), не n=1 как в кейсе J1419."""
import os
import sys
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module
ew = import_module("05_ew_screen")

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "out")

def main():
    """Runs the Stage-2 screen on the known-metallicity set and reports rates."""
    targets = pd.read_csv(os.path.join(OUT, "dla_targets.csv")).reset_index().rename(
        columns={"index": "target_idx"})
    known = pd.read_csv(os.path.join(OUT, "dla_known_metallicity_matches.csv"))
    known = known.drop_duplicates("target_idx")

    cal = known.merge(targets, on="target_idx", suffixes=("", "_t"))
    print(f"Калибровочная выборка: {len(cal)} систем "
          f"({(cal['MH'] < -2.0).sum()} metal-poor, "
          f"{(cal['MH'] >= -2.0).sum()} metal-rich)")

    rows = []
    for row in cal.itertuples():
        rec = ew.process_one(row)
        rec["MH"] = row.MH
        rec["source"] = row.source
        rows.append(rec)
        print(f"  [{len(rows)}/{len(cal)}] ID={row.ID} MH={row.MH:.2f} "
              f"-> {rec.get('status')}, cand={rec.get('candidate_metalpoor')}")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "ew_screen_calibration.csv"), index=False)

    ok = res[res["status"] == "ok"]
    rich = ok[ok["MH"] >= -2.0]
    poor = ok[ok["MH"] < -2.0]
    fn = rich["candidate_metalpoor"].sum()
    tp = poor["candidate_metalpoor"].sum()

    n_rich, n_poor = len(rich), len(poor)
    fn_rate = fn / n_rich if n_rich else float("nan")
    tp_rate = tp / n_poor if n_poor else float("nan")
    # Wilson 95% CI для доли
    from math import sqrt
    def wilson(k, n, z=1.96):
        if n == 0:
            return (float("nan"), float("nan"))
        p = k / n
        denom = 1 + z**2/n
        center = p + z**2/(2*n)
        half = z*sqrt(p*(1-p)/n + z**2/(4*n**2))
        return ((center-half)/denom, (center+half)/denom)

    lo, hi = wilson(fn, n_rich)
    print(f"\n=== КАЛИБРОВКА EW-СКРИНА (n={len(ok)} успешно измерено) ===")
    print(f"False-negative rate (metal-rich [M/H]>=-2.0 ошибочно 'чистые'): "
          f"{fn}/{n_rich} = {fn_rate*100:.1f}% [95% CI {lo*100:.1f}-{hi*100:.1f}%]")
    print(f"True-positive rate (metal-poor [M/H]<-2.0 правильно пойманы):  "
          f"{tp}/{n_poor} = {tp_rate*100:.1f}%")
    print(f"\n-> {os.path.join(OUT, 'ew_screen_calibration.csv')}")
    return fn_rate, (lo, hi), tp_rate

if __name__ == "__main__":
    main()
