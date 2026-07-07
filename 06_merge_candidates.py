#!/usr/bin/env python3
"""Merges the two-stage DR16 DLA narrowing (Stage 1: known metallicity
cross-match; Stage 2: matched-filter EW screen) into out/final_candidates.csv,
sorted by quasar flux (follow-up feasibility). See CLAUDE.md for the full
method history (two earlier, retracted versions of the Stage-2 screen).

Финальное слияние двухступенчатого сужения DR16 DLA, ВЕРСИЯ 3
(2026-07-07, после отзыва box-car и одно-пиксельной версий — см.
CLAUDE.md). Stage 1 (out/dla_known_metallicity_matches.csv, кросс-матч с
Rafelski+2012) даёт системы с уже измеренной металличностью — метал-
бедные среди них (MH<-2.0) уже готовые кандидаты, без EW-скрининга.
Stage 2 (out/dla_ew_screening.csv) — matched-filter EW-скрининг
остальных сайтлайнов (05_ew_screen.py: скан узкого окна +-100 км/с по
+-500 км/с, SOLO 3.36sigma Sidak ИЛИ PAIR 2.0sigma x2 линии согласованно
по скорости). ДОПОЛНИТЕЛЬНЫЙ фильтр качества (решение автора): из
кандидатов Stage 2 берём только те, у кого 3sigma верхний предел по
самой слабо ограниченной линии <0.2Å (rest-frame) — сопоставимо с
медианной информативностью калибровочной выборки Rafelski (0.18Å), где
false-negative rate измерен как 0%. Без этого фильтра медианный SNR
кандидатов Stage 2 — 3.78 (просто "мало данных", а не "чисто").
Итог — один отсортированный список (по SNR/потоку квазара, фактор
осуществимости follow-up спектроскопии высокого разрешения)."""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "out")
UL_QUALITY_CUT = 0.2  # Å rest-frame, 3sigma UL порог для Stage 2

def main():
    """Builds out/final_candidates.csv from Stage 1 + Stage 2, deduped and filtered."""
    targets = pd.read_csv(os.path.join(OUT, "dla_targets.csv")).reset_index().rename(
        columns={"index": "target_idx"})
    known = pd.read_csv(os.path.join(OUT, "dla_known_metallicity_matches.csv"))
    ew = pd.read_csv(os.path.join(OUT, "dla_ew_screening.csv"))

    # ВАЖНО: один QSO (одинаковый ID) может иметь НЕСКОЛЬКО DLA-абсорберов на
    # разных z_abs — join по одному ID размножит строки на все его абсорберы.
    # Матчим по паре (ID, zCNN), которая однозначно указывает на конкретный
    # абсорбер (или по target_idx для Stage 1, где он уже сохранён).

    # Stage 1: уже измеренная металличность, метал-бедные (MH<-2.0) -> кандидаты
    known_poor = known[known["MH"] < -2.0].drop_duplicates("target_idx")
    stage1 = targets[targets["target_idx"].isin(known_poor["target_idx"])].copy()
    stage1 = stage1.merge(known_poor[["target_idx", "MH", "source"]],
                           on="target_idx", how="left")
    stage1["stage"] = "known_metallicity"
    stage1["note"] = stage1.apply(
        lambda r: f"[M/H]={r['MH']:.2f} ({r['source']})", axis=1)  # source is already English

    # Stage 2: matched-filter EW-скрининг, кандидаты без детекции линий
    # И с информативным пределом (см. UL_QUALITY_CUT выше)
    ew_ok = ew[ew["status"] == "ok"]
    print(f"EW-скрининг (v3, matched-filter): {len(ew)} записей, "
          f"{len(ew_ok)} успешно измерено ({len(ew)-len(ew_ok)} fetch/read failed)")
    all_cand = ew_ok[ew_ok["candidate_metalpoor"] == True]
    stage2_cand = all_cand[all_cand["max_3sig_UL"] < UL_QUALITY_CUT]
    print(f"Кандидатов до фильтра качества: {len(all_cand)}, "
          f"после (UL<{UL_QUALITY_CUT}A): {len(stage2_cand)}")
    stage2 = targets.merge(
        stage2_cand[["ID", "zCNN", "n_lines_ok", "max_3sig_UL"]],
        on=["ID", "zCNN"], how="inner")
    stage2["stage"] = "ew_screen"
    stage2["note"] = stage2.apply(
        lambda r: f"0/{int(r['n_lines_ok'])} lines detected (matched-filter), "
                  f"3sigma UL<{r['max_3sig_UL']*1000:.0f} mA", axis=1)

    cols = ["ID", "ra", "dec", "zCNN", "NHI_best", "Flux", "SNR", "Plate",
            "MJD", "Fiber", "stage", "note"]
    final = pd.concat([stage1.reindex(columns=cols),
                        stage2.reindex(columns=cols)], ignore_index=True)

    # Исключаем сайтлайны, совпадающие (<3") с квазарами из собственной
    # PRECISION_DH: эти системы уже полностью охарактеризованы Cooke et al.
    # на Keck (D/H, [O/H] измерены напрямую) — если EW-скрин всё равно
    # пометил такой как кандидата, это подтверждённый ложноположительный
    # (низкое SNR/разрешение SDSS не видит линии, которые Keck видит
    # уверенно). Проверено 2026-07-07: 1 из 7 precision-квазаров вообще
    # попадает в шорт-лист DR16 DLA (SDSSJ1419+0829, z=3.049) — и это
    # именно тот единственный случай, который EW-скрин ложно пометил.
    prec = pd.read_csv(os.path.join(OUT, "cloud_candidates.csv"))
    def angsep_arcsec(ra1, dec1, ra2, dec2):
        r1, d1, r2, d2 = map(np.radians, (ra1, dec1, ra2, dec2))
        c = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(r1 - r2)
        return np.degrees(np.arccos(np.clip(c, -1, 1))) * 3600
    is_known_prec = np.array([
        (angsep_arcsec(row.ra, row.dec, prec["ra"].values, prec["dec"].values) < 3.0).any()
        for row in final.itertuples()
    ])
    n_contam = int(is_known_prec.sum())
    final = final[~is_known_prec].copy()

    final = final.sort_values(["Flux", "SNR"], ascending=False).reset_index(drop=True)
    final.to_csv(os.path.join(OUT, "final_candidates.csv"), index=False)

    print(f"\nStage 1 (известная металличность, MH<-2.0): {len(stage1)}")
    print(f"Stage 2 (EW-скрининг, нет детекции линий):    {len(stage2)}")
    print(f"Исключено как известный ложноположительный (совпадение с "
          f"PRECISION_DH, см. комментарий в коде): {n_contam}")
    print(f"ИТОГО кандидатов: {len(final)} (из {len(targets)} исходных целей)")
    print(f"-> {os.path.join(OUT, 'final_candidates.csv')}")
    print("\nКаждый кандидат — только цель для follow-up спектроскопии; "
          "EW-скрин на SDSS-спектрах имеет подтверждённый ложноположительный "
          "риск (см. код выше), финальную метку даёт только измерение "
          "[O/H]/D/H на новых данных.")
    if len(final):
        print("\nТоп-10 по потоку квазара (самые лёгкие для follow-up):")
        print(final.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
