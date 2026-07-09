#!/usr/bin/env python3
"""v3-trigger Stage 3 final step: merges the DESI DR1 screen results with
the existing SDSS 39-target list into one ranked list, deduping by
position+z_abs (the exact same duplicate-key precedent as the SDSS
Stage1+Stage2 92-vs-66 merge, CLAUDE.md project rule 1) and flagging
cross-survey confirmations - a target independently flagged clean by two
different instruments/pipelines is stronger evidence than either alone
(see CLAUDE.md v3-trigger section, the cross-survey ranking idea)."""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "out")
UL_QUALITY_CUT = 0.2  # same convention as SDSS (06_merge_candidates.py)
TOL_POS_ARCSEC = 3.0
TOL_Z = 0.02

def angsep_arcsec(ra1, dec1, ra2, dec2):
    """Angular separation in arcsec (vectorized)."""
    r1, d1, r2, d2 = map(np.radians, (ra1, dec1, ra2, dec2))
    c = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(r1 - r2)
    return np.degrees(np.arccos(np.clip(c, -1, 1))) * 3600

def build_desi_candidates():
    """Loads the DESI screen output, dedupes, and applies the UL quality cut."""
    d = pd.read_csv(os.path.join(OUT, "desi_ew_screening.csv"))
    d = d.drop_duplicates(subset=["ID", "zDLA"])
    cand = d[(d["candidate_metalpoor"] == True) & (d["max_3sig_UL"] < UL_QUALITY_CUT)].copy()
    cand["survey"] = "DESI"
    cand["z_abs"] = cand["zDLA"]
    cand["brightness"] = cand["SNR_FOREST"]  # DESI's own units, not SDSS-comparable
    # все текущие DESI-кандидаты — Stage 2 (EW-screen); DESI-эквивалента
    # Stage 1 (известная металличность) пока не строился, см. TODO
    cand["provenance"] = "DESI/Stage2"
    cand["note"] = cand.apply(
        lambda r: f"0/4 lines detected (matched-filter), 3sigma UL<{r['max_3sig_UL']*1000:.0f} mA",
        axis=1)
    return cand[["ID", "ra", "dec", "z_abs", "NHI", "SNR_FOREST", "brightness",
                 "survey", "provenance", "note"]]

def build_sdss_candidates():
    """Loads the existing SDSS final_candidates.csv in the merge's common schema."""
    s = pd.read_csv(os.path.join(OUT, "final_candidates.csv"))
    s["survey"] = "SDSS"
    s["z_abs"] = s["zCNN"]
    s["brightness"] = s["Flux"]  # SDSS's own units, not DESI-comparable
    # stage приходит из final_candidates.csv: "known_metallicity" (Stage 1,
    # Rafelski [M/H]<-2) или "ew_screen" (Stage 2, matched-filter).
    s["provenance"] = "SDSS/" + s["stage"].map(
        {"known_metallicity": "Stage1", "ew_screen": "Stage2"})
    return s[["ID", "ra", "dec", "z_abs", "NHI_best", "SNR", "brightness",
              "survey", "provenance", "note"]].rename(
        columns={"NHI_best": "NHI", "SNR": "SNR_native"})

def main():
    """Cross-matches DESI x SDSS candidates, dedupes, flags cross-survey
    confirmations, and writes the merged ranked list."""
    desi = build_desi_candidates().rename(columns={"SNR_FOREST": "SNR_native"})
    sdss = build_sdss_candidates()

    # ВАЖНО: DESI TARGETID — 17-значный int64 (до ~4e16), SDSS ID —
    # float64 из каталога Chabanier. Смешивать их в одной числовой
    # колонке через pd.concat нельзя: pandas апкастит всю колонку до
    # float64, а float64 точно хранит целые только до 2^53 (~9e15) —
    # 87.6% DESI TARGETID (86529/98766 в полном каталоге, 34/38 среди
    # текущих кандидатов) при этом теряют точность молча. Приводим ID
    # к строке ДО конкатенации, а не после — иначе тот же баг просто
    # переедет на репрезентацию.
    desi["ID"] = desi["ID"].astype("int64").astype(str)
    sdss["ID"] = sdss["ID"].astype("int64").astype(str)
    print(f"DESI candidates (post-UL filter): {len(desi)}")
    print(f"SDSS candidates (existing):        {len(sdss)}")

    # Позиция+z_abs дедуп/кросс-матч между обзорами (тот же допуск,
    # что использовался во всём проекте: 3", 0.02 по z).
    cross = []
    for i, d in desi.iterrows():
        sep = angsep_arcsec(d.ra, d.dec, sdss["ra"].values, sdss["dec"].values)
        dz = np.abs(sdss["z_abs"].values - d.z_abs)
        hit = np.where((sep < TOL_POS_ARCSEC) & (dz < TOL_Z))[0]
        if len(hit):
            cross.append((d["ID"], sdss.iloc[hit[0]]["ID"], round(sep[hit[0]], 2)))
    print(f"\nКросс-обзорных совпадений (DESI x SDSS, одна и та же система "
          f"независимо прошла скрин на обоих): {len(cross)}")
    for c in cross:
        print(f"  DESI {c[0]} <-> SDSS {c[1]}  sep={c[2]}\"")

    cross_desi_ids = {c[0] for c in cross}
    cross_sdss_ids = {c[1] for c in cross}
    desi["cross_survey_confirmed"] = desi["ID"].isin(cross_desi_ids)
    sdss["cross_survey_confirmed"] = sdss["ID"].isin(cross_sdss_ids)

    # Дедуп: если система есть в обоих, оставляем ОДНУ строку (SDSS —
    # первичный список, дольше валидирован: калибровка n=59 против n=27,
    # плюс ручной осмотр топ-15, которого DESI ещё не проходил), помечаем
    # cross_survey_confirmed=True и добавляем survey="SDSS+DESI".
    dup_desi_mask = desi["ID"].isin(cross_desi_ids)
    desi_unique = desi[~dup_desi_mask].copy()
    sdss.loc[sdss["ID"].isin(cross_sdss_ids), "survey"] = "SDSS+DESI"

    # brightness_percentile: SDSS Flux и DESI SNR_FOREST — единицы разных
    # обзоров, напрямую не сравнимы (см. коммент в build_*_candidates).
    # Ранжируем ВНУТРИ каждого обзора отдельно (pct=True даёт 0-1 ранг
    # внутри своей группы), затем берём топ-20 по этому общему для обоих
    # обзоров перцентилю — честно и без псевдоточности межобзорной шкалы.
    # ранг внутри sdss/desi_unique целиком (не по колонке "survey" - та
    # уже перезаписана в "SDSS+DESI" для части строк на предыдущем шаге,
    # группировка по ней расколола бы SDSS на две подгруппы неверно)
    sdss["brightness_percentile"] = (sdss["brightness"].rank(pct=True) * 100).round(1)
    desi_unique["brightness_percentile"] = (desi_unique["brightness"].rank(pct=True) * 100).round(1)

    cols = ["ID", "ra", "dec", "z_abs", "NHI", "SNR_native", "brightness",
            "brightness_percentile", "survey", "provenance",
            "cross_survey_confirmed", "note"]
    final = pd.concat([sdss[cols], desi_unique[cols]], ignore_index=True)
    final["top20_feasibility"] = final.index.isin(
        final["brightness_percentile"].nlargest(20).index)

    # dup-check явный (правило 1): по позиции+z внутри итогового списка
    # не должно остаться пар ближе допуска, кроме уже учтённых кросс-пар.
    n_internal_dup = 0
    for i in range(len(final)):
        sep = angsep_arcsec(final.iloc[i].ra, final.iloc[i].dec,
                             final["ra"].values, final["dec"].values)
        dz = np.abs(final["z_abs"].values - final.iloc[i].z_abs)
        n_internal_dup += ((sep < TOL_POS_ARCSEC) & (dz < TOL_Z)).sum() - 1  # minus self
    print(f"\nDup-check остаточных близких пар в объединённом списке "
          f"(должно быть 0): {n_internal_dup}")

    final = final.sort_values(["cross_survey_confirmed", "survey"], ascending=[False, True])
    out_path = os.path.join(OUT, "merged_candidates_sdss_desi.csv")
    final.to_csv(out_path, index=False)

    print(f"\n=== ИТОГ ===")
    print(f"SDSS: {(final['survey']=='SDSS').sum()}, DESI: {(final['survey']=='DESI').sum()}, "
          f"SDSS+DESI (кросс-обзорные): {(final['survey']=='SDSS+DESI').sum()}")
    print(f"Объединённый список: {len(final)} (было бы {len(sdss)+len(desi)} без дедупа "
          f"по {len(cross)} кросс-совпадениям)")
    print(f"-> {out_path}")

    print("\nРаспределение по яркости (внутри каждого обзора, свои единицы):")
    print("SDSS Flux:", sdss["brightness"].describe()[["min","25%","50%","75%","max"]].to_dict())
    print("DESI SNR_FOREST:", desi["brightness"].describe()[["min","25%","50%","75%","max"]].to_dict())

if __name__ == "__main__":
    main()
