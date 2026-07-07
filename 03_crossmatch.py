#!/usr/bin/env python3
"""Positional cross-match: candidate clouds x short GRBs without afterglow.

Позиционный кросс-матч: облака-кандидаты × короткие GRB без послесвечения.
Хронология модели: GRB приходит ПЕРВЫМ, свет облака — в течение ~T_cloud после;
значит архивный GRB на позиции живого абсорбера допустим, обратное — нет.
У GRB нет z, поэтому матч только позиционный; значимость -> 04_background_mc."""
import os, re
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
DATA, OUT = os.path.join(HERE, "data"), os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

def angsep_deg(ra1, dec1, ra2, dec2):
    """Angular separation in degrees (vectorized). Угловое расстояние, градусы (векторизовано)."""
    r1, d1, r2, d2 = map(np.radians, (ra1, dec1, ra2, dec2))
    c = (np.sin(d1)*np.sin(d2) + np.cos(d1)*np.cos(d2)*np.cos(r1 - r2))
    return np.degrees(np.arccos(np.clip(c, -1, 1)))

def load_fermi_tdat(path):
    """Parses a HEASARC TDAT file into name/ra/dec/err_deg/t90.
    Парсер HEASARC TDAT: имя, ra, dec, err_rad(deg), t90."""
    fields, rows, in_data = [], [], False
    with open(path, errors="ignore") as f:
        for line in f:
            if line.startswith("field["):
                # index [1] — имя поля; [0]="field", [2]=тип (char11 и т.п.)
                fields.append(re.split(r"[\[\]=\s]+", line)[1])
            elif line.strip() == "<DATA>":
                in_data = True
            elif line.strip() == "<END>":
                break
            elif in_data:
                # каждая строка данных TDAT кончается лишним "|" -> пустой
                # хвостовой токен; обрезаем по известному числу полей.
                rows.append(line.rstrip("\n").split("|")[:len(fields)])
    df = pd.DataFrame(rows, columns=fields)
    keep = {"name": "name", "ra": "ra", "dec": "dec",
            "error_radius": "err_deg", "t90": "t90"}
    df = df[[c for c in keep if c in df.columns]].rename(columns=keep)
    for c in ("ra", "dec", "err_deg", "t90"):
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["ra", "dec"])

def select_short_grbs(df, t90_max=2.0):
    """Selects short GRBs (t90<=t90_max).
    Короткие GRB. Флаг 'без оптического послесвечения' в GBM-каталоге
    отсутствует — на этапе матча берём все короткие, отсев по послесвечению
    делается по совпавшим кандидатам вручную (их будут единицы)."""
    if "t90" in df:
        df = df[df["t90"] <= t90_max]
    return df.reset_index(drop=True)

def crossmatch(clouds, grbs, extra_deg=0.0):
    """Matches clouds x GRBs where separation < GRB localization error.
    Матч: расстояние < ошибка локализации GRB (+extra на позицию облака)."""
    hits = []
    for _, c in clouds.iterrows():
        d = angsep_deg(c["ra"], c["dec"], grbs["ra"].values, grbs["dec"].values)
        r = grbs["err_deg"].fillna(3.0).values + extra_deg
        for j in np.where(d < r)[0]:
            hits.append({"qso": c.get("qso"), "z_abs": c.get("z_abs"),
                         "grb": grbs.iloc[j]["name"], "sep_deg": round(d[j], 3),
                         "err_deg": r[j], "t90": grbs.iloc[j].get("t90")})
    return pd.DataFrame(hits)

# --- Заготовка ET/CE-эры: containment позиции облака в GW skymap ---
def in_credible_region(skymap_fits, ra, dec, cl=0.9):
    """Checks whether (ra,dec) falls in a GW skymap's credible region.
    Проверка попадания (ra,dec) в credible region карты локализации GW.
    Требует healpy; для текущих детекторов канал пуст по z (см. README)."""
    import healpy as hp
    m = hp.read_map(skymap_fits)
    npix = len(m); nside = hp.npix2nside(npix)
    order = np.argsort(m)[::-1]
    csum = np.cumsum(m[order]); csum /= csum[-1]
    region = np.zeros(npix, bool); region[order[csum <= cl]] = True
    pix = hp.ang2pix(nside, np.radians(90 - dec), np.radians(ra))
    return bool(region[pix])

if __name__ == "__main__":
    clouds = pd.read_csv(os.path.join(OUT, "cloud_candidates.csv"))
    # координаты квазаров нужно добавить в cloud_candidates (ra, dec) —
    # для precision-выборки это 7 объектов, берутся из SIMBAD руками  # TODO
    if not {"ra", "dec"}.issubset(clouds.columns):
        raise SystemExit("Добавь колонки ra, dec (J2000, градусы) в out/cloud_candidates.csv")
    grbs = select_short_grbs(load_fermi_tdat(os.path.join(DATA, "fermigbrst.tdat")))
    print(f"облаков: {len(clouds)}, коротких GRB: {len(grbs)}")
    hits = crossmatch(clouds[clouds["candidate"] == True], grbs)
    hits.to_csv(os.path.join(OUT, "grb_matches.csv"), index=False)
    print(hits.to_string(index=False) if len(hits) else "совпадений нет")
