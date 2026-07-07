#!/usr/bin/env python3
"""Downloads open catalogs: GWTC (reference only), Fermi GBM, Swift BAT, SDSS
DLA, Rafelski+2012 DLA metallicities. All files go to data/. Re-running does
not re-download files that already exist.

Скачивание открытых каталогов: GWTC (справочно), Fermi GBM, Swift BAT, SDSS DLA,
металличности Rafelski+2012. Все файлы кладутся в data/. Повторный запуск не
перекачивает существующее."""
import os, sys, gzip, shutil
import requests

DATA = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA, exist_ok=True)

SOURCES = {
    # Кумулятивный GWTC (CBC-слияния; для петель справочно, см. README п.3)
    "gwtc_events.csv":
        "https://gwosc.org/api/v2/catalogs/GWTC/events?include-default-parameters=true&format=csv",
    # Fermi GBM burst catalog, TDAT-дамп HEASARC  # VERIFY путь
    "fermigbrst.tdat.gz":
        "https://heasarc.gsfc.nasa.gov/FTP/heasarc/dbase/tdat_files/heasarc_fermigbrst.tdat.gz",
    # Swift BAT GRB catalog (сводная таблица)  # VERIFY путь
    "swift_bat_grb.txt":
        "https://swift.gsfc.nasa.gov/results/batgrbcat/summary_cflux/summary_general_info/summary_general.txt",
    # SDSS DR16 DLA (Chabanier+2022). Проверено 2026-07-07: каталог лежит
    # на CDS/VizieR как J/ApJS/258/18, а НЕ на Zenodo (ошибочная посылка
    # в исходном TODO). Формат — CDS ASCII (.dat.gz) с описанием колонок
    # в ReadMe, не FITS; парсинг в 02 (load_sdss_dla) нужно переписать под
    # astropy.io.ascii.read(format="cds", readme=...), см. TODO #4.
    # Имя файла после распаковки (dr16dla.dat) должно совпадать с тем, что
    # указано в File Summary самого ReadMe — иначе astropy.io.ascii(format="cds")
    # не находит таблицу по имени.
    "dr16dla.dat.gz":
        "https://cdsarc.cds.unistra.fr/ftp/J/ApJS/258/18/dr16dla.dat.gz",
    "sdss_dr16_dla_ReadMe":
        "https://cdsarc.cds.unistra.fr/ftp/J/ApJS/258/18/ReadMe",
}

def fetch(name, url, tries=3, timeout=120):
    """Downloads url to data/name with retries; skips if already present."""
    dst = os.path.join(DATA, name)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"[skip] {name} уже есть"); return dst
    for i in range(tries):
        try:
            print(f"[get ] {name} <- {url}")
            r = requests.get(url, timeout=timeout, stream=True)
            r.raise_for_status()
            with open(dst, "wb") as f:
                for chunk in r.iter_content(1 << 16):
                    f.write(chunk)
            print(f"[ ok ] {name}: {os.path.getsize(dst)/1e6:.1f} MB")
            return dst
        except Exception as e:
            print(f"[warn] попытка {i+1}/{tries} не удалась: {e}")
    print(f"[FAIL] {name} — проверь URL (метка VERIFY в скрипте)")
    return None

def fetch_rafelski_metallicity():
    """Builds data/dla_metallicity_compilation.csv (242 DLA metallicities,
    Rafelski et al. 2012, ApJ 755, 89, VizieR J/ApJ/755/89) — 47 new
    measurements (table2) + 195 literature compilation (table3), combined
    with qso/ra/dec/z_abs/[M/H]. Used by 02/06 as the Stage-1 known-
    metallicity cross-match (see CLAUDE.md)."""
    dst = os.path.join(DATA, "dla_metallicity_compilation.csv")
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"[skip] {os.path.basename(dst)} уже есть"); return dst
    import pandas as pd
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astroquery.vizier import Vizier

    print("[get ] Rafelski+2012 (J/ApJ/755/89) <- VizieR")
    v = Vizier(columns=["**"]); v.ROW_LIMIT = -1
    cats = v.get_catalogs("J/ApJ/755/89")
    t1 = cats["J/ApJ/755/89/table1"].to_pandas()  # QSO -> RAJ2000/DEJ2000 (sexagesimal)
    t2 = cats["J/ApJ/755/89/table2"].to_pandas()  # 47 new measurements
    t3 = cats["J/ApJ/755/89/table3"].to_pandas()  # 195 literature compilation

    t1_coords = t1.drop_duplicates(subset="QSO")[["QSO", "RAJ2000", "DEJ2000"]]
    t2m = t2.merge(t1_coords, on="QSO", how="left")
    c = SkyCoord(t2m["RAJ2000"], t2m["DEJ2000"], unit=(u.hourangle, u.deg))
    t2m["ra"], t2m["dec"] = c.ra.deg, c.dec.deg

    m2 = pd.DataFrame({"qso": t2m["QSO"], "ra": t2m["ra"], "dec": t2m["dec"],
                        "z_abs": t2m["z"], "MH": t2m["[M/H]"],
                        "e_MH": t2m["e_[M/H]"], "source": "Rafelski2012_new"})
    m3 = pd.DataFrame({"qso": t3["QSO"], "ra": t3["_RA"], "dec": t3["_DE"],
                        "z_abs": t3["z"], "MH": t3["[M/H]"],
                        "e_MH": t3["e_[M/H]"], "source": "Rafelski2012_lit"})
    master = pd.concat([m2, m3], ignore_index=True).dropna(subset=["ra", "dec", "z_abs"])
    master.to_csv(dst, index=False)
    print(f"[ ok ] {os.path.basename(dst)}: {len(master)} систем с [M/H]")
    return dst

if __name__ == "__main__":
    got = {n: fetch(n, u) for n, u in SOURCES.items()}
    # распаковка .gz-архивов для удобства (имя без .gz)
    for name, gz in got.items():
        if gz and name.endswith(".gz"):
            dst = gz[:-3]
            if not os.path.exists(dst):
                with gzip.open(gz, "rb") as fi, open(dst, "wb") as fo:
                    shutil.copyfileobj(fi, fo)
                print(f"[ ok ] распакован {os.path.basename(dst)}")
    fetch_rafelski_metallicity()
    missing = [n for n, p in got.items() if p is None]
    sys.exit(1 if missing else 0)
