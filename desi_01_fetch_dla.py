#!/usr/bin/env python3
"""Downloads the DESI DR1 DLA Toolkit catalog (Brodzeller et al. 2025,
arXiv:2503.14740, Phys. Rev. D 112, 083510) into data/. URL verified by
direct fetch 2026-07-09 (v3-trigger Stage 1 recon, see CLAUDE.md): the
paper's title says "DR2 Ly-alpha BAO" but its Data Availability section
and the file path/name both confirm this is the DR1 catalog (98766 DLA
candidates), used as an input to the DR2 BAO analysis rather than being
DR2 data itself.

Скачивание каталога DESI DR1 DLA Toolkit — ссылка проверена прямым
фетчем 2026-07-09 (разведка этапа 1 v3-триггера)."""
import os
import requests

DATA = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA, exist_ok=True)

URL = ("https://data.desi.lbl.gov/public/dr1/vac/dr1/dla-toolkit/v2.0/"
       "dlacat-dlatoolkit-dr1-main-dark-v2.0.fits")
DST = os.path.join(DATA, "desi_dr1_dlacat_v2.0.fits")

def fetch(tries=3, timeout=120):
    """Downloads the DESI DR1 DLA catalog to data/, skipping if present."""
    if os.path.exists(DST) and os.path.getsize(DST) > 0:
        print(f"[skip] {os.path.basename(DST)} уже есть")
        return DST
    for i in range(tries):
        try:
            print(f"[get ] {os.path.basename(DST)} <- {URL}")
            r = requests.get(URL, timeout=timeout, stream=True,
                              headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            with open(DST, "wb") as f:
                for chunk in r.iter_content(1 << 16):
                    f.write(chunk)
            print(f"[ ok ] {os.path.basename(DST)}: {os.path.getsize(DST)/1e6:.1f} MB")
            return DST
        except Exception as e:
            print(f"[warn] попытка {i+1}/{tries} не удалась: {e}")
    print(f"[FAIL] {os.path.basename(DST)}")
    return None

if __name__ == "__main__":
    fetch()
