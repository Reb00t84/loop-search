# loop_search — searching open archives for causal-loop candidates

Pipeline for the phenomenology of preprint DOI 10.5281/zenodo.21149560
("self-consistent causal loops from black-hole bounces"). Channel priority
after the pilot analysis (July 2026):

1. **Clouds (main channel, self-sufficient).** Metal-free gas with suppressed
   D/H is the one combination stellar processing cannot produce (processing
   burns D but must also produce metals). Quasar spectral archives sit right
   at the needed z ~ 2–4.5.
2. **Afterglow-less GRBs × cloud positions.** Positional match accounting for
   localization errors; GRBs have no z, so the match is positional only, plus
   a background estimate.
3. **GW channel — NOT GWTC.** The loop-collapse signal is an unmodeled burst,
   not a compact-binary merger; the right archive is cWB/burst searches. A
   null result at z < 1 is predicted by the model itself (loops are locally
   suppressed by the falling density); the channel opens up with ET/CE.
   Script 03 includes a stub for future skymap matching.

## Dependencies
```
pip install numpy pandas astropy requests healpy scipy matplotlib astroquery sparclclient pykoa pyvo
```
(healpy is only needed for the GW-skymap stub; scipy/matplotlib/astroquery
are needed for the Stage-2 threshold calibration, spectrum plots, and the
Rafelski+2012 VizieR fetch, respectively; sparclclient is needed for DESI
per-object spectrum retrieval (`desi_04`, `hires_02`'s DESI branch); pykoa
for the Keck/KOA archive query + download (`hires_01`, `hires_02`); pyvo for
the ESO TAP archive query (`hires_01`) — everything else works without them)

## Run order
```
python 01_fetch_catalogs.py      # downloads catalogs into data/
python 02_filter_clouds.py       # cloud-candidate selection + mixing line + DLA shortlist
python 03_crossmatch.py          # positional match: clouds × GRBs
python 04_background_mc.py       # Monte Carlo background (RA scrambling) and match p-value
python 05_ew_screen.py           # Stage 2: matched-filter EW screen on SDSS spectra (~85 min, background)
python 06_merge_candidates.py    # merges Stage 1+2 -> final candidate list
python 07_calibrate_screen.py    # calibrates Stage-2 false-negative/true-positive rate
python 08_inspect_top.py         # PNG panels for eyeballing the top candidates
python 09_reproduce_boxcar_recheck.py  # regenerates the retracted box-car screen's miss rate (~40 min, background)

# DESI DR1 extension (v3) — ports the same method to a second survey
python desi_01_fetch_dla.py            # downloads the DESI DR1 DLA Toolkit catalog
python desi_02_dv_systematic.py        # zDLA vs metal-line velocity offset, DESI x Rafelski overlap
python desi_03_select_targets.py       # NHI + SNR_FOREST shortlist -> 4708 targets
python desi_04_ew_screen.py            # Stage 2: matched-filter EW screen via SPARCL spectra
python desi_05_calibrate.py            # false-negative calibration on the DESI x Rafelski overlap
python desi_06_merge.py                # merges SDSS 39 + DESI 38 -> 77 ranked candidates
python desi_07_cross_check.py          # cross-survey re-measurement where shortlists overlap
python desi_08_exclude_contradicted.py # drops the 2 cross-survey-contradicted targets -> N=75
python desi_09_inspect.py              # PNG panels + single-pixel recheck for the 38 DESI candidates

# Archival high-resolution follow-up (v4) — checks the 75 against Keck/ESO archives
python hires_01_archival_coverage.py   # 5" cone search: KOA (HIRES/ESI) + ESO (UVES/ESPRESSO/XSHOOTER)
python hires_02_inventory.py           # downloads + inventories usable products per target/line
python hires_03_measure_purity.py      # matched-filter measurement on the real high-res spectra
python hires_04_rescan_window.py       # targeted rescan for the one edge-of-window detection
python hires_05_cii_diagnostics.py     # resolves the CII1334 ambiguity left by hires_04
```

## Status (2026-07-11)
Published through v4 (preprint DOI 10.5281/zenodo.21309962, concept DOI
10.5281/zenodo.21149560 for all versions). `out/merged_candidates_clean.csv`
holds the current ranked target list, **N=75** (10 Stage 1 known-metallicity
+ 27 SDSS + 38 DESI matched-filter non-detections, after excluding 2
cross-survey-contradicted targets — see the DESI section below). Of those
75, a 5″ archival cone search finds 15 with public Keck/ESO spectra already
on hand, 10 of which are usable for measurement; **all 4 usable Stage-2
("clean") targets measured so far come back contaminated on the
higher-resolution data — 0/4 confirmed clean** (see the v4 section below).
`out/final_candidates.csv` (N=39, SDSS-only, published with v2) is left
untouched as a separately-validated artifact, per project precedent — not
silently revised in place.

## Manual checks
- The metallicity selection cut ([O/H] < -1.5 in flag_anomalies) is a
  parameter — tune it deliberately.
- out/merged_candidates_clean.csv (N=75) and out/final_candidates.csv
  (N=39) are shortlists of follow-up spectroscopy TARGETS, not confirmed
  loop candidates; only [O/H]/D/H measured on new high-resolution data can
  give the final label.
- The v4 archival check (n=4) found 100% of measurable Stage-2 "clean"
  survey non-detections were actually contaminated at echelle resolution —
  small-n, not a blanket dismissal of the remaining 71, but a strong prior
  that low-resolution non-detections need direct confirmation before being
  treated as clean.
- A strict Monte Carlo background built from the GBM exposure map (instead
  of the RA-scrambling in 04) was deliberately not done — see CLAUDE.md.

## Outputs
- `data/*.csv|dat.gz` — raw catalogs
- `data/dla_metallicity_compilation.csv` — Rafelski+2012, 242 systems with [M/H]
- `out/cloud_candidates.csv` — precision D/H clouds with anomaly flags
- `out/dla_targets.csv` — shortlist of 7996 DR16 DLA sightlines (SDSS)
- `out/dla_known_metallicity_matches.csv` — Stage 1: Rafelski cross-match (SDSS)
- `out/zcnn_vs_metal_velocity_offset.csv` — zCNN vs. z_metal systematic, SDSS (59 systems)
- `out/dla_ew_screening.csv` — Stage 2 matched-filter: EW/significance for every SDSS target
- `out/ew_screen_calibration.csv` — matched-filter calibration on 59 known-metallicity systems (SDSS)
- `out/final_candidates.csv` — SDSS-only result (v2): N=39, sorted by quasar flux
- `out/inspect/*.png` — spectrum panels for manual review of SDSS candidates
- `out/boxcar_recheck.csv` — regenerated miss-rate check on the retracted box-car screen (32/56 = 57%)
- `out/grb_matches.csv` — positional matches with GRBs
- `out/background.txt` — background expectation and p-value
- `out/desi_targets.csv` — DESI DR1 shortlist of 4708 sightlines (NHI + SNR_FOREST cut)
- `out/desi_zdla_vs_metal_velocity_offset.csv` — zDLA vs. z_metal systematic, DESI (34 systems)
- `out/desi_ew_screening.csv` — DESI Stage 2 matched-filter raw results (66 candidates pre-quality-filter)
- `out/desi_ew_screen_calibration.csv` — matched-filter calibration on the DESI x Rafelski overlap (27 systems)
- `out/merged_candidates_sdss_desi.csv` — SDSS 39 + DESI 38 merged, 77 rows, before cross-survey exclusion
- `out/cross_survey_corroboration.csv` — independent re-measurement of the 5 SDSS/DESI shortlist-overlap targets
- `out/inspect_desi/*.png` — spectrum panels for manual review of DESI candidates
- `out/desi_candidates_recheck.csv` — objective single-pixel recheck for the 38 DESI candidates
- `out/merged_candidates_clean.csv` — RESULT (v3/v4): N=75, ranked, cross-survey-cleaned
- `out/archival_coverage.csv` — v4 Stage 1: 5″ cone-search hits against KOA + ESO (26 rows)
- `out/highres_inventory.csv` — v4 Stage 2a: per-line usability inventory of downloaded archival spectra (141 rows)
- `out/highres_purity.csv` — v4 Stage 2b: matched-filter measurement on real high-res spectra, 40 rows (10 targets x 4 lines), updated in place by `hires_04`/`hires_05` without discarding history
- `out/dark_sector_v5_estimates.md` — order-of-magnitude dark-matter-contamination estimates for the v5 "loop composition" draft section (not yet in any published preprint version)

## DESI extension (v3, published 2026-07-09)
`desi_01`–`desi_09` port the same matched-filter method to DESI DR1
(Brodzeller et al. 2025 DLA catalog, spectra via the SPARCL API — DESI
stores spectra as ~219MB/file HEALPix coadds, not per-object files, so
this needed its own fetcher). 38 DESI candidates merged with the SDSS 39;
a cross-survey check (does the *other* instrument's spectrum, where one
exists at the same position+z, independently corroborate a "clean" call)
caught 2 confirmed contaminants — one with a 9-12σ line DESI sees that
SDSS's own ±500 km/s window missed entirely due to a ~415 km/s SDSS-DESI
z_DLA disagreement for that system, the same failure class that retired
the box-car method, this time between instruments rather than between
z_DLA and the metal-line velocity. Both excluded (`out/final_candidates.csv`
itself is left untouched — this is a merge-stage finding, not grounds to
silently rewrite an already-referenced artifact). The 38 DESI candidates
also went through the same PNG-panel + objective single-pixel due
diligence the SDSS 39 got: 0/38 show a >4σ single-pixel line the
matched-filter missed (SDSS's retracted box-car method was 57%, see
`out/boxcar_recheck.csv`). Result: **N=75** (10 Stage 1 + 27 SDSS/Stage 2
+ 38 DESI/Stage 2) in `out/merged_candidates_clean.csv`, which now also
carries `provenance` and `brightness_percentile`/`top20_feasibility`
columns so it stands as the machine-readable source for the v3 target
table on its own, without a join back to `final_candidates.csv`. Published
as preprint v3, DOI 10.5281/zenodo.21274668. See CLAUDE.md for the full
Stage 1-3.5 writeup, including three SPARCL client bugs hit and worked
around along the way.

## Archival high-resolution follow-up (v4, published 2026-07-11)
`hires_01`–`hires_05` check whether any of the 75 candidates in
`out/merged_candidates_clean.csv` already have public high-/medium-resolution
spectra sitting in the Keck (KOA) or ESO archives — a chance to falsify
non-detections without a telescope proposal. Stage 1 (`hires_01`, 5″ cone
search against KOA HIRES/ESI and ESO UVES/ESPRESSO/XSHOOTER): 15/75 targets
(11 SDSS + 4 DESI) have at least one archival hit, 10 of them high-resolution,
all with a public exposure. Stage 2a (`hires_02`, download + inventory):
10/15 are usable (at least one of the 4 metal lines actually measurable on
the reduced spectrum) — the other 5 are blocked by concrete, named causes
(ESI has no reduced level-1 product at KOA; some HIRES exposures ship in an
older, unsupported MAKEE format; one exposure's wavelength solution failed
outright — see CLAUDE.md for all three). Stage 2b (`hires_03`, matched-filter
measurement reusing `05_ew_screen`'s `find_peak`/`classify` unchanged, for
direct comparability with the SDSS/DESI numbers): 40 line measurements
(10 targets × 4 lines) — 29 detected, 9 upper limits, 2 unavailable.
**All 4 of the usable Stage-2 ("clean") targets come back contaminated on
the higher-resolution spectrum — 0/4 confirmed clean.** An edge-of-window
rescan and a CII1334-blend diagnostic (`hires_04`, `hires_05`) resolved the
one ambiguous case (target 39627762889133674): the apparent 4-line
coincidence sitting at the edge of the ±500 km/s scan window turns out to be
a systematic z_abs offset (3 of the 4 lines cleanly resolve together at
dv≈+650 km/s once the window is widened to ±1500 km/s; the fourth, CII1334,
was a blend with a second, independently-catalogued DLA on the same
sightline — the real line agrees with the other three once measured
locally, at dv≈+650 km/s too). Conclusion: n=4 is too small to generalize,
but it is consistent with what the project already knew about its Stage-2
screen's blind spots at SDSS/DESI resolution — the archival channel is now
exhausted, and confirming any of the remaining 71 candidates needs new
telescope time, not more archive mining. Published as preprint v4, DOI
10.5281/zenodo.21309962 (concept DOI 10.5281/zenodo.21149560 unchanged).
See CLAUDE.md, "v4-триггер", for the full five-stage writeup, including the
KOA/ESO engineering pitfalls (proxy retries, a silent-failure PyKOA download
path that returns without an exception or an output file, a 13x download
speedup after profiling, and resume/marker-file protections that survived
two mid-run environment kills).

## Project notes
`CLAUDE.md` is the project's lab notebook (in Russian), documenting the
hypothesis, pilot-analysis pitfalls, the three iterations of the Stage-2
method (with the reasoning for retracting the first two), the named
"right window, wrong center" failure class, and the project rules
distilled from concrete mistakes caught along the way. `tools/build/`
holds the preprint PDF build recipe (pandoc+xelatex, with built-in QC);
`tools/dark_sector_v5_calc.py` reproduces every number in
`out/dark_sector_v5_estimates.md`.
