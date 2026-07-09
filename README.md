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
pip install numpy pandas astropy requests healpy scipy matplotlib astroquery
```
(healpy is only needed for the GW-skymap stub; scipy/matplotlib/astroquery
are needed for the Stage-2 threshold calibration, spectrum plots, and the
Rafelski+2012 VizieR fetch, respectively — everything else works without them)

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
```

## Status (2026-07-08)
Pipeline 01→08 runs cleanly end to end. **out/final_candidates.csv holds N=39**
targets, after three iterations of the Stage-2 method (box-car sum ->
single pixel -> matched-filter scan; the first two were retracted — see
CLAUDE.md for the full story) plus a quality filter on limit informativeness
(3σ upper limit < 0.2 Å, comparable to the calibration sample). 10 targets
come from known metallicity (Rafelski+2012, [M/H] < -2.0), 29 from the
matched-filter screen of SDSS spectra. Calibration on 59 known-metallicity
systems: 0% false-negative rate. Full method history, both retracted
versions, and why an earlier N=66 is not trustworthy — all in CLAUDE.md.

## Manual checks
- The metallicity selection cut ([O/H] < -1.5 in flag_anomalies) is a
  parameter — tune it deliberately.
- out/final_candidates.csv (N=39) is a shortlist of follow-up spectroscopy
  TARGETS, not confirmed loop candidates; only [O/H]/D/H measured on new
  high-resolution data can give the final label.
- A strict Monte Carlo background built from the GBM exposure map (instead
  of the RA-scrambling in 04) was deliberately not done — see CLAUDE.md.

## Outputs
- `data/*.csv|dat.gz` — raw catalogs
- `data/dla_metallicity_compilation.csv` — Rafelski+2012, 242 systems with [M/H]
- `out/cloud_candidates.csv` — precision D/H clouds with anomaly flags
- `out/dla_targets.csv` — shortlist of 7996 DR16 DLA sightlines
- `out/dla_known_metallicity_matches.csv` — Stage 1: Rafelski cross-match
- `out/zcnn_vs_metal_velocity_offset.csv` — zCNN vs. z_metal systematic (59 systems)
- `out/dla_ew_screening.csv` — Stage 2 matched-filter: EW/significance for every target
- `out/ew_screen_calibration.csv` — matched-filter calibration on 59 known-metallicity systems
- `out/final_candidates.csv` — RESULT: N=39, sorted by quasar flux
- `out/inspect/*.png` — spectrum panels for manual review of candidates
- `out/boxcar_recheck.csv` — regenerated miss-rate check on the retracted box-car screen (32/56 = 57%)
- `out/grb_matches.csv` — positional matches with GRBs
- `out/background.txt` — background expectation and p-value

## DESI extension (v3 trigger, in progress)
`desi_01`–`desi_06` port the same matched-filter method to DESI DR1
(Brodzeller et al. 2025 DLA catalog, spectra via the SPARCL API — DESI
stores spectra as ~219MB/file HEALPix coadds, not per-object files, so
this needed its own fetcher). Current result: 38 DESI candidates merged
with the SDSS 39 into `out/merged_candidates_sdss_desi.csv` (N=77, 0
cross-survey confirmations — expected given each survey's own candidate
rate is well under 1%, not a null result). Not yet manually inspected the
way the SDSS 39 were; see CLAUDE.md for the full Stage 1-3 writeup,
including three SPARCL client bugs hit and worked around along the way.

## Project notes
`CLAUDE.md` is the project's lab notebook (in Russian), documenting the
hypothesis, pilot-analysis pitfalls, the three iterations of the Stage-2
method (with the reasoning for retracting the first two), and six project
rules distilled from concrete mistakes caught along the way.
