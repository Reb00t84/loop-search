#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regenerates fig3_archival_funnel(_ru).pdf and fig4_dv_highres(_ru).pdf for
the v4 preprint. The original generating script was never saved (ad hoc,
in a gitignored dir) - this replaces it, reading numbers straight from the
committed out/*.csv (not hardcoded), fixing three real bugs found in the
old PNGs: raster instead of vector (fig1/fig2 are already PDF - this makes
3/4 consistent with that, no pixelation at zoom/print); the legend in fig4
had no opaque background (axvline showed straight through the text); the
RU labels calqued English ("литературные металлы", "survey-недетекции").

Usage: python3 tools/build/make_figures.py
Writes into "preprint files/" (gitignored, not part of the repo).
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
OUT_DIR = os.path.join(ROOT, "preprint files")
BLUES = ["#9DB8D2", "#7DA3C8", "#5D8CBD", "#3D6FA8"]
RED = "#B2453B"

merged = pd.read_csv(os.path.join(ROOT, "out", "merged_candidates_clean.csv"))
arch = pd.read_csv(os.path.join(ROOT, "out", "archival_coverage.csv"))
inv = pd.read_csv(os.path.join(ROOT, "out", "highres_inventory.csv"))
pur = pd.read_csv(os.path.join(ROOT, "out", "highres_purity.csv"))

N_MERGED = len(merged)
N_ARCHIVAL = arch["ID"].nunique()
N_USABLE = inv[inv["suitability"] == "usable"]["ID"].nunique()
stage2_ids = pur[pur["target_class"] == "contamination"]["ID"].unique()
N_STAGE2 = len(stage2_ids)
N_CONTAM = pur[(pur["target_class"] == "contamination") &
               (pur["status"].isin(["detected", "blend_or_artifact_at_scan_max"]))]["ID"].nunique()
assert (N_MERGED, N_ARCHIVAL, N_USABLE, N_STAGE2, N_CONTAM) == (75, 15, 10, 4, 4), \
    (N_MERGED, N_ARCHIVAL, N_USABLE, N_STAGE2, N_CONTAM)

det = pur[pur["status"].isin(["detected", "blend_or_artifact_at_scan_max"])].copy()
stage1_dv = det[det["target_class"] == "consistent_with_literature"]["dv_kms"].tolist()
stage2_dv = det[det["target_class"] == "contamination"]["dv_kms"].tolist()
N_STAGE1_DET, N_STAGE2_DET = len(stage1_dv), len(stage2_dv)
N_TOTAL_DET = N_STAGE1_DET + N_STAGE2_DET
N_BEYOND = sum(1 for v in stage1_dv + stage2_dv if abs(v) > 300)
assert (N_STAGE1_DET, N_STAGE2_DET, N_TOTAL_DET, N_BEYOND) == (19, 10, 29, 10), \
    (N_STAGE1_DET, N_STAGE2_DET, N_TOTAL_DET, N_BEYOND)


def make_fig3(ru, path):
    if ru:
        labels = ["Объединённый список целей (v3)",
                  "Архивное покрытие, конус 5″ (KOA + ESO)",
                  "Пригодные редуцированные продукты",
                  "Проверяемые цели 2-й ступени\n(недетекции обзора)",
                  "Контаминированы на эшелле-разрешении"]
        xlabel = "целей"
        annotation = "0 подтверждённых чистых"
    else:
        labels = ["Merged target list (v3)",
                  "Archival coverage, 5″ cone (KOA + ESO)",
                  "Usable reduced products",
                  "Testable Stage-2 targets\n(survey non-detections)",
                  "Contaminated at echelle resolution"]
        xlabel = "targets"
        annotation = "0 confirmed clean"

    values = [N_MERGED, N_ARCHIVAL, N_USABLE, N_STAGE2, N_CONTAM]
    colors = BLUES + [RED]

    fig, ax = plt.subplots(figsize=(10.0, 4.7), facecolor="white")
    ax.set_facecolor("white")
    y = range(len(values))[::-1]
    ax.barh(list(y), values, color=colors, height=0.55)
    for yi, v in zip(y, values):
        ax.text(v + 1.0, yi, f"{v}", va="center", ha="left", fontsize=13, fontweight="bold")
    ax.text(values[-1] + 1.0 + (5 if not ru else 6), 0, annotation,
            va="center", ha="left", fontsize=12, style="italic", color=RED)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_xlim(0, 80)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, facecolor="white", edgecolor="none", transparent=False)
    plt.close(fig)


def make_fig4(ru, path):
    if ru:
        title = f"{N_BEYOND}/{N_TOTAL_DET} детекций за пределами |dv| = 300 км/с (штрих)"
        label1 = f"1-я ступень (линии из литературы), n={N_STAGE1_DET}"
        label2 = f"2-я ступень (недетекции обзора), n={N_STAGE2_DET}"
        xlabel = "dv (центроид линии металла − каталожный $z_{abs}$), км/с"
        ylabel = "детекций"
    else:
        title = f"{N_BEYOND}/{N_TOTAL_DET} detections beyond |dv| = 300 km/s (dashed)"
        label1 = f"Stage 1 (literature lines), n={N_STAGE1_DET}"
        label2 = f"Stage 2 (survey non-detections), n={N_STAGE2_DET}"
        xlabel = "dv (metal-line centroid − catalog $z_{abs}$), km/s"
        ylabel = "detections"

    bins = list(range(-500, 701, 100))
    fig, ax = plt.subplots(figsize=(10.0, 4.7), facecolor="white")
    ax.set_facecolor("white")
    ax.hist([stage1_dv, stage2_dv], bins=bins, stacked=True,
            color=[BLUES[1], RED], label=[label1, label2],
            edgecolor="white", linewidth=1.0)
    ax.axvline(-300, color="0.35", linestyle="--", linewidth=1.3, zorder=1)
    ax.axvline(300, color="0.35", linestyle="--", linewidth=1.3, zorder=1)
    leg = ax.legend(loc="upper left", fontsize=12, framealpha=1.0,
                     facecolor="white", edgecolor="0.6")
    leg.set_zorder(10)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, facecolor="white", edgecolor="none", transparent=False)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    make_fig3(False, os.path.join(OUT_DIR, "fig3_archival_funnel.pdf"))
    make_fig3(True, os.path.join(OUT_DIR, "fig3_archival_funnel_ru.pdf"))
    make_fig4(False, os.path.join(OUT_DIR, "fig4_dv_highres.pdf"))
    make_fig4(True, os.path.join(OUT_DIR, "fig4_dv_highres_ru.pdf"))
    print("wrote 4 PDFs to", OUT_DIR)
