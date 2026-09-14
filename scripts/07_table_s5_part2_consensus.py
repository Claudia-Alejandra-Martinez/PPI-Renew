"""
Supplementary Table S5, part 2 — functional consensus.

Consolidates the ShinyGO enrichment results across ensemble realizations and
compares them with the baseline reported in the main figure.

Two decisions worth noting:

No top-N truncation is applied before the consensus. Fold enrichment values
within the enriched set are narrowly distributed, so rank-based truncation is
unstable: restricting to the top 20 gives a Jaccard of 0.379 between two runs
whose complete enriched sets overlap at 0.907. The top-N cut is applied only
at the end, to select the pathways shown in the main figure.

Pathways are matched by KEGG identifier rather than by name, which is
sensitive to formatting differences between exports.

MIN_GENES reproduces the minimum pathway size ShinyGO applies when displaying
results. Without it, pathways with a single mapped gene yield inflated fold
enrichment by arithmetic alone.

Output
------
Table_S5_pathway_robustness.csv    pathways reported in the main figure
Dataset_S5_consensus_all.csv       every pathway, baseline or ensemble
summary_s5.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR      = Path("data")
SHINYGO_DIR   = BASE_DIR / "shinygo"                    # realization exports
BASELINE_FILE = BASE_DIR / "enrichment_baseline.csv"    # baseline export

# Pattern of the realization exports. The baseline must not match it: if it
# did, it would be counted as a realization and detect its own pathways,
# inflating their detection frequency.
REALIZATION_GLOB = "realization_*.csv"

OUT_DIR = Path("output")

FDR_CUTOFF = 0.05
MIN_GENES  = 2
TOP_N      = 10      # pathways reported in the main figure
DECIMALS   = 3


# ---------------------------------------------------------------------------

def load_enrichment(path: Path) -> pd.DataFrame:
    """Read a ShinyGO export and apply the display filters."""
    df = pd.read_csv(path)
    required = {"Enrichment FDR", "Fold Enrichment", "Pathway", "nGenes"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")

    df = df[(df["Enrichment FDR"] <= FDR_CUTOFF)
            & (df["nGenes"] >= MIN_GENES)].copy()
    df["ID"] = df["Pathway"].str.extract(r"(hsa\d+)")
    df["Name"] = (df["Pathway"]
                  .str.replace(r"^Path:hsa\d+\s*", "", regex=True)
                  .str.strip())
    return df.reset_index(drop=True)


def concordance_correlation(x, y) -> float:
    """Lin's concordance correlation coefficient: agreement, not just correlation."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(
        2 * np.cov(x, y, ddof=1)[0, 1]
        / (x.var(ddof=1) + y.var(ddof=1) + (x.mean() - y.mean()) ** 2)
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not BASELINE_FILE.exists():
        raise FileNotFoundError(BASELINE_FILE)
    files = sorted(SHINYGO_DIR.glob(REALIZATION_GLOB))
    if not files:
        raise FileNotFoundError(f"No realization exports in {SHINYGO_DIR}")
    if BASELINE_FILE.name in {p.name for p in files}:
        raise RuntimeError("Baseline file matches the realization pattern")

    baseline = load_enrichment(BASELINE_FILE)
    ensemble = {p.stem: load_enrichment(p) for p in files}
    n_real = len(ensemble)

    sizes = [len(t) for t in ensemble.values()]
    print(f"Baseline: {len(baseline)} pathways (FDR <= {FDR_CUTOFF}, "
          f"k >= {MIN_GENES})")
    print(f"Realizations: {n_real}, {np.mean(sizes):.1f} pathways on average "
          f"(range {min(sizes)}-{max(sizes)})")

    # -- Accumulate across realizations ------------------------------------
    accumulated: dict = {}
    for name, table in ensemble.items():
        for _, row in table.iterrows():
            entry = accumulated.setdefault(
                row["ID"], {"name": row["Name"], "folds": [], "runs": []}
            )
            entry["folds"].append(row["Fold Enrichment"])
            entry["runs"].append(name)

    baseline_map = baseline.set_index("ID")[
        ["Name", "Fold Enrichment", "Enrichment FDR"]
    ].to_dict("index")
    for pid, row in baseline_map.items():
        accumulated.setdefault(pid, {"name": row["Name"], "folds": [], "runs": []})

    records = []
    for pid, entry in accumulated.items():
        base = baseline_map.get(pid)
        detections = len(entry["runs"])
        folds = entry["folds"]
        if base is None:
            status = "Ensemble-only"
        elif detections == n_real:
            status = "Recovered"
        elif detections > 0:
            status = "Partially recovered"
        else:
            status = "Baseline-only"
        records.append({
            "Pathway ID": pid,
            "Pathway": base["Name"] if base else entry["name"],
            "In baseline": base is not None,
            "Baseline fold enrichment": round(base["Fold Enrichment"], DECIMALS) if base else np.nan,
            "Baseline FDR": f"{base['Enrichment FDR']:.2e}" if base else "",
            "Detections": detections,
            "Mean fold enrichment": round(float(np.mean(folds)), DECIMALS) if folds else np.nan,
            "Min fold": round(min(folds), DECIMALS) if folds else np.nan,
            "Max fold": round(max(folds), DECIMALS) if folds else np.nan,
            "Status": status,
        })

    consensus = (pd.DataFrame(records)
                 .sort_values(["Detections", "Mean fold enrichment"],
                              ascending=[False, False])
                 .reset_index(drop=True))
    consensus.to_csv(OUT_DIR / "Dataset_S5_consensus_all.csv",
                     index=False, encoding="utf-8-sig")

    counts = consensus["Status"].value_counts()
    n_baseline = int(consensus["In baseline"].sum())
    n_recovered = int(counts.get("Recovered", 0))
    n_partial = int(counts.get("Partially recovered", 0))

    shared = consensus[consensus["In baseline"]
                       & consensus["Mean fold enrichment"].notna()]
    rho, p_value = spearmanr(shared["Baseline fold enrichment"],
                             shared["Mean fold enrichment"])
    ccc = concordance_correlation(shared["Baseline fold enrichment"],
                                  shared["Mean fold enrichment"])
    median_diff = float(np.median(np.abs(shared["Baseline fold enrichment"]
                                         - shared["Mean fold enrichment"])))

    print(f"\nBaseline pathways: {n_baseline}")
    print(f"  recovered in all {n_real}:   {n_recovered} "
          f"({100 * n_recovered / n_baseline:.0f}%)")
    print(f"  partially recovered:        {n_partial}")
    print(f"  detected in none:           {int(counts.get('Baseline-only', 0))}")
    print(f"  detected at least once:     {n_recovered + n_partial} "
          f"({100 * (n_recovered + n_partial) / n_baseline:.1f}%)")
    print(f"Ensemble-only pathways:       {int(counts.get('Ensemble-only', 0))}")
    print(f"\nFold concordance (n = {len(shared)}): Spearman rho = {rho:.3f} "
          f"(p = {p_value:.2e}), Lin's CCC = {ccc:.3f}, "
          f"median absolute difference = {median_diff:.2f}")

    # -- Table S5: pathways reported in the main figure ---------------------
    top_ids = baseline.nlargest(TOP_N, "Fold Enrichment")["ID"].tolist()
    indexed = consensus.set_index("Pathway ID")
    rows = []
    for rank, pid in enumerate(top_ids, 1):
        row = indexed.loc[pid]
        folds = accumulated[pid]["folds"]
        rows.append({
            "Rank": rank,
            "KEGG pathway": row["Pathway"],
            "Realizations detected": f"{int(row['Detections'])}/{n_real}",
            "Baseline fold enrichment": row["Baseline fold enrichment"],
            "Mean fold enrichment": row["Mean fold enrichment"],
            "Fold ratio": round(row["Mean fold enrichment"]
                                / row["Baseline fold enrichment"], DECIMALS)
                          if folds else np.nan,
            "Range": (f"{min(folds):.{DECIMALS}f}-{max(folds):.{DECIMALS}f}"
                      if folds else "n.a."),
            "Status": row["Status"],
        })

    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "Table_S5_pathway_robustness.csv",
                 index=False, encoding="utf-8-sig")

    n_top_recovered = int((table["Status"] == "Recovered").sum())
    print(f"\n{n_top_recovered} of the {len(table)} pathways in the main figure "
          f"were recovered in every realization\n")
    with pd.option_context("display.width", 200):
        print(table.to_string(index=False))

    summary = {
        "n_realizations": n_real,
        "fdr_cutoff": FDR_CUTOFF,
        "min_genes": MIN_GENES,
        "baseline_pathways": n_baseline,
        "recovered": n_recovered,
        "partially_recovered": n_partial,
        "baseline_only": int(counts.get("Baseline-only", 0)),
        "ensemble_only": int(counts.get("Ensemble-only", 0)),
        "concordance": {"spearman_rho": round(float(rho), 3),
                        "spearman_p": float(p_value),
                        "lin_ccc": round(ccc, 3),
                        "median_absolute_difference": round(median_diff, 2),
                        "n_shared": int(len(shared))},
        "top_pathways_recovered": n_top_recovered,
    }
    (OUT_DIR / "summary_s5.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWritten to {OUT_DIR}")


if __name__ == "__main__":
    main()
