"""
Supplementary Tables S2 and S3 — topological specificity of core mediators.

S2 compares the proportion of proteins whose frequency as a path intermediary
exceeds a degree-preserving null expectation, between the highly convergent
core and the rest of the evaluable interactome.

S3 stratifies the same comparison by degree quartile, to verify that the
over-representation reported in S2 is not driven by the higher connectivity of
core proteins.

A protein is evaluable only if it acts as an intermediary in at least one
candidate shortest path. Proteins occurring exclusively as path endpoints have
an intermediary frequency of zero by design; including them would deflate the
specificity rate with untestable cases.

This analysis characterizes the core rather than pruning it: all core proteins
are retained downstream regardless of their empirical p-value.

Output
------
Table_S2_topological_specificity.csv
Table_S3_degree_stratification.csv
summary_s2_s3.json
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR    = Path("data")
INTERACTOME = BASE_DIR / "ppi_renew_application_all.csv"   # Prot_A, Prot_B
ALL_PATHS   = BASE_DIR / "shortest_paths_2Renew_6Condition.csv"          # all candidate paths
CORE_PATHS  = BASE_DIR / "shortest_paths_2Renew_6Condition.csv"                  # paths passing >=2 / >=6
SCORES      = BASE_DIR / "permutation_scores.csv"          # Nodes, Empirical_P_Value, ...

OUT_DIR = Path("output")

P_CUTOFF   = 0.05
N_QUANTILES = 4

# Expected values from the published analysis, used as a sanity check
EXPECTED_EVALUABLE = 5547
EXPECTED_CORE_NODES = 110


# ---------------------------------------------------------------------------

def load_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep)


def path_column(df: pd.DataFrame) -> str:
    for column in ("Common_Full_Path", "Path"):
        if column in df.columns:
            return column
    raise ValueError(f"No path column found. Available: {list(df.columns)}")


def split_roles(df: pd.DataFrame) -> tuple[set, set]:
    """Return (intermediaries, endpoints) across the paths in a table."""
    column = path_column(df)
    intermediaries, endpoints = set(), set()
    for value in df[column]:
        if not isinstance(value, str):
            continue
        nodes = value.split(" -> ")
        if len(nodes) < 2:
            continue
        endpoints.update((nodes[0], nodes[-1]))
        intermediaries.update(nodes[1:-1])
    return intermediaries, endpoints


def specificity_row(label: str, nodes: list[str], p_values: dict) -> dict:
    significant = sum(1 for n in nodes if p_values[n] <= P_CUTOFF)
    return {
        "Node set": label,
        "Evaluable nodes": len(nodes),
        "Significant (p <= 0.05)": significant,
        "Not significant": len(nodes) - significant,
        "Specificity rate (%)": round(100 * significant / max(1, len(nodes)), 1),
    }


def odds_ratio_ci(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """Woolf 95% confidence interval for the odds ratio."""
    odds = (a / b) / (c / d)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return float(np.exp(np.log(odds) - 1.96 * se)), float(np.exp(np.log(odds) + 1.96 * se))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    graph = nx.from_pandas_edgelist(load_table(INTERACTOME), "Prot_A", "Prot_B")
    degree = dict(graph.degree())

    all_intermediaries, _ = split_roles(load_table(ALL_PATHS))
    core_df = load_table(CORE_PATHS)
    core_intermediaries, core_endpoints = split_roles(core_df)
    core_nodes = core_intermediaries | core_endpoints

    scores = load_table(SCORES)
    if "Nodes" not in scores.columns:
        raise ValueError(f"Scores table lacks a 'Nodes' column: {list(scores.columns)}")
    p_values = scores.set_index("Nodes")["Empirical_P_Value"].to_dict()

    evaluable = sorted(all_intermediaries & set(p_values))
    core = sorted(core_intermediaries & set(evaluable))
    background = sorted(set(evaluable) - set(core))

    missing = (all_intermediaries | core_intermediaries) - set(p_values)
    if missing:
        print(f"WARNING: {len(missing)} intermediaries lack a p-value "
              f"(e.g. {sorted(missing)[:5]})")

    print(f"Interactome: {graph.number_of_nodes():,} nodes")
    print(f"Evaluable intermediaries: {len(evaluable):,}"
          + ("" if len(evaluable) == EXPECTED_EVALUABLE
             else f"  <- expected {EXPECTED_EVALUABLE}"))
    print(f"Core: {len(core_nodes)} nodes"
          + ("" if len(core_nodes) == EXPECTED_CORE_NODES
             else f"  <- expected {EXPECTED_CORE_NODES}")
          + f" ({len(core)} evaluable, "
          f"{len(core_endpoints - core_intermediaries)} endpoint-only)")

    # -- Table S2 ----------------------------------------------------------
    s2 = pd.DataFrame([
        specificity_row("All evaluable nodes", evaluable, p_values),
        specificity_row("Interactome background", background, p_values),
        specificity_row("Highly convergent core", core, p_values),
    ])
    s2.to_csv(OUT_DIR / "Table_S2_topological_specificity.csv",
              index=False, encoding="utf-8-sig")

    a = int(s2.loc[2, "Significant (p <= 0.05)"])
    b = int(s2.loc[2, "Not significant"])
    c = int(s2.loc[1, "Significant (p <= 0.05)"])
    d = int(s2.loc[1, "Not significant"])
    odds, p_fisher = fisher_exact([[a, b], [c, d]], alternative="greater")
    ci_low, ci_high = odds_ratio_ci(a, b, c, d)
    rate_ratio = s2.loc[2, "Specificity rate (%)"] / s2.loc[1, "Specificity rate (%)"]

    print(f"\n{s2.to_string(index=False)}")
    print(f"\nFisher's exact test (one-sided): OR = {odds:.2f} "
          f"(95% CI {ci_low:.2f}-{ci_high:.2f}), p = {p_fisher:.2e}")
    print(f"Rate ratio: {rate_ratio:.1f}-fold")

    # -- Table S3 ----------------------------------------------------------
    # Degree quartiles are half-open, so that a node of degree exactly equal to
    # a boundary belongs to a single quartile.
    edges = np.quantile([degree.get(n, 0) for n in evaluable],
                        np.linspace(0, 1, N_QUANTILES + 1))
    rows = []
    for i in range(N_QUANTILES):
        low, high = edges[i], edges[i + 1]
        last = i == N_QUANTILES - 1
        in_bin = (lambda n: low <= degree.get(n, 0) <= high) if last else \
                 (lambda n: low <= degree.get(n, 0) < high)

        core_bin = [n for n in core if in_bin(n)]
        back_bin = [n for n in background if in_bin(n)]
        core_sig = sum(1 for n in core_bin if p_values[n] <= P_CUTOFF)
        back_sig = sum(1 for n in back_bin if p_values[n] <= P_CUTOFF)

        core_pct = 100 * core_sig / len(core_bin) if core_bin else None
        back_pct = 100 * back_sig / len(back_bin) if back_bin else None
        label_high = int(high) if last else int(high) - 1

        rows.append({
            "Degree quartile": f"Q{i + 1} ({int(low)}-{label_high})",
            "Core n": len(core_bin),
            "Core significant (%)": round(core_pct, 1) if core_pct is not None else "n.a.",
            "Background n": len(back_bin),
            "Background significant (%)": round(back_pct, 1) if back_pct is not None else "n.a.",
            "Core / Background ratio": (round(core_pct / back_pct, 2)
                                        if (core_pct is not None and back_pct) else "n.a."),
        })

    s3 = pd.DataFrame(rows)
    s3.to_csv(OUT_DIR / "Table_S3_degree_stratification.csv",
              index=False, encoding="utf-8-sig")

    core_deg = np.array([degree.get(n, 0) for n in core])
    back_deg = np.array([degree.get(n, 0) for n in background])
    _, p_mw = mannwhitneyu(core_deg, back_deg, alternative="two-sided")

    print(f"\n{s3.to_string(index=False)}")
    print(f"\nMedian degree: core {np.median(core_deg):.1f}, "
          f"background {np.median(back_deg):.1f} "
          f"(Mann-Whitney U, p = {p_mw:.2e})")

    # -- Summary -----------------------------------------------------------
    summary = {
        "p_cutoff": P_CUTOFF,
        "evaluable_nodes": len(evaluable),
        "core_nodes_total": len(core_nodes),
        "core_nodes_evaluable": len(core),
        "core_endpoint_only": len(core_endpoints - core_intermediaries),
        "fisher": {"odds_ratio": round(float(odds), 2),
                   "ci_low": round(ci_low, 2), "ci_high": round(ci_high, 2),
                   "p_value": float(p_fisher)},
        "rate_ratio": round(float(rate_ratio), 2),
        "median_degree": {"core": float(np.median(core_deg)),
                          "background": float(np.median(back_deg)),
                          "mannwhitney_p": float(p_mw)},
    }
    (OUT_DIR / "summary_s2_s3.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWritten to {OUT_DIR}")


if __name__ == "__main__":
    main()
