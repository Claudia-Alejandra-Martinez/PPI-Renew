"""
Supplementary Table S5, part 1 — ensemble generation.

Shortest-path algorithms on unweighted graphs return a single representative
route when several paths of identical minimum length connect the same node
pair. This script quantifies how much that arbitrary choice matters, by
regenerating the retained network under different tie resolutions.

Uniformly distributed micro-weights (0 <= w <= 1e-4) are added to every edge,
so edge cost becomes 1 + w. Because w is several orders of magnitude smaller
than the unit cost, hop count remains the dominant term and biological path
lengths are unchanged; only tie resolution varies between realizations.

Each realization is filtered with the same topological criteria as the main
analysis, and its protein list is written for submission to ShinyGO. Part 2
consolidates the enrichment results.

Note on filter order: tail frequencies are accumulated over all candidate
paths and counted once per component, before the endpoint filters are applied.
Reversing that order discards most paths, because a downstream segment shared
across the full candidate set is far more common than one shared among the
survivors of the endpoint filters.

Output
------
gene_lists/realization_XX.txt    protein lists for ShinyGO
network_stability.csv            node and path overlap with the reference
ensemble_networks.json           retained paths per realization
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

try:
    import nx_cugraph as nx_cg
    GPU_AVAILABLE = True
except ImportError:
    nx_cg = None
    GPU_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR    = Path("data")
INTERACTOME = BASE_DIR / "ppi_renew_application_all.csv"   # Prot_A, Prot_B
TARGETS     = BASE_DIR / "targets.csv"                     # Protein, Compound, Neighborhood_1
CONDITIONS  = BASE_DIR / "targets_diseases.csv"            # Protein, Compound, Neighborhood_1

OUT_DIR   = Path("output")
LISTS_DIR = OUT_DIR / "gene_lists"

USE_GPU = True

# Network-level metrics were unchanged between 10 and 30 realizations
# (mean node Jaccard 0.449; 45 proteins detected in every realization),
# indicating that the estimate had converged at 10.
N_REALIZATIONS = 10
MAX_WEIGHT     = 1e-4
SEED_BASE      = 1000

# Topological filters, identical to the main analysis
MAX_HOPS         = 5          # 5 hops = paths of up to 6 nodes
MIN_LEN, MAX_LEN = 2, 6
MIN_COMPONENTS   = 2
MIN_CONDITIONS   = 6
MIN_TAIL_FREQ    = 2

# Cutoff must admit a 5-hop path carrying its maximum micro-weight
CUTOFF = MAX_HOPS * (1 + MAX_WEIGHT * 1.5)

# Expected values from the main analysis, used as a sanity check
EXPECTED_PATHS = 72
EXPECTED_NODES = 110


# ---------------------------------------------------------------------------

def load_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep)


def load_inputs() -> dict:
    """Build the interactome and the target-to-component mappings."""
    graph = nx.from_pandas_edgelist(load_table(INTERACTOME), "Prot_A", "Prot_B")
    nodes = set(graph)

    targets = load_table(TARGETS)
    targets = targets[targets["Neighborhood_1"] == "Target"]
    conditions = load_table(CONDITIONS)
    conditions = conditions[conditions["Neighborhood_1"] == "Target"]

    return {
        "edges": list(graph.edges()),
        "components": sorted(targets["Compound"].unique()),
        "component_targets": {
            c: sorted({p for p in g["Protein"].unique() if p in nodes})
            for c, g in targets.groupby("Compound")
        },
        "condition_targets": sorted(
            {p for p in conditions["Protein"].unique() if p in nodes}
        ),
        "components_per_protein": (
            targets.groupby("Protein")["Compound"]
            .apply(lambda s: sorted(set(s))).to_dict()
        ),
        "conditions_per_protein": (
            conditions.groupby("Protein")["Compound"]
            .apply(lambda s: sorted(set(s))).to_dict()
        ),
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
    }


def build_realization(seed: int | None, data: dict):
    """
    Compute one realization. A seed of None yields the unweighted reference.

    Returns (paths, nodes, metrics).
    """
    module = nx_cg if (USE_GPU and GPU_AVAILABLE) else nx

    edge_df = pd.DataFrame(data["edges"], columns=["Prot_A", "Prot_B"])
    if seed is None:
        edge_df["w"] = 1.0
    else:
        rng = np.random.default_rng(seed)
        edge_df["w"] = 1.0 + rng.uniform(0.0, MAX_WEIGHT, size=len(edge_df))

    graph = module.from_pandas_edgelist(edge_df, "Prot_A", "Prot_B", edge_attr="w")
    condition_set = set(data["condition_targets"])
    sources = sorted({p for ps in data["component_targets"].values() for p in ps})

    paths_by_source = {}
    for source in sources:
        try:
            reachable = module.single_source_dijkstra_path(
                graph, source, cutoff=CUTOFF, weight="w"
            )
        except Exception:
            continue
        paths_by_source[source] = [
            tuple(p) for target, p in reachable.items()
            if target in condition_set and MIN_LEN <= len(p) <= MAX_LEN
        ]
        del reachable

    tail_freq: Counter = Counter()
    survivors: set = set()
    n_candidates = 0

    for component in data["components"]:
        component_paths = set()
        for source in data["component_targets"].get(component, []):
            component_paths.update(paths_by_source.get(source, []))
        n_candidates += len(component_paths)

        for path in component_paths:
            tail_freq[path[1:]] += 1
            if (len(data["components_per_protein"].get(path[0], [])) >= MIN_COMPONENTS
                    and len(data["conditions_per_protein"].get(path[-1], [])) >= MIN_CONDITIONS):
                survivors.add(path)

    paths = sorted(p for p in survivors if tail_freq[p[1:]] >= MIN_TAIL_FREQ)
    nodes = sorted({n for p in paths for n in p})

    return paths, nodes, {
        "candidates": n_candidates,
        "after_endpoints": len(survivors),
        "paths": len(paths),
        "nodes": len(nodes),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LISTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Backend: {'nx_cugraph' if (USE_GPU and GPU_AVAILABLE) else 'networkx'}")
    data = load_inputs()
    print(f"Interactome: {data['n_nodes']:,} nodes, {data['n_edges']:,} edges")
    print(f"{N_REALIZATIONS} realizations, w ~ U(0, {MAX_WEIGHT:g})\n")

    names = ["reference"] + [f"realization_{i + 1:02d}" for i in range(N_REALIZATIONS)]
    seeds = [None] + [SEED_BASE + i for i in range(N_REALIZATIONS)]

    paths, nodes, metrics = {}, {}, {}
    for name, seed in zip(names, seeds):
        start = time.time()
        p, n, m = build_realization(seed, data)
        paths[name], nodes[name], metrics[name] = set(p), n, m
        print(f"{name:16s} paths={m['paths']:4d} nodes={m['nodes']:4d} "
              f"({time.time() - start:.1f}s)")

    ref = metrics["reference"]
    if (ref["paths"], ref["nodes"]) != (EXPECTED_PATHS, EXPECTED_NODES):
        print(f"\nWARNING: reference gave {ref['paths']} paths / {ref['nodes']} "
              f"nodes; expected {EXPECTED_PATHS} / {EXPECTED_NODES}")

    # -- Network-level stability -------------------------------------------
    ref_nodes, ref_paths = set(nodes["reference"]), paths["reference"]
    rows = []
    for name in names[1:]:
        n, p = set(nodes[name]), paths[name]
        rows.append({
            "Realization": name,
            "Nodes": len(n),
            "Nodes shared with reference": len(n & ref_nodes),
            "Node Jaccard": round(len(n & ref_nodes) / max(1, len(n | ref_nodes)), 3),
            "Paths": len(p),
            "Path Jaccard": round(len(p & ref_paths) / max(1, len(p | ref_paths)), 3),
        })
    stability = pd.DataFrame(rows)
    stability.to_csv(OUT_DIR / "network_stability.csv",
                     index=False, encoding="utf-8-sig")

    detections = Counter(n for name in names[1:] for n in nodes[name])
    core = sum(1 for v in detections.values() if v == N_REALIZATIONS)
    invariant = len({metrics[n]["paths"] for n in names}) == 1

    print(f"\n{stability.to_string(index=False)}")
    print(f"\nMean node Jaccard: {stability['Node Jaccard'].mean():.3f}")
    print(f"Mean path Jaccard: {stability['Path Jaccard'].mean():.3f}")
    print(f"Path count invariant across realizations: {invariant}")
    print(f"Proteins detected in every realization: {core} of {len(detections)}")

    # -- Gene lists for ShinyGO --------------------------------------------
    for name in names:
        (LISTS_DIR / f"{name}.txt").write_text("\n".join(nodes[name]),
                                               encoding="utf-8")
    (OUT_DIR / "ensemble_networks.json").write_text(
        json.dumps({n: sorted(" -> ".join(p) for p in paths[n]) for n in names},
                   indent=2), encoding="utf-8")

    print(f"\nGene lists written to {LISTS_DIR}")
    print("Submit each list to ShinyGO and save the full CSV export using the "
          "same base name, then run part 2.")


if __name__ == "__main__":
    main()
