"""
Supplementary Table S1 — topological sensitivity analysis.

Evaluates how many paths and nodes are retained as the output pleiotropy
threshold varies from >= 1 to >= 8 condition networks, while the input
convergence requirement is held constant at >= 2 Renew components.

The shortest paths are computed once and the thresholds are then applied to
the same candidate set, so that differences across rows reflect only the
threshold and not variation in path finding.

Output
------
Table_S1_threshold_sensitivity.csv
"""

from __future__ import annotations

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

OUT_DIR = Path("output")
USE_GPU = True

MAX_HOPS         = 5          # 5 hops = paths of up to 6 nodes
MIN_LEN, MAX_LEN = 2, 6
MIN_COMPONENTS   = 2          # held constant across all rows
MIN_TAIL_FREQ    = 2
CONDITION_RANGE  = range(1, 9)   # >= 1 to >= 8
N_CONDITIONS     = 8             # total condition networks, for coverage %

# Expected values from the published analysis, used as a sanity check
EXPECTED = {4: (11628, 2930), 5: (1918, 974), 6: (72, 110), 7: (0, 0)}


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
        "graph": graph,
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
    }


def candidate_paths(data: dict) -> tuple[list[tuple], Counter]:
    """
    Compute the candidate shortest paths once.

    Returns the deduplicated path set and the tail frequencies accumulated
    over all candidates, counted once per component. The tail frequencies must
    be computed on the full candidate set rather than on the survivors of the
    endpoint filters, since a downstream segment shared across the complete set
    is far more common than one shared among survivors.
    """
    module = nx_cg if (USE_GPU and GPU_AVAILABLE) else nx
    graph = module.from_pandas_edgelist(
        pd.DataFrame(data["edges"], columns=["Prot_A", "Prot_B"]),
        "Prot_A", "Prot_B",
    )
    condition_set = set(data["condition_targets"])
    sources = sorted({p for ps in data["component_targets"].values() for p in ps})

    paths_by_source = {}
    for source in sources:
        try:
            reachable = module.single_source_dijkstra_path(
                graph, source, cutoff=MAX_HOPS
            )
        except Exception:
            continue
        paths_by_source[source] = [
            tuple(p) for target, p in reachable.items()
            if target in condition_set and MIN_LEN <= len(p) <= MAX_LEN
        ]
        del reachable

    tail_freq: Counter = Counter()
    all_paths: set = set()
    for component in data["components"]:
        component_paths = set()
        for source in data["component_targets"].get(component, []):
            component_paths.update(paths_by_source.get(source, []))
        for path in component_paths:
            tail_freq[path[1:]] += 1
        all_paths.update(component_paths)

    return sorted(all_paths), tail_freq


def apply_thresholds(paths, tail_freq, data, min_conditions: int):
    """Filter the candidate paths at a given output pleiotropy threshold."""
    retained = [
        p for p in paths
        if len(data["components_per_protein"].get(p[0], [])) >= MIN_COMPONENTS
        and len(data["conditions_per_protein"].get(p[-1], [])) >= min_conditions
        and tail_freq[p[1:]] >= MIN_TAIL_FREQ
    ]
    nodes = {n for p in retained for n in p}
    return len(retained), len(nodes)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Backend: {'nx_cugraph' if (USE_GPU and GPU_AVAILABLE) else 'networkx'}")
    data = load_inputs()
    print(f"Interactome: {data['graph'].number_of_nodes():,} nodes, "
          f"{data['graph'].number_of_edges():,} edges")

    start = time.time()
    paths, tail_freq = candidate_paths(data)
    print(f"Candidate paths: {len(paths):,} ({time.time() - start:.1f}s)")

    rows = []
    for threshold in CONDITION_RANGE:
        n_paths, n_nodes = apply_thresholds(paths, tail_freq, data, threshold)
        rows.append({
            "Condition threshold": f">= {threshold}",
            "Condition coverage (%)": round(100 * threshold / N_CONDITIONS, 1),
            "Retained short paths": n_paths,
            "Retained nodes": n_nodes,
        })
        flag = ""
        if threshold in EXPECTED:
            exp_p, exp_n = EXPECTED[threshold]
            flag = "" if (n_paths, n_nodes) == (exp_p, exp_n) else \
                   f"  <- expected {exp_p}/{exp_n}"
        print(f"  >= {threshold}: {n_paths:>6,} paths, {n_nodes:>5,} nodes{flag}")

    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "Table_S1_threshold_sensitivity.csv",
                 index=False, encoding="utf-8-sig")
    print(f"\n{table.to_string(index=False)}")
    print(f"\nWritten to {OUT_DIR}")


if __name__ == "__main__":
    main()
