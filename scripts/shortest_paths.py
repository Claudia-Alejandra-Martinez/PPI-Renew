"""
Network Pharmacology Shortest-Path & Convergence Filter Pipeline

This script constitutes the computational core of the network pharmacology workflow. It is designed to identify and isolate the most
direct biological signaling cascades bridging multicomponent formulation targets (Renew®) and physiological condition networks.

Output
------
nodes_scores.csv
shortest_paths_all.csv
filtered_shortest_paths.csv
20260902_shortest_paths_2Renew_6Condition.csv
""" 

import pandas as pd
import nx_cugraph as nx 
from collections import Counter, defaultdict

# --- Load data and build the graph ---
Data = pd.read_csv("Renew_Contition_PPI.csv")

# Create the graph from the DataFrame; nx_cugraph mimics the NetworkX API
G = nx.from_pandas_edgelist(Data, 'Prot_A', 'Prot_B')

# --- Load Targets and Targets_Diseases tables ---
Targets = pd.read_csv("input_table_1.csv")
# Filter only the Targets from table 1 (Neighborhood_1 == "Target")
Targets_filtered = Targets[Targets['Neighborhood_1'] == 'Target']

Targets_2 = pd.read_csv("Renew_PPI_tags.csv")
# Filter only the Targets from table 2
Targets_Diseases = Targets_2[Targets_2['Neighborhood_1'] == 'Target']
# List of Targets from table 2
TD = list(Targets_Diseases['Protein'])

# Get the list of unique Compounds from table 1 and 2
Compounds = Targets['Compound'].unique().tolist()
Compounds_2 = Targets_2['Compound'].unique().tolist()

# --- Calculate shortest paths using Dijkstra ONLY FOR the targets in table 1 ---
# Extract all targets from table 1
all_targets_table1 = list(Targets_filtered['Protein'].unique())
# In nx_cugraph, to get the nodes it is better to iterate over the graph (e.g., list(G))
source_targets = [t for t in all_targets_table1 if t in list(G)]
all_paths = {src: nx.single_source_dijkstra_path(G, src) for src in source_targets}

# ====================================================
# 1. Get candidate paths by Compound and calculate node scores
# ====================================================
compound_candidate_paths = {}   # Dictionary: Compound -> list of candidate paths (each path is a list)
all_candidate_paths = []        # Union of all candidate paths (for score calculation)

for comp in Compounds:
    # Use only the filtered Targets (table 1) to form the starting group
    group_proteins = list(Targets_filtered[Targets_filtered['Compound'] == comp]['Protein'])
    candidate_paths = []
    for src in group_proteins:
        if src not in all_paths:
            continue
        for tgt in TD:
            if tgt not in all_paths[src]:
                continue
            path = all_paths[src][tgt]
            # Filter paths with a length between 2 and 6 nodes
            if 2 <= len(path) <= 6:
                candidate_paths.append(path)
                
    # Remove duplicates by converting each path to a tuple and back to a list
    candidate_paths = [list(p) for p in {tuple(p) for p in candidate_paths}]
    compound_candidate_paths[comp] = candidate_paths
    all_candidate_paths.extend(candidate_paths)

# Calculate node scores using multipliers based on path length
multipliers = {2: 10000.0, 3: 1000.0, 4: 100.0, 5: 10.0, 6: 1.0}
node_scores = Counter()
for path in sorted(all_candidate_paths, key=lambda p: (len(p), p)):
    length = len(path)
    if length in multipliers:
        node_scores.update({node: multipliers[length] for node in path})

# Constraint: obtain only the nodes that participate in the candidate paths
candidate_nodes = set()
for path in all_candidate_paths:
    candidate_nodes.update(path)
candidate_nodes = sorted(candidate_nodes)

# Create the scores table only for the nodes that appear in the candidate paths
scores = [node_scores.get(node, 0) for node in candidate_nodes]
df_scores = pd.DataFrame({'Nodes': candidate_nodes, 'Path_Related_Score': scores})

# Export the node scores output
df_scores.to_csv("nodes_scores.csv", index=False)


# ====================================================
# 2. Prepare the candidate paths table with additional columns:
# ====================================================

# Precalculate protein -> Compounds mappings (as a string)
# For Targets (which contains the Targets from table 1)
target_compounds_dict = (
    Targets_filtered.groupby("Protein")["Compound"]
    .unique()
    .apply(lambda comps: ", ".join(comps))
    .to_dict()
)

# For Targets_2 (from table 2)
targets2_compounds_dict = (
    Targets_Diseases.groupby("Protein")["Compound"]
    .unique()
    .apply(lambda comps: ", ".join(comps))
    .to_dict()
)

rows_paths = []
for comp, paths in compound_candidate_paths.items():
    for path in paths:
        # First node of the path
        first_target = path[0]
        # Search in the precalculated dictionary
        target_compounds_str = target_compounds_dict.get(first_target, "")
        
        # Last protein of the path (tail)
        last_protein = path[-1] if len(path) >= 2 else None
        if last_protein:
            associated_compounds2_str = targets2_compounds_dict.get(last_protein, "")
        else:
            associated_compounds2_str = ""
            
        rows_paths.append({
            "Target_Compounds": target_compounds_str,
            "Compound": comp,
            "PathLength": len(path),
            "Path": " -> ".join(path),
            "Compounds_2": associated_compounds2_str
        })

df_paths = pd.DataFrame(rows_paths)
# Reorder columns so that "Target_Compounds" is first
df_paths = df_paths[["Target_Compounds", "Compound", "PathLength", "Path", "Compounds_2"]]

# Export the paths output
df_paths.to_csv("shortest_paths_all.csv", index=False)


# ====================================================
# 3. Convergence & Common Tails Filtering
# ====================================================

# --- Precalculate "raw" dictionaries to get the associated Compound lists ---
target_compounds_dict_raw = Targets_filtered.groupby("Protein")["Compound"].unique().to_dict()
targets2_compounds_dict_raw = Targets_Diseases.groupby("Protein")["Compound"].unique().to_dict()

# --- 1. Calculate the frequency of each tail (path[1:]) among ALL candidate paths ---
tail_frequency_table1 = Counter()
for comp in Compounds:
    candidate_paths = compound_candidate_paths.get(comp, [])
    for path in candidate_paths:
        if len(path) >= 2:
            tail = tuple(path[1:])
            tail_frequency_table1[tail] += 1

# Define as common those tails that appear at least 2 times
final_common_tails = {tail for tail, freq in tail_frequency_table1.items() if freq >= 2}

# --- 2. Filter candidate paths that meet the multi-objective convergence conditions ---
filtered_final_paths = []  # Each element will be a tuple: (path, target_comps_list, compounds2_list)
for comp in Compounds:
    for path in compound_candidate_paths.get(comp, []):
        if len(path) < 2:
            continue
        tail = tuple(path[1:])
        if tail not in final_common_tails:
            continue
            
        first_target = path[0]
        last_protein = path[-1]
        
        target_comps = target_compounds_dict_raw.get(first_target, [])
        compounds2 = targets2_compounds_dict_raw.get(last_protein, [])
        
        # Dual-convergence criteria: >= 2 input compounds, >= 6 disease outputs
        if len(target_comps) >= 2 and len(compounds2) >= 6:
            filtered_final_paths.append((path, target_comps, compounds2))

# --- 3. Prepare the intermediate output including data to group by tail ---
paths_details = []
for path, target_comps, compounds2 in filtered_final_paths:
    detail = {
        "Common_Full_Path": " -> ".join(path),
        "Path_Length": len(path),
        "Target_Compounds": ", ".join(target_comps),
        "Compounds_2": ", ".join(compounds2),
        "Tail": tuple(path[1:]),  # This data is for grouping
        "First_Node": path[0]     # First node of the path
    }
    paths_details.append(detail)

# --- 4. Determine if all paths that contain a specific tail start with the same Target ---
tail_to_first_nodes = defaultdict(set)
for rec in paths_details:
    tail_to_first_nodes[rec["Tail"]].add(rec["First_Node"])

# Add the "Starts_Same_Target" column
for rec in paths_details:
    first_nodes = tail_to_first_nodes[rec["Tail"]]
    rec["Starts_Same_Target"] = "Same" if len(first_nodes) == 1 else "Different"

# --- 5. Create the final DataFrame ---
df_common_full = pd.DataFrame(paths_details)

# Export the final converged paths output
df_common_full.to_csv("filtered_shortest_paths.csv", index=False)


# ====================================================
# 4. Generate connections output in Cytoscape format
# ====================================================

# Auxiliary function to get the connection label (type) given the edge index and path length.
def get_edge_label(i, path_length):
    # If the path has only 2 nodes it is considered "Input_Output"
    if path_length == 2:
        return "Input_Output"
    else:
        if i == 0:
            return "Input_First_Neighbor"
        else:
            # i=1 -> "First_Neighbor_Second_Neighbor", etc.
            ordinal = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh"]
            if i < len(ordinal):
                return f"{ordinal[i-1]}_Neighbor_{ordinal[i]}" if i < len(ordinal) else f"Neighbor_{i}_{i+1}"
            else:
                return f"Neighbor_{i}_{i+1}"

# It is assumed that 'filtered_final_paths' is the list of tuples obtained in Part C
edges_data = []
for (path, target_comps, compounds2) in filtered_final_paths:
    path_str = " -> ".join(path)  # Complete path string for reference
    path_length = len(path)       # Path length
    n = path_length
    
    # Iterate over each pair of consecutive nodes in the path
    for i in range(n - 1):
        from_node = path[i]
        to_node = path[i+1]
        edge_label = get_edge_label(i, n)
        
        # Two rows are created for the same connection, reversing the order (bidirectional representation)
        connection1 = f"{to_node} (interacts with) {from_node}"
        connection2 = f"{from_node} (interacts with) {to_node}"
        
        edges_data.append({
            "Path": path_str,
            "Path_Length": path_length,
            "Edge": connection1,
            "Edge_Label": edge_label
        })
        edges_data.append({
            "Path": path_str,
            "Path_Length": path_length,
            "Edge": connection2,
            "Edge_Label": edge_label
        })

# Convert the connections list to a DataFrame
df_edges = pd.DataFrame(edges_data)

# Export the new output to import directly into Cytoscape
df_edges.to_csv("20260902_shortest_paths_2Renew_6Condition.csv", index=False)
