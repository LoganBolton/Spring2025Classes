import torch
import numpy as np
from torch_geometric.data import Data
import os
import json
import pandas as pd
import string
from collections import defaultdict # Import defaultdict for easier grouping


def get_node_token_mapping(tokens, num_nodes):
    """
    Gets the indices where each node appears in token list.
    Assumes that nodes are in the format 'A', 'B', ..., 'Z', etc.
    """
    node_token_map = {i: [] for i in range(num_nodes)}
    # Generate expected node identifiers ('A', 'B', ..., up to num_nodes)
    node_identifiers = list(string.ascii_uppercase[:num_nodes])

    for token_idx, token in enumerate(tokens):
        cleaned_token = token.strip(':') # Example cleaning

        if cleaned_token in node_identifiers:
            node_idx = node_identifiers.index(cleaned_token)
            node_token_map[node_idx].append(token_idx)
    return node_token_map

def aggregate_attention(raw_attn_matrix, node_token_map, num_nodes):
    """
    Aggregates token-level attention to node-level attention, respecting causal masking.

    Args:
        raw_attn_matrix (torch.Tensor): Token attention [num_tokens, num_tokens].
                                         Assumes raw_attn_matrix[i, j] is attention FROM token i TO token j.
        node_token_map (dict): Map from node_idx to list of token_indices.
        num_nodes (int): Number of nodes.

    Returns:
        torch.Tensor: Node-level attention matrix [num_nodes, num_nodes].
    """

    # Initialize on the same device as the input matrix
    device = raw_attn_matrix.device
    node_attn_matrix = torch.zeros((num_nodes, num_nodes), dtype=raw_attn_matrix.dtype, device=device)

    for u in range(num_nodes):  # Source node index
        for v in range(num_nodes):  # Target node index
            u_tokens = node_token_map.get(u, [])
            v_tokens = node_token_map.get(v, [])

            # Filter tokens to be within matrix bounds (though usually not needed if map is correct)
            valid_u_tokens = [idx for idx in u_tokens if idx < raw_attn_matrix.shape[0]]
            valid_v_tokens = [idx for idx in v_tokens if idx < raw_attn_matrix.shape[1]] # Assuming square

            sum_attentions = torch.tensor(0.0, dtype=raw_attn_matrix.dtype, device=device)
            count_attentions = 0

            for i in valid_u_tokens:    # i = source token index
                for j in valid_v_tokens:  # j = target token index
                    # Apply the causal mask condition: token j must be <= token i
                    if j <= i:
                        # Check bounds just in case (should be redundant with valid_ checks)
                        if i < raw_attn_matrix.shape[0] and j < raw_attn_matrix.shape[1]:
                            sum_attentions += raw_attn_matrix[i, j]
                            count_attentions += 1

            if count_attentions > 0:
                node_attn_matrix[u, v] = sum_attentions / count_attentions

    return node_attn_matrix

source_dir = 'attention_matrices/arg_3'
metadata_path = f'{source_dir}/metadata.json'
save_dir = f'{source_dir}/combined'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    

data = []
with open(metadata_path, 'r') as f:
    data = json.load(f)


# get all the graphs for each graph ID into a dictionary
# convert each one to aggregated attention matrix (tokens to english)
# have a dictionary of graph_id to all aggregated attention matrices
aggregate_graphs = {}
original_metadata_grouped = defaultdict(list) # Stores original metadata dicts grouped by id

for graph in data:
    graph_id = graph['graph_id']
    original_metadata_grouped[graph_id].append(graph)
    if graph_id not in aggregate_graphs:
        aggregate_graphs[graph_id] = []
    raw_matrix_path = graph['attention_matrix_path']
    
    raw_matrix = np.load(raw_matrix_path)
    # num_nodes = graph['num_nodes']
    max_nodes = graph['max_nodes']
    tokens = graph['tokens']
    source_nodes = graph['source']
    target_nodes = graph['target']
    
    node_token_map = get_node_token_mapping(tokens, max_nodes)
    aggregate_matrix = aggregate_attention(torch.tensor(raw_matrix), node_token_map, max_nodes)
   
    NODE_FEATURE_DIM = 16 
    node_features = torch.ones((max_nodes, NODE_FEATURE_DIM), dtype=torch.float)
    edge_index_ground_truth = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

    aggregate_graphs[graph_id].append(aggregate_matrix)
    
# averages the aggregated matrices for each graph_id
averaged_aggregate_graphs = {}
for graph_id, matrix_list in aggregate_graphs.items():
    stacked_matrices = torch.stack(matrix_list, dim=0)
    averaged_matrix = torch.mean(stacked_matrices, dim=0)

    averaged_aggregate_graphs[graph_id] = averaged_matrix
    print(f"Averaged {len(matrix_list)} matrices for graph_id {graph_id}.")

new_metadata_list = []
combined_metadata_save_path = os.path.join(save_dir, 'combined_metadata.json') # Path for the new metadata file

print("\nSaving averaged matrices and creating combined metadata...")

# save new averaged, aggregated matrices to simple adjacency matrix
# also put it all in a nice metadata file
for graph_id, averaged_matrix in averaged_aggregate_graphs.items():
    # --- Save the averaged matrix ---
    save_path = os.path.join(save_dir, f"averaged_id_{graph_id}.pt")
    try:
        torch.save(averaged_matrix, save_path)
        print(f"Saved averaged matrix for graph_id {graph_id} to {save_path}.")
    except Exception as e:
        print(f"Error saving averaged matrix for graph_id {graph_id} to {save_path}: {e}")
        continue # Skip metadata creation if saving failed

    # --- Create the new metadata entry ---
    # 1. Get a representative metadata entry (e.g., the first one for this graph_id)
    representative_meta = original_metadata_grouped[graph_id][0].copy() # Use copy to avoid modifying the original

    # 2. Get all original attention matrix paths for this graph_id
    original_paths = [meta['attention_matrix_path'] for meta in original_metadata_grouped[graph_id]]

    # 3. Remove the old specific attention matrix path
    if 'attention_matrix_path' in representative_meta:
        del representative_meta['attention_matrix_path']

    # 4. Add the new fields
    representative_meta['averaged_attention_matrix_path'] = os.path.relpath(save_path, start=os.path.dirname(combined_metadata_save_path)) # Store relative path
    # Or use absolute path: representative_meta['averaged_attention_matrix_path'] = os.path.abspath(save_path)
    representative_meta['original_attention_matrix_paths'] = original_paths
    representative_meta['num_averaged_samples'] = len(original_paths) # Add count of averaged graphs

    # 5. Add the new metadata entry to our list
    new_metadata_list.append(representative_meta)

# --- Save the new combined metadata ---
with open(combined_metadata_save_path, 'w') as f:
    json.dump(new_metadata_list, f, indent=2)
print(f"\nSaved combined metadata for {len(new_metadata_list)} graph IDs to {combined_metadata_save_path}.")

print("done")
