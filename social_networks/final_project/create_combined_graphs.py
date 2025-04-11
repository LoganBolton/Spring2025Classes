import torch
import numpy as np
from transformers import LlamaForCausalLM, AutoTokenizer
from torch_geometric.data import Data
import random
import os
import json
import pandas as pd
import string

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
    Aggregates token-level attention to node-level attention.

    Args:
        raw_attn_matrix (torch.Tensor): Token attention [num_tokens, num_tokens].
        node_token_map (dict): Map from node_idx to list of token_indices.
        num_nodes (int): Number of nodes.
        method (str): Aggregation method ('average', 'max', 'sum').

    Returns:
        torch.Tensor: Node-level attention matrix [num_nodes, num_nodes].
    """
    node_attn_matrix = torch.zeros((num_nodes, num_nodes), dtype=raw_attn_matrix.dtype)

    for u in range(num_nodes):
        for v in range(num_nodes):
            u_tokens = node_token_map.get(u, [])
            v_tokens = node_token_map.get(v, [])

            if not u_tokens or not v_tokens:
                # If a node wasn't found in tokens, its attention is 0
                node_attn_matrix[u, v] = 0
                continue

            # Extract the submatrix of attentions between tokens of u and v
            # Ensure indices are within bounds of raw_attn_matrix
            valid_u_tokens = [idx for idx in u_tokens if idx < raw_attn_matrix.shape[0]]
            valid_v_tokens = [idx for idx in v_tokens if idx < raw_attn_matrix.shape[1]]

            if not valid_u_tokens or not valid_v_tokens:
                 node_attn_matrix[u, v] = 0
                 continue
            
            # Use advanced indexing to get all relevant token-token attentions
            token_attentions = raw_attn_matrix[valid_u_tokens][:, valid_v_tokens]


            if token_attentions.numel() == 0:
                 node_attn_matrix[u, v] = 0 # Should not happen if valid tokens exist, but safe check
                 continue

            # Perform aggregation
            node_attn_matrix[u, v] = torch.mean(token_attentions)

    return node_attn_matrix

source_dir = 'attention_matrices/demo1'
metadata_path = f'{source_dir}/metadata.json'
save_dir = 'attention_matrices/demo1/combined'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    

data = []
with open(metadata_path, 'r') as f:
    data = json.load(f)


# get all the graphs for each graph ID into a dictionary
# convert each one to aggregated attention matrix (tokens to english)
# have a dictionary of graph_id to all aggregated attention matrices
aggregate_graphs = {}
for graph in data:
    if graph['graph_id'] not in aggregate_graphs:
        aggregate_graphs[graph['graph_id']] = []
    rax_matrix_path = graph['attention_matrix_path']
    
    raw_matrix = np.load(rax_matrix_path)
    num_nodes = graph['num_nodes']
    tokens = graph['tokens']
    
    node_token_map = get_node_token_mapping(tokens, num_nodes)
    aggregate_matrix = aggregate_attention(torch.tensor(raw_matrix), node_token_map, num_nodes)
    aggregate_graphs[graph['graph_id']].append(aggregate_matrix)
    

