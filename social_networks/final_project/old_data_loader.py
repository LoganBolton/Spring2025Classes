import json
import os
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
import string # To help with node identification

def get_node_token_mapping(tokens, num_nodes):
    """
    Maps node indices (0 to num_nodes-1) to lists of token indices
    where the corresponding node identifier appears.
    Assumes nodes are identified by uppercase letters starting from 'A'.
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

def aggregate_attention(raw_attn_matrix, node_token_map, num_nodes, method='average'):
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
            if method == 'average':
                node_attn_matrix[u, v] = torch.mean(token_attentions)
            elif method == 'max':
                node_attn_matrix[u, v] = torch.max(token_attentions)
            elif method == 'sum':
                node_attn_matrix[u, v] = torch.sum(token_attentions)
            else:
                raise ValueError(f"Unknown aggregation method: {method}")

    return node_attn_matrix


NODE_FEATURE_DIM = 16 # Choose a dimension (e.g., 16, 32, 64) - Hyperparameter


class GraphAttentionDataset(Dataset):
    """
    PyTorch Geometric Dataset for loading graph structures and their
    corresponding LLM attention matrices. Includes token-to-node attention aggregation.
    """
    def __init__(self, metadata_path, root_dir=None, aggregation_method='average', transform=None, pre_transform=None):
        self.metadata_path = metadata_path
        self.root_dir = ""
        self.aggregation_method = aggregation_method
        
        try:
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Metadata file not found at: {self.metadata_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Error decoding JSON from: {self.metadata_path}")
            
        self.processed_data = []
        self._process_metadata()
        
        super().__init__(root_dir, transform, pre_transform)

    def _process_metadata(self):
        print(f"Processing {len(self.metadata)} graph samples...")
        skipped_count = 0
        for idx, item in enumerate(self.metadata):
            # Basic checks for essential metadata
            num_nodes = item.get("num_nodes")
            attn_matrix_rel_path = item.get("attention_matrix_path")
            tokens = item.get("tokens")
            source_nodes = item.get("source")
            target_nodes = item.get("target")

            if None in [num_nodes, attn_matrix_rel_path, tokens, source_nodes, target_nodes]:
                print(f"Warning: Skipping item {idx} due to missing essential metadata (num_nodes, path, tokens, or edges).")
                skipped_count += 1
                continue

            # --- 1. Load Raw Attention Matrix ---
            attn_matrix_abs_path = os.path.join(self.root_dir, attn_matrix_rel_path)
            try:
                raw_attention_matrix = torch.from_numpy(np.load(attn_matrix_abs_path)).float()
            except FileNotFoundError:
                print(f"Warning: Attention matrix file not found for item {idx} at: {attn_matrix_abs_path}. Skipping.")
                skipped_count += 1
                continue
            except Exception as e:
                print(f"Warning: Error loading attention matrix for item {idx} at {attn_matrix_abs_path}: {e}. Skipping.")
                skipped_count += 1
                continue

            # --- 2. Aggregate Token Attention to Node Attention ---
            node_token_map = get_node_token_mapping(tokens, num_nodes)
            # print(node_token_map)
            
            # I think this should be fine, but gonna leave it for now 
            # Check if mapping found tokens for all nodes
            all_nodes_found = all(bool(node_token_map.get(i)) for i in range(num_nodes))
            if not all_nodes_found:
                 print(f"Warning: Skipping item {idx} because not all nodes were found in tokens.")
                 skipped_count += 1
                 continue # Skip if we can't map all nodes

            # Perform aggregation
            try:
                 node_attention_matrix = aggregate_attention(
                     raw_attention_matrix,
                     node_token_map,
                     num_nodes,
                     method=self.aggregation_method
                 )
            except Exception as e:
                 print(f"Warning: Error aggregating attention for item {idx}: {e}. Skipping.")
                 skipped_count += 1
                 continue


            # --- 3. Node Information ---
            # node_features = torch.eye(num_nodes, dtype=torch.float)
            node_features = torch.ones((num_nodes, NODE_FEATURE_DIM), dtype=torch.float)

            # --- 4. Ground Truth Edges ---
            if len(source_nodes) != len(target_nodes):
                print(f"Warning: Skipping item {idx} due source/target length mismatch.")
                skipped_count += 1
                continue
            edge_index_ground_truth = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
            
            # Validate edge indices
            if edge_index_ground_truth.numel() > 0: # Check if there are any edges
                 max_node_idx = edge_index_ground_truth.max()
                 if max_node_idx >= num_nodes:
                      print(f"Warning: Item {idx}: Max node index in edges ({max_node_idx}) "
                            f"exceeds num_nodes ({num_nodes}). Check metadata. Skipping.")
                      skipped_count += 1
                      continue
            
            # --- 5. Create PyG Data object ---
            data = Data(
                x=node_features,
                edge_index=edge_index_ground_truth, # Ground truth edges
                attn_matrix=node_attention_matrix,  # Shape: [num_nodes, num_nodes]
                num_nodes=num_nodes,
            )
            
            # Optional: Add a check for the processed matrix shape
            if data.attn_matrix.shape != (num_nodes, num_nodes):
                 print(f"ERROR: Item {idx}: Aggregated attention matrix shape {data.attn_matrix.shape} "
                       f"doesn't match expected ({num_nodes}, {num_nodes}). Check aggregation logic.")
                 # This would indicate a bug in the aggregation logic
                 skipped_count += 1
                 continue

            self.processed_data.append(data)

        print(f"Successfully processed {len(self.processed_data)} graph samples.")
        if skipped_count > 0:
            print(f"Skipped {skipped_count} samples due to warnings or errors during processing.")

    def len(self):
        return len(self.processed_data)

    def get(self, idx):
        return self.processed_data[idx]

    def process(self):
        # This method is usually for processing raw files into processed files.
        # Since we process in __init__ and store in memory, this can be empty
        # or could be adapted if you want to save/load processed data to disk.
        pass

    # Required by PyG Dataset
    @property
    def raw_file_names(self):
        # Should ideally return list of raw files (e.g., metadata.json, .npy paths)
        # For simplicity, just return the metadata file.
        return [os.path.basename(self.metadata_path)]

    @property
    def processed_file_names(self):
        # We are processing in memory, so this might not be strictly needed
        # unless you want to save/load processed data later.
        # Returning a placeholder.
        return ['dummy_processed.pt'] 
    
# # --- Example Usage ---
# if __name__ == '__main__':
#     metadata_file = '/Users/log/Github/Spring2025Classes/social_networks/final_project/attention_matrices/metadata.json'
    
#     if not os.path.exists(metadata_file):
#         print(f"Error: Metadata file not found at {metadata_file}")
#     else:
#         try:
#             # You can choose the aggregation method here
#             dataset = GraphAttentionDataset(metadata_path=metadata_file, aggregation_method='average') # or 'max' or 'sum'
            
#             if len(dataset) > 0:
#                 print(f"\nDataset loaded successfully with {len(dataset)} graphs.")
#                 first_graph = dataset[0]
                
#                 print("\n--- Example: First Graph Data (After Aggregation) ---")
#                 print(first_graph)
#                 print(f"Number of nodes: {first_graph.num_nodes}")
#                 print(f"Node features (x) shape: {first_graph.x.shape}")
#                 print(f"Ground truth edge_index shape: {first_graph.edge_index.shape}")
#                 # This shape should now match [num_nodes, num_nodes]
#                 print(f"Aggregated Attention matrix (attn_matrix) shape: {first_graph.attn_matrix.shape}") 
                
#                 # Verify the aggregated matrix values (optional)
#                 # print("\nAggregated Attention matrix (top-left 5x5):")
#                 # print(first_graph.attn_matrix[:5, :5])
#             else:
#                 print("\nDataset loaded, but it contains no valid graph samples after processing.")
#                 print("Check warnings during processing for details.")

#         except Exception as e:
#             print(f"\nAn error occurred during dataset creation or access: {e}")