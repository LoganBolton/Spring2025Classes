import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import networkx as nx
from scipy.stats import ks_2samp # For comparing distributions
from sklearn.metrics import jaccard_score, normalized_mutual_info_score, adjusted_rand_score # For edge set and community comparison
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings # To suppress specific warnings if needed

# --- Community Detection Libraries ---
try:
    import community as community_louvain # pip install python-louvain
except ImportError:
    print("Warning: `python-louvain` library not found. Louvain community detection will be skipped.")
    print("Install using: pip install python-louvain")
    community_louvain = None

import networkx.algorithms.community as nx_comm

# --- Import necessary components from the training script ---
# Define the MLP Model (Copied from training script)
class MLPPredictAdj(nn.Module):
    def __init__(self, input_size, hidden_channels=128, output_size=None):
        super().__init__()
        if output_size is None:
            output_size = input_size # Predict flattened adjacency of same size
        self.fc1 = nn.Linear(input_size, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.fc3 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc4 = nn.Linear(hidden_channels // 2, hidden_channels// 4)
        self.fc_out = nn.Linear(hidden_channels // 4, output_size)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        adj_logits = self.fc_out(x)
        return adj_logits

# Standard PyTorch Dataset (Copied and potentially adapted)
class AdjacencyDatasetEval(Dataset):
    def __init__(self, data_list):
        # data_list should be a list of tuples:
        # (flattened_attn_matrix_padded, gt_adj_directed_padded, graph_id, original_num_nodes)
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        attn_flat_pad, gt_adj_pad, graph_id, orig_n_nodes = self.data_list[idx]
        return attn_flat_pad, gt_adj_pad, graph_id, orig_n_nodes

# --- Configuration Parameters (Should match training script where applicable) ---
RUN_PATH = 'attention_matrices/no_args_7_1b/combined'
METADATA_PATH = f'/Users/log/Github/Spring2025Classes/social_networks/final_project/{RUN_PATH}/combined_metadata.json'
DATA_DIR = f'/Users/log/Github/Spring2025Classes/social_networks/final_project/'
BATCH_SIZE = 128 # Can be larger for evaluation
HIDDEN_CHANNELS = 4096 # Must match the trained model
TEST_SPLIT = 0.2 # Used to identify the same test set
RANDOM_SEED = 42
# --- Directory containing the trained model and output for this script ---
TRAINING_OUTPUT_DIR = f'{RUN_PATH}/mlp_baseline_training_output'
EVALUATION_OUTPUT_DIR = f'{RUN_PATH}/mlp_baseline_evaluation_output_v3' # New output dir for V3 (with communities)
MODEL_PATH = os.path.join(TRAINING_OUTPUT_DIR, 'best_model.pt') # Path to the saved model
PREDICTION_THRESHOLD = 0.5 # Threshold to convert sigmoid outputs to binary edges
# --- End Configuration Parameters ---

# --- Helper Functions for SNA ---

def calculate_sna_metrics(G):
    """
    Calculates various SNA metrics for a given NetworkX graph,
    including community detection results.
    Returns:
        dict: Dictionary of metrics. Includes partitions if calculated.
    """
    metrics = {}
    num_nodes = G.number_of_nodes()

    # Default values for empty or trivial graphs
    metrics['num_nodes'] = num_nodes
    metrics['num_edges'] = 0
    metrics['density'] = 0.0
    metrics['avg_clustering'] = 0.0
    metrics['transitivity'] = 0.0 # Global clustering
    metrics['degree_mean'] = 0.0
    metrics['degree_std'] = 0.0
    metrics['in_degree_mean'] = 0.0
    metrics['in_degree_std'] = 0.0
    metrics['out_degree_mean'] = 0.0
    metrics['out_degree_std'] = 0.0
    metrics['betweenness_mean'] = 0.0
    metrics['betweenness_std'] = 0.0
    metrics['closeness_mean'] = 0.0
    metrics['closeness_std'] = 0.0
    metrics['local_clustering_mean'] = 0.0
    metrics['local_clustering_std'] = 0.0
    metrics['num_weakly_connected_components'] = 0
    metrics['largest_weakly_connected_component_size'] = 0
    metrics['num_strongly_connected_components'] = 0
    metrics['largest_strongly_connected_component_size'] = 0
    metrics['diameter_lcc'] = 0
    metrics['calculated_diameter_on_lcc'] = False

    # Community Detection Defaults
    metrics['num_communities_louvain'] = 0
    metrics['partition_louvain'] = None # Store the actual partition
    metrics['num_communities_lpa'] = 0
    metrics['partition_lpa'] = None # Store the actual partition

    # Lists for distribution analysis
    metrics['degree_values'] = []
    metrics['in_degree_values'] = []
    metrics['out_degree_values'] = []
    metrics['betweenness_values'] = []
    metrics['closeness_values'] = []
    metrics['local_clustering_values'] = []

    if num_nodes == 0:
        return metrics # Return defaults for empty graph

    # --- Standard Metrics (as before) ---
    metrics['num_edges'] = G.number_of_edges()
    try:
        metrics['density'] = nx.density(G)
    except ZeroDivisionError:
        metrics['density'] = 0.0

    try:
        metrics['avg_clustering'] = nx.average_clustering(G)
        metrics['transitivity'] = nx.transitivity(G)
        local_clustering_dict = nx.clustering(G)
        metrics['local_clustering_values'] = list(local_clustering_dict.values())
        if metrics['local_clustering_values']:
             metrics['local_clustering_mean'] = np.mean(metrics['local_clustering_values'])
             metrics['local_clustering_std'] = np.std(metrics['local_clustering_values'])
    except (ZeroDivisionError, nx.NetworkXError):
         metrics['avg_clustering'], metrics['transitivity'] = 0.0, 0.0
         metrics['local_clustering_values'] = [0.0] * num_nodes
         metrics['local_clustering_mean'], metrics['local_clustering_std'] = 0.0, 0.0

    degree_dict = dict(G.degree())
    in_degree_dict = dict(G.in_degree())
    out_degree_dict = dict(G.out_degree())
    metrics['degree_values'] = list(degree_dict.values())
    metrics['in_degree_values'] = list(in_degree_dict.values())
    metrics['out_degree_values'] = list(out_degree_dict.values())

    if num_nodes > 1 and metrics['num_edges'] > 0 :
         k_approx = min(num_nodes, 100) if num_nodes > 100 else None
         try:
             betweenness_dict = nx.betweenness_centrality(G, k=k_approx, normalized=True, endpoints=False)
         except Exception: betweenness_dict = {n: 0.0 for n in G.nodes()}
         try:
             closeness_dict = nx.closeness_centrality(G)
         except Exception: closeness_dict = {n: 0.0 for n in G.nodes()}
    else:
         betweenness_dict = {n: 0.0 for n in G.nodes()}
         closeness_dict = {n: 0.0 for n in G.nodes()}
    metrics['betweenness_values'] = list(betweenness_dict.values())
    metrics['closeness_values'] = list(closeness_dict.values())

    metrics['degree_mean'] = np.mean(metrics['degree_values']) if metrics['degree_values'] else 0
    metrics['degree_std'] = np.std(metrics['degree_values']) if metrics['degree_values'] else 0
    metrics['in_degree_mean'] = np.mean(metrics['in_degree_values']) if metrics['in_degree_values'] else 0
    metrics['in_degree_std'] = np.std(metrics['in_degree_values']) if metrics['in_degree_values'] else 0
    metrics['out_degree_mean'] = np.mean(metrics['out_degree_values']) if metrics['out_degree_values'] else 0
    metrics['out_degree_std'] = np.std(metrics['out_degree_values']) if metrics['out_degree_values'] else 0
    metrics['betweenness_mean'] = np.mean(metrics['betweenness_values']) if metrics['betweenness_values'] else 0
    metrics['betweenness_std'] = np.std(metrics['betweenness_values']) if metrics['betweenness_values'] else 0
    metrics['closeness_mean'] = np.mean(metrics['closeness_values']) if metrics['closeness_values'] else 0
    metrics['closeness_std'] = np.std(metrics['closeness_values']) if metrics['closeness_values'] else 0

    wcc = list(nx.weakly_connected_components(G))
    if wcc:
        metrics['num_weakly_connected_components'] = len(wcc)
        largest_wcc_nodes = max(wcc, key=len)
        metrics['largest_weakly_connected_component_size'] = len(largest_wcc_nodes)
        if metrics['largest_weakly_connected_component_size'] > 1:
            lcc_subgraph = G.subgraph(largest_wcc_nodes).copy()
            try:
                if nx.is_weakly_connected(lcc_subgraph):
                     metrics['diameter_lcc'] = nx.diameter(lcc_subgraph)
                     metrics['calculated_diameter_on_lcc'] = True
                else: metrics['diameter_lcc'], metrics['calculated_diameter_on_lcc'] = -1, True
            except Exception: metrics['diameter_lcc'], metrics['calculated_diameter_on_lcc'] = -1, True
        else: metrics['diameter_lcc'], metrics['calculated_diameter_on_lcc'] = 0, True
    else: metrics['num_weakly_connected_components'], metrics['largest_weakly_connected_component_size'], metrics['diameter_lcc'], metrics['calculated_diameter_on_lcc'] = 0, 0, 0, False

    scc = list(nx.strongly_connected_components(G))
    if scc:
        metrics['num_strongly_connected_components'] = len(scc)
        metrics['largest_strongly_connected_component_size'] = len(max(scc, key=len))
    else: metrics['num_strongly_connected_components'], metrics['largest_strongly_connected_component_size'] = 0, 0

    # --- Community Detection ---
    if num_nodes > 0 and metrics['num_edges'] > 0: # Only run if graph is non-trivial
        # 1. Louvain Algorithm (on Undirected Graph)
        if community_louvain:
            try:
                # Louvain works best on undirected graphs without self-loops
                G_undirected = G.to_undirected()
                G_undirected.remove_edges_from(nx.selfloop_edges(G_undirected))
                if G_undirected.number_of_edges() > 0:
                    partition = community_louvain.best_partition(G_undirected, random_state=RANDOM_SEED) # Add random state for reproducibility
                    metrics['partition_louvain'] = partition
                    metrics['num_communities_louvain'] = len(set(partition.values()))
                else: # Handle case where undirected graph has no edges
                    metrics['partition_louvain'] = {node: i for i, node in enumerate(G.nodes())} # Each node is a community
                    metrics['num_communities_louvain'] = num_nodes
            except Exception as e:
                # print(f"Warning: Louvain community detection failed: {e}")
                metrics['partition_louvain'] = None # Indicate failure
                metrics['num_communities_louvain'] = -1 # Use -1 to indicate failure vs 0 communities

        # 2. Asynchronous Label Propagation (on Directed Graph)
        try:
            communities_generator = nx_comm.label_propagation_communities(G)
            communities_list = list(communities_generator)
            if communities_list:
                # Convert list of sets to partition dictionary (node -> community_id)
                partition = {}
                for i, community_nodes in enumerate(communities_list):
                    for node in community_nodes:
                        partition[node] = i
                metrics['partition_lpa'] = partition
                metrics['num_communities_lpa'] = len(communities_list)
            else: # Should not happen if graph has nodes/edges, but handle anyway
                 metrics['partition_lpa'] = {node: i for i, node in enumerate(G.nodes())}
                 metrics['num_communities_lpa'] = num_nodes
        except Exception as e:
            # print(f"Warning: Label Propagation community detection failed: {e}")
            metrics['partition_lpa'] = None # Indicate failure
            metrics['num_communities_lpa'] = -1 # Use -1 for failure

    elif num_nodes > 0: # Handle graph with nodes but no edges
        metrics['partition_louvain'] = {node: i for i, node in enumerate(G.nodes())}
        metrics['num_communities_louvain'] = num_nodes
        metrics['partition_lpa'] = {node: i for i, node in enumerate(G.nodes())}
        metrics['num_communities_lpa'] = num_nodes

    return metrics


def compare_sna_metrics(metrics_gt, metrics_pred):
    """
    Compares SNA metrics between ground truth and prediction.
    Does NOT compare community partitions directly (NMI/ARI done in main loop).
    Calculates difference in the *number* of communities found.
    """
    comparison = {}

    # Basic properties difference
    comparison['diff_num_nodes'] = metrics_pred.get('num_nodes', 0) - metrics_gt.get('num_nodes', 0)
    comparison['diff_num_edges'] = metrics_pred.get('num_edges', 0) - metrics_gt.get('num_edges', 0)
    comparison['diff_density'] = metrics_pred.get('density', 0) - metrics_gt.get('density', 0)
    comparison['diff_avg_clustering'] = metrics_pred.get('avg_clustering', 0) - metrics_gt.get('avg_clustering', 0)
    comparison['diff_transitivity'] = metrics_pred.get('transitivity', 0) - metrics_gt.get('transitivity', 0)
    comparison['diff_diameter_lcc'] = metrics_pred.get('diameter_lcc', -1) - metrics_gt.get('diameter_lcc', -1)
    # print("gt diameter_lcc", metrics_gt.get('diameter_lcc', -1))
    # print("pred diameter_lcc", metrics_pred.get('diameter_lcc', -1))

    # Centrality mean differences
    comparison['diff_degree_mean'] = metrics_pred.get('degree_mean', 0) - metrics_gt.get('degree_mean', 0)
    comparison['diff_in_degree_mean'] = metrics_pred.get('in_degree_mean', 0) - metrics_gt.get('in_degree_mean', 0)
    comparison['diff_out_degree_mean'] = metrics_pred.get('out_degree_mean', 0) - metrics_gt.get('out_degree_mean', 0)
    comparison['diff_betweenness_mean'] = metrics_pred.get('betweenness_mean', 0) - metrics_gt.get('betweenness_mean', 0)
    comparison['diff_closeness_mean'] = metrics_pred.get('closeness_mean', 0) - metrics_gt.get('closeness_mean', 0)
    comparison['diff_local_clustering_mean'] = metrics_pred.get('local_clustering_mean', 0) - metrics_gt.get('local_clustering_mean', 0)

    # NEW: Difference in number of communities
    comparison['diff_num_communities_louvain'] = metrics_pred.get('num_communities_louvain', -1) - metrics_gt.get('num_communities_louvain', -1)
    comparison['diff_num_communities_lpa'] = metrics_pred.get('num_communities_lpa', -1) - metrics_gt.get('num_communities_lpa', -1)


    # Kolmogorov-Smirnov test for centrality and clustering distributions (p-value)
    for key in ['degree', 'in_degree', 'out_degree', 'betweenness', 'closeness', 'local_clustering']:
        gt_vals = metrics_gt.get(f'{key}_values', [])
        pred_vals = metrics_pred.get(f'{key}_values', [])
        if len(gt_vals) > 1 and len(pred_vals) > 1:
            if len(set(gt_vals)) > 1 or len(set(pred_vals)) > 1:
                 try:
                     stat, p_val = ks_2samp(gt_vals, pred_vals)
                     comparison[f'ks_{key}_pvalue'] = p_val
                 except ValueError: comparison[f'ks_{key}_pvalue'] = np.nan
            else: comparison[f'ks_{key}_pvalue'] = 1.0 if set(gt_vals) == set(pred_vals) else 0.0
        else: comparison[f'ks_{key}_pvalue'] = np.nan


    # Connected components difference
    comparison['diff_num_wcc'] = metrics_pred.get('num_weakly_connected_components', 0) - metrics_gt.get('num_weakly_connected_components', 0)
    comparison['diff_largest_wcc_size'] = metrics_pred.get('largest_weakly_connected_component_size', 0) - metrics_gt.get('largest_weakly_connected_component_size', 0)
    comparison['diff_num_scc'] = metrics_pred.get('num_strongly_connected_components', 0) - metrics_gt.get('num_strongly_connected_components', 0)
    comparison['diff_largest_scc_size'] = metrics_pred.get('largest_strongly_connected_component_size', 0) - metrics_gt.get('largest_strongly_connected_component_size', 0)

    return comparison


# --- Function to load data (unchanged from previous version) ---
def load_data_for_evaluation(metadata_path, data_dir):
    """Loads data from metadata JSON and prepares tensors for MLP evaluation."""
    print(f"Loading metadata from: {metadata_path}")
    try:
        with open(metadata_path, 'r') as f:
            metadata_list = json.load(f)
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {metadata_path}")
        return [], -1
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {metadata_path}")
        return [], -1

    print(f"Found {len(metadata_list)} graph entries.")
    eval_data_list = []
    skipped_count = 0
    target_max_nodes = -1

    # First pass to find the maximum max_nodes across all valid graphs (for padding)
    for metadata in metadata_list:
         relative_path = metadata.get('averaged_attention_matrix_path', None)
         gt_adj_orig = metadata.get('gt_adjacency', None)
         if relative_path is None or gt_adj_orig is None:
             continue
         current_max = metadata.get('max_nodes', 0)
         if current_max > target_max_nodes:
             target_max_nodes = current_max

    if target_max_nodes <= 0 and metadata_list:
        print("Error: Could not determine a valid max_nodes from metadata containing both attention and GT. Cannot proceed.")
        return [], -1
    elif target_max_nodes <= 0:
        print("Warning: No valid graph entries found in metadata.")
        return [], -1
    print(f"Determined target max_nodes for padding: {target_max_nodes}")

    for i, metadata in enumerate(tqdm(metadata_list, desc="Loading graph data for Evaluation")):
        relative_path = metadata.get('averaged_attention_matrix_path', None)
        gt_adj_orig = metadata.get('gt_adjacency', None)
        max_nodes_meta = metadata.get('max_nodes', None)

        if relative_path is None or gt_adj_orig is None or max_nodes_meta is None:
            skipped_count += 1
            continue
        attn_matrix_path = os.path.join(data_dir, RUN_PATH, relative_path)

        try:
            avg_attn_matrix = torch.load(attn_matrix_path).float()
            gt_adjacency_directed = torch.tensor(gt_adj_orig).float()
            current_max_nodes = max_nodes_meta

            if current_max_nodes <= 0:
                skipped_count += 1
                continue
            if current_max_nodes != gt_adjacency_directed.shape[0] or \
               current_max_nodes != gt_adjacency_directed.shape[1] or \
               current_max_nodes != avg_attn_matrix.shape[0] or \
               current_max_nodes != avg_attn_matrix.shape[1]:
                 skipped_count += 1
                 continue
            if target_max_nodes < current_max_nodes:
                 skipped_count += 1
                 continue

            attn_matrix_padded = avg_attn_matrix
            gt_adj_padded = gt_adjacency_directed
            if current_max_nodes < target_max_nodes:
                pad_size = target_max_nodes - current_max_nodes
                attn_matrix_padded = F.pad(avg_attn_matrix, (0, pad_size, 0, pad_size), "constant", 0)
                gt_adj_padded = F.pad(gt_adjacency_directed, (0, pad_size, 0, pad_size), "constant", 0)

            expected_shape = (target_max_nodes, target_max_nodes)
            if attn_matrix_padded.shape != expected_shape or gt_adj_padded.shape != expected_shape:
                 skipped_count += 1
                 continue

            attn_flat_padded = attn_matrix_padded.flatten()
            graph_id = metadata.get('graph_id', f'graph_{i}')
            eval_data_list.append((attn_flat_padded, gt_adj_padded, graph_id, current_max_nodes))

        except FileNotFoundError:
            skipped_count += 1
        except Exception as e:
            # print(f"Warning: Error processing graph {metadata.get('graph_id', i)}: {e}") # Keep for debugging
            skipped_count += 1

    print(f"Successfully prepared {len(eval_data_list)} graphs for Evaluation.")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} graphs due to errors or missing/inconsistent data.")
    if not eval_data_list:
        print("Error: No valid graph data could be prepared for evaluation.")
        return [], -1

    return eval_data_list, target_max_nodes


# --- Main execution block ---
if __name__ == "__main__":
    print("--- Starting MLP Evaluation Script (v3 with Community Detection) ---")
    print(f"Evaluating model: {MODEL_PATH}")
    print(f"Using metadata: {METADATA_PATH}")
    print(f"Output Directory: {EVALUATION_OUTPUT_DIR}")

    # Optional: Suppress warnings from libraries if needed
    # warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    os.makedirs(EVALUATION_OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Data
    all_eval_data, target_max_nodes = load_data_for_evaluation(METADATA_PATH, DATA_DIR)
    if not all_eval_data or target_max_nodes <= 0: exit()
    input_feature_size = target_max_nodes * target_max_nodes

    # Split Data
    if len(all_eval_data) < 2 or (TEST_SPLIT > 0 and len(all_eval_data) * TEST_SPLIT < 1):
        print("Warning: Not enough data for train/test split. Using all data for testing.")
        test_eval_data = all_eval_data
    else:
        try:
            _, test_eval_data = train_test_split(all_eval_data, test_size=TEST_SPLIT, random_state=RANDOM_SEED)
        except ValueError: test_eval_data = all_eval_data # Fallback if split fails
    print(f"Loaded {len(test_eval_data)} samples for evaluation (test split).")
    if not test_eval_data: exit("Test set is empty.")

    # DataLoader
    test_dataset = AdjacencyDatasetEval(test_eval_data)
    effective_batch_size = min(BATCH_SIZE, len(test_dataset))
    if effective_batch_size <= 0: exit("Error: Effective batch size is zero.")
    test_loader = DataLoader(test_dataset, batch_size=effective_batch_size, shuffle=False)

    # Load Model
    model = MLPPredictAdj(input_size=input_feature_size,
                          hidden_channels=HIDDEN_CHANNELS,
                          output_size=input_feature_size).to(device)
    if not os.path.exists(MODEL_PATH): exit(f"Error: Model not found at {MODEL_PATH}.")
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print(f"Successfully loaded trained model from {MODEL_PATH}")
    except Exception as e: exit(f"Error loading model state_dict: {e}.")


    print("\n--- Starting SNA Evaluation on Test Set ---")
    all_results = []
    edge_predictions_flat = []
    edge_ground_truth_flat = []


    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Evaluating Batches"):
            attn_flat_batch, gt_adj_padded_batch, graph_ids_batch, orig_n_nodes_batch = batch_data
            attn_flat_batch = attn_flat_batch.to(device)
            logits_batch = model(attn_flat_batch)
            probs_batch = torch.sigmoid(logits_batch)
            preds_adj_flat_batch = (probs_batch > PREDICTION_THRESHOLD).int()

            for i in range(attn_flat_batch.size(0)):
                graph_id = graph_ids_batch[i]
                try: original_n = int(orig_n_nodes_batch[i].item())
                except AttributeError: original_n = int(orig_n_nodes_batch[i])
                N = target_max_nodes

                gt_adj_padded = gt_adj_padded_batch[i].cpu().numpy()
                pred_adj_flat = preds_adj_flat_batch[i].cpu()
                pred_adj_padded = pred_adj_flat.view(N, N).numpy()

                gt_adj_sub = gt_adj_padded[:original_n, :original_n]
                pred_adj_sub = pred_adj_padded[:original_n, :original_n]

                # --- Edge Jaccard & Flattening ---
                if original_n > 0:
                    mask = np.ones((original_n, original_n), dtype=bool)
                    np.fill_diagonal(mask, 0)
                    gt_edges_masked = gt_adj_sub[mask]
                    pred_edges_masked = pred_adj_sub[mask]
                    edge_ground_truth_flat.extend(gt_edges_masked)
                    edge_predictions_flat.extend(pred_edges_masked)
                else:
                    gt_edges_masked = np.array([])
                    pred_edges_masked = np.array([])

                # --- Create NetworkX Graphs ---
                G_gt = nx.from_numpy_array(gt_adj_sub, create_using=nx.DiGraph)
                G_pred = nx.from_numpy_array(pred_adj_sub, create_using=nx.DiGraph)

                # --- Calculate SNA Metrics (including partitions) ---
                metrics_gt = calculate_sna_metrics(G_gt)
                metrics_pred = calculate_sna_metrics(G_pred)

                # --- Compare Basic Metrics (number of communities diff) ---
                comparison_metrics = compare_sna_metrics(metrics_gt, metrics_pred)

                # --- Community Partition Comparison (NMI, ARI) ---
                nmi_louvain, ari_louvain = np.nan, np.nan
                nmi_lpa, ari_lpa = np.nan, np.nan
                nodes_list = list(G_gt.nodes()) # Should be consistent for GT and Pred

                if original_n > 0 and nodes_list: # Only compare if graph is not empty
                    # Louvain Comparison
                    part_gt_lvn = metrics_gt.get('partition_louvain')
                    part_pred_lvn = metrics_pred.get('partition_louvain')
                    if part_gt_lvn is not None and part_pred_lvn is not None:
                        # Ensure all nodes are covered, use -1 for missing nodes if any (shouldn't happen with current setup)
                        labels_gt = [part_gt_lvn.get(node, -1) for node in nodes_list]
                        labels_pred = [part_pred_lvn.get(node, -1) for node in nodes_list]
                        # Check if labels are valid (not all -1)
                        if not all(l == -1 for l in labels_gt) and not all(l == -1 for l in labels_pred):
                             try:
                                 # NMI might throw warning if only one cluster found, handle it
                                 with warnings.catch_warnings():
                                     warnings.simplefilter("ignore", category=UserWarning)
                                     nmi_louvain = normalized_mutual_info_score(labels_gt, labels_pred, average_method='arithmetic')
                                 ari_louvain = adjusted_rand_score(labels_gt, labels_pred)
                             except ValueError as e:
                                 # print(f"NMI/ARI calculation error for Louvain on {graph_id}: {e}")
                                 pass # Keep as NaN

                    # LPA Comparison
                    part_gt_lpa = metrics_gt.get('partition_lpa')
                    part_pred_lpa = metrics_pred.get('partition_lpa')
                    if part_gt_lpa is not None and part_pred_lpa is not None:
                        labels_gt = [part_gt_lpa.get(node, -1) for node in nodes_list]
                        labels_pred = [part_pred_lpa.get(node, -1) for node in nodes_list]
                        if not all(l == -1 for l in labels_gt) and not all(l == -1 for l in labels_pred):
                            try:
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore", category=UserWarning)
                                    nmi_lpa = normalized_mutual_info_score(labels_gt, labels_pred, average_method='arithmetic')
                                ari_lpa = adjusted_rand_score(labels_gt, labels_pred)
                            except ValueError as e:
                                # print(f"NMI/ARI calculation error for LPA on {graph_id}: {e}")
                                pass # Keep as NaN

                # --- Store Results ---
                graph_results = {
                    'graph_id': graph_id,
                    # Store scalar metrics (exclude lists and partitions)
                    **{f'gt_{k}': v for k, v in metrics_gt.items() if not isinstance(v, (list, dict)) and k not in ['partition_louvain', 'partition_lpa']},
                    **{f'pred_{k}': v for k, v in metrics_pred.items() if not isinstance(v, (list, dict)) and k not in ['partition_louvain', 'partition_lpa']},
                    **comparison_metrics,
                    'nmi_louvain': nmi_louvain, # Add NMI/ARI scores
                    'ari_louvain': ari_louvain,
                    'nmi_lpa': nmi_lpa,
                    'ari_lpa': ari_lpa,
                }

                # Graph-specific edge Jaccard
                if original_n > 0 and gt_edges_masked.size > 0:
                    graph_results['edge_jaccard'] = jaccard_score(gt_edges_masked.astype(int), pred_edges_masked.astype(int), average='binary', zero_division=1.0)
                elif original_n > 0: graph_results['edge_jaccard'] = 1.0
                else: graph_results['edge_jaccard'] = 1.0

                all_results.append(graph_results)

    print("\n--- Evaluation Complete ---")

    # --- Aggregate and Summarize Results ---
    if not all_results: exit("No results were generated.")
    results_df = pd.DataFrame(all_results)

    # Overall Edge Jaccard
    if edge_ground_truth_flat:
        overall_edge_jaccard = jaccard_score(np.array(edge_ground_truth_flat).astype(int), np.array(edge_predictions_flat).astype(int), average='binary', zero_division=1.0)
        print(f"\nOverall Edge Jaccard Score (Micro, non-diagonal): {overall_edge_jaccard:.4f}")
    else: print("\nCould not calculate Overall Edge Jaccard Score.")

    # Average Metric Differences
    print("\n--- Average Metric Differences (Prediction - Ground Truth) ---")
    diff_cols = [col for col in results_df.columns if col.startswith('diff_')]
    avg_diffs = results_df[diff_cols].mean(skipna=True)
    print(avg_diffs)

    print("\n--- Average Absolute Metric Differences ---")
    abs_diffs = results_df[diff_cols].abs().mean(skipna=True)
    print(abs_diffs)

    # Average KS Test p-values
    print("\n--- Average KS Test p-values (Lower p-value means distributions likely differ) ---")
    ks_cols = [col for col in results_df.columns if col.startswith('ks_') and col.endswith('_pvalue')]
    avg_ks_pvalues = results_df[ks_cols].mean(skipna=True)
    print(avg_ks_pvalues)

    # Average Per-Graph Scores
    print("\n--- Average Per-Graph Scores (Macro Averages) ---")
    avg_graph_jaccard = results_df['edge_jaccard'].mean(skipna=True)
    avg_nmi_louvain = results_df['nmi_louvain'].mean(skipna=True)
    avg_ari_louvain = results_df['ari_louvain'].mean(skipna=True)
    avg_nmi_lpa = results_df['nmi_lpa'].mean(skipna=True)
    avg_ari_lpa = results_df['ari_lpa'].mean(skipna=True)
    print(f"Avg Edge Jaccard: {avg_graph_jaccard:.4f}")
    print(f"Avg NMI (Louvain): {avg_nmi_louvain:.4f}")
    print(f"Avg ARI (Louvain): {avg_ari_louvain:.4f}")
    print(f"Avg NMI (LPA):    {avg_nmi_lpa:.4f}")
    print(f"Avg ARI (LPA):    {avg_ari_lpa:.4f}")


    # Save detailed results
    results_save_path = os.path.join(EVALUATION_OUTPUT_DIR, 'sna_evaluation_results_v3.csv') # V3
    results_df.to_csv(results_save_path, index=False)
    print(f"\nDetailed evaluation results saved to: {results_save_path}")

    # --- Generate Plots ---
    print("\nGenerating comparison plots...")
    plot_dir = os.path.join(EVALUATION_OUTPUT_DIR, 'plots_v3') # V3
    os.makedirs(plot_dir, exist_ok=True)

    try:
        # --- Plot GT vs Predicted Distributions ---
        # ADDED: num_communities_louvain, num_communities_lpa
        base_metrics_for_dist_plot = [
            'num_edges', 'density',
            'avg_clustering', 'transitivity', 'local_clustering_mean',
            'degree_mean', 'in_degree_mean', 'out_degree_mean',
            'betweenness_mean', 'closeness_mean',
            'num_weakly_connected_components', 'largest_weakly_connected_component_size',
            'num_strongly_connected_components', 'largest_strongly_connected_component_size',
            'diameter_lcc',
            'num_communities_louvain', 'num_communities_lpa' # Added community counts
        ]

        print("Generating GT vs Predicted distribution plots...")
        for base_metric in base_metrics_for_dist_plot:
            gt_col = f'gt_{base_metric}'
            pred_col = f'pred_{base_metric}'
            if gt_col in results_df.columns and pred_col in results_df.columns:
                gt_data = results_df[gt_col].replace([np.inf, -np.inf, -1], np.nan).dropna() # Filter -1 (failure indicator)
                pred_data = results_df[pred_col].replace([np.inf, -np.inf, -1], np.nan).dropna()
                if gt_data.empty and pred_data.empty: continue

                plt.figure(figsize=(10, 6))
                if not gt_data.empty: sns.histplot(gt_data, color='skyblue', label='Ground Truth', kde=True, stat="density", linewidth=0, alpha=0.6)
                if not pred_data.empty: sns.histplot(pred_data, color='lightcoral', label='Predicted', kde=True, stat="density", linewidth=0, alpha=0.6)
                title_metric = base_metric.replace("_", " ").title()
                plt.title(f'Distribution Comparison: {title_metric}')
                plt.xlabel(f'{title_metric} Value')
                plt.ylabel('Density')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f'{base_metric}_distribution_comparison.png'))
                plt.close()

        # --- Scatter plot of GT vs Pred ---
        # ADDED: num_communities_louvain, num_communities_lpa
        print("Generating GT vs Predicted scatter plots...")
        scatter_metrics = [
            ('gt_num_edges', 'pred_num_edges'), ('gt_density', 'pred_density'),
            ('gt_avg_clustering', 'pred_avg_clustering'), ('gt_transitivity', 'pred_transitivity'),
            ('gt_diameter_lcc', 'pred_diameter_lcc'),
            ('gt_num_communities_louvain', 'pred_num_communities_louvain'), # Added community counts
            ('gt_num_communities_lpa', 'pred_num_communities_lpa')
         ]
        for gt_col, pred_col in scatter_metrics:
            if gt_col in results_df.columns and pred_col in results_df.columns:
                 valid_data = results_df[[gt_col, pred_col]].replace([np.inf, -np.inf, -1], np.nan).dropna()
                 if not valid_data.empty:
                     plt.figure(figsize=(6, 6))
                     jitter_amount = valid_data.std().min() * 0.05
                     if np.isfinite(jitter_amount) and jitter_amount > 0:
                          jittered_gt = valid_data[gt_col] + np.random.uniform(-jitter_amount, jitter_amount, size=len(valid_data))
                          jittered_pred = valid_data[pred_col] + np.random.uniform(-jitter_amount, jitter_amount, size=len(valid_data))
                     else: jittered_gt, jittered_pred = valid_data[gt_col], valid_data[pred_col]

                     plt.scatter(jittered_gt, jittered_pred, alpha=0.5, s=15)
                     min_val = min(valid_data[gt_col].min(), valid_data[pred_col].min())
                     max_val = max(valid_data[gt_col].max(), valid_data[pred_col].max())
                     plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.75, zorder=0, label='y=x')
                     title_metric = gt_col.replace("gt_", "").replace("_", " ").title()
                     plt.title(f'Predicted vs Ground Truth: {title_metric}')
                     plt.xlabel(f'Ground Truth {title_metric}')
                     plt.ylabel(f'Predicted {title_metric}')
                     plt.legend()
                     plt.grid(True, linestyle='--', alpha=0.6)
                     plt.tight_layout()
                     plt.savefig(os.path.join(plot_dir, f'{gt_col.replace("gt_", "")}_scatter.png'))
                     plt.close()

        # --- Histograms for Comparison Scores (Jaccard, NMI, ARI) ---
        print("Generating comparison score distribution plots...")
        score_cols = ['edge_jaccard', 'nmi_louvain', 'ari_louvain', 'nmi_lpa', 'ari_lpa']
        for score_col in score_cols:
             if score_col in results_df.columns:
                 score_data = results_df[score_col].dropna()
                 if not score_data.empty:
                     plt.figure(figsize=(8, 5))
                     sns.histplot(score_data, kde=True, bins=20)
                     title_score = score_col.replace("_", " ").upper()
                     plt.title(f'Distribution of {title_score} (Per Graph)')
                     plt.xlabel(f'{title_score}')
                     plt.ylabel('Frequency')
                     plt.xlim([-1.1, 1.1] if 'ari' in score_col else [-0.1, 1.1]) # Adjust xlim for ARI
                     plt.tight_layout()
                     plt.savefig(os.path.join(plot_dir, f'{score_col}_distribution.png'))
                     plt.close()

        print(f"Plots saved to: {plot_dir}")

    except Exception as e:
        print(f"Warning: Could not generate plots. Error: {e}")
        import traceback
        traceback.print_exc()

    print("--- Evaluation Script Finished ---")