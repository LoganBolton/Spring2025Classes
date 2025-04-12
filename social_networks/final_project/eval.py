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
from sklearn.metrics import jaccard_score # For edge set comparison
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- Import necessary components from the training script ---
# Assuming the training script is named 'mlp_baseline_train.py'
# If it's in the same directory, you can potentially import directly,
# but it's safer to copy the necessary class definition here.

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
        self.fc_out = nn.Linear(hidden_channels // 4, output_size) # Original fc3 equivalent (output)

        self.dropout = nn.Dropout(p=0.5) # Dropout might not be needed for eval, but keep for consistency

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
RUN_PATH = 'attention_matrices/no_args_6/combined'
METADATA_PATH = f'/Users/log/Github/Spring2025Classes/social_networks/final_project/{RUN_PATH}/combined_metadata.json'
DATA_DIR = f'/Users/log/Github/Spring2025Classes/social_networks/final_project/'
BATCH_SIZE = 8 # Can be larger for evaluation
HIDDEN_CHANNELS = 2048 # Must match the trained model
TEST_SPLIT = 0.2 # Used to identify the same test set
RANDOM_SEED = 42
# --- Directory containing the trained model and output for this script ---
TRAINING_OUTPUT_DIR = f'{RUN_PATH}/mlp_baseline_training_output'
EVALUATION_OUTPUT_DIR = f'{RUN_PATH}/mlp_baseline_evaluation_output' # New output dir
MODEL_PATH = os.path.join(TRAINING_OUTPUT_DIR, 'best_model.pt') # Path to the saved model
PREDICTION_THRESHOLD = 0.5 # Threshold to convert sigmoid outputs to binary edges
# --- End Configuration Parameters ---

# --- Helper Functions for SNA ---

def calculate_sna_metrics(G):
    """Calculates various SNA metrics for a given NetworkX graph."""
    metrics = {}
    num_nodes = G.number_of_nodes()
    if num_nodes == 0:
        # Handle empty graphs gracefully
        metrics['num_nodes'] = 0
        metrics['num_edges'] = 0
        metrics['density'] = 0
        metrics['avg_clustering'] = 0
        metrics['degree_mean'] = 0
        metrics['degree_std'] = 0
        metrics['in_degree_mean'] = 0
        metrics['in_degree_std'] = 0
        metrics['out_degree_mean'] = 0
        metrics['out_degree_std'] = 0
        metrics['betweenness_mean'] = 0
        metrics['betweenness_std'] = 0
        metrics['closeness_mean'] = 0
        metrics['closeness_std'] = 0
        # metrics['eigenvector_mean'] = 0 # Can be problematic on disconnected graphs
        # metrics['eigenvector_std'] = 0
        metrics['num_weakly_connected_components'] = 0
        metrics['largest_weakly_connected_component_size'] = 0
        metrics['num_strongly_connected_components'] = 0
        metrics['largest_strongly_connected_component_size'] = 0
        return metrics

    metrics['num_nodes'] = num_nodes
    metrics['num_edges'] = G.number_of_edges()
    metrics['density'] = nx.density(G)

    # Clustering (for directed graphs, this is average clustering)
    try:
        metrics['avg_clustering'] = nx.average_clustering(G)
    except ZeroDivisionError:
         metrics['avg_clustering'] = 0.0 # Handle cases with no triangles

    # --- Centrality Measures (Calculate distributions) ---
    degree_dict = dict(G.degree())
    in_degree_dict = dict(G.in_degree())
    out_degree_dict = dict(G.out_degree())
    # Check if graph is non-empty before calculating centrality
    if num_nodes > 1 and metrics['num_edges'] > 0 :
         # Use k=min(num_nodes, 100) for approximation if graph is large
         k_approx = min(num_nodes, 100) if num_nodes > 100 else None
         try:
            betweenness_dict = nx.betweenness_centrality(G, k=k_approx, normalized=True, endpoints=False)
         except Exception: # Catch potential issues with disconnected graphs
            betweenness_dict = {n: 0.0 for n in G.nodes()}

         try:
            closeness_dict = nx.closeness_centrality(G)
         except Exception: # Catch potential issues with disconnected graphs
            closeness_dict = {n: 0.0 for n in G.nodes()}

        # # Eigenvector centrality can fail on graphs with multiple components or be slow
        # try:
        #     # Increase max_iter, tol might help convergence issues
        #     eigenvector_dict = nx.eigenvector_centrality_numpy(G, max_iter=500, tol=1e-04)
        # except (nx.NetworkXError, nx.PowerIterationFailedConvergence, np.linalg.LinAlgError) as e:
        #     print(f"Warning: Eigenvector centrality calculation failed: {e}. Setting to 0.")
        #     eigenvector_dict = {n: 0.0 for n in G.nodes()} # Assign 0 if fails

    else: # Handle single node or edgeless graph cases
         betweenness_dict = {n: 0.0 for n in G.nodes()}
         closeness_dict = {n: 0.0 for n in G.nodes()}
         # eigenvector_dict = {n: 0.0 for n in G.nodes()}


    # Store centralities as lists for distribution analysis later if needed
    metrics['degree_values'] = list(degree_dict.values())
    metrics['in_degree_values'] = list(in_degree_dict.values())
    metrics['out_degree_values'] = list(out_degree_dict.values())
    metrics['betweenness_values'] = list(betweenness_dict.values())
    metrics['closeness_values'] = list(closeness_dict.values())
    # metrics['eigenvector_values'] = list(eigenvector_dict.values())

    # Store mean and std for easier comparison
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
    # metrics['eigenvector_mean'] = np.mean(metrics['eigenvector_values']) if metrics['eigenvector_values'] else 0
    # metrics['eigenvector_std'] = np.std(metrics['eigenvector_values']) if metrics['eigenvector_values'] else 0


    # --- Connected Components ---
    # Weakly connected components (like undirected connectivity)
    wcc = list(nx.weakly_connected_components(G))
    metrics['num_weakly_connected_components'] = len(wcc)
    metrics['largest_weakly_connected_component_size'] = len(max(wcc, key=len)) if wcc else 0

    # Strongly connected components (paths in both directions)
    scc = list(nx.strongly_connected_components(G))
    metrics['num_strongly_connected_components'] = len(scc)
    metrics['largest_strongly_connected_component_size'] = len(max(scc, key=len)) if scc else 0


    return metrics

def compare_sna_metrics(metrics_gt, metrics_pred):
    """Compares SNA metrics between ground truth and prediction."""
    comparison = {}

    # Basic properties difference
    comparison['diff_num_nodes'] = metrics_pred.get('num_nodes', 0) - metrics_gt.get('num_nodes', 0)
    comparison['diff_num_edges'] = metrics_pred.get('num_edges', 0) - metrics_gt.get('num_edges', 0)
    comparison['diff_density'] = metrics_pred.get('density', 0) - metrics_gt.get('density', 0)
    comparison['diff_avg_clustering'] = metrics_pred.get('avg_clustering', 0) - metrics_gt.get('avg_clustering', 0)

    # Centrality mean differences
    comparison['diff_degree_mean'] = metrics_pred.get('degree_mean', 0) - metrics_gt.get('degree_mean', 0)
    comparison['diff_in_degree_mean'] = metrics_pred.get('in_degree_mean', 0) - metrics_gt.get('in_degree_mean', 0)
    comparison['diff_out_degree_mean'] = metrics_pred.get('out_degree_mean', 0) - metrics_gt.get('out_degree_mean', 0)
    comparison['diff_betweenness_mean'] = metrics_pred.get('betweenness_mean', 0) - metrics_gt.get('betweenness_mean', 0)
    comparison['diff_closeness_mean'] = metrics_pred.get('closeness_mean', 0) - metrics_gt.get('closeness_mean', 0)
    # comparison['diff_eigenvector_mean'] = metrics_pred.get('eigenvector_mean', 0) - metrics_gt.get('eigenvector_mean', 0)

    # Kolmogorov-Smirnov test for centrality distributions (p-value)
    # Low p-value suggests distributions are significantly different
    for key in ['degree', 'in_degree', 'out_degree', 'betweenness', 'closeness']: #, 'eigenvector']:
        gt_vals = metrics_gt.get(f'{key}_values', [])
        pred_vals = metrics_pred.get(f'{key}_values', [])
        if len(gt_vals) > 1 and len(pred_vals) > 1: # KS test requires at least 2 samples
             # ks test requires non-empty arrays
            if gt_vals and pred_vals:
                stat, p_val = ks_2samp(gt_vals, pred_vals)
                comparison[f'ks_{key}_pvalue'] = p_val
            else:
                comparison[f'ks_{key}_pvalue'] = None # Indicate test not possible
        else:
            comparison[f'ks_{key}_pvalue'] = None # Test not meaningful for < 2 samples

    # Connected components difference
    comparison['diff_num_wcc'] = metrics_pred.get('num_weakly_connected_components', 0) - metrics_gt.get('num_weakly_connected_components', 0)
    comparison['diff_largest_wcc_size'] = metrics_pred.get('largest_weakly_connected_component_size', 0) - metrics_gt.get('largest_weakly_connected_component_size', 0)
    comparison['diff_num_scc'] = metrics_pred.get('num_strongly_connected_components', 0) - metrics_gt.get('num_strongly_connected_components', 0)
    comparison['diff_largest_scc_size'] = metrics_pred.get('largest_strongly_connected_component_size', 0) - metrics_gt.get('largest_strongly_connected_component_size', 0)

    return comparison


# --- MODIFIED: Function to load data specifically for evaluation ---
def load_data_for_evaluation(metadata_path, data_dir):
    """
    Loads data from metadata JSON and prepares tensors for MLP evaluation.
    Crucially, it keeps the original directed GT adjacency and original node count.
    """
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
    eval_data_list = [] # List to store tuples (attn_flat_pad, gt_adj_pad, graph_id, orig_n_nodes)
    skipped_count = 0
    target_max_nodes = -1

    # First pass to find the maximum max_nodes across all valid graphs (for padding)
    for metadata in metadata_list:
         relative_path = metadata.get('averaged_attention_matrix_path', None)
         if relative_path is None:
             continue # Skip if path missing
         current_max = metadata.get('max_nodes', 0)
         if current_max > target_max_nodes:
             target_max_nodes = current_max

    if target_max_nodes <= 0 and metadata_list:
        print("Error: Could not determine a valid max_nodes from metadata. Cannot proceed.")
        return [], -1
    print(f"Determined target max_nodes for padding: {target_max_nodes}")

    for i, metadata in enumerate(tqdm(metadata_list, desc="Loading graph data for Evaluation")):
        relative_path = metadata.get('averaged_attention_matrix_path', None)
        gt_adj_orig = metadata.get('gt_adjacency', None)
        max_nodes = metadata.get('max_nodes', None)

        if relative_path is None or gt_adj_orig is None:
            print(f"Warning: Missing essential data (attn path, node list, or GT adj) for graph {metadata.get('graph_id', i)}. Skipping.")
            skipped_count += 1
            continue
        attn_matrix_path = os.path.join(data_dir, RUN_PATH, relative_path)

        try:
            avg_attn_matrix = torch.load(attn_matrix_path).float()
            # IMPORTANT: Use the original directed adjacency matrix
            gt_adjacency_directed = torch.tensor(gt_adj_orig).float()

            # Use max_nodes from metadata for *this specific graph* before padding
            current_max_nodes = metadata.get('max_nodes', 0)

            # Basic validation before padding check
            if current_max_nodes <= 0:
                print(f"Warning: Invalid max_nodes ({current_max_nodes}) for graph {metadata.get('graph_id', i)}. Skipping.")
                skipped_count += 1
                continue
            if max_nodes != gt_adjacency_directed.shape[0] or max_nodes != avg_attn_matrix.shape[0]:
                 print(f"Warning: Node list length ({max_nodes}) doesn't match matrix dimensions ({gt_adjacency_directed.shape[0]},{avg_attn_matrix.shape[0]}) for graph {metadata.get('graph_id', i)}. Skipping.")
                 skipped_count += 1
                 continue

            # Ensure this graph's max_nodes isn't larger than the overall target
            if target_max_nodes < current_max_nodes:
                 print(f"Warning: Graph {metadata['graph_id']} has max_nodes={current_max_nodes}, exceeding determined target max_nodes {target_max_nodes}. Skipping.")
                 skipped_count += 1
                 continue

            # Pad matrices to target_max_nodes if necessary
            attn_matrix_padded = avg_attn_matrix
            gt_adj_padded = gt_adjacency_directed
            if current_max_nodes < target_max_nodes:
                pad_size = target_max_nodes - current_max_nodes
                attn_matrix_padded = F.pad(avg_attn_matrix, (0, pad_size, 0, pad_size), "constant", 0)
                gt_adj_padded = F.pad(gt_adjacency_directed, (0, pad_size, 0, pad_size), "constant", 0)


            # Shape verification AFTER padding
            expected_shape = (target_max_nodes, target_max_nodes)
            if attn_matrix_padded.shape != expected_shape:
                 print(f"Warning: Padded Attention matrix shape mismatch for graph {metadata['graph_id']}. Expected {expected_shape}, got {attn_matrix_padded.shape}. Skipping.")
                 skipped_count += 1
                 continue
            if gt_adj_padded.shape != expected_shape:
                 print(f"Warning: Padded GT Adjacency shape mismatch for graph {metadata['graph_id']}. Expected {expected_shape}, got {gt_adj_padded.shape}. Skipping.")
                 skipped_count += 1
                 continue

            # Flatten the *padded* attention matrix for MLP input
            attn_flat_padded = attn_matrix_padded.flatten() # Shape [N*N]

            graph_id = metadata['graph_id'] # Store graph ID

            # Store the padded attention (flat), padded GT (matrix), graph ID, and original node count
            eval_data_list.append((attn_flat_padded, gt_adj_padded, graph_id, max_nodes))

        except FileNotFoundError:
            print(f"Warning: Attention matrix file not found: {attn_matrix_path}. Skipping graph {metadata.get('graph_id', i)}.")
            skipped_count += 1
        except Exception as e:
            print(f"Warning: Error processing graph {metadata.get('graph_id', i)}: {e}. Skipping.")
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
    print("--- Starting MLP Evaluation Script ---")
    print(f"Evaluating model: {MODEL_PATH}")
    print(f"Using metadata: {METADATA_PATH}")
    print(f"Output Directory: {EVALUATION_OUTPUT_DIR}")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    os.makedirs(EVALUATION_OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Data using the evaluation-specific function
    all_eval_data, target_max_nodes = load_data_for_evaluation(METADATA_PATH, DATA_DIR)

    if not all_eval_data or target_max_nodes <= 0:
        print("Failed to load data or determine max_nodes. Exiting.")
        exit()

    input_feature_size = target_max_nodes * target_max_nodes # MLP input is N*N

    # Split data to get the *same* test set as used during training
    # We don't need the training set here, just the test set.
    _, test_eval_data = train_test_split(all_eval_data, test_size=TEST_SPLIT, random_state=RANDOM_SEED)
    print(f"Loaded {len(test_eval_data)} samples for evaluation (test split).")

    if not test_eval_data:
        print("Test set is empty. Cannot perform evaluation. Exiting.")
        exit()

    # Create standard PyTorch Dataset and DataLoader for the test set
    test_dataset = AdjacencyDatasetEval(test_eval_data)
    # Handle potentially small test set batch size
    effective_batch_size = min(BATCH_SIZE, len(test_dataset))
    test_loader = DataLoader(test_dataset, batch_size=effective_batch_size, shuffle=False) # No shuffle for eval

    # Instantiate MLP model
    model = MLPPredictAdj(input_size=input_feature_size,
                          hidden_channels=HIDDEN_CHANNELS,
                          output_size=input_feature_size).to(device)

    # Load the trained model weights
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Trained model not found at {MODEL_PATH}. Exiting.")
        exit()
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval() # Set model to evaluation mode
        print(f"Successfully loaded trained model from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model state_dict: {e}. Exiting.")
        exit()


    print("\n--- Starting SNA Evaluation on Test Set ---")
    all_results = []
    edge_predictions_flat = []
    edge_ground_truth_flat = []


    with torch.no_grad():
        for attn_flat_batch, gt_adj_padded_batch, graph_ids_batch, orig_n_nodes_batch in tqdm(test_loader, desc="Evaluating Batches"):

            attn_flat_batch = attn_flat_batch.to(device)

            # Get model predictions (logits)
            logits_batch = model(attn_flat_batch) # Shape: [B, N*N]
            probs_batch = torch.sigmoid(logits_batch)
            preds_adj_flat_batch = (probs_batch > PREDICTION_THRESHOLD).int() # Shape: [B, N*N]

            # Process each graph in the batch
            for i in range(attn_flat_batch.size(0)):
                graph_id = graph_ids_batch[i].item()
                original_n = orig_n_nodes_batch[i].item()
                N = target_max_nodes # Padded size

                # Get padded GT and prediction for this sample
                gt_adj_padded = gt_adj_padded_batch[i].cpu().numpy() # Shape [N, N]
                pred_adj_flat = preds_adj_flat_batch[i].cpu() # Shape [N*N]
                pred_adj_padded = pred_adj_flat.view(N, N).numpy() # Shape [N, N]

                # --- Extract the actual subgraphs based on original_n ---
                gt_adj_sub = gt_adj_padded[:original_n, :original_n]
                pred_adj_sub = pred_adj_padded[:original_n, :original_n]

                # --- Store flat edges for Jaccard calculation (ignoring padding) ---
                # Only consider edges between the original nodes
                mask = np.ones((original_n, original_n), dtype=bool)
                np.fill_diagonal(mask, 0) # Exclude self-loops if desired (optional)

                edge_ground_truth_flat.extend(gt_adj_sub[mask])
                edge_predictions_flat.extend(pred_adj_sub[mask])


                # --- Create NetworkX graphs (Directed) ---
                # IMPORTANT: Use DiGraph as we assume the original graphs were directed
                G_gt = nx.from_numpy_array(gt_adj_sub, create_using=nx.DiGraph)
                G_pred = nx.from_numpy_array(pred_adj_sub, create_using=nx.DiGraph)

                # --- Calculate SNA metrics for both graphs ---
                metrics_gt = calculate_sna_metrics(G_gt)
                metrics_pred = calculate_sna_metrics(G_pred)

                # --- Compare the metrics ---
                comparison_metrics = compare_sna_metrics(metrics_gt, metrics_pred)

                # --- Store results ---
                graph_results = {
                    'graph_id': graph_id,
                    **{f'gt_{k}': v for k, v in metrics_gt.items() if not isinstance(v, list)}, # Store non-list metrics
                    **{f'pred_{k}': v for k, v in metrics_pred.items() if not isinstance(v, list)},
                    **comparison_metrics
                }
                # Add edge Jaccard Index for this specific graph
                if G_gt.number_of_nodes() > 0: # Avoid division by zero for empty graphs
                     gt_edges = set(G_gt.edges())
                     pred_edges = set(G_pred.edges())
                     intersection_len = len(gt_edges.intersection(pred_edges))
                     union_len = len(gt_edges.union(pred_edges))
                     graph_results['edge_jaccard'] = intersection_len / union_len if union_len > 0 else 1.0 # Perfect score if no edges in either
                else:
                    graph_results['edge_jaccard'] = 1.0 # Empty graph matches empty graph


                all_results.append(graph_results)

                # Optionally, save centrality distributions for later plotting (can consume memory)
                # graph_results['gt_degree_dist'] = metrics_gt.get('degree_values', [])
                # graph_results['pred_degree_dist'] = metrics_pred.get('degree_values', [])
                # ... add others if needed ...

    print("\n--- Evaluation Complete ---")

    # --- Aggregate and Summarize Results ---
    results_df = pd.DataFrame(all_results)

    # Calculate overall Edge Jaccard Score (Micro-average)
    # Note: jaccard_score needs binary labels {0, 1}
    if edge_ground_truth_flat: # Ensure list is not empty
        overall_edge_jaccard = jaccard_score(np.array(edge_ground_truth_flat).astype(int),
                                            np.array(edge_predictions_flat).astype(int),
                                            average='binary', # Treat as binary classification (edge exists or not)
                                            zero_division=0) # Return 0 if no true/pred edges
        print(f"\nOverall Edge Jaccard Score (Micro, non-diagonal): {overall_edge_jaccard:.4f}")
    else:
        print("\nCould not calculate Overall Edge Jaccard Score (no edges found).")


    # Calculate average differences and other summary stats
    print("\n--- Average Metric Differences (Prediction - Ground Truth) ---")
    diff_cols = [col for col in results_df.columns if col.startswith('diff_')]
    avg_diffs = results_df[diff_cols].mean()
    print(avg_diffs)

    print("\n--- Average Absolute Metric Differences ---")
    abs_diffs = results_df[diff_cols].abs().mean()
    print(abs_diffs)

    print("\n--- Average KS Test p-values (Lower p-value means distributions likely differ) ---")
    ks_cols = [col for col in results_df.columns if col.startswith('ks_') and col.endswith('_pvalue')]
    avg_ks_pvalues = results_df[ks_cols].mean(skipna=True) # skipna in case some tests weren't possible
    print(avg_ks_pvalues)

    print("\n--- Average Edge Jaccard Score (Macro) ---")
    avg_graph_jaccard = results_df['edge_jaccard'].mean()
    print(f"{avg_graph_jaccard:.4f}")


    # Save detailed results to CSV
    results_save_path = os.path.join(EVALUATION_OUTPUT_DIR, 'sna_evaluation_results.csv')
    results_df.to_csv(results_save_path, index=False)
    print(f"\nDetailed evaluation results saved to: {results_save_path}")

    # --- Optional: Generate Plots ---
    print("\nGenerating comparison plots...")
    plot_dir = os.path.join(EVALUATION_OUTPUT_DIR, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    try:
        # Plot distribution of differences for key metrics
        metrics_to_plot = ['diff_num_edges', 'diff_density', 'diff_avg_clustering',
                           'diff_degree_mean', 'diff_betweenness_mean', 'diff_closeness_mean',
                           'edge_jaccard']
        for metric in metrics_to_plot:
            if metric in results_df.columns:
                plt.figure(figsize=(8, 5))
                sns.histplot(results_df[metric], kde=True)
                plt.title(f'Distribution of {metric.replace("diff_", "Difference in ")}')
                plt.xlabel(metric)
                plt.ylabel('Frequency')
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f'{metric}_distribution.png'))
                plt.close()

        # Scatter plot of GT vs Pred for key metrics
        scatter_metrics = [('gt_num_edges', 'pred_num_edges'),
                           ('gt_density', 'pred_density'),
                           ('gt_avg_clustering', 'pred_avg_clustering')]
        for gt_col, pred_col in scatter_metrics:
            if gt_col in results_df.columns and pred_col in results_df.columns:
                 plt.figure(figsize=(6, 6))
                 # Add slight jitter to see overlapping points
                 jitter_amount = results_df[[gt_col, pred_col]].std().min() * 0.05
                 jittered_gt = results_df[gt_col] + np.random.uniform(-jitter_amount, jitter_amount, size=len(results_df))
                 jittered_pred = results_df[pred_col] + np.random.uniform(-jitter_amount, jitter_amount, size=len(results_df))
                 plt.scatter(jittered_gt, jittered_pred, alpha=0.5)
                 # Add y=x line
                 lims = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
                 plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
                 plt.title(f'Predicted vs Ground Truth: {gt_col.replace("gt_", "")}')
                 plt.xlabel(f'Ground Truth {gt_col.replace("gt_", "")}')
                 plt.ylabel(f'Predicted {pred_col.replace("pred_", "")}')
                 plt.axis('equal') # Ensure equal aspect ratio
                 plt.grid(True, linestyle='--', alpha=0.6)
                 plt.tight_layout()
                 plt.savefig(os.path.join(plot_dir, f'{gt_col.replace("gt_", "")}_scatter.png'))
                 plt.close()

        print(f"Plots saved to: {plot_dir}")

    except Exception as e:
        print(f"Warning: Could not generate plots. Error: {e}")


    print("--- Evaluation Script Finished ---")