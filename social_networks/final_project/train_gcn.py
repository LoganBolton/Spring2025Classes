import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, BCEWithLogitsLoss
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader # Correct import
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm
import pandas as pd

# --- Configuration Parameters ---
RUN_PATH = 'attention_matrices/arg_3/combined'
METADATA_PATH = f'/Users/log/Github/Spring2025Classes/social_networks/final_project/{RUN_PATH}/combined_metadata.json'
DATA_DIR = f'/Users/log/Github/Spring2025Classes/social_networks/final_project/'
EPOCHS = 25
LEARNING_RATE = 0.001
BATCH_SIZE = 8
HIDDEN_CHANNELS = 32
TEST_SPLIT = 0.2
RANDOM_SEED = 42
OUTPUT_DIR = f'{RUN_PATH}/gcn_training_output_hardcoded'
# --- End Configuration Parameters ---


# Define the GCN Model
class GCNPredictAdj(torch.nn.Module):
    def __init__(self, num_nodes, hidden_channels=16):
        super().__init__()
        # Input feature dimension is num_nodes (since x is [N, N])
        self.conv1 = GCNConv(num_nodes, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # Output maps hidden features back to edge prediction scores for each node
        self.lin_out = Linear(hidden_channels, num_nodes)

    def forward(self, x, edge_index):
        # x shape: [B*N, N] after PyG batching (N=num_nodes)
        # edge_index shape: [2, num_edges_in_batch]
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        # h shape: [B*N, hidden_channels]
        adj_logits = self.lin_out(h)
        # adj_logits shape: [B*N, N]
        return adj_logits

# Function to load data
def load_data(metadata_path, data_dir):
    """Loads data from metadata JSON and prepares PyG Data objects."""
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
    pyg_data_list = []
    skipped_count = 0
    target_max_nodes = -1

    for metadata in metadata_list:
         current_max = metadata.get('max_nodes', 0)
         if current_max > target_max_nodes:
             target_max_nodes = current_max
    if target_max_nodes <= 0 and metadata_list:
        print("Error: Could not determine a valid max_nodes from metadata. Cannot proceed.")
        return [], -1
    print(f"Determined target max_nodes for padding: {target_max_nodes}")

    for i, metadata in enumerate(tqdm(metadata_list, desc="Loading graph data")):
        relative_path = metadata.get('averaged_attention_matrix_path', None)
        if relative_path is None:
            print(f"Warning: 'averaged_attention_matrix_path' missing for graph {metadata.get('graph_id', i)}. Skipping.")
            skipped_count += 1
            continue
        attn_matrix_path = os.path.join(data_dir, RUN_PATH, relative_path)

        try:
            avg_attn_matrix = torch.load(attn_matrix_path).float()
            gt_adjacency = torch.tensor(metadata['gt_adjacency']).float()

            max_nodes = metadata['max_nodes'] # Original max_nodes for this graph

            if target_max_nodes < max_nodes:
                 print(f"Warning: Graph {metadata['graph_id']} has {max_nodes} nodes, exceeding determined target max_nodes {target_max_nodes}. Skipping.")
                 skipped_count += 1
                 continue

            # Pad matrices if max_nodes < target_max_nodes
            if max_nodes < target_max_nodes:
                print("ahhhhh")
                # pad_size = target_max_nodes - max_nodes
                # avg_attn_matrix = F.pad(avg_attn_matrix, (0, pad_size, 0, pad_size), "constant", 0)
                # gt_adjacency = F.pad(gt_adjacency, (0, pad_size, 0, pad_size), "constant", 0)
            elif avg_attn_matrix.shape[0] != target_max_nodes or avg_attn_matrix.shape[1] != target_max_nodes:
                 print(f"Warning: Matrix shape mismatch for graph {metadata['graph_id']} even after potential padding. Expected ({target_max_nodes},{target_max_nodes}), got {avg_attn_matrix.shape}. Skipping.")
                 skipped_count += 1
                 continue
            elif gt_adjacency.shape[0] != target_max_nodes or gt_adjacency.shape[1] != target_max_nodes:
                 # This check might be redundant if padding covers it, but good sanity check
                 print(f"Warning: GT Adjacency shape mismatch for graph {metadata['graph_id']}. Expected ({target_max_nodes},{target_max_nodes}), got {gt_adjacency.shape}. Skipping.")
                 skipped_count += 1
                 continue


            node_features = avg_attn_matrix # Shape [target_max_nodes, target_max_nodes]

            adj_full = torch.ones((target_max_nodes, target_max_nodes), dtype=torch.long)
            adj_full.fill_diagonal_(0)
            edge_index = adj_full.nonzero(as_tuple=False).t().contiguous() # Shape [2, N*(N-1)]

            # *** RENAME 'y' to 'adj_target' ***
            adj_target = torch.clamp(gt_adjacency + gt_adjacency.t(), max=1) # Shape [target_max_nodes, target_max_nodes]

            # *** ADD SHAPE VERIFICATION PRINT ***
            # print(f"  Graph {metadata['graph_id']}: x shape: {node_features.shape}, adj_target shape: {adj_target.shape}, edge_index shape: {edge_index.shape}")

            # Ensure shapes are definitely correct before creating Data object
            expected_shape = (target_max_nodes, target_max_nodes)
            if node_features.shape != expected_shape or adj_target.shape != expected_shape:
                 print(f"Critical Error: Shape mismatch just before Data creation for graph {metadata['graph_id']}. x: {node_features.shape}, adj_target: {adj_target.shape}. Expected: {expected_shape}. Skipping.")
                 skipped_count += 1
                 continue

            data = Data(x=node_features, edge_index=edge_index, adj_target=adj_target)
            data.graph_id = metadata['graph_id']
            pyg_data_list.append(data)

        except FileNotFoundError:
            print(f"Warning: Attention matrix file not found: {attn_matrix_path}. Skipping graph {metadata.get('graph_id', i)}.")
            skipped_count += 1
        except Exception as e:
            print(f"Warning: Error processing graph {metadata.get('graph_id', i)}: {e}. Skipping.")
            skipped_count += 1

    print(f"Successfully loaded {len(pyg_data_list)} graphs.")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} graphs due to errors, missing paths, shape mismatches, or exceeding max_nodes.")
    if not pyg_data_list:
        print("Error: No valid graph data could be loaded.")
        return [], -1

    return pyg_data_list, target_max_nodes

# Training function
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    num_samples = 0 # Keep track of total samples processed
    for data in loader: # data is now a Batch object
        data = data.to(device)
        optimizer.zero_grad()

        # data.x is [B*N, N], data.edge_index is [2, num_edges_in_batch]
        logits = model(data.x, data.edge_index) # Output shape: [B*N, N]

        # *** Access target using 'adj_target' ***
        # data.adj_target will be shape [B*N, N] after collation if 'x' was handled that way
        # Need to reshape target to match logits [B*N, N] if criterion expects it
        target = data.adj_target.float() # Shape [B*N, N]

        # Ensure target and logits shapes match for loss calculation
        if logits.shape != target.shape:
             print(f"Shape mismatch Error in train: logits {logits.shape}, target {target.shape}")
             # Maybe reshape target? This depends on how collation actually worked.
             # If target was stacked differently, need to adjust.
             # Assuming standard node-feature-like collation for now.
             # If target is [B, N, N], need to reshape: target = target.view(-1, target.shape[-1])
             try:
                 target = target.view(logits.shape) # Attempt to reshape
             except RuntimeError as reshape_error:
                 print(f"Cannot reshape target {target.shape} to {logits.shape}. Collation might be unexpected. Error: {reshape_error}")
                 continue # Skip batch if shapes are fundamentally wrong

        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs # Loss per graph
        num_samples += data.num_graphs

    return total_loss / num_samples if num_samples > 0 else 0


# Evaluation function
@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0
    all_preds_flat = []
    all_targets_flat = []
    num_samples = 0

    for data in loader:
        data = data.to(device)

        logits_raw = model(data.x, data.edge_index) # Shape [B*N, N]

        # *** Access target using 'adj_target' ***
        target_raw = data.adj_target.float() # Shape [B*N, N] ??

        # Reshape logits and targets back to per-graph adjacency matrices [B, N, N]
        num_graphs = data.num_graphs
        num_nodes = data.x.shape[1] # N from input x [B*N, N] - This assumes x was collated correctly! Better: use target_max_nodes
        N = target_max_nodes # Use the known max_nodes
        if data.x.shape[0] != num_graphs * N:
             print(f"Warning: Unexpected shape for data.x in evaluate: {data.x.shape}. Expected num_nodes: {num_graphs * N}")
             # Fallback or error
             N = data.x.shape[1] # Use N from x as fallback
             # continue # Safer to skip batch if shapes look wrong

        # If target_raw collated same way as logits [B*N, N]
        if target_raw.shape == logits_raw.shape:
             logits = logits_raw.view(num_graphs, N, N)
             target = target_raw.view(num_graphs, N, N)
        # If target_raw somehow remained [B, N, N] (less likely with PyG default)
        elif target_raw.shape == (num_graphs, N, N):
             logits = logits_raw.view(num_graphs, N, N) # Still need to reshape logits
             target = target_raw # Already in correct shape
        else:
             print(f"Shape mismatch Error in evaluate: logits_raw {logits_raw.shape}, target_raw {target_raw.shape}")
             continue # Skip batch

        # Symmetrize logits for undirected prediction
        logits_sym = (logits + logits.permute(0, 2, 1)) / 2

        loss = criterion(logits_sym, target) # Loss requires [B, N, N] inputs if calculated per graph
        total_loss += loss.item() * num_graphs

        preds_prob = torch.sigmoid(logits_sym)
        preds_binary = (preds_prob > threshold).int()

        # Flatten predictions and targets, ignoring diagonal for metrics
        mask = torch.ones_like(target[0], dtype=torch.bool).fill_diagonal_(0).to(device) # Mask for one graph [N,N]

        for i in range(num_graphs):
             pred_flat = torch.masked_select(preds_binary[i], mask) # Shape [N*(N-1)]
             target_flat = torch.masked_select(target[i], mask)   # Shape [N*(N-1)]
             all_preds_flat.append(pred_flat.cpu())
             all_targets_flat.append(target_flat.cpu())

        num_samples += num_graphs

    if not all_preds_flat: # Handle empty evaluation set
        print("Warning: No evaluation samples processed.")
        return 0.0, 0.0, 0.0

    # Concatenate all flattened results
    all_preds_tensor = torch.cat(all_preds_flat, dim=0).numpy()
    all_targets_tensor = torch.cat(all_targets_flat, dim=0).numpy()

    # Calculate metrics across all flattened non-diagonal entries
    accuracy = accuracy_score(all_targets_tensor, all_preds_tensor)
    f1 = f1_score(all_targets_tensor, all_preds_tensor, average='binary', zero_division=0)

    # print("\n--- Evaluation Metrics ---")
    # print(f"Accuracy (element-wise, non-diagonal): {accuracy:.4f}")
    # print(f"F1 Score (element-wise, non-diagonal): {f1:.4f}")

    # try:
    #     print("\nClassification Report (element-wise, non-diagonal):")
    #     print(classification_report(all_targets_tensor, all_preds_tensor, target_names=['No Edge', 'Edge'], zero_division=0))
    # except ValueError as e:
    #     print(f"Could not generate classification report: {e}")

    avg_loss = total_loss / num_samples if num_samples > 0 else 0
    return avg_loss, accuracy, f1


# --- Main execution block ---
if __name__ == "__main__":
    print("--- Starting GCN Training Script ---")
    print(f"Metadata Path: {METADATA_PATH}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Epochs: {EPOCHS}, LR: {LEARNING_RATE}, Batch Size: {BATCH_SIZE}")
    print(f"Hidden Channels: {HIDDEN_CHANNELS}, Test Split: {TEST_SPLIT}, Seed: {RANDOM_SEED}")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Data - Pass target_max_nodes explicitly now
    all_data, determined_max_nodes = load_data(METADATA_PATH, DATA_DIR)

    if not all_data or determined_max_nodes <= 0:
        print("Failed to load data or determine max_nodes. Exiting.")
        exit()

    # Use the globally determined max_nodes for the model and evaluation logic
    global target_max_nodes # Make it globally accessible for evaluate function
    target_max_nodes = determined_max_nodes
    print(f"Model will be built for max_nodes = {target_max_nodes}")


    train_data, test_data = train_test_split(all_data, test_size=TEST_SPLIT, random_state=RANDOM_SEED)
    print(f"Split data: {len(train_data)} training samples, {len(test_data)} test samples.")

    test_batch_size = min(BATCH_SIZE, len(test_data)) if len(test_data) > 0 else 1
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

    model = GCNPredictAdj(num_nodes=target_max_nodes, hidden_channels=HIDDEN_CHANNELS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Use pos_weight if dataset is imbalanced (many more non-edges than edges)
    # Example: calculate pos_weight = num_neg_samples / num_pos_samples from training data
    criterion = BCEWithLogitsLoss() # Consider adding pos_weight later if needed

    print("\n--- Starting Training ---")
    best_test_f1 = -1.0
    best_epoch = -1
    history = {'epoch':[], 'train_loss': [], 'test_loss': [], 'test_accuracy': [], 'test_f1': []}

    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, train_loader, optimizer, criterion, device)

        test_loss, test_acc, test_f1 = 0.0, 0.0, 0.0
        if test_data:
            test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['test_accuracy'].append(test_acc)
        history['test_f1'].append(test_f1)

        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}', end='')
        if test_data:
             print(f', Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}')
        else:
             print(" (No test data for evaluation)")

        if test_data and test_f1 > best_test_f1:
            best_test_f1 = test_f1
            best_epoch = epoch
            model_save_path = os.path.join(OUTPUT_DIR, 'best_model.pt')
            torch.save(model.state_dict(), model_save_path)
            print(f"*** New best model saved to {model_save_path} (F1: {best_test_f1:.4f}) ***")
        elif not test_data and epoch == EPOCHS:
            model_save_path = os.path.join(OUTPUT_DIR, 'last_model.pt')
            torch.save(model.state_dict(), model_save_path)
            print(f"*** No test data. Saved last model to {model_save_path} ***")

    print(f"\n--- Training Complete ---")
    if test_data:
        print(f"Best Test F1 Score: {best_test_f1:.4f} at Epoch {best_epoch}")
    else:
        print("Training finished. No test data was available for validation.")

    history_df = pd.DataFrame(history)
    history_save_path = os.path.join(OUTPUT_DIR, 'training_history.csv')
    history_df.to_csv(history_save_path, index=False)
    print(f"Training history saved to {history_save_path}")

    if test_data and best_epoch != -1 :
        print("\n--- Final Evaluation on Test Set using Best Model ---")
        try:
            # Load best model state
            best_model_path = os.path.join(OUTPUT_DIR, 'best_model.pt')
            if os.path.exists(best_model_path):
                 model.load_state_dict(torch.load(best_model_path, map_location=device)) # Ensure loading to correct device
                 final_loss, final_acc, final_f1 = evaluate(model, test_loader, criterion, device)
                 print(f'Final Results - Test Loss: {final_loss:.4f}, Test Acc: {final_acc:.4f}, Test F1: {final_f1:.4f}')
            else:
                 print("Could not find best_model.pt for final evaluation.")

        except Exception as e:
            print(f"Error during final evaluation: {e}")

    elif not test_data:
         print("\nSkipping final evaluation as no test data was provided.")

        # --- Final Evaluation and Prediction Visualization ---
    if test_data and best_epoch != -1 :
        best_model_path = os.path.join(OUTPUT_DIR, 'best_model.pt')
        if os.path.exists(best_model_path):
            try:
                model.load_state_dict(torch.load(best_model_path, map_location=device))
                # --- Add Prediction Printing Here ---
                print("\n--- Sample Predictions from Test Set (using best model) ---")
                model.eval() # Ensure model is in eval mode
                threshold = 0.5 # Use the same threshold as in evaluation
                num_examples_to_print = 3 # How many examples to show

                printed_count = 0
                with torch.no_grad():
                    # Get a batch (or iterate through loader if needed)
                    for data in test_loader: # Iterate to get potentially different graphs
                        if printed_count >= num_examples_to_print:
                            break

                        data = data.to(device)
                        logits_raw = model(data.x, data.edge_index)
                        target_raw = data.adj_target.float() # Assuming [B*N, N] or similar

                        # Reshape back to [B, N, N]
                        num_graphs_in_batch = data.num_graphs
                        N = target_max_nodes # Use global max_nodes

                        # Defensive shape check and reshape
                        try:
                            logits = logits_raw.view(num_graphs_in_batch, N, N)
                            target = target_raw.view(num_graphs_in_batch, N, N)
                        except RuntimeError as e:
                            print(f"Skipping batch for printing due to reshape error: {e}. Logits shape: {logits_raw.shape}, Target shape: {target_raw.shape}")
                            continue

                        # Process logits to binary predictions
                        logits_sym = (logits + logits.permute(0, 2, 1)) / 2
                        preds_prob = torch.sigmoid(logits_sym)
                        preds_binary = (preds_prob > threshold).int()

                        # Iterate through graphs in the current batch
                        for i in range(num_graphs_in_batch):
                            if printed_count >= num_examples_to_print:
                                break

                            graph_idx_in_batch = i
                            # Try to get original graph ID if available and correctly collated
                            original_graph_id = data.graph_id[graph_idx_in_batch].item() if hasattr(data, 'graph_id') and data.graph_id is not None else f"Index {printed_count}"


                            print(f"\n--- Example {printed_count + 1} (Graph ID: {original_graph_id}) ---")

                            gt_matrix = target[graph_idx_in_batch].cpu().numpy().astype(int)
                            pred_matrix = preds_binary[graph_idx_in_batch].cpu().numpy()

                            print("Ground Truth Adjacency:")
                            print(gt_matrix)
                            print("\nPredicted Adjacency:")
                            print(pred_matrix)
                            print("-" * 20)

                            printed_count += 1
                        # Ensure outer loop breaks if count reached within inner loop
                        if printed_count >= num_examples_to_print:
                            break

            except Exception as e:
                print(f"Error during final evaluation or prediction printing: {e}")
        else:
             print("Could not find best_model.pt for final evaluation and prediction printing.")

    elif not test_data:
         print("\nSkipping final evaluation and prediction printing as no test data was provided.")

    print("--- Script Finished ---")