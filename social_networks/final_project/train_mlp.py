import json
import os
import numpy as np
import torch
import torch.nn as nn # Use torch.nn
import torch.nn.functional as F
from torch.nn import Linear, BCEWithLogitsLoss # Keep Linear and Loss
# Removed torch_geometric imports
# from torch_geometric.nn import GCNConv
# from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader # Correct import
from torch.utils.data import Dataset, DataLoader # Use standard PyTorch DataLoader and Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm
import pandas as pd

# --- Configuration Parameters ---
RUN_PATH = 'attention_matrices/arg_3/combined'
METADATA_PATH = f'/Users/log/Github/Spring2025Classes/social_networks/final_project/{RUN_PATH}/combined_metadata.json'
DATA_DIR = f'/Users/log/Github/Spring2025Classes/social_networks/final_project/'
EPOCHS = 200
LEARNING_RATE = 0.002
BATCH_SIZE = 8
HIDDEN_CHANNELS = 2048 # MLP might need more capacity
TEST_SPLIT = 0.2
RANDOM_SEED = 42
# --- MODIFIED: Changed output directory name ---
OUTPUT_DIR = f'{RUN_PATH}/mlp_baseline_training_output'
# --- End Configuration Parameters ---


# Define the MLP Model
class MLPPredictAdj(nn.Module):
    def __init__(self, input_size, hidden_channels=128, output_size=None):
        super().__init__()
        if output_size is None:
            output_size = input_size # Predict flattened adjacency of same size
        self.fc1 = Linear(input_size, hidden_channels)
        self.fc2 = Linear(hidden_channels, hidden_channels)
        self.fc3 = Linear(hidden_channels, hidden_channels // 2)
        self.fc4 = Linear(hidden_channels // 2, hidden_channels// 4)
        self.fc_out = Linear(hidden_channels // 4, output_size) # Original fc3 equivalent (output)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # x shape: [B, N*N] (flattened attention matrix)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = F.relu(self.fc3(x))
        x = self.dropout(x)

        x = F.relu(self.fc4(x))

        # Output logits for flattened adjacency matrix
        adj_logits = self.fc_out(x) # Shape: [B, N*N]
        return adj_logits

# --- MODIFIED: Standard PyTorch Dataset ---
class AdjacencyDataset(Dataset):
    def __init__(self, data_list):
        # data_list should be a list of tuples: (flattened_attn_matrix, flattened_gt_adj, graph_id)
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        attn_flat, adj_flat, graph_id = self.data_list[idx]
        # Return tensors and potentially the graph_id if needed later
        return attn_flat, adj_flat, graph_id

# --- MODIFIED: Function to load data for MLP ---
def load_data_mlp(metadata_path, data_dir):
    """Loads data from metadata JSON and prepares flattened tensors for MLP."""
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
    mlp_data_list = [] # List to store tuples (attn_flat, adj_flat, graph_id)
    skipped_count = 0
    target_max_nodes = -1

    # First pass to find the maximum max_nodes across all valid graphs
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

    for i, metadata in enumerate(tqdm(metadata_list, desc="Loading graph data for MLP")):
        relative_path = metadata.get('averaged_attention_matrix_path', None)
        if relative_path is None:
            print(f"Warning: 'averaged_attention_matrix_path' missing for graph {metadata.get('graph_id', i)}. Skipping.")
            skipped_count += 1
            continue
        attn_matrix_path = os.path.join(data_dir, RUN_PATH, relative_path)

        try:
            avg_attn_matrix = torch.load(attn_matrix_path).float()
            gt_adjacency = torch.tensor(metadata['gt_adjacency']).float()

            max_nodes = metadata.get('max_nodes', 0) # Original max_nodes for this graph

            # Basic validation before padding check
            if max_nodes <= 0:
                print(f"Warning: Invalid max_nodes ({max_nodes}) for graph {metadata.get('graph_id', i)}. Skipping.")
                skipped_count += 1
                continue

            if target_max_nodes < max_nodes:
                 print(f"Warning: Graph {metadata['graph_id']} has {max_nodes} nodes, exceeding determined target max_nodes {target_max_nodes}. Skipping.")
                 skipped_count += 1
                 continue

            # Pad matrices if necessary BEFORE checking shapes
            if max_nodes < target_max_nodes:
                pad_size = target_max_nodes - max_nodes
                avg_attn_matrix = F.pad(avg_attn_matrix, (0, pad_size, 0, pad_size), "constant", 0)
                gt_adjacency = F.pad(gt_adjacency, (0, pad_size, 0, pad_size), "constant", 0)
                # Update max_nodes after padding for shape check consistency
                max_nodes = target_max_nodes

            # Shape verification AFTER padding
            expected_shape = (target_max_nodes, target_max_nodes)
            if avg_attn_matrix.shape != expected_shape:
                 print(f"Warning: Attention matrix shape mismatch for graph {metadata['graph_id']} after potential padding. Expected {expected_shape}, got {avg_attn_matrix.shape}. Skipping.")
                 skipped_count += 1
                 continue
            if gt_adjacency.shape != expected_shape:
                 print(f"Warning: GT Adjacency shape mismatch for graph {metadata['graph_id']} after potential padding. Expected {expected_shape}, got {gt_adjacency.shape}. Skipping.")
                 skipped_count += 1
                 continue

            # Flatten the matrices for MLP input/output
            attn_flat = avg_attn_matrix.flatten() # Shape [N*N]
            # Ensure GT is undirected for target (like in GCN version)
            adj_target_undirected = torch.clamp(gt_adjacency + gt_adjacency.t(), max=1)
            adj_flat = adj_target_undirected.flatten() # Shape [N*N]

            graph_id = metadata['graph_id'] # Store graph ID

            mlp_data_list.append((attn_flat, adj_flat, graph_id))

        except FileNotFoundError:
            print(f"Warning: Attention matrix file not found: {attn_matrix_path}. Skipping graph {metadata.get('graph_id', i)}.")
            skipped_count += 1
        except Exception as e:
            print(f"Warning: Error processing graph {metadata.get('graph_id', i)}: {e}. Skipping.")
            skipped_count += 1

    print(f"Successfully prepared {len(mlp_data_list)} graphs for MLP.")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} graphs due to errors, missing paths, shape mismatches, or exceeding max_nodes.")
    if not mlp_data_list:
        print("Error: No valid graph data could be prepared for MLP.")
        return [], -1

    return mlp_data_list, target_max_nodes

# --- MODIFIED: Training function for MLP ---
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    num_samples = 0 # Keep track of total samples processed
    for attn_flat_batch, adj_flat_batch, _ in loader: # Unpack batch, ignore graph_id here
        attn_flat_batch = attn_flat_batch.to(device)
        adj_flat_batch = adj_flat_batch.to(device) # Target shape [B, N*N]

        optimizer.zero_grad()
        logits = model(attn_flat_batch) # Output shape: [B, N*N]

        # Ensure target and logits shapes match for loss calculation
        if logits.shape != adj_flat_batch.shape:
             print(f"Shape mismatch Error in train: logits {logits.shape}, target {adj_flat_batch.shape}")
             continue # Skip batch if shapes are wrong

        loss = criterion(logits, adj_flat_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * attn_flat_batch.size(0) # Loss per sample in batch
        num_samples += attn_flat_batch.size(0)

    return total_loss / num_samples if num_samples > 0 else 0


# --- MODIFIED: Evaluation function for MLP ---
@torch.no_grad()
def evaluate(model, loader, criterion, device, N, threshold=0.5): # Pass N (max_nodes)
    model.eval()
    total_loss = 0
    all_preds_flat_masked = []
    all_targets_flat_masked = []
    num_samples = 0

    # Create the diagonal mask once
    mask = torch.ones((N, N), dtype=torch.bool).fill_diagonal_(0).to(device)
    mask_flat = mask.flatten() # Flattened mask [N*N]

    for attn_flat_batch, adj_flat_batch, _ in loader: # Unpack batch
        attn_flat_batch = attn_flat_batch.to(device)
        adj_flat_batch = adj_flat_batch.to(device) # Target shape [B, N*N]
        batch_size = attn_flat_batch.size(0)

        logits = model(attn_flat_batch) # Shape [B, N*N]

        # Ensure target and logits shapes match for loss calculation
        if logits.shape != adj_flat_batch.shape:
             print(f"Shape mismatch Error in evaluate: logits {logits.shape}, target {adj_flat_batch.shape}")
             continue # Skip batch

        loss = criterion(logits, adj_flat_batch) # Calculate loss on the full flattened output
        total_loss += loss.item() * batch_size

        preds_prob = torch.sigmoid(logits)
        preds_binary = (preds_prob > threshold).int() # Shape [B, N*N]

        # Apply the flattened mask to ignore diagonal elements for metrics
        # Expand mask_flat to match batch size: [1, N*N] -> [B, N*N]
        batch_mask_flat = mask_flat.unsqueeze(0).expand_as(preds_binary)

        preds_masked = torch.masked_select(preds_binary, batch_mask_flat) # Select non-diagonal elements
        targets_masked = torch.masked_select(adj_flat_batch, batch_mask_flat) # Select corresponding targets

        all_preds_flat_masked.append(preds_masked.cpu())
        all_targets_flat_masked.append(targets_masked.cpu())

        num_samples += batch_size

    if not all_preds_flat_masked: # Handle empty evaluation set
        print("Warning: No evaluation samples processed.")
        return 0.0, 0.0, 0.0

    # Concatenate all masked (non-diagonal) results
    all_preds_tensor = torch.cat(all_preds_flat_masked, dim=0).numpy()
    all_targets_tensor = torch.cat(all_targets_flat_masked, dim=0).numpy()

    # Calculate metrics across all flattened non-diagonal entries
    accuracy = accuracy_score(all_targets_tensor, all_preds_tensor)
    f1 = f1_score(all_targets_tensor, all_preds_tensor, average='binary', zero_division=0)

    avg_loss = total_loss / num_samples if num_samples > 0 else 0
    return avg_loss, accuracy, f1


# --- Main execution block ---
if __name__ == "__main__":
    print("--- Starting MLP Baseline Training Script ---") # Changed title
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

    # Load Data using MLP-specific function
    all_data_mlp, determined_max_nodes = load_data_mlp(METADATA_PATH, DATA_DIR)

    if not all_data_mlp or determined_max_nodes <= 0:
        print("Failed to load data or determine max_nodes. Exiting.")
        exit()

    # Use the globally determined max_nodes for the model and evaluation logic
    target_max_nodes = determined_max_nodes # Keep this name for consistency
    print(f"Model will be built for max_nodes = {target_max_nodes}")
    input_feature_size = target_max_nodes * target_max_nodes # MLP input is N*N


    # Split data (list of tuples)
    train_mlp_data, test_mlp_data = train_test_split(all_data_mlp, test_size=TEST_SPLIT, random_state=RANDOM_SEED)
    print(f"Split data: {len(train_mlp_data)} training samples, {len(test_mlp_data)} test samples.")

    # Create standard PyTorch Datasets
    train_dataset = AdjacencyDataset(train_mlp_data)
    test_dataset = AdjacencyDataset(test_mlp_data)

    # Create standard PyTorch DataLoaders
    test_batch_size = min(BATCH_SIZE, len(test_dataset)) if len(test_dataset) > 0 else 1
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # Handle empty test set for loader
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False) if test_dataset else None


    # Instantiate MLP model
    model = MLPPredictAdj(input_size=input_feature_size,
                          hidden_channels=HIDDEN_CHANNELS,
                          output_size=input_feature_size).to(device) # Output size is also N*N
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = BCEWithLogitsLoss() # Same loss function is suitable

    print("\n--- Starting Training ---")
    best_test_f1 = -1.0
    best_epoch = -1
    history = {'epoch':[], 'train_loss': [], 'test_loss': [], 'test_accuracy': [], 'test_f1': []}

    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, train_loader, optimizer, criterion, device)

        test_loss, test_acc, test_f1 = 0.0, 0.0, 0.0
        # Check if test_loader exists and has data
        if test_loader and len(test_loader.dataset) > 0:
            test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device, N=target_max_nodes)
            history['test_loss'].append(test_loss)
            history['test_accuracy'].append(test_acc)
            history['test_f1'].append(test_f1)
        else:
            # Append placeholder values if no test data
             history['test_loss'].append(float('nan'))
             history['test_accuracy'].append(float('nan'))
             history['test_f1'].append(float('nan'))


        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)


        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}', end='')
        if test_loader and len(test_loader.dataset) > 0:
             print(f', Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}')
             # Save best model based on Test F1
             if test_f1 > best_test_f1:
                best_test_f1 = test_f1
                best_epoch = epoch
                model_save_path = os.path.join(OUTPUT_DIR, 'best_model.pt')
                torch.save(model.state_dict(), model_save_path)
                print(f"*** New best model saved to {model_save_path} (F1: {best_test_f1:.4f}) ***")

        else:
             print(" (No test data for evaluation)")
             # Save last model if no test data
             if epoch == EPOCHS:
                 model_save_path = os.path.join(OUTPUT_DIR, 'last_model.pt')
                 torch.save(model.state_dict(), model_save_path)
                 print(f"*** No test data. Saved last model to {model_save_path} ***")


    print(f"\n--- Training Complete ---")
    if test_loader and len(test_loader.dataset) > 0:
        print(f"Best Test F1 Score: {best_test_f1:.4f} at Epoch {best_epoch}")
    else:
        print("Training finished. No test data was available for validation.")

    # Save training history
    history_df = pd.DataFrame(history)
    history_save_path = os.path.join(OUTPUT_DIR, 'training_history.csv')
    history_df.to_csv(history_save_path, index=False)
    print(f"Training history saved to {history_save_path}")

    # --- Final Evaluation and Prediction Visualization ---
    # Check if we had test data and a best model was saved
    if test_loader and len(test_loader.dataset) > 0 and best_epoch != -1:
        print("\n--- Final Evaluation on Test Set using Best Model ---")
        best_model_path = os.path.join(OUTPUT_DIR, 'best_model.pt')
        if os.path.exists(best_model_path):
            try:
                model.load_state_dict(torch.load(best_model_path, map_location=device))
                final_loss, final_acc, final_f1 = evaluate(model, test_loader, criterion, device, N=target_max_nodes)
                print(f'Final Results - Test Loss: {final_loss:.4f}, Test Acc (non-diag): {final_acc:.4f}, Test F1 (non-diag): {final_f1:.4f}')

                # --- Add Prediction Printing Here ---
                print("\n--- Sample Predictions from Test Set (using best model) ---")
                model.eval() # Ensure model is in eval mode
                threshold = 0.5 # Use the same threshold as in evaluation
                num_examples_to_print = 3 # How many examples to show

                printed_count = 0
                # Store graph IDs seen during printing to avoid duplicates if batch size > 1
                printed_graph_ids = set()

                with torch.no_grad():
                    # Iterate through loader to get different graphs if needed
                    for attn_flat_batch, adj_flat_batch, graph_ids_batch in test_loader:
                        if printed_count >= num_examples_to_print:
                            break

                        attn_flat_batch = attn_flat_batch.to(device)
                        # adj_flat_batch = adj_flat_batch.to(device) # Target needed for GT

                        logits = model(attn_flat_batch) # Shape [B, N*N]
                        preds_prob = torch.sigmoid(logits)
                        preds_binary_flat = (preds_prob > threshold).int() # Shape [B, N*N]

                        # Iterate through samples in the batch
                        for i in range(attn_flat_batch.size(0)):
                            current_graph_id = graph_ids_batch[i].item() # Get graph ID

                            # Skip if we already printed this graph ID or enough examples
                            if current_graph_id in printed_graph_ids or printed_count >= num_examples_to_print:
                                continue

                            print(f"\n--- Example {printed_count + 1} (Graph ID: {current_graph_id}) ---")

                            # Reshape target and prediction back to N x N matrices
                            N = target_max_nodes
                            gt_matrix_flat = adj_flat_batch[i]
                            pred_matrix_flat = preds_binary_flat[i]

                            gt_matrix = gt_matrix_flat.view(N, N).cpu().numpy().astype(int)
                            pred_matrix = pred_matrix_flat.view(N, N).cpu().numpy()

                            print("Ground Truth Adjacency (Undirected):")
                            print(gt_matrix)
                            print("\nPredicted Adjacency:")
                            print(pred_matrix)
                            print("-" * 20)

                            printed_graph_ids.add(current_graph_id)
                            printed_count += 1

                        # Ensure outer loop breaks if count reached within inner loop
                        if printed_count >= num_examples_to_print:
                            break

            except Exception as e:
                print(f"Error during final evaluation or prediction printing: {e}")
        else:
             print("Could not find best_model.pt for final evaluation and prediction printing.")

    elif not (test_loader and len(test_loader.dataset) > 0):
         print("\nSkipping final evaluation and prediction printing as no test data was provided.")

    print("--- Script Finished ---")