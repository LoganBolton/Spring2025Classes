import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BCEWithLogitsLoss
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import pandas as pd
import time # For timing trials

# --- Configuration Parameters ---
RUN_PATH = 'attention_matrices/no_args_7_1b/combined'
METADATA_PATH = f'/Users/log/Github/Spring2025Classes/social_networks/final_project/{RUN_PATH}/combined_metadata.json'
DATA_DIR = f'/Users/log/Github/Spring2025Classes/social_networks/final_project/'
# --- Reduced EPOCHS for faster HPO demo ---
EPOCHS = 150 # Reduced epochs for HPO example, increase for real runs
# BATCH_SIZE = 8 # Will be defined in hyperparameter sets
TEST_SPLIT = 0.2
VAL_SPLIT = 0.1 # 10% of the original training data
RANDOM_SEED = 42
# --- MODIFIED: Output directory for HPO results ---
OUTPUT_DIR_BASE = f'{RUN_PATH}/mlp_baseline_hpo_output'
# --- End Configuration Parameters ---

# --- NEW: Define Hyperparameter Sets to Try ---
# Example sets: trying different learning rates and hidden channels
hyperparameter_sets = [
    {'lr': 0.003, 'hidden': 1024, 'batch_size': 8, 'dropout': 0.5},
    {'lr': 0.002, 'hidden': 2048, 'batch_size': 16, 'dropout': 0.6},
    {'lr': 0.002, 'hidden': 2048, 'batch_size': 16, 'dropout': 0.3},
    {'lr': 0.001, 'hidden': 2048, 'batch_size': 8, 'dropout': 0.5},
    {'lr': 0.002, 'hidden': 4096, 'batch_size': 8, 'dropout': 0.5},
    {'lr': 0.001, 'hidden': 4096, 'batch_size': 8, 'dropout': 0.5},
    {'lr': 0.001, 'hidden': 4096, 'batch_size': 16, 'dropout': 0.5},
    {'lr': 0.001, 'hidden': 8192, 'batch_size': 8, 'dropout': 0.5},
    # Add more combinations as needed
]

# Define the MLP Model (Allowing dropout configuration)
class MLPPredictAdj(nn.Module):
    def __init__(self, input_size, hidden_channels=128, output_size=None, dropout_p=0.5): # Added dropout_p
        super().__init__()
        if output_size is None:
            output_size = input_size # Predict flattened adjacency of same size
        self.fc1 = Linear(input_size, hidden_channels)
        self.fc2 = Linear(hidden_channels, hidden_channels)
        self.fc3 = Linear(hidden_channels, hidden_channels // 2)
        self.fc4 = Linear(hidden_channels // 2, hidden_channels // 4)
        self.fc_out = Linear(hidden_channels // 4, output_size)

        self.dropout = nn.Dropout(p=dropout_p) # Use configurable dropout

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

# --- Standard PyTorch Dataset (Unchanged) ---
class AdjacencyDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        attn_flat, adj_flat, graph_id = self.data_list[idx]
        return attn_flat, adj_flat, graph_id

# --- Function to load data for MLP (Unchanged) ---
def load_data_mlp(metadata_path, data_dir):
    # (Code is identical to the previous version - omitted for brevity)
    # ... (same loading logic as before) ...
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
            # print(f"Warning: 'averaged_attention_matrix_path' missing for graph {metadata.get('graph_id', i)}. Skipping.")
            skipped_count += 1
            continue
        attn_matrix_path = os.path.join(data_dir, RUN_PATH, relative_path)

        try:
            avg_attn_matrix = torch.load(attn_matrix_path).float()
            gt_adjacency = torch.tensor(metadata['gt_adjacency']).float()

            max_nodes = metadata.get('max_nodes', 0) # Original max_nodes for this graph

            # Basic validation before padding check
            if max_nodes <= 0:
                # print(f"Warning: Invalid max_nodes ({max_nodes}) for graph {metadata.get('graph_id', i)}. Skipping.")
                skipped_count += 1
                continue

            if target_max_nodes < max_nodes:
                #  print(f"Warning: Graph {metadata['graph_id']} has {max_nodes} nodes, exceeding determined target max_nodes {target_max_nodes}. Skipping.")
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
                #  print(f"Warning: Attention matrix shape mismatch for graph {metadata['graph_id']} after potential padding. Expected {expected_shape}, got {avg_attn_matrix.shape}. Skipping.")
                 skipped_count += 1
                 continue
            if gt_adjacency.shape != expected_shape:
                #  print(f"Warning: GT Adjacency shape mismatch for graph {metadata['graph_id']} after potential padding. Expected {expected_shape}, got {gt_adjacency.shape}. Skipping.")
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
            # print(f"Warning: Attention matrix file not found: {attn_matrix_path}. Skipping graph {metadata.get('graph_id', i)}.")
            skipped_count += 1
        except Exception as e:
            # print(f"Warning: Error processing graph {metadata.get('graph_id', i)}: {e}. Skipping.")
            skipped_count += 1

    print(f"Successfully prepared {len(mlp_data_list)} graphs for MLP.")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} graphs during loading.")
    if not mlp_data_list:
        print("Error: No valid graph data could be prepared for MLP.")
        return [], -1

    return mlp_data_list, target_max_nodes

# --- Training function for MLP (Unchanged) ---
def train(model, loader, optimizer, criterion, device):
    # (Code is identical to the previous version - omitted for brevity)
    # ... (same training logic as before) ...
    model.train()
    total_loss = 0
    num_samples = 0 # Keep track of total samples processed
    if not loader: # Handle empty loader case
        return 0.0
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

    return total_loss / num_samples if num_samples > 0 else 0.0


# --- Evaluation function for MLP (Unchanged) ---
@torch.no_grad()
def evaluate(model, loader, criterion, device, N, threshold=0.5):
    # (Code is identical to the previous version - omitted for brevity)
    # ... (same evaluation logic as before) ...
    model.eval()
    total_loss = 0
    all_preds_flat_masked = []
    all_targets_flat_masked = []
    num_samples = 0

    if not loader: # Handle empty loader case
        return 0.0, 0.0, 0.0

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

        try:
            loss = criterion(logits, adj_flat_batch) # Calculate loss on the full flattened output
            total_loss += loss.item() * batch_size
        except Exception as e:
             print(f"Error calculating loss in evaluate: {e}")
             print(f"Logits shape: {logits.shape}, Target shape: {adj_flat_batch.shape}")
             continue


        preds_prob = torch.sigmoid(logits)
        preds_binary = (preds_prob > threshold).int() # Shape [B, N*N]

        # Apply the flattened mask to ignore diagonal elements for metrics
        # Expand mask_flat to match batch size: [1, N*N] -> [B, N*N]
        batch_mask_flat = mask_flat.unsqueeze(0).expand_as(preds_binary)

        # Ensure mask is on the same device
        batch_mask_flat = batch_mask_flat.to(preds_binary.device)

        preds_masked = torch.masked_select(preds_binary, batch_mask_flat) # Select non-diagonal elements
        targets_masked = torch.masked_select(adj_flat_batch.int(), batch_mask_flat) # Select corresponding targets

        all_preds_flat_masked.append(preds_masked.cpu())
        all_targets_flat_masked.append(targets_masked.cpu())

        num_samples += batch_size

    if not all_preds_flat_masked or num_samples == 0: # Handle empty evaluation set
        # print("Warning: No evaluation samples processed.")
        return 0.0, 0.0, 0.0

    # Concatenate all masked (non-diagonal) results
    all_preds_tensor = torch.cat(all_preds_flat_masked, dim=0).numpy()
    all_targets_tensor = torch.cat(all_targets_flat_masked, dim=0).numpy()

    if len(all_targets_tensor) == 0: # Check if after masking, there's anything left
        return total_loss / num_samples if num_samples > 0 else 0.0, 0.0, 0.0


    # Calculate metrics across all flattened non-diagonal entries
    accuracy = accuracy_score(all_targets_tensor, all_preds_tensor)
    f1 = f1_score(all_targets_tensor, all_preds_tensor, average='binary', zero_division=0)

    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    return avg_loss, accuracy, f1


# --- NEW: Function to run a single training and validation trial ---
def run_training_trial(params, train_loader, val_loader, device, input_feature_size, target_max_nodes, epochs, trial_output_dir):
    """Trains and validates a model for one set of hyperparameters."""
    print(f"\n--- Starting Trial ---")
    print(f"Params: {params}")
    start_time = time.time()

    # Create output dir for this specific trial's artifacts (like best model)
    os.makedirs(trial_output_dir, exist_ok=True)

    # Instantiate model and optimizer with current trial's parameters
    model = MLPPredictAdj(input_size=input_feature_size,
                          hidden_channels=params['hidden'],
                          output_size=input_feature_size,
                          dropout_p=params['dropout']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    criterion = BCEWithLogitsLoss()

    best_trial_val_f1 = -1.0
    best_epoch_for_trial = -1
    trial_best_model_path = os.path.join(trial_output_dir, 'best_model_trial.pt')
    history = {'epoch':[], 'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_f1': []}


    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1 = 0.0, 0.0, 0.0
        if val_loader:
            val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device, N=target_max_nodes)
        else: # Handle case with no validation data
            val_loss, val_acc, val_f1 = float('nan'), float('nan'), float('nan')


        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        history['val_f1'].append(val_f1)

        # Simple print for progress within trial
        if epoch % 10 == 0 or epoch == epochs: # Print every 10 epochs or last epoch
            print(f'  Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}', end='')
            if val_loader:
                print(f', Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}', end='')
            print()


        # Save the best model *for this trial* based on validation F1
        if val_loader and val_f1 > best_trial_val_f1:
            best_trial_val_f1 = val_f1
            best_epoch_for_trial = epoch
            torch.save(model.state_dict(), trial_best_model_path)
            # print(f'    > New best Val F1 for trial: {best_trial_val_f1:.4f} at epoch {epoch}')


    end_time = time.time()
    print(f"--- Trial Finished ---")
    print(f"Best Validation F1 for this trial: {best_trial_val_f1:.4f} (at epoch {best_epoch_for_trial})")
    print(f"Trial Duration: {end_time - start_time:.2f} seconds")

    # Save trial history
    history_df = pd.DataFrame(history)
    history_save_path = os.path.join(trial_output_dir, 'trial_training_history.csv')
    history_df.to_csv(history_save_path, index=False)
    print(f"Trial history saved to {history_save_path}")

    return best_trial_val_f1, trial_best_model_path # Return best score and path to best model


# --- Main execution block ---
if __name__ == "__main__":
    print("--- Starting MLP Baseline Hyperparameter Optimization ---")
    print(f"Metadata Path: {METADATA_PATH}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Output Base Directory: {OUTPUT_DIR_BASE}")
    print(f"Epochs per trial: {EPOCHS}, Test Split: {TEST_SPLIT}, Val Split: {VAL_SPLIT}, Seed: {RANDOM_SEED}")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    # Output directory for the overall HPO process results
    os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load and Prepare Data ONCE ---
    all_data_mlp, determined_max_nodes = load_data_mlp(METADATA_PATH, DATA_DIR)

    if not all_data_mlp or determined_max_nodes <= 0:
        print("Failed to load data or determine max_nodes. Exiting.")
        exit()

    target_max_nodes = determined_max_nodes
    print(f"Model will be built for max_nodes = {target_max_nodes}")
    input_feature_size = target_max_nodes * target_max_nodes

    # --- Split data into train, validation, and test ONCE ---
    train_val_mlp_data, test_mlp_data = train_test_split(
        all_data_mlp, test_size=TEST_SPLIT, random_state=RANDOM_SEED
    )
    train_mlp_data, val_mlp_data = [], []
    if len(train_val_mlp_data) > 0:
        val_split_relative = VAL_SPLIT / (1.0 - TEST_SPLIT) if (1.0 - TEST_SPLIT) > 0 else 0
        if val_split_relative >= 1.0 or val_split_relative <= 0:
             print(f"Adjusted validation split ({val_split_relative:.2f}) invalid or covers all/no data. Using all for training.")
             train_mlp_data = train_val_mlp_data
             val_mlp_data = [] # Ensure val_mlp_data is empty list
        else:
             train_mlp_data, val_mlp_data = train_test_split(
                 train_val_mlp_data, test_size=val_split_relative, random_state=RANDOM_SEED
             )
    else:
         # train_val_mlp_data was empty
         test_mlp_data = all_data_mlp # Or handle as error, depending on requirement

    print(f"Split data: {len(train_mlp_data)} training, {len(val_mlp_data)} validation, {len(test_mlp_data)} test samples.")

    # Create Datasets ONCE
    train_dataset = AdjacencyDataset(train_mlp_data) if train_mlp_data else None
    val_dataset = AdjacencyDataset(val_mlp_data) if val_mlp_data else None
    test_dataset = AdjacencyDataset(test_mlp_data) if test_mlp_data else None

    # Note: DataLoaders might need recreation if batch_size is tuned per trial
    # We will create them inside the loop for simplicity here.
    test_loader = None # Create test loader once at the end
    if test_dataset:
        test_batch_size = min(hyperparameter_sets[0]['batch_size'], len(test_dataset)) # Use a default BS for now
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)


    # --- Loop Through Hyperparameter Sets ---
    best_overall_val_f1 = -1.0
    best_params = None
    best_model_path_overall = None
    hpo_results = []

    print(f"\n--- Starting Hyperparameter Search ({len(hyperparameter_sets)} trials) ---")

    for i, params in enumerate(hyperparameter_sets):
        trial_output_dir = os.path.join(OUTPUT_DIR_BASE, f'trial_{i+1}')

        # Create DataLoaders for this trial (using trial's batch size)
        current_batch_size = params['batch_size']
        train_loader = None
        if train_dataset:
             loader_batch_size = min(current_batch_size, len(train_dataset))
             if loader_batch_size > 0:
                 train_loader = DataLoader(train_dataset, batch_size=loader_batch_size, shuffle=True)

        val_loader = None
        if val_dataset:
             loader_batch_size = min(current_batch_size, len(val_dataset))
             if loader_batch_size > 0:
                 val_loader = DataLoader(val_dataset, batch_size=loader_batch_size, shuffle=False)


        # Run the training and validation trial
        trial_val_f1, trial_model_path = run_training_trial(
            params,
            train_loader,
            val_loader,
            device,
            input_feature_size,
            target_max_nodes,
            EPOCHS,
            trial_output_dir
        )

        # Store results
        result_entry = params.copy()
        result_entry['trial_num'] = i + 1
        result_entry['best_val_f1'] = trial_val_f1
        hpo_results.append(result_entry)

        # Check if this trial is the best so far
        if trial_val_f1 > best_overall_val_f1:
            print(f"*** Found new best hyperparameters (Val F1: {trial_val_f1:.4f})! ***")
            best_overall_val_f1 = trial_val_f1
            best_params = params
            best_model_path_overall = trial_model_path # Store path to the best model file

    print(f"\n--- Hyperparameter Search Complete ---")

    # Save HPO results summary
    hpo_results_df = pd.DataFrame(hpo_results)
    hpo_summary_path = os.path.join(OUTPUT_DIR_BASE, 'hpo_summary.csv')
    hpo_results_df.to_csv(hpo_summary_path, index=False)
    print(f"HPO summary saved to {hpo_summary_path}")

    if best_params:
        print(f"\nBest hyperparameters found:")
        print(best_params)
        print(f"Best Validation F1 achieved: {best_overall_val_f1:.4f}")
    else:
        print("\nNo successful trials completed or no validation data available.")
        exit() # Exit if no best model found

    # --- Final Evaluation on Test Set using the Overall Best Model ---
    print("\n--- Final Evaluation on Test Set using Best Hyperparameters ---")
    if best_model_path_overall and os.path.exists(best_model_path_overall) and test_loader:
        print(f"Loading best model from: {best_model_path_overall}")
        # Re-instantiate the model with the best parameters
        final_model = MLPPredictAdj(input_size=input_feature_size,
                                    hidden_channels=best_params['hidden'],
                                    output_size=input_feature_size,
                                    dropout_p=best_params['dropout']).to(device)
        try:
            final_model.load_state_dict(torch.load(best_model_path_overall, map_location=device))
            criterion = BCEWithLogitsLoss() # Need criterion for loss calc in evaluate

            # Adjust test loader batch size based on best params if needed
            final_test_batch_size = min(best_params['batch_size'], len(test_dataset))
            if final_test_batch_size <= 0: final_test_batch_size = 1 # Ensure batch size > 0
            final_test_loader = DataLoader(test_dataset, batch_size=final_test_batch_size, shuffle=False)


            final_loss, final_acc, final_f1 = evaluate(final_model, final_test_loader, criterion, device, N=target_max_nodes)
            print(f'Final Test Results - Loss: {final_loss:.4f}, Acc (non-diag): {final_acc:.4f}, F1 (non-diag): {final_f1:.4f}')

            # (Optional: Add prediction printing here using final_model and final_test_loader)
            # ... prediction printing code from previous version ...

        except Exception as e:
            print(f"Error during final evaluation: {e}")
            import traceback
            traceback.print_exc()
    elif not test_loader:
        print("Skipping final test evaluation as no test data was available.")
    elif not best_model_path_overall or not os.path.exists(best_model_path_overall):
         print(f"Skipping final test evaluation as the best model file was not found ('{best_model_path_overall}').")

    print("\n--- Script Finished ---")