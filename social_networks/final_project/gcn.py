import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Module, Linear, BCEWithLogitsLoss
from torch.optim import Adam
from torch_geometric.data import Dataset, Data, DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling, to_undirected
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import string
import time # For timing training
from torch_geometric.loader import dataloader


# --- 1. Configuration & Constants ---
METADATA_PATH = '/Users/log/Github/Spring2025Classes/social_networks/final_project/attention_matrices/metadata.json'
# Absolute path to the directory containing the .npy files, derived from METADATA_PATH
BASE_ATTN_DIR = os.path.dirname(METADATA_PATH)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# GCN Hyperparameters
HIDDEN_CHANNELS = 64
OUT_CHANNELS = 32 # Embedding size after GCN layers
LEARNING_RATE = 0.005
EPOCHS = 100 # Start with a reasonable number, adjust based on convergence
BATCH_SIZE = 16 # Adjust based on memory

# --- 2. Custom PyG Dataset ---

class AttentionGraphDataset(Dataset):
    def __init__(self, metadata_path, base_attn_dir, transform=None, pre_transform=None):
        super().__init__(transform, pre_transform)
        self.base_attn_dir = base_attn_dir
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Pre-calculate node name to index mapping (A=0, B=1, ...)
        self.node_names = list(string.ascii_uppercase)

    def len(self):
        return len(self.metadata)

    def get(self, idx):
        item = self.metadata[idx]
        num_nodes = item['num_nodes']

        # --- Ground Truth Edges ---
        source_nodes = torch.tensor(item['source'], dtype=torch.long)
        target_nodes = torch.tensor(item['target'], dtype=torch.long)
        # Ensure edge_index is [2, num_edges]
        ground_truth_edge_index = torch.stack([source_nodes, target_nodes], dim=0)

        # --- Node Features (Using simple identity matrix) ---
        # Requires fixed num_nodes per graph or padding/masking if variable
        # For simplicity, assume fixed max_nodes or handle variability
        # Let's use simple learnable embeddings initialized randomly later,
        # or identity features for now.
        # x = torch.eye(num_nodes, dtype=torch.float)
        # Using node indices as initial features - model can learn embeddings
        x = torch.arange(num_nodes, dtype=torch.float).unsqueeze(1) # Simple index features
        # Or use identity matrix if preferred:
        # x = torch.eye(num_nodes, dtype=torch.float)


        # --- Attention Matrix Processing ---
        attn_matrix_path = item['attention_matrix_path']
        try:
            # Load the pre-averaged attention matrix (seq_len, seq_len)
            raw_attn = np.load(attn_matrix_path)
            raw_attn = torch.from_numpy(raw_attn).float()
        except FileNotFoundError:
            print(f"Error: Attention matrix not found at {attn_matrix_path}")
            # Return None or raise error? Let's return None and handle it in DataLoader collation
            return None
        except Exception as e:
            print(f"Error loading {attn_matrix_path}: {e}")
            return None


        # Map tokens to node IDs (Simplified Approach)
        tokens = item['tokens']
        token_to_node_idx = {} # Store first token index for each node
        node_name_to_id = {name: i for i, name in enumerate(self.node_names[:num_nodes])}

        for i, token in enumerate(tokens):
            # Handle potential tokenization variations (e.g., ' A', 'A:', ':A')
            cleaned_token = token.strip().replace(":", "")
            if cleaned_token in node_name_to_id:
                node_id = node_name_to_id[cleaned_token]
                if node_id not in token_to_node_idx: # Store only the first occurrence
                    token_to_node_idx[node_id] = i

        # Create attention-derived adjacency (sparse format for PyG)
        attn_edge_index_list = []
        attn_edge_weight_list = []

        # Check if all nodes were found in tokens
        if len(token_to_node_idx) != num_nodes:
            print(f"Warning: Found tokens for {len(token_to_node_idx)} out of {num_nodes} nodes in graph {idx}.")
            found_node_ids = list(token_to_node_idx.keys())
            # Create lists first
            source_indices_list = []
            target_indices_list = []
            for src_node_id in found_node_ids:
                 for tgt_node_id in found_node_ids:
                      source_indices_list.append(src_node_id)
                      target_indices_list.append(tgt_node_id)
            # Convert lists to tensors
            source_indices = torch.tensor(source_indices_list, dtype=torch.long)
            target_indices = torch.tensor(target_indices_list, dtype=torch.long)

        else:
             # Create edges for all possible pairs if all nodes were found (already tensors)
             source_indices = torch.arange(num_nodes).repeat_interleave(num_nodes)
             target_indices = torch.arange(num_nodes).repeat(num_nodes)

        # Now, source_indices and target_indices are *always* tensors
        for i in range(len(source_indices)): # or source_indices.size(0)
            # .item() is now safe to call as we ensured they are tensors
            src_node_id = source_indices[i].item()
            tgt_node_id = target_indices[i].item()

            # Check if both nodes were found in the tokens
            # (This check might be slightly redundant now, but harmless)
            if src_node_id in token_to_node_idx and tgt_node_id in token_to_node_idx:
                src_token_idx = token_to_node_idx[src_node_id]
                tgt_token_idx = token_to_node_idx[tgt_node_id]

                # Check if indices are within bounds of the loaded attention matrix
                if src_token_idx < raw_attn.shape[0] and tgt_token_idx < raw_attn.shape[1]:
                    weight = raw_attn[src_token_idx, tgt_token_idx]
                    attn_edge_index_list.append([src_node_id, tgt_node_id])
                    attn_edge_weight_list.append(weight)
                else:
                     print(f"Warning: Token index out of bounds for graph {idx}. Src: {src_token_idx}, Tgt: {tgt_token_idx}, Shape: {raw_attn.shape}")

        if not attn_edge_index_list:
             print(f"Warning: No attention edges created for graph {idx}. Skipping?")
             # Handle empty graphs - maybe add self-loops with weight 0?
             # For now return None, filter later
             return None

        attn_edge_index = torch.tensor(attn_edge_index_list, dtype=torch.long).t()
        attn_edge_weight = torch.tensor(attn_edge_weight_list, dtype=torch.float)

        # Normalize attention weights (optional but often helpful)
        # Example: Row normalization
        # attn_edge_weight = attn_edge_weight / (torch.sum(attn_edge_weight) + 1e-6) # Simple normalization
        # Or use scatter_add for proper row normalization if needed

        data = Data(
            x=x,
            edge_index=ground_truth_edge_index, # Ground truth for supervision
            attn_edge_index=attn_edge_index,     # Structure derived from attention
            attn_edge_weight=attn_edge_weight, # Weights derived from attention
            num_nodes=num_nodes
        )
        return data

# Filter out None values potentially returned by get() if errors occurred
def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    # Default collate behavior if batch is not empty
    return dataloader.Collater(follow_batch=[], exclude_keys=[])(batch)


# --- 3. GCN Model Definition ---

class GCNEdgePredictor(Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # Node Embedding Layers using Attention Structure
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

        # Edge Predictor (simple dot product or MLP)
        # Using a simple linear layer approach for directed edges
        # self.predictor = Linear(out_channels * 2, 1) # For MLP predictor

    def encode(self, x, edge_index, edge_weight):
        # Use the attention-derived edges for message passing
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

    def decode(self, z, edge_label_index):
        # Predict scores for edges defined by edge_label_index
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        # Dot product for simplicity
        scores = (src * dst).sum(dim=-1)
        return scores

        # # MLP predictor example
        # combined = torch.cat([src, dst], dim=-1)
        # return self.predictor(combined).squeeze(-1)

    def forward(self, x, attn_edge_index, attn_edge_weight):
        # This forward is primarily for getting embeddings if needed separately
        # The main logic is split into encode and decode for link prediction tasks
        return self.encode(x, attn_edge_index, attn_edge_weight)

# --- 4. Training Loop ---

def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        if data is None: continue # Skip batch if collation resulted in None
        data = data.to(DEVICE)
        optimizer.zero_grad()

        # Ensure x has appropriate dimensions if using simple index features
        # If x was torch.arange(n).unsqueeze(1), GCN expects features per node.
        # We might need an Embedding layer if input_channels > 1
        # Or adjust GCN input_channels=1 if using index features directly
        # For torch.eye, in_channels = num_nodes (potentially large)
        # Let's assume node features 'x' are already suitable (e.g. torch.eye or learned embeddings)
        # We need to determine the 'in_channels' for GCN based on 'data.x'
        # If using torch.arange(num_nodes).unsqueeze(1), in_channels = 1
        # If using torch.eye(num_nodes), in_channels = data.num_nodes (can vary per batch!) - Requires different handling

        # Let's redefine x creation in Dataset to use random features or embeddings
        # In Dataset.get():
        # feature_dim = 16 # Or some other fixed dimension
        # x = torch.randn(num_nodes, feature_dim, dtype=torch.float)

        # Recalculate GCN in_channels based on chosen features
        # Assuming features are now fixed dim (e.g., 16 defined above, or based on data.x.size(1))

        z = model.encode(data.x, data.attn_edge_index, data.attn_edge_weight)

        # Prepare positive and negative edges for loss calculation
        # Use ground truth edges as positive examples
        pos_edge_index = data.edge_index

        # Sample negative edges (edges not present in the ground truth)
        if data.num_nodes > 1: # negative_sampling needs at least 2 nodes
             neg_edge_index = negative_sampling(
                 edge_index=pos_edge_index,
                 num_nodes=data.num_nodes,
                 num_neg_samples=pos_edge_index.size(1) # Sample as many negatives as positives
            )
        else: # Handle graphs with 0 or 1 node
             neg_edge_index = torch.empty((2,0), dtype=torch.long, device=DEVICE)


        # Combine positive and negative edges for scoring
        edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)

        # Create labels (1 for positive, 0 for negative)
        pos_label = torch.ones(pos_edge_index.size(1))
        neg_label = torch.zeros(neg_edge_index.size(1))
        edge_label = torch.cat([pos_label, neg_label], dim=0).to(DEVICE)

        # Decode scores for these edges
        out = model.decode(z, edge_label_index)

        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs # Multiply by batch size (num graphs in batch)

    return total_loss / len(loader.dataset) # Average loss per graph


# --- 5. Evaluation Function ---

@torch.no_grad()
def evaluate(model, loader, threshold=0.5):
    model.eval()
    all_preds = []
    all_labels = []

    for data in loader:
        if data is None: continue
        data = data.to(DEVICE)

        z = model.encode(data.x, data.attn_edge_index, data.attn_edge_weight)

        # Use ground truth positive edges
        pos_edge_index = data.edge_index
        
        # Sample negative edges for evaluation (can sample more exhaustively here if needed)
        if data.num_nodes > 1:
             neg_edge_index = negative_sampling(
                 edge_index=pos_edge_index,
                 num_nodes=data.num_nodes,
                 num_neg_samples=pos_edge_index.size(1) # Match positive count for balanced metrics, or sample more
            )
        else:
            neg_edge_index = torch.empty((2,0), dtype=torch.long, device=DEVICE)


        edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        pos_label = torch.ones(pos_edge_index.size(1))
        neg_label = torch.zeros(neg_edge_index.size(1))
        edge_label = torch.cat([pos_label, neg_label], dim=0).cpu() # Move labels to CPU for sklearn

        scores = model.decode(z, edge_label_index)
        preds = (torch.sigmoid(scores) > threshold).cpu() # Apply sigmoid and threshold

        all_preds.append(preds)
        all_labels.append(edge_label)

    if not all_labels:
        print("Warning: No data evaluated.")
        return {"precision": 0, "recall": 0, "f1": 0, "auc": 0}

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    # Calculate AUC using scores before thresholding
    # Need to collect raw scores if calculating AUC accurately
    # Re-running decode just for scores for simplicity here:
    auc_scores = []
    auc_labels = []
    for data in loader:
         if data is None: continue
         data = data.to(DEVICE)
         z = model.encode(data.x, data.attn_edge_index, data.attn_edge_weight)
         pos_edge_index = data.edge_index
         if data.num_nodes > 1:
             neg_edge_index = negative_sampling(pos_edge_index, data.num_nodes, pos_edge_index.size(1))
         else:
             neg_edge_index = torch.empty((2,0), dtype=torch.long, device=DEVICE)
         edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
         pos_label = torch.ones(pos_edge_index.size(1))
         neg_label = torch.zeros(neg_edge_index.size(1))
         edge_label = torch.cat([pos_label, neg_label], dim=0).cpu()
         
         scores = model.decode(z, edge_label_index).cpu()
         auc_scores.append(torch.sigmoid(scores)) # Use probabilities for AUC
         auc_labels.append(edge_label)
         
    if auc_scores:    
        auc_scores = torch.cat(auc_scores, dim=0).numpy()
        auc_labels = torch.cat(auc_labels, dim=0).numpy()
        auc = roc_auc_score(auc_labels, auc_scores) if len(np.unique(auc_labels)) > 1 else 0.5 # Handle cases with only one class
    else:
        auc = 0.0


    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }

# --- 6. Main Execution ---

if __name__ == '__main__':
    print("Loading dataset...")
    dataset = AttentionGraphDataset(METADATA_PATH, BASE_ATTN_DIR)

    # Filter out None items from the dataset before creating DataLoader
    valid_indices = [i for i, data in enumerate(dataset) if data is not None]
    if len(valid_indices) != len(dataset):
        print(f"Filtered out {len(dataset) - len(valid_indices)} graphs due to loading/processing errors.")
        dataset = dataset.index_select(valid_indices) # Create subset

    if len(dataset) == 0:
        print("Error: No valid graphs found in the dataset. Exiting.")
        exit()

    print(f"Dataset size: {len(dataset)}")

    # Determine input channels from the first valid graph's features
    # Make sure all graphs have the same feature dimension!
    # Add a check or ensure consistency in the Dataset class.
    # Example: Using fixed random features of size 16
    feature_dim = 16
    def set_random_features(data, feature_dim=16):
        if data is not None:
            data.x = torch.randn(data.num_nodes, feature_dim, dtype=torch.float)
        return data
    
    # Apply the feature setting transformation
    dataset.transform = lambda data: set_random_features(data, feature_dim)

    # Re-check dataset after transform
    valid_indices = [i for i, data in enumerate(dataset) if data is not None]
    dataset = dataset.index_select(valid_indices)
     
    if len(dataset) == 0:
        print("Error: No valid graphs after setting features. Exiting.")
        exit()
        
    print(f"Dataset size after feature setting: {len(dataset)}")

    # Split dataset (optional, simple sequential split for example)
    # A random split is generally better:
    # generator = torch.Generator().manual_seed(42)
    # train_size = int(0.8 * len(dataset))
    # val_size = len(dataset) - train_size
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)

    # Using sequential split for simplicity:
    split_idx = int(0.8 * len(dataset))
    train_dataset = dataset[:split_idx]
    val_dataset = dataset[split_idx:] # Use remaining for validation/testing

    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Error: Not enough data for train/validation split. Need more data.")
        # Adjust split or handle small dataset case
        if len(dataset)>0 :
             train_dataset = dataset # Use all for training if val is empty
             val_dataset = dataset # Use all for validation as well
             print("Warning: Using the entire dataset for both training and validation.")
        else:
             exit()


    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Initialize Model, Optimizer, Criterion
    # Determine in_channels based on the actual feature dimension used
    in_channels = feature_dim # Should match the dimension set in the transform
    model = GCNEdgePredictor(in_channels, HIDDEN_CHANNELS, OUT_CHANNELS).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = BCEWithLogitsLoss() # Suitable for binary edge prediction scores

    print("Starting training...")
    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        avg_loss = train(model, train_loader, optimizer, criterion)
        val_metrics = evaluate(model, val_loader)
        print(f'Epoch: {epoch:03d}, Loss: {avg_loss:.4f}, '
              f'Val Precision: {val_metrics["precision"]:.4f}, '
              f'Val Recall: {val_metrics["recall"]:.4f}, '
              f'Val F1: {val_metrics["f1"]:.4f}, '
              f'Val AUC: {val_metrics["auc"]:.4f}')
              
    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")

    # Final evaluation on the validation set
    final_metrics = evaluate(model, val_loader)
    print("\nFinal Validation Metrics:")
    print(f" Precision: {final_metrics['precision']:.4f}")
    print(f" Recall:    {final_metrics['recall']:.4f}")
    print(f" F1 Score:  {final_metrics['f1']:.4f}")
    print(f" AUC:       {final_metrics['auc']:.4f}")

    # --- Optional: Save the trained model ---
    # torch.save(model.state_dict(), 'gcn_attention_model.pth')
    # print("Model saved to gcn_attention_model.pth")