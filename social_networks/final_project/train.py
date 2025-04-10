import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader # Use DataLoader for batching
from torch_geometric.nn import GCNConv
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import string

# Assume data_loader.py is in the same directory or accessible
from data_loader import GraphAttentionDataset, get_node_token_mapping, aggregate_attention, NODE_FEATURE_DIM

# --- Helper Function ---

def dense_attn_to_sparse(attn_matrix):
    """
    Converts a dense node-level attention matrix into sparse edge_index
    and edge_weight format for PyG GCN layers.
    Includes self-loops.
    """
    num_nodes = attn_matrix.shape[0]
    source_nodes = []
    target_nodes = []
    edge_weights = []

    # Add edges for all pairs (i, j), including self-loops (i, i)
    for i in range(num_nodes):
        for j in range(num_nodes):
            weight = attn_matrix[i, j].item()
            # Optional: Add a threshold to only include edges with significant attention
            # if weight > some_threshold:
            source_nodes.append(i)
            target_nodes.append(j)
            edge_weights.append(weight)

    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    edge_weight = torch.tensor(edge_weights, dtype=torch.float)
    
    # Normalize edge weights (optional but often helpful)
    # Example: row-normalize (sum of weights outgoing from a node is 1)
    # You might need scatter_add from torch_scatter for efficient normalization
    # For simplicity, let's skip normalization here, but consider it.
    # from torch_scatter import scatter_add
    # edge_weight_sum = scatter_add(edge_weight, edge_index[0], dim=0, dim_size=num_nodes)
    # edge_weight_sum = edge_weight_sum[edge_index[0]] # Expand sums back to edge shape
    # edge_weight = edge_weight / (edge_weight_sum + 1e-6) # Normalize

    return edge_index, edge_weight

def create_ground_truth_adj(edge_index, num_nodes):
    """Creates a dense ground truth adjacency matrix (torch tensor)."""
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    # Ensure edge_index is not empty before accessing elements
    if edge_index.numel() > 0:
         adj[edge_index[0], edge_index[1]] = 1.0
    return adj

# --- GCN Model Definition ---

class GraphReconstructionGCN(nn.Module):
    def __init__(self, num_node_features, hidden_dim, embedding_dim):
        super().__init__()
        # GCN layers to process graph structure based on attention
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embedding_dim)
        # No specific edge prediction head here; we'll use embeddings directly

    def forward(self, x, attn_edge_index, attn_edge_weight):
        """
        Forward pass to get node embeddings.

        Args:
            x (Tensor): Node features [num_nodes, num_node_features].
            attn_edge_index (LongTensor): Edge indices derived from attention matrix
                                          [2, num_attention_edges].
            attn_edge_weight (Tensor): Edge weights from attention matrix
                                       [num_attention_edges].

        Returns:
            Tensor: Node embeddings [num_nodes, embedding_dim].
        """
        h = self.conv1(x, attn_edge_index, attn_edge_weight)
        h = F.relu(h)
        h = F.dropout(h, p=0.5, training=self.training) # Dropout for regularization
        h = self.conv2(h, attn_edge_index, attn_edge_weight)
        # Output node embeddings directly
        return h

# --- Edge Predictor (using dot product) ---

class DotProductPredictor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, node_embeddings):
        """
        Predicts edge scores for all node pairs using dot product.

        Args:
            node_embeddings (Tensor): Node embeddings [num_nodes, embedding_dim].

        Returns:
            Tensor: Predicted edge scores (logits) [num_nodes, num_nodes].
        """
        # Calculate scores for all pairs (i, j)
        # score(i, j) = embedding_i^T * embedding_j
        scores = torch.matmul(node_embeddings, node_embeddings.t())
        return scores

# --- Training and Evaluation Functions ---

def train_epoch(gcn_model, predictor, loader, optimizer, criterion, device):
    gcn_model.train()
    predictor.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        # --- Prepare GCN inputs using the attention matrix ---
        # Handle potential batches (although loader might yield single graphs if batch_size=1)
        # PyG Batch object handles node features 'x' correctly.
        # We need to process attn_matrix for potentially multiple graphs in a batch.
        # For simplicity, let's assume batch_size=1 for now.
        # If using batch_size > 1, processing attn_matrix needs careful handling
        # of graph boundaries using data.batch or data.ptr.
        # The dense_attn_to_sparse needs to be adapted or called per graph.
        # --> Let's stick to batch_size=1 for this example implementation.
        if hasattr(data, 'batch') and data.batch is not None and data.batch.max() > 0:
            print("Warning: Batch size > 1 detected. This example primarily supports batch_size=1 for simplicity in handling attn_matrix.")
            # For batch > 1, you'd loop through graphs in the batch or adapt dense_attn_to_sparse
            # to handle the Batch object structure (using data.ptr). Skipping batch > 1 handling here.
            continue # Skip batches for now

        if not hasattr(data, 'attn_matrix'):
             print(f"Warning: Skipping data point missing 'attn_matrix'. Index in batch: {data}") # Might need more info
             continue

        attn_edge_index, attn_edge_weight = dense_attn_to_sparse(data.attn_matrix)
        attn_edge_index = attn_edge_index.to(device)
        attn_edge_weight = attn_edge_weight.to(device)

        # --- Forward pass ---
        node_embeddings = gcn_model(data.x, attn_edge_index, attn_edge_weight)
        predicted_scores = predictor(node_embeddings) # Shape: [N, N] (logits)

        # --- Prepare Ground Truth ---
        ground_truth_adj = create_ground_truth_adj(data.edge_index, data.num_nodes).to(device) # Shape: [N, N]

        # --- Calculate Loss ---
        # BCEWithLogitsLoss expects logits and targets of the same shape
        loss = criterion(predicted_scores, ground_truth_adj)

        # --- Backpropagation ---
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs # data.num_graphs is 1 if batch_size=1

    return total_loss / len(loader.dataset)

# threshold is how confident it has to be to make a prediction
def evaluate(gcn_model, predictor, loader, criterion, device, threshold=0.55):
    gcn_model.eval()
    predictor.eval()
    total_loss = 0
    all_preds = []
    all_true = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            # Similar batching assumption as in training
            if hasattr(data, 'batch') and data.batch is not None and data.batch.max() > 0:
                print("Warning: Skipping batch > 1 in evaluation.")
                continue
                
            if not hasattr(data, 'attn_matrix'):
                 print(f"Warning: Skipping data point missing 'attn_matrix' during evaluation.")
                 continue

            # Prepare GCN inputs
            attn_edge_index, attn_edge_weight = dense_attn_to_sparse(data.attn_matrix)
            attn_edge_index = attn_edge_index.to(device)
            attn_edge_weight = attn_edge_weight.to(device)

            # Forward pass
            node_embeddings = gcn_model(data.x, attn_edge_index, attn_edge_weight)
            predicted_scores = predictor(node_embeddings) # Shape: [N, N] (logits)

            # Prepare Ground Truth
            ground_truth_adj = create_ground_truth_adj(data.edge_index, data.num_nodes).to(device) # Shape: [N, N]

            # Calculate Loss
            loss = criterion(predicted_scores, ground_truth_adj)
            total_loss += loss.item() * data.num_graphs

            # Get Predictions
            # Apply sigmoid to get probabilities, then threshold
            predicted_probs = torch.sigmoid(predicted_scores)
            predicted_labels = (predicted_probs > threshold).float()

            # Store predictions and true labels (flattened)
            # Exclude diagonal (self-loops) from evaluation metrics? Optional.
            # For now, include them.
            all_preds.append(predicted_labels.view(-1).cpu().numpy())
            all_true.append(ground_truth_adj.view(-1).cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)

    # Concatenate predictions and true labels from all graphs
    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)

    # Calculate metrics
    # Use zero_division=0 to avoid warnings when a class isn't predicted/present
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true, all_preds, average='binary', zero_division=0
    )
    accuracy = accuracy_score(all_true, all_preds)

    return avg_loss, accuracy, precision, recall, f1

# --- Main Training Script ---

if __name__ == '__main__':
    # --- Configuration ---
    METADATA_PATH = '/Users/log/Github/Spring2025Classes/social_networks/final_project/attention_matrices/metadata.json'
    # ROOT_DIR should be the directory containing the 'attention_matrices' folder
    # If metadata paths are relative like "attention_matrices/avg_attn_0.npy",
    # ROOT_DIR should be the parent of 'attention_matrices'.
    # In your case, it seems ROOT_DIR is '/Users/log/Github/Spring2025Classes/social_networks/final_project/'
    # The DataLoader seems to handle path joining correctly if you set root_dir=""
    # and the paths in metadata are relative *to the location of metadata.json*
    # Let's try setting root_dir to the *parent* of the attention_matrices folder.
    ROOT_DIR = '/Users/log/Github/Spring2025Classes/social_networks/final_project' # Adjust if needed
    AGGREGATION_METHOD = 'average' # or 'max', 'sum'
    HIDDEN_DIM = 64          # Hidden dimension for GCN layers
    EMBEDDING_DIM = 32       # Output dimension of GCN (node embeddings)
    LEARNING_RATE = 0.005
    EPOCHS = 100             # Increase for real training
    BATCH_SIZE = 1           # Keep batch size 1 for simplicity with attn_matrix handling
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # --- Load Dataset ---
    print("Loading dataset...")
    try:
        # Pass the directory *containing* 'attention_matrices' as root_dir
        dataset = GraphAttentionDataset(
            metadata_path=METADATA_PATH,
            root_dir=ROOT_DIR, # This helps dataset locate the .npy files
            aggregation_method=AGGREGATION_METHOD
        )
    except FileNotFoundError as e:
        print(f"Error loading dataset: {e}")
        print(f"Please ensure METADATA_PATH ('{METADATA_PATH}') is correct.")
        print(f"And ROOT_DIR ('{ROOT_DIR}') correctly points to the parent directory of 'attention_matrices'.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during dataset loading: {e}")
        exit()

    if len(dataset) == 0:
        print("Dataset loaded, but contains no valid samples after processing.")
        print("Check warnings printed during dataset initialization.")
        exit()

    print(f"Dataset loaded with {len(dataset)} graphs.")

    # Optional: Split dataset (e.g., 80% train, 20% test)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # Use PyG DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) # No shuffle for test

    # --- Initialize Model, Optimizer, Loss ---
    # Assuming node features are identity matrices (one-hot encoding) created in the dataset
    first_data = dataset[0] # Get a sample to determine feature size
    num_node_features = first_data.num_node_features

    gcn_model = GraphReconstructionGCN(
        num_node_features=NODE_FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        embedding_dim=EMBEDDING_DIM
    ).to(DEVICE)

    predictor = DotProductPredictor().to(DEVICE)

    # Combine parameters from both models for the optimizer
    optimizer = optim.Adam(
        list(gcn_model.parameters()) + list(predictor.parameters()),
        lr=LEARNING_RATE
        )
        
    # Loss function for binary edge classification
    criterion = nn.BCEWithLogitsLoss() # Handles sigmoid internally, more stable

    # --- Training Loop ---
    print("\nStarting Training...")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(gcn_model, predictor, train_loader, optimizer, criterion, DEVICE)
        
        # Evaluate on test set periodically (e.g., every 10 epochs)
        if epoch % 10 == 0 or epoch == EPOCHS:
            test_loss, accuracy, precision, recall, f1 = evaluate(
                gcn_model, predictor, test_loader, criterion, DEVICE
            )
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | "
                  f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        else:
             print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f}")


    # --- Final Evaluation ---
    print("\nTraining finished. Evaluating on Test Set...")
    final_loss, final_accuracy, final_precision, final_recall, final_f1 = evaluate(
        gcn_model, predictor, test_loader, criterion, DEVICE
    )
    print("\n--- Final Test Metrics ---")
    print(f"Loss:      {final_loss:.4f}")
    print(f"Accuracy:  {final_accuracy:.4f}")
    print(f"Precision: {final_precision:.4f}")
    print(f"Recall:    {final_recall:.4f}")
    print(f"F1-Score:  {final_f1:.4f}")

    # --- Optional: Save the model ---
    # torch.save(gcn_model.state_dict(), 'gcn_model.pth')
    # torch.save(predictor.state_dict(), 'predictor_model.pth')
    # print("\nModels saved.")