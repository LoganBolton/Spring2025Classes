# model.py
"""
Contains the GCN model definition for edge prediction, using aggregated
attention matrices for message passing and learnable node embeddings.
"""

import torch
import torch.nn.functional as F
from torch.nn import Module, Embedding, Linear # Linear might be needed for other decoders
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse

class GCNEdgePredictorWithEmbeddings(Module):
    """
    GCN model with initial learnable node embeddings, using attention-derived
    matrices for message passing and predicting edge existence.
    """
    def __init__(self, num_nodes, embedding_dim, hidden_channels, out_channels):
        """
        Args:
            num_nodes (int): The maximum number of nodes in any graph in the dataset.
                             Used to size the Embedding layer.
            embedding_dim (int): The dimensionality of the initial node embeddings.
            hidden_channels (int): The number of channels in the hidden GCN layer.
            out_channels (int): The dimensionality of the final node embeddings (output of GCN).
        """
        super().__init__()
        # Learnable embeddings for each node index (0 to num_nodes-1)
        self.embedding = Embedding(num_nodes, embedding_dim)

        # GCN layers
        self.conv1 = GCNConv(embedding_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels) # Output node embeddings
        
        # Optional: Initialize weights? (e.g., Xavier initialization)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        self.embedding.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def encode(self, node_indices, attn_matrix):
        """
        Generates embeddings, then runs GCN layers using attention matrix.
        Args:
            node_indices (Tensor): Node indices [num_nodes] (e.g., torch.arange(num_nodes)).
            attn_matrix (Tensor): Aggregated node-level attention [num_nodes, num_nodes].
        Returns:
            Tensor: Node embeddings [num_nodes, out_channels].
        """
        # Get initial node features from embeddings
        # Make sure node_indices are within the bounds of the Embedding layer
        if node_indices.max() >= self.embedding.num_embeddings:
             raise ValueError(f"Node index {node_indices.max()} is out of bounds for embedding layer size {self.embedding.num_embeddings}")
        x = self.embedding(node_indices)

        # Convert attention matrix to sparse graph structure
        # Use try-except for potential issues in dense_to_sparse if attn_matrix is weird
        try:
             attn_edge_index, attn_edge_weight = dense_to_sparse(attn_matrix)
        except Exception as e:
             print(f"Error during dense_to_sparse conversion: {e}")
             print(f"Attention matrix shape: {attn_matrix.shape}")
             # Handle error appropriately, maybe return zero embeddings or raise
             raise e # Re-raise for now

        # Ensure edge weights are non-negative for stability
        attn_edge_weight = F.relu(attn_edge_weight) 

        # Pass through GCN layers
        x = self.conv1(x, attn_edge_index, attn_edge_weight)
        x = F.relu(x)
        # Consider adding dropout only during training if needed later
        # x = F.dropout(x, p=0.5, training=self.training) 
        x = self.conv2(x, attn_edge_index, attn_edge_weight)
        return x

    def decode(self, z, edge_label_index):
        """ 
        Predicts edge scores using dot product between node embeddings.
        Args:
            z (Tensor): Node embeddings [num_nodes, out_channels].
            edge_label_index (Tensor): Edge indices for prediction [2, num_edges_to_predict].
        Returns:
            Tensor: Predicted edge scores (logits) [num_edges_to_predict].
        """
        # Check if edge_label_index is empty
        if edge_label_index.numel() == 0:
            return torch.empty(0, device=z.device) 
            
        # Ensure indices in edge_label_index are valid for z
        max_idx = edge_label_index.max()
        if max_idx >= z.shape[0]:
             raise ValueError(f"Index {max_idx} in edge_label_index out of bounds for node embeddings shape {z.shape}")

        node_i_emb = z[edge_label_index[0]]
        node_j_emb = z[edge_label_index[1]]
        # Calculate dot product
        scores = (node_i_emb * node_j_emb).sum(dim=-1)
        return scores

    # Note: We don't define a forward pass here that takes a `Data` object directly.
    # The training loop will typically call encode() and decode() separately.
    # This provides more flexibility, especially with negative sampling.