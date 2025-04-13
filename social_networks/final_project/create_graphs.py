import torch
import numpy as np
from transformers import LlamaForCausalLM, AutoTokenizer
from torch_geometric.data import Data
import random
import os
import json

# Load Llama Model
model_name = "meta-llama/Llama-3.2-1B"
# model_name = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(
    model_name,
    output_attentions=True,
    device_map="cpu",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)

# Utilitiy functions
def create_pertubations(source, target):
    sources, targets = [], []
    
    for i in range(len(source)):
        new_source = []
        new_target = []
        for k in range(len(source)):
            wrapped_idx = (i+k)%len(source)
            wrapped_source = source[wrapped_idx]
            wrapped_target = target[wrapped_idx]
            
            new_source.append(wrapped_source)
            new_target.append(wrapped_target)
        sources.append(new_source)
        targets.append(new_target)
    return sources, targets

def create_prompts(sources, targets):
    prompts = []
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    for i in range(len(sources)):
        source = sources[i]
        target = targets[i]
        prompt = ""
        for j in range(len(source)):
            source_node = alphabet[source[j]]
            target_node = alphabet[target[j]]
            prompt += f"{source_node}->{target_node}\n"
        prompts.append(prompt)
    return prompts

def create_gt_adjacency(source, target, num_nodes):
    # Find the size of the matrix
    n = num_nodes
    
    # Create an n×n matrix filled with zeros
    matrix = np.zeros((n, n), dtype=int)
    
    # Set the connections
    for s, t in zip(source, target):
        matrix[s][t] = 1
    
    return matrix
    
num_nodes = 3
P_CONNECTION = 0.25
NUM_GRAPHS = 1

source_dir = 'attention_matrices/example_fig'
metadata_path = f'{source_dir}/metadata.json'
if not os.path.exists(source_dir):
    os.makedirs(source_dir)


for graph_id in range(NUM_GRAPHS):
    # Generate random GT adjacency matrix
    # source = []
    # target = []
    # for node_1 in range(num_nodes):
    #     for node_2 in range(num_nodes-1):
    #         if node_1 == node_2:
    #             continue
    #         if random.random() < P_CONNECTION:
    #             source.append(node_1)
    #             target.append(node_2)
    source = [0, 1, 2]
    target = [1, 2, 0]
    
    sources, targets = create_pertubations(source, target)
    prompts = create_prompts(sources, targets)
    gt_adjacency = create_gt_adjacency(source, target, num_nodes)
    
    
    # Generate all variations of a graph
    for i in range(len(prompts)):
        source = sources[i]
        target = targets[i]
        prompt = prompts[i]

        inputs = tokenizer(prompt, return_tensors="pt")

        # Run inference and extract attention weights
        with torch.no_grad():
            outputs = model(**inputs)
        attentions = outputs.attentions

        ################ WORKING CODE NO TOP ARGS
        # Extract attention maps for all layers
        num_layers = len(attentions)
        attention_matrix = attentions[0][0].detach().numpy()

        for row in attention_matrix:
            top_indices = np.argpartition(row, -3)[-3:]
            top_values = row[top_indices]
            row.fill(0)  # Zero out the row
            row[top_indices] = top_values  # Set top 3 values

        # Average across heads
        avg_attention_matrix = np.mean(attention_matrix, axis=0)

        # Save the matrix to a file for later GCN training
        # attention_file_path = f'attention_matrices/arg_3_avg/avg_attn_{z}.npy'
        attention_file_path = f'{source_dir}/avg_attn_{graph_id}_{i}.npy'
        np.save(attention_file_path, avg_attention_matrix)
        #################### WORKING CODE
        
        ############# TOP K PER ROW #############
        # # 1. Select the attention tensor for the desired layer (e.g., layer 0)
        # # attentions[layer_index] shape: (batch_size, num_heads, seq_len, seq_len)
        # # We assume batch_size=1 here
        # TOP_K = 3
        # layer_attention = attentions[0][0] # Shape: (num_heads, seq_len, seq_len)
        # num_heads, seq_len, _ = layer_attention.shape

        # # 2. Create a tensor to store the filtered attention matrices for each head
        # filtered_layer_attention = torch.zeros_like(layer_attention)

        # # 3. Iterate through each head
        # for head_idx in range(num_heads):
        #     head_matrix = layer_attention[head_idx] # Shape: (seq_len, seq_len)

        #     # 4. Iterate through each row (query token) in the current head's matrix
        #     for row_idx in range(seq_len):
        #         row_attentions = head_matrix[row_idx] # Shape: (seq_len)

        #         # Check if k is greater than the number of elements in the row
        #         current_k = min(TOP_K, seq_len)
        #         if current_k <= 0: continue # Skip if row is empty or k is non-positive

        #         # 5. Find the top K values and their indices in this row
        #         # topk returns (values, indices)
        #         top_values, top_indices = torch.topk(row_attentions, k=current_k)

        #         # 6. Create a zero vector for the filtered row
        #         # filtered_row = torch.zeros_like(row_attentions) # Not needed if modifying filtered_layer_attention directly

        #         # 7. Place the top K values into the corresponding indices of the filtered matrix
        #         # Use scatter_ for potential efficiency or direct indexing
        #         # filtered_layer_attention[head_idx, row_idx, top_indices] = top_values # Direct indexing is often clearer
        #         filtered_layer_attention[head_idx, row_idx].scatter_(0, top_indices, top_values)


        # # 8. Average the filtered attention matrices across all heads
        # # filtered_layer_attention shape: (num_heads, seq_len, seq_len)
        # avg_filtered_attention = torch.mean(filtered_layer_attention, dim=0) # Average over dim 0 (heads)
        # # avg_filtered_attention shape: (seq_len, seq_len)

        # # 9. Convert to NumPy array for saving
        # avg_attention_matrix_np = avg_filtered_attention.cpu().detach().numpy()

        # # 10. Save the final averaged and filtered matrix
        # # Make sure graph_id and i (or z) are correctly defined for your loop
        # attention_file_path = f'{source_dir}/avg_attn_{graph_id}_{i}.npy' # Use i from your loop
        # np.save(attention_file_path, avg_attention_matrix_np)
        # print(f"Saved filtered and averaged attention matrix to: {attention_file_path}")
        ############# TOP K PER ROW #############

        base_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        # print(base_tokens)
        
        metadata_entry = {
            "graph_id": graph_id,
            "attention_matrix_path": attention_file_path,
            "max_nodes": num_nodes,
            "num_nodes": len(set(source+target)),
            "num_edges": len(source),
            "connection_probability": P_CONNECTION,
            "layer": '0',
            "head": 'all',
            "gt_adjacency": gt_adjacency.tolist(),
            "source": source,
            "target": target,
            "tokens": base_tokens,
            "prompt": prompt,
        }
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            if not isinstance(metadata, list):
                metadata = [metadata] 
            metadata.append(metadata_entry)
        else:
            metadata = [metadata_entry]
        
        # Write the updated metadata to file
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)