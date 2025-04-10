import torch
import numpy as np
from transformers import LlamaForCausalLM, AutoTokenizer
from torch_geometric.data import Data
import random
import os
import json

model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(
    model_name,
    output_attentions=True,
    device_map="cpu",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)


num_nodes = 10
max_possible_connections = num_nodes*(num_nodes-1)
P_CONNECTION = 0.25
NUM_GRAPHS = 1
# metadata_path = 'attention_matrices/arg_3_avg/metadata.json'
metadata_path = 'attention_matrices/demo1/metadata.json'


for z in range(NUM_GRAPHS):
    # source = []
    # target = []
    # for node_1 in range(num_nodes):
    #     for node_2 in range(num_nodes-1):
    #         if node_1 == node_2:
    #             continue
    #         if random.random() < P_CONNECTION:
    #             source.append(node_1)
    #             target.append(node_2)
            
    # # print(source, target)
    # alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    # prompt = ""
    # for i in range(len(source)):
    #     source_node = alphabet[source[i]]
    #     target_node = alphabet[target[i]]
    #     prompt += f"{source_node}->{target_node}\n"
    # if prompt == "":
    #     continue
    # else:
    #     prompt = prompt[0:-1] # get rid of last \n
    prompt = """A -> B
    A -> C
    C -> B
    D -> B
    E -> D"""
    source = [0, 0, 2, 3, 4]
    target = [1, 2, 1, 1, 3]
    print(prompt)


    inputs = tokenizer(prompt, return_tensors="pt")

    # Run inference and extract attention weights
    with torch.no_grad():
        outputs = model(**inputs)
    attentions = outputs.attentions

    # Extract attention maps for all layers
    num_layers = len(attentions)
    attention_matrix = attentions[0][0].detach().numpy()
    # Retain only the top 3 values in each row
    for row in attention_matrix:
        top_indices = np.argpartition(row, -3)[-3:]
        top_values = row[top_indices]
        row.fill(0)  # Zero out the row
        row[top_indices] = top_values  # Set top 3 values

    # Average across heads
    avg_attention_matrix = np.mean(attention_matrix, axis=0)

    # Save the matrix to a file for later GCN training
    # attention_file_path = f'attention_matrices/arg_3_avg/avg_attn_{z}.npy'
    attention_file_path = f'attention_matrices/demo1/avg_attn_{z}.npy'
    np.save(attention_file_path, avg_attention_matrix)

    # Confirm saving
    # print("Average attention matrix from layer 0 saved as 'avg_attention_matrix_layer0.npy'")
    # print("Matrix shape:", avg_attention_matrix.shape)

    base_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    # print(base_tokens)
    
    metadata_entry = {
        "attention_matrix_path": attention_file_path,
        "max_nodes": num_nodes,
        "num_nodes": len(set(source+target)),
        "num_edges": len(source),
        "connection_probability": P_CONNECTION,
        "layer": '0',
        "head": 'all',
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
    