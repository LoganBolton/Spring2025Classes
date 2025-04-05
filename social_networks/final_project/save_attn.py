import torch
import numpy as np
from transformers import LlamaForCausalLM, AutoTokenizer

# Load model and tokenizer locally
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(
    model_name,
    output_attentions=True,
    device_map="cpu",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)

# Prepare input text
text = """A: B, C, D, E
C: B
D: B
E: D"""
inputs = tokenizer(text, return_tensors="pt")

# Run inference and extract attention weights
with torch.no_grad():
    outputs = model(**inputs)
attentions = outputs.attentions

# Extract average attention from the first layer
layer_idx = 0
avg_attention_matrix = attentions[layer_idx][0].mean(dim=0).detach().numpy()

# Save the matrix to a file for later GCN training
np.save("avg_attention_matrix_layer0.npy", avg_attention_matrix)

# Confirm saving
print("Average attention matrix from layer 0 saved as 'avg_attention_matrix_layer0.npy'")
print("Matrix shape:", avg_attention_matrix.shape)