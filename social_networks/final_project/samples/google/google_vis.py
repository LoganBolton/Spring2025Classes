import torch
from transformers import LlamaForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

# Step 1: Load model and tokenizer locally
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(
    model_name,
    output_attentions=True,
    device_map="cpu",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)

# Step 2: Prepare input text
text = """A -> B
A -> C
C -> B
D -> B
E -> D"""

inputs = tokenizer(text, return_tensors="pt")
base_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# def clean_tokens(tokens):
#     return [t.replace('Ġ', ' ')
#             .replace('<|begin_of_text|>', '[BOS]')
#             .replace('Ċ', '\n')
#             for t in tokens]

def clean_tokens(tokens):
    return [t.replace('Ġ', ' ')
            .replace('<|begin_of_text|>', '[BOS]')
            .replace('Ċ', '\n')
            for t in tokens]

tokens = clean_tokens(base_tokens)
tokens = tokens[1:] # Remove [BOS] token

# Step 3: Run inference and extract attention weights
with torch.no_grad():
    outputs = model(**inputs)
attentions = outputs.attentions
num_layers = len(attentions)
num_heads = attentions[0].shape[1]
seq_len = len(tokens)

# Step 5: Print model info
print(f"Model: {model_name}")
print(f"Layers: {num_layers}, Heads per layer: {num_heads}")
print(f"Sequence length: {seq_len}")
print(f"Tokens: {tokens}")

# Modified: Plot the average attention of the second layer (index 1) across all heads
def plot_second_layer_average():
    """
    Create a single heatmap showing the average attention weights 
    of the second layer (index 1) across all attention heads
    """
    plt.figure(figsize=(8, 6))
    
    # Get the second layer (index 1) and average across all heads (dim=1)
    second_layer = attentions[1][0].detach().numpy()  # [heads, seq_len, seq_len]
    avg_attention = np.mean(second_layer, axis=0)     # Average across heads
    
    # Remove the last row and column (EOS token)
    # avg_attention = avg_attention[:-1, :-1]
    
    # Create mask to hide upper triangular (future tokens)
    mask_upper_triangular = np.triu(np.ones_like(avg_attention), k=1)  # k=1 keeps diagonal visible
    
    # Plot with reversed color map (red for high values, blue for low)
    ax = sns.heatmap(
        avg_attention,
        annot=False,
        cmap="RdBu_r",  # Red-Blue reversed: red for high values, blue for low
        xticklabels=tokens,
        yticklabels=tokens,
        fmt=".2f",
        annot_kws={'size': 8},
        mask=mask_upper_triangular,
    )
    
    plt.title(f"Layer 2 Mean")
    plt.ylabel("Query Tokens")
    plt.xlabel("Key Tokens")
    plt.xticks(rotation=45, ha="right")
    
    plt.tight_layout()
    plt.savefig("second_layer_average_attention.png", dpi=300)
    # plt.show()

# Run the new visualization function
plot_second_layer_average()