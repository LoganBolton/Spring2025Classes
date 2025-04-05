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
# text = """A -> B
# A -> C
# C -> B
# D -> B
# E -> D"""
# text = """A to B
# A to C
# C to B
# D to B
# E to D"""
text = """A: B, C, D, E
C: B
D: B
E: D"""
inputs = tokenizer(text, return_tensors="pt")
base_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

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

# Optional: To visualize multiple heads from different layers
def plot_multiple_heads_and_layers(
    layers_to_show=[0], 
    heads_to_show=[0,3,12]):
    rows = len(layers_to_show)
    cols = len(heads_to_show)
    
    plt.figure(figsize=(cols * 5, rows * 4))
    highlight_positions = []
    
    for i, layer_idx in enumerate(layers_to_show):
        for j, head_idx in enumerate(heads_to_show):
            plt.subplot(rows, cols, i * cols + j + 1)
            
            attn_weights = attentions[layer_idx][0, head_idx].detach().numpy()
            # Remove the last row and column (EOS token)
            attn_weights = attn_weights[:-1, :-1]
            mask_upper_triangular = np.triu(np.ones_like(attn_weights), k=1)  # k=1 keeps diagonal visible
            
            ax = sns.heatmap(
                attn_weights,
                annot=False,
                cmap="Blues",
                xticklabels=tokens,  # Only show on first column
                yticklabels=tokens,  # Only show on last row
                fmt=".2f",
                annot_kws={'size': 6},
                mask=mask_upper_triangular,
            )
        
            # Add black rectangles around the cells you want to highlight
            for pos in highlight_positions:
                row, col = pos
                # Draw rectangle with black border around the cell
                rect = plt.Rectangle((col, row), 1, 1, fill=False, edgecolor='gray', lw=2)
                ax.add_patch(rect)
                
            plt.title(f"Layer {layer_idx}, Head {head_idx}")
            
            if j == 0:  # Only add y-label on first column
                plt.ylabel("Query Tokens")
                plt.yticks(fontsize=6)
            if i == rows-1:  # Only add x-label on last row
                plt.xlabel("Key Tokens")
                plt.xticks(rotation=90, ha="right", fontsize=6)
    
    plt.tight_layout()
    plt.savefig("multi_head_layer_attention.png", dpi=300)
    # plt.show()

# Uncomment to run this additional visualization
plot_multiple_heads_and_layers()