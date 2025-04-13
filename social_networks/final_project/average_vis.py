import torch
from transformers import LlamaForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
text = """A->B
B->C
C->A"""
# text = """B->C
# C->A
# A->B"""
# text = """C->A
# A->B
# B->C"""

inputs = tokenizer(text, return_tensors="pt")
base_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

def clean_tokens(tokens):
    return [t.replace('Ġ', ' ')
            .replace('<|begin_of_text|>', '[BOS]')
            .replace('Ċ', '\n')
            for t in tokens]

tokens = clean_tokens(base_tokens)

# Step 3: Run inference and extract attention weights
with torch.no_grad():
    outputs = model(**inputs)
attentions = outputs.attentions  # List of tensors: one per layer
num_layers = len(attentions)
num_heads = attentions[0].shape[1]
seq_len = len(tokens)

# Step 4: Print model info
print(f"Model: {model_name}")
print(f"Layers: {num_layers}, Heads per layer: {num_heads}")
print(f"Sequence length: {seq_len}")
print(f"Tokens: {tokens}")

# Step 5: Plot multiple heads across layers
def plot_multiple_heads_and_layers(layers_to_show=[0], heads_to_show=[0, 1, 2]):
    rows = len(layers_to_show)
    cols = len(heads_to_show)
    plt.figure(figsize=(cols * 5, rows * 4))
    
    for i, layer_idx in enumerate(layers_to_show):
        for j, head_idx in enumerate(heads_to_show):
            plt.subplot(rows, cols, i * cols + j + 1)
            attn_weights = attentions[layer_idx][0, head_idx].detach().numpy()
            # attn_weights = attn_weights[:-1, :-1]
            mask_upper_triangular = np.triu(np.ones_like(attn_weights), k=1)

            ax = sns.heatmap(
                attn_weights,
                annot=False,
                cmap="Blues",
                xticklabels=tokens,
                yticklabels=tokens,
                mask=mask_upper_triangular
            )

            plt.title(f"Layer {layer_idx}, Head {head_idx}")
            if j == 0:
                plt.ylabel("Query Tokens")
            if i == rows - 1:
                plt.xlabel("Key Tokens")
                plt.xticks(rotation=90, ha="right", fontsize=6)
    
    plt.tight_layout()
    plt.savefig("multi_head_layer_attention.png", dpi=300)
    # plt.show()

# Step 6: Plot average across heads in a given layer
def plot_average_attention_across_heads(layer_idx=0):
    attn_tensor = attentions[layer_idx][0]  # (num_heads, seq_len, seq_len)
    avg_attn = attn_tensor.mean(dim=0).detach().numpy()  # (seq_len, seq_len)
    # avg_attn = avg_attn[:-1, :-1]
    mask_upper_triangular = np.triu(np.ones_like(avg_attn), k=1)

    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(
        avg_attn,
        annot=False,
        cmap="viridis",
        xticklabels=tokens,
        yticklabels=tokens,
        mask=mask_upper_triangular
    )

    plt.title(f"Average Attention (Layer {layer_idx})")
    plt.ylabel("Query Tokens")
    plt.xlabel("Key Tokens")
    plt.xticks(rotation=0, ha="center", fontsize=8)
    plt.yticks(fontsize=8, rotation=0)
    plt.tight_layout()
    plt.savefig(f"average_attention_layer_{layer_idx}.png", dpi=300)
    # plt.show()

# Run visualizations
plot_multiple_heads_and_layers()
plot_average_attention_across_heads(layer_idx=0)