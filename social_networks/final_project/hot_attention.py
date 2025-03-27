import torch
from transformers import LlamaForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
# Import Polygon for clipping
from matplotlib.patches import Polygon

# --- (Keep all the code from Step 1 to Step 5, including the working find_tag_indices_multi_token_v3) ---

# Step 1: Load model and tokenizer locally
model_name = "meta-llama/Llama-3.2-1B"
# Using a cached version if available, otherwise download
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        output_attentions=True,
        device_map="cpu",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="eager" # Add to suppress warning
    )
    print(f"Loaded model {model_name} from local cache.")
except Exception as e:
    print(f"Could not load from local cache: {e}. Downloading...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        output_attentions=True,
        device_map="cpu",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager" # Add to suppress warning
    )


# Step 2: Prepare input text
text = """**Reformatted Question:** A machine is set up in such a way that it will <fact1>short circuit if both the black wire and the red wire touch the battery at the same time</fact1>. The machine will <fact2>not short circuit if just one of these wires touches the battery</fact2>. The black wire is designated as the one that is supposed to touch the battery, while the red wire is supposed to remain in some other part of the machine. One day, the <fact3>black wire and the red wire both end up touching the battery at the same time</fact3>. There is a short circuit. Did the <fact4>black wire cause the short circuit</fact4>? Options: - Yes - No

**Answer:** The question states that the machine <fact1>short circuits only when both the black and red wires touch the battery simultaneously</fact1>. It also specifies that <fact2>touching the battery with only one wire will not cause a short circuit</fact2>.  While the <fact3>black wire did touch the battery</fact3>, it was the <fact3>simultaneous contact of both wires</fact3> that triggered the short circuit. Therefore, the <fact4>black wire alone did not cause the short circuit</fact4>. The answer is {No}."""

inputs = tokenizer(text, return_tensors="pt")
base_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

def clean_tokens(tokens):
    return [t.replace('Ġ', ' ')
            .replace('<|begin_of_text|>', '[BOS]')
            .replace('Ċ', '\n')
            for t in tokens]

tokens_with_bos = clean_tokens(base_tokens)
tokens = tokens_with_bos[1:] # Remove [BOS] token

# ---- Find Tag Indices Function (v3 - Keep as is) ----
def find_tag_indices_multi_token_v3(token_list):
    all_occurrences = {}
    temp_starts = {}
    instance_counters = {}
    i = 0
    while i < len(token_list):
        current_token = token_list[i].strip()
        is_start, tag_num_start, start_tag_len = False, None, 0
        if current_token == '<':
            if i + 3 < len(token_list):
                token2, token3, token4 = token_list[i+1].strip().lower(), token_list[i+2].strip(), token_list[i+3].strip()
                if token2 == 'fact' and token3.isdigit() and token4.startswith('>'):
                    is_start, tag_num_start, start_tag_len = True, token3, 4
        if is_start:
            base_tag_name = f"fact{tag_num_start}"
            if base_tag_name not in temp_starts: temp_starts[base_tag_name] = []
            temp_starts[base_tag_name].append(i)
            i += start_tag_len
            continue
        is_end, tag_num_end, end_tag_len = False, None, 0
        if current_token == '<':
            if i + 4 < len(token_list):
                token2, token3, token4, token5 = token_list[i+1].strip(), token_list[i+2].strip().lower(), token_list[i+3].strip(), token_list[i+4].strip()
                if token2 == '/' and token3 == 'fact' and token4.isdigit() and token5.startswith('>'):
                    is_end, tag_num_end, end_tag_len = True, token4, 5
        elif current_token == '</':
            if i + 3 < len(token_list):
                token2, token3, token4 = token_list[i+1].strip().lower(), token_list[i+2].strip(), token_list[i+3].strip()
                if token2 == 'fact' and token3.isdigit() and token4.startswith('>'):
                    is_end, tag_num_end, end_tag_len = True, token3, 4
        if is_end:
            base_tag_name = f"fact{tag_num_end}"
            if base_tag_name in temp_starts and temp_starts[base_tag_name]:
                start_index = temp_starts[base_tag_name].pop(0)
                # Adjust end index for finding tags: it's the index AFTER the closing tag starts
                end_index_tag_content = i # Index where '</fact...' starts
                instance_counters[base_tag_name] = instance_counters.get(base_tag_name, 0) + 1
                unique_tag_name = f"{base_tag_name}_{instance_counters[base_tag_name]}"
                # Store the start of the tag opening '<' and the start of the tag closing '</'
                all_occurrences[unique_tag_name] = {'start_tag_open': start_index, 'start_tag_close': end_index_tag_content}
                i += end_tag_len
                continue
        i += 1

    # Refine indices to represent the *content* span for plotting
    plot_indices = {}
    for tag_name, loc in all_occurrences.items():
        start_content_idx = -1
        # Find the '>' of the opening tag
        temp_i = loc['start_tag_open']
        while temp_i < len(token_list) and temp_i < loc['start_tag_close']:
             if token_list[temp_i].strip().endswith('>'):
                 start_content_idx = temp_i + 1
                 break
             temp_i += 1

        if start_content_idx != -1:
             # The end index for plotting should be the start of the closing tag
             plot_indices[tag_name] = {'start': start_content_idx, 'end': loc['start_tag_close']} # 'end' is exclusive for spans/lines

    return plot_indices


tag_locations = find_tag_indices_multi_token_v3(tokens)
print("\nFound Tag Locations (Content Indices for Plotting):")
if not tag_locations:
    print("  No tags found.")
for tag, loc in tag_locations.items():
    print(f"  {tag}: Start Index={loc['start']}, End Index (Exclusive)={loc['end']}")
    # print(f"     Tokens: {tokens[loc['start']:loc['end']]}") # Optional: verify tokens


# Step 3: Run inference and extract attention weights
with torch.no_grad():
    outputs = model(**inputs)
attentions = outputs.attentions
num_layers = model.config.num_hidden_layers
num_heads = model.config.num_attention_heads # Use config for heads
seq_len = len(tokens)

# Step 5: Print model info
print(f"\nModel: {model_name}")
print(f"Layers: {num_layers}, Heads per layer: {num_heads}")
print(f"Sequence length: {seq_len}")


# --- Modified Plotting Function ---
def plot_multiple_heads_and_layers(
    tag_indices,
    layers_to_show=[0],
    heads_to_show=[0],
    line_color='red',
    line_style='--',
    line_width=1, # Adjusted line width slightly
    shade_color='grey',
    shade_alpha=0.25
):
    rows = len(layers_to_show)
    cols = len(heads_to_show)
    token_count = len(tokens)

    base_width = max(12, token_count / 5)
    base_height = max(12, token_count / 6)
    plt.figure(figsize=(cols * base_width, rows * base_height))

    # Define the vertices for the lower triangle clipping path (including diagonal)
    # Coordinates are (x, y)
    clip_verts = [(0, 0), (token_count, token_count), (0, token_count), (0, 0)]


    for i, layer_idx in enumerate(layers_to_show):
        if layer_idx >= num_layers:
             print(f"Warning: Layer index {layer_idx} invalid. Skipping.")
             continue

        for j, head_idx in enumerate(heads_to_show):
            if head_idx >= num_heads:
                print(f"Warning: Head index {head_idx} invalid. Skipping.")
                continue

            ax = plt.subplot(rows, cols, i * cols + j + 1)

            # --- Create the clip polygon specific to this axes's data transform ---
            # This is important because each subplot might have slightly different transforms
            clip_polygon = Polygon(clip_verts, transform=ax.transData, facecolor='none', edgecolor='none')
            # --- End clip polygon creation ---

            attn_weights = attentions[layer_idx][0, head_idx].detach().cpu().numpy()
            attn_weights_plot = attn_weights[:token_count, :token_count]

            # Mask for upper triangle (future tokens) - applied to heatmap
            mask_upper = np.triu(np.ones_like(attn_weights_plot, dtype=bool), k=1)

            # Draw Heatmap FIRST
            sns.heatmap(
                attn_weights_plot,
                cmap="Blues",
                xticklabels=False,
                yticklabels=False,
                cbar=False,       # <--- Removed the colorbar entirely
                mask=mask_upper, # Apply causal mask to heatmap
                ax=ax,
                square=True,
                linewidths=0, # Remove cell borders for clarity
                # zorder=0 # Ensure heatmap is at the bottom
            )

            # Set ticks and labels AFTER heatmap
            tick_positions = np.arange(token_count) + 0.5
            ax.set_xticks(tick_positions)
            ax.set_yticks(tick_positions)
            fontsize = max(3, 10 - token_count // 30)
            ax.set_xticklabels(tokens, rotation=90, fontsize=fontsize, ha='center')
            ax.set_yticklabels(tokens, rotation=0, fontsize=fontsize, va='center')


            # ---- Add Shading and Lines for Tags (Clipped) ----
            span_zorder = 1  # Draw spans above heatmap but below lines
            line_zorder = 2  # Draw lines on top of spans

            if tag_indices:
                for tag_name, loc in tag_indices.items():
                    start_idx = loc['start']
                    end_idx = loc['end'] # This is exclusive

                    if start_idx < token_count and end_idx <= token_count and start_idx < end_idx:
                        # --- Add Shading (Clipped) ---
                        # Draw vertical span (column shading)
                        span_v = ax.axvspan(start_idx, end_idx, color=shade_color, alpha=shade_alpha, zorder=span_zorder, lw=0)
                        # Draw horizontal span (row shading)
                        span_h = ax.axhspan(start_idx, end_idx, color=shade_color, alpha=shade_alpha, zorder=span_zorder, lw=0)

                        # Apply the lower-triangle clip path to the spans
                        span_v.set_clip_path(clip_polygon)
                        span_h.set_clip_path(clip_polygon)
                        # --- End Shading ---

                        # --- Draw Lines (Segmented for Lower Triangle) ---
                        # Vertical line at start_idx (from diagonal down)
                        ax.plot([start_idx, start_idx], [start_idx, token_count], color=line_color, linestyle=line_style, linewidth=line_width, zorder=line_zorder)
                        # Vertical line at end_idx (from diagonal down)
                        ax.plot([end_idx, end_idx], [end_idx, token_count], color=line_color, linestyle=line_style, linewidth=line_width, zorder=line_zorder)
                        # Horizontal line at start_idx (from left edge to diagonal)
                        ax.plot([0, start_idx], [start_idx, start_idx], color=line_color, linestyle=line_style, linewidth=line_width, zorder=line_zorder)
                        # Horizontal line at end_idx (from left edge to diagonal)
                        ax.plot([0, end_idx], [end_idx, end_idx], color=line_color, linestyle=line_style, linewidth=line_width, zorder=line_zorder)
                        # --- End Lines ---
                    else:
                        print(f"Warning: Tag '{tag_name}' indices ({start_idx}, {end_idx}) invalid or out of bounds for token count {token_count}. Not drawing for this tag.")
            # ---- End Shading and Lines ----

            # Set plot limits AFTER drawing everything to ensure clipping works correctly
            ax.set_xlim(0, token_count)
            ax.set_ylim(token_count, 0) # Inverted y-axis typical for matrices

            ax.set_title(f"Layer {layer_idx}, Head {head_idx}", fontsize=10)


    plt.tight_layout(pad=2.5)
    plt.savefig("multi_head_layer_attention_shaded_tags_clipped.png", dpi=300, bbox_inches='tight') # New filename
    print("\nSaved plot to multi_head_layer_attention_shaded_tags_clipped.png")
    # plt.show()


# --- Run Visualization ---
num_model_layers = model.config.num_hidden_layers
num_model_heads = model.config.num_attention_heads
plot_multiple_heads_and_layers(
    tag_locations,
    layers_to_show=[0], # Example: Layer 0
    heads_to_show=[0],    # Example: Head 0
    shade_color='grey',
    shade_alpha=0.25
)