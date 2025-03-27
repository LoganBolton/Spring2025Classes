import torch
from transformers import LlamaForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
# Import Polygon for clipping
from matplotlib.patches import Polygon
import re # Import regular expressions module
import os # Import os module for directory creation

# --- (Keep Step 1: Model and Tokenizer Loading - No changes needed) ---
# Step 1: Load model and tokenizer locally
model_name = "meta-llama/Llama-3.1-8B"
# model_name = "meta-llama/Llama-3.2-1B"
# Using a cached version if available, otherwise download
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        output_attentions=True,
        device_map="cpu", # Keep on CPU if memory is limited
        torch_dtype=torch.float32, # Use float32 for broader compatibility if needed
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
        device_map="cpu", # Keep on CPU
        torch_dtype=torch.float32, # Use float32
        low_cpu_mem_usage=True,
        attn_implementation="eager" # Add to suppress warning
    )

# --- (Keep Step 2: Text Prep and Token Cleaning - No changes needed) ---
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


# --- (Keep Find Tag Indices Function v4 - No changes needed) ---
def find_tag_indices_multi_token_v4(token_list):
    all_occurrences = {}
    temp_starts = {}
    instance_counters = {}
    i = 0
    while i < len(token_list):
        current_token = token_list[i].strip()
        is_start, tag_num_start, start_tag_len = False, None, 0
        tag_start_index = -1 # Store the index of '<'

        # Check for <factN> variations
        if current_token == '<':
            if i + 3 < len(token_list):
                token2, token3, token4 = token_list[i+1].strip().lower(), token_list[i+2].strip(), token_list[i+3].strip()
                if token2 == 'fact' and token3.isdigit() and token4.startswith('>'):
                    is_start, tag_num_start, start_tag_len = True, token3, 4
                    tag_start_index = i # Found start tag '<' at index i
        elif current_token.startswith('<fact') and current_token.endswith('>') and len(current_token) > 6:
             match = re.match(r'<fact(\d+)>', current_token)
             if match:
                 is_start, tag_num_start, start_tag_len = True, match.group(1), 1
                 tag_start_index = i # Found start tag '<' at index i

        if is_start:
            base_tag_name = f"fact{tag_num_start}"
            if base_tag_name not in temp_starts: temp_starts[base_tag_name] = []
            # Store the starting index of the tag itself ('<')
            temp_starts[base_tag_name].append(tag_start_index)
            i += start_tag_len # Move past the opening tag tokens
            continue

        is_end, tag_num_end, end_tag_len = False, None, 0
        closing_tag_start_index = -1 # Store index of '<' in '</fact...>'

         # Check for </factN> variations
        if current_token == '<':
            if i + 4 < len(token_list):
                token2, token3, token4, token5 = token_list[i+1].strip(), token_list[i+2].strip().lower(), token_list[i+3].strip(), token_list[i+4].strip()
                if token2 == '/' and token3 == 'fact' and token4.isdigit() and token5.startswith('>'):
                    is_end, tag_num_end, end_tag_len = True, token4, 5
                    closing_tag_start_index = i # Found closing tag '<' at index i
        elif current_token == '</':
             if i + 3 < len(token_list):
                 token2, token3, token4 = token_list[i+1].strip().lower(), token_list[i+2].strip(), token_list[i+3].strip()
                 if token2 == 'fact' and token3.isdigit() and token4.startswith('>'):
                     is_end, tag_num_end, end_tag_len = True, token3, 4
                     closing_tag_start_index = i # Found closing tag '<' at index i
        elif current_token.startswith('</fact') and current_token.endswith('>') and len(current_token) > 7:
             match = re.match(r'</fact(\d+)>', current_token)
             if match:
                 is_end, tag_num_end, end_tag_len = True, match.group(1), 1
                 closing_tag_start_index = i # Found closing tag '<' at index i

        if is_end:
            base_tag_name = f"fact{tag_num_end}"
            if base_tag_name in temp_starts and temp_starts[base_tag_name]:
                # Retrieve the start index of the opening tag ('<')
                tag_start_idx_actual = temp_starts[base_tag_name].pop(0)
                # Calculate the end index *after* the closing tag ('>')
                tag_end_idx_actual = closing_tag_start_index + end_tag_len

                instance_counters[base_tag_name] = instance_counters.get(base_tag_name, 0) + 1
                unique_tag_name = f"{base_tag_name}_{instance_counters[base_tag_name]}"
                # Store the indices covering the entire tag pair
                all_occurrences[unique_tag_name] = {
                    'tag_start': tag_start_idx_actual,
                    'tag_end': tag_end_idx_actual
                 }
                i += end_tag_len # Move past the closing tag tokens
                continue
        i += 1

    return all_occurrences


tag_locations = find_tag_indices_multi_token_v4(tokens) # Use the new function
print("\nFound Tag Locations (Indices including tags):")
if not tag_locations:
    print("  No tags found.")
for tag, loc in tag_locations.items():
    print(f"  {tag}: Start Index={loc['tag_start']}, End Index (Exclusive)={loc['tag_end']}")


# --- Find 'Answer' token index ---
target_token = 'Answer' # Note the leading space from cleaning
try:
    # Find the index *after* removing BOS
    answer_token_index = tokens.index(target_token)
    print(f"\nFound '{target_token}' token at index: {answer_token_index}")
except ValueError:
    print(f"\nWarning: '{target_token}' token not found in the token list.")
    answer_token_index = -1 # Use -1 to indicate not found


# --- (Keep Step 3: Inference and Attention Extraction - No changes needed) ---
# Step 3: Run inference and extract attention weights
with torch.no_grad():
    outputs = model(**inputs)
attentions = outputs.attentions
num_layers = model.config.num_hidden_layers
num_heads = model.config.num_attention_heads # Use config for heads
seq_len = len(tokens)


# --- (Keep Step 5: Print Model Info - No changes needed) ---
# Step 5: Print model info
print(f"\nModel: {model_name}")
print(f"Layers: {num_layers}, Heads per layer: {num_heads}")
print(f"Sequence length: {seq_len}")


# --- Modified Plotting Function (Manual Grid, Saves Individual Files, Adds Black Line) ---
def plot_individual_attention_maps(
    tag_indices,
    answer_idx, # <-- Add parameter for answer index
    layers_to_show=[0],
    heads_to_show=[0],
    output_dir="hot", # Directory to save images
    base_filename="attention_map", # Base name for saved files
    line_color='red',       # Color for tag boundary lines
    line_style='--',      # Style for tag boundary lines
    line_width=1,         # Width for tag boundary lines
    grid_line_color='black', # Color for the cell grid lines
    grid_line_width=0.15,    # Width for the cell grid lines
    shade_color='grey',
    shade_alpha=0.25,
    number_fontsize=30,
    number_alpha=0.4,
    number_color='black',
    answer_line_color='black', # <-- Color for the answer line
    answer_line_width=1.5,      # <-- Width for the answer line
    answer_line_style='-'       # <-- Style for the answer line (solid)
):
    token_count = len(tokens)
    base_width = max(12, token_count / 4) # Adjust figsize for single plot
    base_height = max(12, token_count / 5) # Adjust figsize for single plot

    # --- Create output directory if it doesn't exist ---
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error creating directory {output_dir}: {e}")
            return # Exit if directory cannot be created
    # --- End directory creation ---

    clip_verts = [(0, 0), (token_count, token_count), (0, token_count), (0, 0)]

    for layer_idx in layers_to_show:
        if not (0 <= layer_idx < num_layers):
             print(f"Warning: Layer index {layer_idx} invalid (max is {num_layers-1}). Skipping.")
             continue

        for head_idx in heads_to_show:
            if not (0 <= head_idx < num_heads):
                print(f"Warning: Head index {head_idx} invalid (max is {num_heads-1}). Skipping.")
                continue

            print(f"Generating plot for Layer {layer_idx}, Head {head_idx}...")

            # --- Create a NEW figure and axes for EACH plot ---
            fig, ax = plt.subplots(figsize=(base_width, base_height))
            # --- End new figure ---

            clip_polygon = Polygon(clip_verts, transform=ax.transData, facecolor='none', edgecolor='none')
            attn_weights = attentions[layer_idx][0, head_idx].detach().cpu().numpy()
            attn_weights_plot = attn_weights[:token_count, :token_count]
            mask_upper = np.triu(np.ones_like(attn_weights_plot, dtype=bool), k=1)

            # --- Z-order definitions ---
            heatmap_zorder = 0.5
            grid_zorder = 1.0      # Grid lines on top of heatmap colors
            span_zorder = 1.5      # Shaded spans on top of grid
            number_zorder = 2.0    # Numbers on top of spans
            line_zorder = 2.5      # Red lines for tags
            answer_line_zorder = 2.6 # Answer line slightly above tag lines
            # --- Plot heatmap WITHOUT internal grid lines ---
            sns.heatmap(
                attn_weights_plot,
                cmap="Blues",
                xticklabels=False,
                yticklabels=False,
                cbar=False,
                mask=mask_upper,
                ax=ax,
                square=True,
                linewidths=0,     # <-- Set linewidths to 0
                linecolor='none', # <-- Set linecolor to none
                zorder=heatmap_zorder # Set base zorder for heatmap
            )

            # --- Manually draw grid lines ONLY for the lower triangle ---
            for k in range(1, token_count):
                # Vertical line segments (from x=k, starting at y=k-1 down to y=token_count)
                ax.plot([k, k], [k - 1, token_count], color=grid_line_color, linewidth=grid_line_width, zorder=grid_zorder)
                # Horizontal line segments (from y=k, starting at x=0 across to x=k)
                ax.plot([0, k], [k, k], color=grid_line_color, linewidth=grid_line_width, zorder=grid_zorder)
            # --- End manual grid drawing ---


            tick_positions = np.arange(token_count) + 0.5
            ax.set_xticks(tick_positions)
            ax.set_yticks(tick_positions)
            fontsize = max(3, 10 - token_count // 30)
            ax.set_xticklabels(tokens, rotation=90, fontsize=fontsize, ha='center')
            ax.set_yticklabels(tokens, rotation=0, fontsize=fontsize, va='center')

            # --- Apply Tag Highlighting (using updated z-orders) ---
            if tag_indices:
                for tag_name, loc in tag_indices.items():
                    start_idx_tag = loc['tag_start']
                    end_idx_tag = loc['tag_end']

                    if start_idx_tag >= 0 and end_idx_tag <= token_count and start_idx_tag < end_idx_tag:
                        match = re.search(r'fact(\d+)', tag_name)
                        fact_num_str = match.group(1) if match else None

                        # Shaded spans
                        span_v = ax.axvspan(start_idx_tag, end_idx_tag, color=shade_color, alpha=shade_alpha, zorder=span_zorder, lw=0)
                        span_h = ax.axhspan(start_idx_tag, end_idx_tag, color=shade_color, alpha=shade_alpha, zorder=span_zorder, lw=0)
                        span_v.set_clip_path(clip_polygon)
                        span_h.set_clip_path(clip_polygon)

                        # Fact numbers
                        if fact_num_str:
                            center_x = (start_idx_tag + end_idx_tag) / 2.0
                            center_y = (start_idx_tag + end_idx_tag) / 2.0
                            dynamic_fontsize = max(5, min(number_fontsize, (end_idx_tag - start_idx_tag) * 1.5))

                            ax.text(center_x, center_y, fact_num_str,
                                    ha='center', va='center',
                                    fontsize=dynamic_fontsize+20, # Increased size slightly
                                    color=number_color,
                                    alpha=number_alpha,
                                    zorder=number_zorder, # Ensure numbers are above spans/grid
                                    clip_on=True) # Clip numbers to axes bounds

                        # Red boundary lines
                        # Vertical lines (start at diagonal, go down)
                        ax.plot([start_idx_tag, start_idx_tag], [start_idx_tag, token_count], color=line_color, linestyle=line_style, linewidth=line_width, zorder=line_zorder) # Vertical start
                        ax.plot([end_idx_tag, end_idx_tag], [end_idx_tag, token_count], color=line_color, linestyle=line_style, linewidth=line_width, zorder=line_zorder)     # Vertical end
                        # Horizontal lines (start at left, go to diagonal)
                        ax.plot([0, start_idx_tag], [start_idx_tag, start_idx_tag], color=line_color, linestyle=line_style, linewidth=line_width, zorder=line_zorder) # Horizontal start
                        ax.plot([0, end_idx_tag], [end_idx_tag, end_idx_tag], color=line_color, linestyle=line_style, linewidth=line_width, zorder=line_zorder)     # Horizontal end
                    else:
                        # Warning printed only once per invalid tag during finding, no need here unless debugging
                        pass
            # --- End Tag Highlighting ---

            # --- Add Solid Black Line after 'Answer' token ---
            if answer_idx != -1:
                line_pos = answer_idx + 1 # The boundary is *after* the token index
                if 0 < line_pos < token_count: # Ensure the line position is valid
                    # Vertical line: starts at x=line_pos, from y=line_pos (diagonal) down to y=token_count (bottom)
                    ax.plot([line_pos, line_pos], [line_pos, token_count],
                            color=answer_line_color,
                            linestyle=answer_line_style,
                            linewidth=answer_line_width,
                            zorder=answer_line_zorder) # Ensure it's drawn on top

                    # Horizontal line: starts at y=line_pos, from x=0 (left) across to x=line_pos (diagonal)
                    ax.plot([0, line_pos], [line_pos, line_pos],
                            color=answer_line_color,
                            linestyle=answer_line_style,
                            linewidth=answer_line_width,
                            zorder=answer_line_zorder) # Ensure it's drawn on top
            # --- End Answer Line ---


            ax.set_xlim(0, token_count)
            ax.set_ylim(token_count, 0)
            ax.set_title(f"Layer {layer_idx}, Head {head_idx}", fontsize=10)

            # --- Construct unique filename and save ---
            filename = os.path.join(output_dir, f"{base_filename}_layer_{layer_idx}_head_{head_idx}.png")
            try:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"   Saved plot to {filename}")
            except Exception as e:
                print(f"   Error saving plot {filename}: {e}")
            # --- End saving ---

            # --- Close the figure to free memory ---
            plt.close(fig)
            # --- End closing ---

    print("\nFinished generating individual attention map images.")


# --- Run Visualization (Saving Individual Files) ---
num_model_layers = model.config.num_hidden_layers
num_model_heads = model.config.num_attention_heads

plot_individual_attention_maps(
    tag_locations,
    answer_idx=answer_token_index, # Pass the found index
    layers_to_show=[0],     # Example: Layers 0
    heads_to_show=[8,9,10,11,12,13,14,15,16],      # Example: Head 3
    output_dir="hot/individual_maps", # Save to a new subdirectory
    base_filename="llama3.1_8b_attention", # Custom base name
    shade_color='grey',
    shade_alpha=0.15,
    number_fontsize=30,
    number_alpha=0.4,
    number_color='red',
    grid_line_color='black', # Explicitly define grid color
    grid_line_width=0.15,    # Explicitly define grid width
    answer_line_color='black', # Make answer line black
    answer_line_width=2.0      # Make answer line slightly thicker
)