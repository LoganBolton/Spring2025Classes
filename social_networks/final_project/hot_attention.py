import torch
from transformers import LlamaForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from matplotlib.patches import Polygon
import re
import os
from typing import Optional, List

# --- (Keep existing code: Model/Tokenizer Loading, Text Prep, Tag Finding, Inference) ---
# [ ... Your existing setup code ... ]
# Step 1: Load model and tokenizer locally
model_name = "meta-llama/Llama-3.1-8B"
# model_name = "meta-llama/Llama-3.2-1B" # Using smaller model for faster testing

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        output_attentions=True,
        device_map="cpu", # Keep on CPU for wider compatibility if memory is limited
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="eager" # Ensure we get standard attention tensors
    )
    print(f"Loaded model {model_name} from local cache.")
except Exception as e:
    print(f"Could not load from local cache: {e}. Downloading...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        output_attentions=True,
        device_map="cpu", # Keep on CPU for wider compatibility if memory is limited
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager" # Ensure we get standard attention tensors
    )

# text = """Question: For every 12 cans you recycle, you receive $0.50, and for every 5 kilograms of newspapers, you receive $1.50. If your family collected 144 cans and 20 kilograms of newspapers, how much money would you receive?
# Answer_Reasoning: There are 144/12 = 12 sets of 12 cans that the family collected.\nSo, the family would receive $0.50 x 12 = 6 for the cans.\nThere are 20/5 = 4 sets of 5 kilograms of newspapers that the family collected.\nSo, the family would receive $1.50 x 4 = 6 for the newspapers.\nTherefore, the family would receive a total of $6 + $6 = 12.
# Final_Answer: {"""

text = """Question: <fact1>For every 12 cans you recycle</fact1>, <fact2>you receive $0.50</fact2>, <fact3>and for every 5 kilograms of newspapers</fact3>, <fact4>you receive $1.50</fact4>. <fact5>If your family collected 144 cans and 20 kilograms of newspapers</fact5>, <fact6>how much money would you receive</fact6>?
Answer_Reasoning: There are <fact5>144</fact5>/<fact1>12</fact1> = 12 sets of 12 cans that the family collected.\nSo, the family would receive <fact2>$0.50</fact2> x <fact1>12</fact1> = $6 for the cans.\nThere are <fact5>20</fact5>/<fact3>5</fact3> = 4 sets of 5 kilograms of newspapers that the family collected.\nSo, the family would receive <fact4>$1.50</fact4> x 4 = 6 for the newspapers.\nTherefore, the family would receive a total of $6 + $6 = 12.
Final_Answer: {"""

inputs = tokenizer(text, return_tensors="pt")
base_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

def clean_tokens(tokens):
    return [t.replace('Ġ', ' ')
            .replace('<|begin_of_text|>', '[BOS]')
            .replace('Ċ', '\n')
            for t in tokens]

tokens_with_bos = clean_tokens(base_tokens)
tokens = tokens_with_bos[1:] # Remove [BOS] token


# Find Tag Indices Function v4 (Keep as is)
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
                tag_start_idx_actual = temp_starts[base_tag_name].pop(0)
                tag_end_idx_actual = closing_tag_start_index + end_tag_len
                instance_counters[base_tag_name] = instance_counters.get(base_tag_name, 0) + 1
                unique_tag_name = f"{base_tag_name}_{instance_counters[base_tag_name]}"
                all_occurrences[unique_tag_name] = {
                    'tag_start': tag_start_idx_actual,
                    'tag_end': tag_end_idx_actual
                 }
                i += end_tag_len
                continue
        i += 1

    return all_occurrences

tag_locations = find_tag_indices_multi_token_v4(tokens)
print("\nFound Tag Locations (Indices including tags):")
if not tag_locations:
    print("  No tags found.")
for tag, loc in tag_locations.items():
    print(f"  {tag}: Start Index={loc['tag_start']}, End Index (Exclusive)={loc['tag_end']}")

# Find 'Answer' token index (Keep as is)
target_token = 'Answer' # Note the leading space from cleaning
try:
    answer_token_index = tokens.index(target_token)
    print(f"\nFound '{target_token}' token at index: {answer_token_index}")
except ValueError:
    print(f"\nWarning: '{target_token}' token not found in the token list.")
    answer_token_index = -1

# Step 3: Run inference and extract attention weights (Keep as is)
with torch.no_grad():
    outputs = model(**inputs)
attentions = outputs.attentions # Tuple of tensors, one for each layer
num_layers = model.config.num_hidden_layers
num_heads = model.config.num_attention_heads
seq_len = len(tokens) # Use actual token length after removing BOS

# Step 5: Print model info (Keep as is)
print(f"\nModel: {model_name}")
print(f"Layers: {num_layers}, Heads per layer: {num_heads}")
print(f"Sequence length: {seq_len}")


# --- Modified Plotting Function with Upper Triangle White Mask ---
def plot_individual_attention_maps(
    tag_indices,
    answer_idx,
    layers_to_show: List[int] = [0],
    heads_to_show: List[int] = [0],
    output_dir: str = "hot",
    base_filename: str = "attention_map",
    top_n: Optional[int] = None,
    normalize_rows: bool = False,
    cmap: str = "Blues",
    line_color: str = 'red',
    line_style: str = '--',
    line_width: float = 1,
    grid_line_color: str = 'black',
    grid_line_width: float = 0.15,
    shade_color: str = 'grey',
    shade_alpha: float = 0.25,
    number_fontsize: int = 30,
    number_alpha: float = 0.4,
    number_color: str = 'black',
    answer_line_color: str = 'black',
    answer_line_width: float = 1.5,
    answer_line_style: str = '-'
):
    """
    Generates and saves attention heatmaps, masking the upper triangle with white.
    (Args documentation remains the same)
    """
    token_count = len(tokens)
    base_width = max(12, token_count / 4)
    base_height = max(12, token_count / 5)

    if top_n is not None and normalize_rows:
        print("Warning: Both top_n and normalize_rows are specified. Prioritizing top_n.")
        normalize_rows = False

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error creating directory {output_dir}: {e}")
            return

    for layer_idx in layers_to_show:
        if not (0 <= layer_idx < num_layers):
             print(f"Warning: Layer index {layer_idx} invalid (max is {num_layers-1}). Skipping.")
             continue

        layer_attentions = attentions[layer_idx]

        for head_idx in heads_to_show:
            if not (0 <= head_idx < num_heads):
                print(f"Warning: Head index {head_idx} invalid (max is {num_heads-1}). Skipping.")
                continue

            print(f"Generating plot for Layer {layer_idx}, Head {head_idx}...")

            fig, ax = plt.subplots(figsize=(base_width, base_height))
            ax.set_facecolor('white') # Set background

            # --- Z-order Definitions (Adjusted) ---
            # Ensure mask is above spans but below grid/lines/text
            heatmap_zorder = 0.5
            span_zorder = 1.0          # Spans drawn above heatmap
            upper_mask_zorder = 1.5    # White mask above spans
            grid_zorder = 2.0          # Grid above mask
            number_zorder = 3.0        # Numbers above grid
            line_zorder = 4.0          # Tag lines above numbers
            answer_line_zorder = 4.1   # Answer line above tag lines

            # Extract and prepare attention weights (same logic as before)
            attn_weights = layer_attentions[0, head_idx, 1:, 1:].detach().cpu().numpy()
            attn_weights_plot_orig = attn_weights[:token_count, :token_count]

            data_to_plot = np.copy(attn_weights_plot_orig)
            plot_title_suffix = "(Raw Attention)"
            mode_suffix = "_raw"

            # --- Top-N / Normalization Logic (Keep as is) ---
            if top_n is not None and top_n > 0:
                # [ ... Same top_n logic ... ]
                print(f"   Selecting Top {top_n} attention values per row...")
                top_n_data = np.full_like(attn_weights_plot_orig, np.nan)
                for i in range(token_count):
                    row_slice = attn_weights_plot_orig[i, :i+1]
                    num_valid_attentions = i + 1
                    current_top_n = min(top_n, num_valid_attentions)
                    if current_top_n > 0:
                         top_indices_in_slice = np.argsort(row_slice)[-current_top_n:]
                         top_n_data[i, top_indices_in_slice] = row_slice[top_indices_in_slice]
                data_to_plot = top_n_data
                plot_title_suffix = f"(Top {top_n} Attention)"
                mode_suffix = f"_top{top_n}"

            elif normalize_rows:
                # [ ... Same normalize_rows logic ... ]
                print("   Applying row-wise normalization...")
                attn_weights_normalized = np.copy(attn_weights_plot_orig)
                epsilon = 1e-9
                for i in range(token_count):
                    row_slice = attn_weights_normalized[i, :i+1]
                    max_val = np.max(row_slice) if len(row_slice) > 0 else 0
                    if max_val > epsilon:
                         attn_weights_normalized[i, :i+1] = row_slice / max_val
                    else:
                         attn_weights_normalized[i, :i+1] = 0.0
                    if i + 1 < token_count:
                        attn_weights_normalized[i, i+1:] = np.nan
                data_to_plot = np.tril(attn_weights_normalized, k=0)
                data_to_plot[np.triu_indices_from(data_to_plot, k=1)] = np.nan
                plot_title_suffix = "(Row Normalized)"
                mode_suffix = "_normalized"

            else:
                 # Raw attention: Ensure upper triangle is NaN for masking in heatmap
                 data_to_plot = np.tril(attn_weights_plot_orig, k=0)
                 data_to_plot[np.triu_indices_from(data_to_plot, k=1)] = np.nan

            plot_title = f"Layer {layer_idx}, Head {head_idx} {plot_title_suffix}"

            # --- Plot Heatmap ---
            # NaN values are masked by seaborn automatically
            sns.heatmap(
                data_to_plot,
                cmap=cmap,
                xticklabels=False,
                yticklabels=False,
                cbar=True,
                mask=np.isnan(data_to_plot),
                ax=ax,
                square=True,
                linewidths=0,
                linecolor='none',
                zorder=heatmap_zorder,
            )

            # --- Apply Tag Highlighting (No Clipping Here) ---
            # Draw the shaded spans *before* the white mask is applied
            if tag_indices:
                for tag_name, loc in tag_indices.items():
                    start_idx_tag = loc['tag_start']
                    end_idx_tag = loc['tag_end']

                    if 0 <= start_idx_tag < end_idx_tag <= token_count:
                        # Draw full spans, they will be covered by the white mask later
                        ax.axvspan(start_idx_tag, end_idx_tag, color=shade_color, alpha=shade_alpha, zorder=span_zorder, lw=0, clip_on=True)
                        ax.axhspan(start_idx_tag, end_idx_tag, color=shade_color, alpha=shade_alpha, zorder=span_zorder, lw=0, clip_on=True)
                        # Note: Boundary lines and numbers are drawn *later* with higher z-order
            diag_offset = 0.00 
            # Vertices: (Top-left near diagonal), (Top-right), (Bottom-right near diagonal)
            top_y_margin = -1.50 # No need to pull down from top? Labels are below.
            right_x_margin = -1.5 # Pull left from right edge

            upper_triangle_verts = [
                # Start near diagonal, just right of y-axis, at the top edge
                (diag_offset, top_y_margin),

                # Go across near top edge, but stop short of the right axis
                (token_count - right_x_margin, top_y_margin),

                # Go down the right side (pulled inwards), stopping just above the diagonal
                (token_count - right_x_margin, token_count - right_x_margin - diag_offset)
            ]
            # Create the polygon patch
            upper_mask_polygon = Polygon(
                upper_triangle_verts,
                facecolor='white',      # Use white to mask
                edgecolor='none',       # No border for the mask itself
                closed=True,
                zorder=upper_mask_zorder # Ensure it's above spans, below grid/lines
            )
            # Add the mask to the plot
            ax.add_patch(upper_mask_polygon)
            # --- ^ ^ ^ End White Mask Addition ^ ^ ^ ---


            # --- Draw grid lines for lower triangle ---
            # Make sure zorder is higher than the mask
            for k in range(1, token_count):
                # Vertical lines (only below or on diagonal)
                ax.plot([k, k], [k - 1, token_count], color=grid_line_color, linewidth=grid_line_width, zorder=grid_zorder)
                # Horizontal lines (only left of or on diagonal)
                ax.plot([0, k + 1], [k, k], color=grid_line_color, linewidth=grid_line_width, zorder=grid_zorder) # Extend slightly to ensure full coverage left-to-diag

            # --- Set Ticks and Labels (Keep as is) ---
            tick_positions = np.arange(token_count) + 0.5
            ax.set_xticks(tick_positions)
            ax.set_yticks(tick_positions)
            fontsize = max(3, 10 - token_count // 30)
            ax.set_xticklabels(tokens, rotation=90, fontsize=fontsize, ha='center')
            ax.set_yticklabels(tokens, rotation=0, fontsize=fontsize, va='center')

            # --- Add Fact Numbers and Boundary Lines (After Mask) ---
            if tag_indices:
                for tag_name, loc in tag_indices.items():
                    start_idx_tag = loc['tag_start']
                    end_idx_tag = loc['tag_end']

                    if 0 <= start_idx_tag < end_idx_tag <= token_count:
                        match = re.search(r'fact(\d+)', tag_name)
                        fact_num_str = match.group(1) if match else None

                        # Fact numbers (Only draw if center is in lower triangle/diagonal)
                        if fact_num_str:
                            center_x = (start_idx_tag + end_idx_tag) / 2.0
                            center_y = (start_idx_tag + end_idx_tag) / 2.0
                            # Condition: y >= x for lower triangle/diagonal (remember y-axis inverted)
                            # Use a small tolerance if needed: center_y >= center_x - 0.01
                            if center_y >= center_x:
                                dynamic_fontsize = max(5, min(number_fontsize, (end_idx_tag - start_idx_tag) * 1.5))
                                ax.text(center_x, center_y, fact_num_str,
                                        ha='center', va='center',
                                        fontsize=dynamic_fontsize + 20,
                                        color=number_color,
                                        alpha=number_alpha,
                                        zorder=number_zorder, # Ensure above mask & grid
                                        clip_on=True)

                        # Boundary lines (drawn over mask/grid)
                        ax.plot([start_idx_tag, start_idx_tag], [max(start_idx_tag, 0), token_count], color=line_color, linestyle=line_style, linewidth=line_width, zorder=line_zorder, clip_on=True)
                        ax.plot([end_idx_tag, end_idx_tag], [max(end_idx_tag, 0), token_count], color=line_color, linestyle=line_style, linewidth=line_width, zorder=line_zorder, clip_on=True)
                        ax.plot([0, min(start_idx_tag, token_count)], [start_idx_tag, start_idx_tag], color=line_color, linestyle=line_style, linewidth=line_width, zorder=line_zorder, clip_on=True)
                        ax.plot([0, min(end_idx_tag, token_count)], [end_idx_tag, end_idx_tag], color=line_color, linestyle=line_style, linewidth=line_width, zorder=line_zorder, clip_on=True)
            # --- End Fact Numbers/Lines ---

            # Add Solid Line after 'Answer' token (drawn over mask/grid)
            if answer_idx != -1:
                line_pos = answer_idx + 1
                if 0 < line_pos < token_count:
                    ax.plot([line_pos, line_pos], [line_pos, token_count],
                            color=answer_line_color, linestyle=answer_line_style, linewidth=answer_line_width,
                            zorder=answer_line_zorder, clip_on=True)
                    ax.plot([0, line_pos], [line_pos, line_pos],
                            color=answer_line_color, linestyle=answer_line_style, linewidth=answer_line_width,
                            zorder=answer_line_zorder, clip_on=True)

            # Final Plot Adjustments (keep as is)
            ax.set_xlim(0, token_count)
            ax.set_ylim(token_count, 0) # Inverted y-axis
            ax.set_title(plot_title, fontsize=10)
            plt.tight_layout()

            # Construct unique filename and save (keep as is)
            filename = os.path.join(output_dir, f"{base_filename}_layer_{layer_idx}_head_{head_idx}{mode_suffix}.png")
            try:
                plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"   Saved plot to {filename}")
            except Exception as e:
                print(f"   Error saving plot {filename}: {e}")

            plt.close(fig)

    print("\nFinished generating individual attention map images.")


# --- Run Visualization (Same parameters as before) ---
N_TOP_VALUES = 5
SELECTED_LAYERS = [0]
# SELECTED_HEADS = [0]
SELECTED_HEADS = [i for i in range(32)]
output_directory_top_n = f"cot/individual_maps_top{N_TOP_VALUES}"

plot_individual_attention_maps(
    tag_locations,
    answer_idx=answer_token_index,
    layers_to_show=SELECTED_LAYERS,
    heads_to_show=SELECTED_HEADS,
    output_dir=output_directory_top_n,
    base_filename=f"{model_name.split('/')[-1]}_attention",
    top_n=N_TOP_VALUES,
    normalize_rows=False,
    cmap="Blues",
    shade_color='grey',
    shade_alpha=0.15, # Keep alpha relatively low
    number_fontsize=25,
    number_alpha=0.5,
    number_color='blue',
    grid_line_color='grey',
    grid_line_width=0.1,
    answer_line_color='green',
    answer_line_width=1.5,
    line_color='purple',
    line_width=1.0
)