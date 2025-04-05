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
# model_name = "meta-llama/Llama-3.1-8B"
model_name = "meta-llama/Llama-3.2-1B" # Using 1B for faster local testing

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        output_attentions=True,
        device_map="cpu",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="eager" # Use eager for easier debugging if needed
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
        attn_implementation="eager"
    )

# --- (Keep Step 2: Text Prep and Token Cleaning - No changes needed) ---
# Step 2: Prepare input text
text = """**Reformatted Question:** A machine is set up in such a way that it will <fact1>short circuit if both the black wire and the red wire touch the battery at the same time</fact1>. The machine will <fact2>not short circuit if just one of these wires touches the battery</fact2>. The black wire is designated as the one that is supposed to touch the battery, while the red wire is supposed to remain in some other part of the machine. One day, the <fact3>black wire and the red wire both end up touching the battery at the same time</fact3>. There is a short circuit. Did the <fact4>black wire cause the short circuit</fact4>? Options: - Yes - No

**Answer:** The question states that the machine <fact1>short circuits only when both the black and red wires touch the battery simultaneously</fact1>. It also specifies that <fact2>touching the battery with only one wire will not cause a short circuit</fact2>.  While the <fact3>black wire did touch the battery</fact3>, it was the <fact3>simultaneous contact of both wires</fact3> that triggered the short circuit. Therefore, the <fact4>black wire alone did not cause the short circuit</fact4>. The answer is {No}."""

inputs = tokenizer(text, return_tensors="pt")
base_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

def clean_tokens(tokens):
    cleaned = []
    for t in tokens:
        t = t.replace(' ', ' ') # Replaces the specific SentencePiece underscore with a standard space
        # Handle special tokens if necessary (adjust based on tokenizer)
        if t == '<|begin_of_text|>':
            cleaned.append('[BOS]')
        # elif t == '</s>': # Example for end of sentence token if present
        #     cleaned.append('[EOS]')
        elif t == 'Ċ': # Handle potential newline tokens
             cleaned.append('\n')
        # Add specific handling for other special tokens if they appear
        elif t in ['<0x0A>', '<0x0D>']: # Example: Handle newline hex codes if tokenizer uses them
             cleaned.append('\n')
        else:
            cleaned.append(t)
    return cleaned


tokens_with_bos = clean_tokens(base_tokens)
tokens = tokens_with_bos[1:] # Remove [BOS] token


# --- (Keep Find Tag Indices Function v4 - No changes needed) ---
def find_tag_indices_multi_token_v4(token_list):
    all_occurrences = {}
    temp_starts = {}
    instance_counters = {}
    i = 0
    while i < len(token_list):
        current_token = token_list[i].strip() # Use strip() for easier comparison
        is_start, tag_num_start, start_tag_len = False, None, 0
        tag_start_index = -1 # Store the index of '<'

        # Check for <factN> variations (More robust checks)
        if current_token == '<' and i + 3 < len(token_list):
                token2, token3, token4 = token_list[i+1].strip().lower(), token_list[i+2].strip(), token_list[i+3].strip()
                if token2 == 'fact' and token3.isdigit() and token4.startswith('>'):
                    is_start, tag_num_start, start_tag_len = True, token3, 4
                    tag_start_index = i
        elif current_token.startswith('<fact') and current_token.endswith('>') and len(current_token) > 6:
             match = re.match(r'<fact(\d+)>', current_token)
             if match:
                 is_start, tag_num_start, start_tag_len = True, match.group(1), 1
                 tag_start_index = i

        if is_start:
            base_tag_name = f"fact{tag_num_start}"
            if base_tag_name not in temp_starts: temp_starts[base_tag_name] = []
            temp_starts[base_tag_name].append(tag_start_index)
            i += start_tag_len
            continue

        is_end, tag_num_end, end_tag_len = False, None, 0
        closing_tag_start_index = -1

         # Check for </factN> variations (More robust checks)
        if current_token == '<' and i + 4 < len(token_list):
                token2, token3, token4, token5 = token_list[i+1].strip(), token_list[i+2].strip().lower(), token_list[i+3].strip(), token_list[i+4].strip()
                if token2 == '/' and token3 == 'fact' and token4.isdigit() and token5.startswith('>'):
                    is_end, tag_num_end, end_tag_len = True, token4, 5
                    closing_tag_start_index = i
        elif current_token == '</' and i + 3 < len(token_list):
                 token2, token3, token4 = token_list[i+1].strip().lower(), token_list[i+2].strip(), token_list[i+3].strip()
                 if token2 == 'fact' and token3.isdigit() and token4.startswith('>'):
                     is_end, tag_num_end, end_tag_len = True, token3, 4
                     closing_tag_start_index = i
        elif current_token.startswith('</fact') and current_token.endswith('>') and len(current_token) > 7:
             match = re.match(r'</fact(\d+)>', current_token)
             if match:
                 is_end, tag_num_end, end_tag_len = True, match.group(1), 1
                 closing_tag_start_index = i

        if is_end:
            base_tag_name = f"fact{tag_num_end}"
            if base_tag_name in temp_starts and temp_starts[base_tag_name]:
                tag_start_idx_actual = temp_starts[base_tag_name].pop(0)
                tag_end_idx_actual = closing_tag_start_index + end_tag_len # Exclusive end index
                instance_counters[base_tag_name] = instance_counters.get(base_tag_name, 0) + 1
                unique_tag_name = f"{base_tag_name}_{instance_counters[base_tag_name]}"
                all_occurrences[unique_tag_name] = {
                    'tag_start': tag_start_idx_actual,
                    'tag_end': tag_end_idx_actual
                 }
                i += end_tag_len
                continue
        i += 1

    # Check for unmatched start tags
    for tag, starts in temp_starts.items():
        if starts:
            print(f"Warning: Unmatched start tags found for {tag} at indices {starts}")

    return all_occurrences

tag_locations = find_tag_indices_multi_token_v4(tokens)
print("\nFound Tag Locations (Indices including tags):")
if not tag_locations:
    print("  No tags found.")
for tag, loc in tag_locations.items():
    print(f"  {tag}: Start Index={loc['tag_start']}, End Index (Exclusive)={loc['tag_end']}")


# --- Find 'Answer' token index (More Robust Check) ---
target_token_options = [' Answer', 'Answer', ':', '\nAnswer'] # Added newline possibility
answer_token_index = -1
search_start_index = 150 # Start searching after the initial facts section

try:
    star_indices = [i for i, x in enumerate(tokens) if x.strip() == '**']
    potential_star_indices = [idx for idx in star_indices if idx > search_start_index]

    if potential_star_indices:
        # Find the '**' that is most likely followed by 'Answer' (usually the first one after facts)
        star_index = potential_star_indices[0]
        print(f"\nFound potential '**' marker for Answer at index {star_index}.")

        # Look for target tokens immediately after the stars
        found_answer = False
        for i, option in enumerate(target_token_options):
             # Check token immediately after '**'
             check_idx_1 = star_index + 1
             if check_idx_1 < len(tokens) and tokens[check_idx_1] == option:
                 answer_token_index = check_idx_1
                 print(f"Found answer token '{option}' at index: {answer_token_index}")
                 found_answer = True
                 break

             # Check token after potential space/colon after '**'
             check_idx_2 = star_index + 2
             if check_idx_2 < len(tokens) and tokens[check_idx_2] == option:
                 answer_token_index = check_idx_2
                 print(f"Found answer token '{option}' at index: {answer_token_index}")
                 found_answer = True
                 break

             # Special check if Answer is tokenized like ' An', 'swer' (after '**')
             if option == ' Answer' and check_idx_1 < len(tokens) - 1:
                 if tokens[check_idx_1] == ' An' and tokens[check_idx_1+1] == 'swer':
                     answer_token_index = check_idx_1 # Mark the start token ' An'
                     print(f"Found multi-token answer ' An', 'swer' starting at index: {answer_token_index}")
                     found_answer = True
                     break
                 # Check after potential space/colon
                 if check_idx_2 < len(tokens) -1 and tokens[check_idx_2] == ' An' and tokens[check_idx_2+1] == 'swer':
                     answer_token_index = check_idx_2 # Mark the start token ' An'
                     print(f"Found multi-token answer ' An', 'swer' starting at index: {answer_token_index}")
                     found_answer = True
                     break

        if not found_answer:
             print(f"Warning: Could not reliably find 'Answer' token near '**' marker at index {star_index}.")
             print("Tokens near marker:", tokens[max(0, star_index-1):min(len(tokens), star_index+6)])

    else:
         print(f"Warning: Could not find '**' marker after index {search_start_index}.")


except Exception as e:
    print(f"\nError during 'Answer' token search: {e}")
    answer_token_index = -1


# --- (Keep Step 3: Inference and Attention Extraction - No changes needed) ---
# Step 3: Run inference and extract attention weights
with torch.no_grad():
    outputs = model(**inputs)
attentions = outputs.attentions # Tuple of tensors, one for each layer
num_layers = model.config.num_hidden_layers
num_heads = model.config.num_attention_heads
seq_len = len(tokens) # Length excluding BOS


# --- (Keep Step 5: Print Model Info - No changes needed) ---
# Step 5: Print model info
print(f"\nModel: {model_name}")
print(f"Layers: {num_layers}, Heads per layer: {num_heads}")
print(f"Sequence length (excluding BOS): {seq_len}")


# --- (Keep plot_individual_attention_maps function - Largely unchanged) ---
# (No functional changes needed here based on the request, but parameter names are consistent)
def plot_individual_attention_maps(
    tag_indices,
    answer_idx,
    layers_to_show=[0],
    heads_to_show=[0],
    output_dir="hot",
    normalize_rows=False,
    line_color='red', # Default tag line color
    line_style='--',
    line_width=1,
    grid_line_color='black', # Default grid color
    grid_line_width=0.15,
    shade_color='grey', # Default shade color
    shade_alpha=0.25,
    number_fontsize=30,
    number_alpha=0.4,
    number_color='black', # Default number color
    answer_line_color='black', # Default answer line color
    answer_line_width=1.5,
    answer_line_style='-',
    cmap="Blues" # Default colormap
):
    token_count = len(tokens)
    base_width = max(12, token_count / 4)
    base_height = max(12, token_count / 5)

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error creating directory {output_dir}: {e}")
            return

    clip_verts = [(0, 0), (token_count, token_count), (0, token_count), (0, 0)]
    safe_model_name = model_name.split('/')[-1].replace('-', '_') # Get model name for filename

    for layer_idx in layers_to_show:
        if not (0 <= layer_idx < num_layers):
             print(f"Warning: Layer index {layer_idx} invalid (max is {num_layers-1}). Skipping.")
             continue
        if layer_idx >= len(attentions) or attentions[layer_idx] is None:
            print(f"Warning: No attention data found for Layer {layer_idx}. Skipping.")
            continue

        for head_idx in heads_to_show:
            if not (0 <= head_idx < num_heads):
                print(f"Warning: Head index {head_idx} invalid (max is {num_heads-1}). Skipping.")
                continue

            print(f"Generating plot for Layer {layer_idx}, Head {head_idx}...")
            fig, ax = plt.subplots(figsize=(base_width, base_height))
            clip_polygon = Polygon(clip_verts, transform=ax.transData, facecolor='none', edgecolor='none')

            try:
                attn_weights_all = attentions[layer_idx][0, head_idx].detach().cpu().numpy()
                attn_weights_plot_orig = attn_weights_all[1:, 1:]
                if attn_weights_plot_orig.shape[0] != token_count or attn_weights_plot_orig.shape[1] != token_count:
                     attn_weights_plot_orig = attn_weights_plot_orig[:token_count, :token_count]
            except IndexError as e:
                 print(f"  Error extracting attention weights for Layer {layer_idx}, Head {head_idx}: {e}. Skipping head.")
                 plt.close(fig)
                 continue

            if normalize_rows:
                attn_weights_normalized = np.copy(attn_weights_plot_orig)
                epsilon = 1e-9
                for i in range(min(token_count, attn_weights_normalized.shape[0])):
                    row_slice = attn_weights_normalized[i, :i+1]
                    max_val = np.max(row_slice) if row_slice.size > 0 else 0
                    if max_val > epsilon:
                         attn_weights_normalized[i, :i+1] = row_slice / max_val
                    else:
                         attn_weights_normalized[i, :i+1] = 0.0
                    if i + 1 < attn_weights_normalized.shape[1]:
                        attn_weights_normalized[i, i+1:] = 0.0
                data_to_plot = attn_weights_normalized
                plot_title = f"{safe_model_name} - Layer {layer_idx}, Head {head_idx} (Row Normalized)"
            else:
                data_to_plot = attn_weights_plot_orig
                plot_title = f"{safe_model_name} - Layer {layer_idx}, Head {head_idx} (Raw Attention)"

            if data_to_plot.shape != (token_count, token_count):
                print(f"  Error: data_to_plot shape mismatch. Skipping head.")
                plt.close(fig)
                continue

            mask_upper = np.triu(np.ones_like(data_to_plot, dtype=bool), k=1)
            heatmap_zorder, grid_zorder, span_zorder, number_zorder, line_zorder, answer_line_zorder = 0.5, 1.0, 1.5, 2.0, 2.5, 2.6

            try:
                sns.heatmap( data_to_plot, cmap=cmap, xticklabels=False, yticklabels=False, cbar=True, mask=mask_upper, ax=ax, square=True, linewidths=0, linecolor='none', zorder=heatmap_zorder)
            except Exception as e:
                print(f"  Error during heatmap creation: {e}. Skipping head.")
                plt.close(fig)
                continue

            # --- Grid, Ticks, Tags, Answer Line ---
            for k in range(1, token_count):
                ax.plot([k, k], [max(k - 1, 0), token_count], color=grid_line_color, linewidth=grid_line_width, zorder=grid_zorder)
                ax.plot([0, min(k, token_count)], [k, k], color=grid_line_color, linewidth=grid_line_width, zorder=grid_zorder)

            tick_positions = np.arange(token_count) + 0.5
            ax.set_xticks(tick_positions)
            ax.set_yticks(tick_positions)
            fontsize = max(3, 10 - token_count // 30)
            try:
                ax.set_xticklabels(tokens, rotation=90, fontsize=fontsize, ha='center')
                ax.set_yticklabels(tokens, rotation=0, fontsize=fontsize, va='center')
            except Exception as e:
                print(f"  Warning: Error setting tick labels: {e}")
                ax.set_xticklabels([])
                ax.set_yticklabels([])

            if tag_indices:
                for tag_name, loc in tag_indices.items():
                    start_idx_tag = loc['tag_start']
                    end_idx_tag = loc['tag_end']
                    if 0 <= start_idx_tag < end_idx_tag <= token_count:
                        match = re.search(r'fact(\d+)', tag_name)
                        fact_num_str = match.group(1) if match else None
                        span_v = ax.axvspan(start_idx_tag, end_idx_tag, color=shade_color, alpha=shade_alpha, zorder=span_zorder, lw=0)
                        span_h = ax.axhspan(start_idx_tag, end_idx_tag, color=shade_color, alpha=shade_alpha, zorder=span_zorder, lw=0)
                        span_v.set_clip_path(clip_polygon)
                        span_h.set_clip_path(clip_polygon)
                        if fact_num_str:
                            center_x = (start_idx_tag + end_idx_tag) / 2.0
                            center_y = (start_idx_tag + end_idx_tag) / 2.0
                            dynamic_fontsize = max(5, min(number_fontsize, (end_idx_tag - start_idx_tag) * 1.5))
                            ax.text(center_x, center_y, fact_num_str, ha='center', va='center', fontsize=dynamic_fontsize, color=number_color, alpha=number_alpha, zorder=number_zorder, clip_on=True)
                        ax.plot([start_idx_tag, start_idx_tag], [max(start_idx_tag, 0), token_count], color=line_color, linestyle=line_style, linewidth=line_width, zorder=line_zorder)
                        ax.plot([end_idx_tag, end_idx_tag], [max(end_idx_tag, 0), token_count], color=line_color, linestyle=line_style, linewidth=line_width, zorder=line_zorder)
                        ax.plot([0, min(start_idx_tag, token_count)], [start_idx_tag, start_idx_tag], color=line_color, linestyle=line_style, linewidth=line_width, zorder=line_zorder)
                        ax.plot([0, min(end_idx_tag, token_count)], [end_idx_tag, end_idx_tag], color=line_color, linestyle=line_style, linewidth=line_width, zorder=line_zorder)

            if answer_idx != -1:
                line_pos = answer_idx + 1
                if 0 < line_pos < token_count:
                    ax.plot([line_pos, line_pos], [line_pos, token_count], color=answer_line_color, linestyle=answer_line_style, linewidth=answer_line_width, zorder=answer_line_zorder)
                    ax.plot([0, line_pos], [line_pos, line_pos], color=answer_line_color, linestyle=answer_line_style, linewidth=answer_line_width, zorder=answer_line_zorder)

            ax.set_xlim(0, token_count)
            ax.set_ylim(token_count, 0)
            ax.set_title(plot_title, fontsize=12)

            norm_suffix = "_normalized" if normalize_rows else "_raw"
            filename = os.path.join(output_dir, f"{safe_model_name}_attention_layer_{layer_idx}_head_{head_idx}{norm_suffix}.png")
            try:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
            except Exception as e:
                print(f"   Error saving plot {filename}: {e}")
            plt.close(fig)

    print(f"\nFinished generating {len(layers_to_show) * len(heads_to_show)} individual attention map images.")


# --- CORRECTED and Parameterized Function: Plot Average Attention Map ---
def plot_average_attention_map(
    tag_indices,
    answer_idx,
    layer_idx,
    output_dir="hot/average_maps",
    normalize_rows=False,
    # --- Colors are now parameters ---
    line_color='red',        # Tag boundary lines
    line_style='--',
    line_width=1,
    grid_line_color='black', # Grid lines
    grid_line_width=0.15,
    shade_color='grey',      # Tag shading
    shade_alpha=0.25,
    number_fontsize=30,
    number_alpha=0.4,
    number_color='black',    # Fact numbers
    answer_line_color='black',# Line after 'Answer'
    answer_line_width=1.5,
    answer_line_style='-',
    cmap="Blues"             # Heatmap colormap
):
    token_count = len(tokens)
    base_width = max(12, token_count / 4)
    base_height = max(12, token_count / 5)
    safe_model_name = model_name.split('/')[-1].replace('-', '_')

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error creating directory {output_dir}: {e}")
            return

    if not (0 <= layer_idx < num_layers):
         print(f"Error: Layer index {layer_idx} invalid. Cannot plot average.")
         return
    if layer_idx >= len(attentions) or attentions[layer_idx] is None:
        print(f"Warning: No attention data for Layer {layer_idx}. Cannot plot average.")
        return

    print(f"Generating average attention plot for Layer {layer_idx}...")

    # --- Calculate Average Attention (Corrected) ---
    try:
        layer_attentions_tensor = attentions[layer_idx]
        average_attention_tensor = torch.mean(layer_attentions_tensor, dim=1, keepdim=False)
        average_attention_np_all = average_attention_tensor[0].detach().cpu().numpy()
        avg_attn_plot_orig = average_attention_np_all[1:, 1:]
        if avg_attn_plot_orig.shape[0] != token_count or avg_attn_plot_orig.shape[1] != token_count:
            avg_attn_plot_orig = avg_attn_plot_orig[:token_count, :token_count]
        if avg_attn_plot_orig.shape != (token_count, token_count):
             raise ValueError(f"Final shape incorrect: {avg_attn_plot_orig.shape}")
    except Exception as e:
        print(f"  Error calculating average attention: {e}. Cannot generate plot.")
        return

    fig, ax = plt.subplots(figsize=(base_width, base_height))
    clip_verts = [(0, 0), (token_count, token_count), (0, token_count), (0, 0)]
    clip_polygon = Polygon(clip_verts, transform=ax.transData, facecolor='none', edgecolor='none')

    # --- Apply Row Normalization ---
    if normalize_rows:
        if not isinstance(avg_attn_plot_orig, np.ndarray) or avg_attn_plot_orig.ndim != 2:
             print(f"  Error: Invalid array for normalization. Skipping.")
             data_to_plot = avg_attn_plot_orig
             plot_title = f"Average Attention - {safe_model_name} Layer {layer_idx} (Raw - Norm Failed)"
        else:
            attn_weights_normalized = np.copy(avg_attn_plot_orig)
            epsilon = 1e-9
            for i in range(min(token_count, attn_weights_normalized.shape[0])):
                row_slice = attn_weights_normalized[i, :i+1]
                max_val = np.max(row_slice) if row_slice.size > 0 else 0
                if max_val > epsilon: attn_weights_normalized[i, :i+1] = row_slice / max_val
                else: attn_weights_normalized[i, :i+1] = 0.0
                if i + 1 < attn_weights_normalized.shape[1]: attn_weights_normalized[i, i+1:] = 0.0
            data_to_plot = attn_weights_normalized
            plot_title = f"Average Attention - {safe_model_name} Layer {layer_idx} (Row Normalized)"
    else:
        data_to_plot = avg_attn_plot_orig
        plot_title = f"Average Attention - {safe_model_name} Layer {layer_idx} (Raw)"

    if data_to_plot.shape != (token_count, token_count):
        print(f"  Error: Final data shape incorrect. Cannot plot.")
        plt.close(fig)
        return

    mask_upper = np.triu(np.ones_like(data_to_plot, dtype=bool), k=1)
    heatmap_zorder, grid_zorder, span_zorder, number_zorder, line_zorder, answer_line_zorder = 0.5, 1.0, 1.5, 2.0, 2.5, 2.6

    try:
        sns.heatmap(data_to_plot, cmap=cmap, xticklabels=False, yticklabels=False, cbar=True, mask=mask_upper, ax=ax, square=True, linewidths=0, linecolor='none', zorder=heatmap_zorder)
    except Exception as e:
        print(f"  Error during heatmap creation: {e}. Aborting plot.")
        plt.close(fig)
        return

    # --- Grid, Ticks, Tags, Answer Line (Using parameters) ---
    for k in range(1, token_count):
        ax.plot([k, k], [max(k - 1, 0), token_count], color=grid_line_color, linewidth=grid_line_width, zorder=grid_zorder)
        ax.plot([0, min(k, token_count)], [k, k], color=grid_line_color, linewidth=grid_line_width, zorder=grid_zorder)

    tick_positions = np.arange(token_count) + 0.5
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    fontsize = max(3, 10 - token_count // 30)
    try:
        ax.set_xticklabels(tokens, rotation=90, fontsize=fontsize, ha='center')
        ax.set_yticklabels(tokens, rotation=0, fontsize=fontsize, va='center')
    except Exception as e:
        print(f"  Warning: Error setting tick labels: {e}")
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # Use passed color parameters for tags and numbers
    if tag_indices:
        for tag_name, loc in tag_indices.items():
            start_idx_tag = loc['tag_start']
            end_idx_tag = loc['tag_end']
            if 0 <= start_idx_tag < end_idx_tag <= token_count:
                match = re.search(r'fact(\d+)', tag_name)
                fact_num_str = match.group(1) if match else None
                # Use shade_color, shade_alpha
                span_v = ax.axvspan(start_idx_tag, end_idx_tag, color=shade_color, alpha=shade_alpha, zorder=span_zorder, lw=0)
                span_h = ax.axhspan(start_idx_tag, end_idx_tag, color=shade_color, alpha=shade_alpha, zorder=span_zorder, lw=0)
                span_v.set_clip_path(clip_polygon)
                span_h.set_clip_path(clip_polygon)
                if fact_num_str:
                    center_x = (start_idx_tag + end_idx_tag) / 2.0
                    center_y = (start_idx_tag + end_idx_tag) / 2.0
                    dynamic_fontsize = max(5, min(number_fontsize, (end_idx_tag - start_idx_tag) * 1.5))
                    # Use number_color, number_alpha
                    ax.text(center_x, center_y, fact_num_str, ha='center', va='center', fontsize=dynamic_fontsize, color=number_color, alpha=number_alpha, zorder=number_zorder, clip_on=True)
                # Use line_color, line_style, line_width
                ax.plot([start_idx_tag, start_idx_tag], [max(start_idx_tag, 0), token_count], color=line_color, linestyle=line_style, linewidth=line_width, zorder=line_zorder)
                ax.plot([end_idx_tag, end_idx_tag], [max(end_idx_tag, 0), token_count], color=line_color, linestyle=line_style, linewidth=line_width, zorder=line_zorder)
                ax.plot([0, min(start_idx_tag, token_count)], [start_idx_tag, start_idx_tag], color=line_color, linestyle=line_style, linewidth=line_width, zorder=line_zorder)
                ax.plot([0, min(end_idx_tag, token_count)], [end_idx_tag, end_idx_tag], color=line_color, linestyle=line_style, linewidth=line_width, zorder=line_zorder)

    # Use passed color parameters for answer line
    if answer_idx != -1:
        line_pos = answer_idx + 1
        if 0 < line_pos < token_count:
            ax.plot([line_pos, line_pos], [line_pos, token_count], color=answer_line_color, linestyle=answer_line_style, linewidth=answer_line_width, zorder=answer_line_zorder)
            ax.plot([0, line_pos], [line_pos, line_pos], color=answer_line_color, linestyle=answer_line_style, linewidth=answer_line_width, zorder=answer_line_zorder)
    # --- End Plot Elements ---

    ax.set_xlim(0, token_count)
    ax.set_ylim(token_count, 0)
    ax.set_title(plot_title, fontsize=12)

    norm_suffix = "_normalized" if normalize_rows else "_raw"
    filename = os.path.join(output_dir, f"{safe_model_name}_avg_attention_layer_{layer_idx}{norm_suffix}.png")
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"   Saved average plot to {filename}")
    except Exception as e:
        print(f"   Error saving average plot {filename}: {e}")
    plt.close(fig)

    print(f"Finished generating average attention map for layer {layer_idx}.")


# --- Run Visualization ---

# === Plot Average Attention for Layer 0 (Normalized - Viridis) ===
print("\n--- Generating Average Head Plots ---")
plot_average_attention_map(
    tag_indices=tag_locations,
    answer_idx=answer_token_index,
    layer_idx=0,
    output_dir="hot/average_maps",
    normalize_rows=True,
    cmap="viridis",          # Viridis heatmap
    # Colors for Viridis (contrast well)
    line_color='red',        # Red lines stand out
    shade_color='grey',
    shade_alpha=0.2,         # Slightly higher alpha can work here
    number_color='white',    # White numbers stand out on dark parts
    number_alpha=0.6,        # Make numbers slightly more solid
    grid_line_color='white', # White grid lines
    answer_line_color='cyan',# Bright answer line
    answer_line_width=2.0
)



print("\nScript finished.")