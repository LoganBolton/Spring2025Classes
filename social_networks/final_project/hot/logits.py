import torch
from transformers import LlamaForCausalLM, AutoTokenizer
import torch.nn.functional as F

# Step 1: Load model and tokenizer
# model_name = "meta-llama/Llama-3.1-8B"
model_name = "meta-llama/Llama-3.2-1B" # Using smaller model for faster testing


try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        output_attentions=True,
        device_map="cpu",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="eager"
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

# CoT
# text = """Answer with a number only. Question: For every 12 cans you recycle, you receive $0.50, and for every 5 kilograms of newspapers, you receive $1.50. If your family collected 144 cans and 20 kilograms of newspapers, how much money would you receive?
# Answer_Reasoning: There are 144/12 = 12 sets of 12 cans that the family collected.\nSo, the family would receive $0.50 x 12 = 6 for the cans.\nThere are 20/5 = 4 sets of 5 kilograms of newspapers that the family collected.\nSo, the family would receive $1.50 x 4 = 6 for the newspapers.\nTherefore, the family would receive a total of $6 + $6 = 12.
# Final_Answer: {"""

# hot
text = """Answer with a number only. Question: <fact1>For every 12 cans you recycle</fact1>, <fact2>you receive $0.50</fact2>, <fact3>and for every 5 kilograms of newspapers</fact3>, <fact4>you receive $1.50</fact4>. <fact5>If your family collected 144 cans and 20 kilograms of newspapers</fact5>, <fact6>how much money would you receive</fact6>?
Answer_Reasoning: There are <fact5>144</fact5>/<fact1>12</fact1> = 12 sets of 12 cans that the family collected.\nSo, the family would receive <fact2>$0.50</fact2> x <fact1>12</fact1> = $6 for the cans.\nThere are <fact5>20</fact5>/<fact3>5</fact3> = 4 sets of 5 kilograms of newspapers that the family collected.\nSo, the family would receive <fact4>$1.50</fact4> x 4 = 6 for the newspapers.\nTherefore, the family would receive a total of $6 + $6 = 12.
Final_Answer: {"""

# Step 3: Tokenize and run inference
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"]

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits  # shape: (1, seq_len, vocab_size)

# Step 4: Convert logits to log probabilities
log_probs = F.log_softmax(logits, dim=-1)  # shape: (1, seq_len, vocab_size)

# Step 5: Get log probs for the next token
next_token_log_probs = log_probs[0, -1]  # shape: (vocab_size,)

# Step 6: Display top-k next-token predictions
top_k = 10
top_log_probs, top_indices = torch.topk(next_token_log_probs, top_k)
top_tokens = tokenizer.convert_ids_to_tokens(top_indices.tolist())

# print("\nTop Next Token Predictions:")
# for token, log_prob in zip(top_tokens, top_log_probs):
#     print(f"Token: {token:>15} | Log Prob: {log_prob.item():.4f}")


top_probs = torch.exp(top_log_probs)

# Display top-k tokens and their actual probabilities
print("\nTop Next Token Predictions (with probabilities):")
for token, prob in zip(top_tokens, top_probs):
    print(f"Token: {token:>15} | Probability: {prob.item():.4f}")
    
# CoT
# Token:               / | Probability: 0.0582
# Token:              eq | Probability: 0.0514
# Token:               1 | Probability: 0.0338
# Token:              12 | Probability: 0.0271

# HoT 
# Token:              12 | Probability: 0.1000
# Token:          answer | Probability: 0.0385
# Token:               6 | Probability: 0.0343