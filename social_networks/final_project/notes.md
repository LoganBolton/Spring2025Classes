# Model Sizes

## Llama 3.2 1B
16 layers
32 heads

## Gemma 2 2B
26 layers
8 heads

## Qwen 2.5 0.5B
24 layers
14 heads

## Qwen 2.5 1.5 B
28 layers
12 heads

# Analysis
## layer 0, head 0
You can look at a node and determine its parent

## layer 0, head 3
you can look at an arrow and determine its parent

## layer 0, head 12
A parent node to all the child nodes in a graph


# Notation
- to and -> are basically the same
- Parents of: is complicated


# Training the model
Given the immediate drop to ln(2) and the flatline, the interaction between the attention weights (attn_edge_weight) and the GCN normalization is the strongest suspect. Start by printing the weight statistics and testing the model with attn_edge_weight=None.

# Things To Try
- Probably need to switch to ->
- Use argmax of attention
- Augment with all the different positons
    - Its probably attending too much to the first pair
    - Try mixing it up by having multiple different generations with different orderings and then avg togethr
- Analysis of average on toy:
    - Everything attending to either start token or just self attention


# WIP TODO
- Change eval so that it uses the same test/val split



# Ablations
Looks like just using all nodes doesn't work ...

## No args
Epoch: 050, Train Loss: 0.6466, Test Loss: 0.6543, Test Acc: 0.5500, Test F1: 0.0000

--- Training Complete ---
Best Test F1 Score: 0.2479 at Epoch 1
Training history saved to gcn_training_output_hardcoded/training_history.csv

--- Final Evaluation on Test Set using Best Model ---
Final Results - Test Loss: 0.6875, Test Acc: 0.5111, Test F1: 0.2479

--- Final Evaluation on Test Set using Best Model ---
Loaded best model from gcn_training_output_hardcoded/best_model.pt
Final Results - Test Loss: 0.6875, Test Acc: 0.5111, Test F1: 0.2479

## top k == 3

Epoch: 050, Train Loss: 0.6611, Test Loss: 0.6607, Test Acc: 0.5368, Test F1: 0.0000

--- Training Complete ---
Best Test F1 Score: 0.3175 at Epoch 1
Training history saved to attention_matrices/arg_3/combined/gcn_training_output_hardcoded/training_history.csv

--- Final Evaluation on Test Set using Best Model ---
Final Results - Test Loss: 0.6876, Test Acc: 0.5474, Test F1: 0.3175

## MLP 
looks like top 3 is marginally better?
- This is with max 6 nodes


- Seems to be that doing top 3 when nodes == 10 is not the move

### top k == 3
--- Training Complete ---
Best Test F1 Score: 0.7483 at Epoch 178
Training history saved to attention_matrices/arg_3/combined/mlp_baseline_training_output/training_history.csv

--- Final Evaluation on Test Set using Best Model ---
Final Results - Test Loss: 0.4674, Test Acc (non-diag): 0.8150, Test F1 (non-diag): 0.7483

### no args
--- Training Complete ---
Best Test F1 Score: 0.7336 at Epoch 197
Training history saved to attention_matrices/no_args/combined/mlp_baseline_training_output/training_history.csv

--- Final Evaluation on Test Set using Best Model ---
Final Results - Test Loss: 0.4701, Test Acc (non-diag): 0.7833, Test F1 (non-diag): 0.7336

## size 10
taking the max of 3 is actually better
### arg 3
epoch,train_loss,test_loss,test_accuracy,test_f1
200,0.389309224486351,0.6456100821495057,0.6011111111111112,0.5184439973172368

### no args
Epoch: 200, Train Loss: 0.4577, Test Loss: 0.6107, Test Acc: 0.5769, Test F1: 0.4366

## size 7
### 1000 training, no args, no val
Best Test F1 Score: 0.7275 at Epoch 190
Training history saved to attention_matrices/no_args_7_1b/combined/mlp_baseline_training_output/training_history.csv

--- Final Evaluation on Test Set using Best Model ---
Final Results - Test Loss: 0.3801, Test Acc (non-diag): 0.7752, Test F1 (non-diag): 0.7275

**with good params**
EPOCHS = 200
LEARNING_RATE = 0.001
BATCH_SIZE = 32
HIDDEN_CHANNELS = 4096 # MLP might need more capacity
TEST_SPLIT = 0.2
RANDOM_SEED = 42

--- Training Complete ---
Best Test F1 Score: 0.8519 at Epoch 200
Training history saved to attention_matrices/no_args_7_1b/combined/mlp_baseline_training_output/training_history.csv

--- Final Evaluation on Test Set using Best Model ---
Final Results - Test Loss: 0.2988, Test Acc (non-diag): 0.8829, Test F1 (non-diag): 0.8519


--- Final Evaluation on Test Set using Best Model ---
Final Results - Test Loss: 0.2393, Test Acc (non-diag): 0.9031, Test F1 (non-diag): 0.8776

keep it at 128


# best training run

EPOCHS = 200
LEARNING_RATE = 0.001
BATCH_SIZE = 128
HIDDEN_CHANNELS = 4096 # MLP might need more capacity
TEST_SPLIT = 0.2
VAL_SPLIT = 0.1 # 10% of the original training data
RANDOM_SEED = 42


--- Training Complete ---
Best Test F1 Score: 0.8816 at Epoch 187
Training history saved to attention_matrices/no_args_7_1b/combined/mlp_baseline_training_output/training_history.csv

--- Final Evaluation on Test Set using Best Model ---
Final Results - Test Loss: 0.2136, Test Acc (non-diag): 0.9064, Test F1 (non-diag): 0.8816

--- Sample Predictions from Test Set (using best model) ---

--- Example 1 (Graph ID: 521) ---
Ground Truth Adjacency (Undirected):
[[0 1 0 0 0 1 1]
 [1 0 1 0 0 0 1]
 [0 1 0 0 0 0 0]
 [0 0 0 0 1 1 1]
 [0 0 0 1 0 0 0]
 [1 0 0 1 0 0 0]
 [1 1 0 1 0 0 0]]

Predicted Adjacency:
[[0 0 0 1 0 0 1]
 [0 0 1 0 0 0 1]
 [0 1 0 0 0 0 0]
 [1 0 0 0 1 1 1]
 [0 0 0 1 0 0 0]
 [0 0 0 1 0 0 0]
 [1 1 0 1 0 0 0]]
--------------------

--- Example 2 (Graph ID: 737) ---
Ground Truth Adjacency (Undirected):
[[0 0 1 1 0 0 0]
 [0 0 0 0 0 0 0]
 [1 0 0 1 0 0 0]
 [1 0 1 0 0 1 0]
 [0 0 0 0 0 0 0]
 [0 0 0 1 0 0 0]
 [0 0 0 0 0 0 0]]

Predicted Adjacency:
[[0 0 1 1 0 0 0]
 [0 0 0 0 0 0 0]
 [1 0 0 1 0 1 0]
 [1 0 1 0 0 0 0]
 [0 0 0 0 0 0 0]
 [0 0 1 0 0 0 0]
 [0 0 0 0 0 0 0]]
--------------------

--- Example 3 (Graph ID: 740) ---
Ground Truth Adjacency (Undirected):
[[0 1 0 0 0 0 0]
 [1 0 1 0 1 0 0]
 [0 1 0 1 0 1 0]
 [0 0 1 0 0 0 0]
 [0 1 0 0 0 1 0]
 [0 0 1 0 1 0 1]
 [0 0 0 0 0 1 0]]

Predicted Adjacency:
[[0 1 0 0 0 0 0]
 [1 0 1 0 1 0 0]
 [0 1 0 1 0 0 0]
 [0 0 1 0 0 1 0]
 [0 1 0 0 0 1 0]
 [0 0 0 1 1 0 1]
 [0 0 0 0 0 1 0]]



 # good graphs

 --- Example 1 (Graph ID: 521) ---
Ground Truth Adjacency (Undirected):
[[0 1 0 0 0 1 1]
 [1 0 1 0 0 0 1]
 [0 1 0 0 0 0 0]
 [0 0 0 0 1 1 1]
 [0 0 0 1 0 0 0]
 [1 0 0 1 0 0 0]
 [1 1 0 1 0 0 0]]

Predicted Adjacency:
[[0 0 0 1 0 0 1]
 [0 0 1 0 0 0 1]
 [0 1 0 0 0 0 0]
 [1 0 0 0 1 1 1]
 [0 0 0 1 0 0 0]
 [0 0 0 1 0 0 0]
 [1 1 0 1 0 0 0]]
--------------------

--- Example 2 (Graph ID: 737) ---
Ground Truth Adjacency (Undirected):
[[0 0 1 1 0 0 0]
 [0 0 0 0 0 0 0]
 [1 0 0 1 0 0 0]
 [1 0 1 0 0 1 0]
 [0 0 0 0 0 0 0]
 [0 0 0 1 0 0 0]
 [0 0 0 0 0 0 0]]

Predicted Adjacency:
[[0 0 1 1 0 0 0]
 [0 0 0 0 0 0 0]
 [1 0 0 1 0 1 0]
 [1 0 1 0 0 0 0]
 [0 0 0 0 0 0 0]
 [0 0 1 0 0 0 0]
 [0 0 0 0 0 0 0]]
--------------------

--- Example 3 (Graph ID: 740) ---
Ground Truth Adjacency (Undirected):
[[0 1 0 0 0 0 0]
 [1 0 1 0 1 0 0]
 [0 1 0 1 0 1 0]
 [0 0 1 0 0 0 0]
 [0 1 0 0 0 1 0]
 [0 0 1 0 1 0 1]
 [0 0 0 0 0 1 0]]

Predicted Adjacency:
[[0 1 0 0 0 0 0]
 [1 0 1 0 1 0 0]
 [0 1 0 1 0 0 0]
 [0 0 1 0 0 1 0]
 [0 1 0 0 0 1 0]
 [0 0 0 1 1 0 1]
 [0 0 0 0 0 1 0]]
--------------------