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
- add a gt matrix 


# Random
Right now I'm looking at every node, no matter its position. Maybe its best to only look at nodes that come before arrow token? maybe after?
- If it sucks
    - Change from all nodes to just looking at parent or child nodes
    - Kinda more boring though
    - Hope it can get both