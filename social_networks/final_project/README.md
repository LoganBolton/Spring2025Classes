# Project Summary

## Research Question
_Is it possible to reverse engineer properties of a text prompt based off the attention map of an LLM?_

## Related Work
Other Arxiv papers have shown that you can train a transformer to reverse engineer the actual text input with okayish accuracy just based off the attention values across all heads/layers


# Method

## Summary
- Feed in a directed graph to an LLM in a simple text format. 
- Train a MLP on attention values and have it generate a new directed graph 
- Compare the generated graph to the original graph using a variety of social network analysis tools

## Setup
### Data
**create_graphs.py**
- Generate an adjacency matrix for a directed graph
- Generate every equivalent form of that graph in text format
**create_combined_graphs.py**
- Extract the attention from an LLM on all heads/layers with these prompts
- Condense token attention to just the attention to and from each node
- Average across all these graphs

### Training
- Split up all datasets into a test train split
- Train MLP model to predict original adjacency matrix based off averaged attention matrix

### Evaluation
- Compare generate graph to original graph
- Social analysis methods:
    - Centrality Measure
    - Node/Edge Count
    - Clustering
    - ????
- Traditional ML Methods:
    - Accuracy
    - FP/FN, etc
    - F1 Score