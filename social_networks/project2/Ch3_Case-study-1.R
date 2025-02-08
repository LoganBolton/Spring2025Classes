# Set working directory and load data
setwd("/Users/szs0398/Library/CloudStorage/OneDrive-AuburnUniversity/Teaching/25S_5740-6740/R-files")

# Install tidyr package if not already installed
install.packages("tidyr")
library(igraph)

D <- read.table("wiki-Vote.txt")

head(D)
tail(D)

# Create the graph
g <- graph_from_data_frame(edges_df, directed = TRUE)

# Check if the graph is strongly connected
strongly_connected <- is_connected(g, mode = "strong")
print(strongly_connected)

# Analyze strong components
scc <- components(g, mode = "strong")
scc


# Extract and plot the largest component
largest_comp_ids <- which(scc$membership == which.max(table(scc$membership)))
largest_component <- induced_subgraph(g, largest_comp_ids)

# Improved plotting function: No vertex labels and increased node distance
plot_network <- function(g, layout_type = layout_with_kk) {
  # Use Kamada-Kawai layout for better spacing between nodes
  layout <- layout_type(g)
  
  # Scale the layout to further spread nodes apart
  layout <- layout * 3  # Multiply by a factor to increase node spacing
  
  plot(g,
       layout = layout_nicely(g),
       vertex.size = 1,               # Moderate vertex size
       vertex.color = "lightblue",    # Light color for vertices
       vertex.label = NA,             # Remove vertex labels
       edge.arrow.size = 0.04,         # Smaller arrowheads for clarity
       edge.width = 0.07,              # Thin edges for less clutter
       edge.color = "black",       # Uniform dark gray edges
       main = "Graph with Increased Node Spacing and No Labels")
}


# Plot full and largest networks side by side
par(mfrow = c(1, 2))  # Set up side-by-side plots
plot_network(g)  # Full network
title(sub = "Full Network")
plot_network(largest_component)  # Largest component
title(sub = "Largest Strongly Connected Component")


# Calculate and plot degree distributions
in_deg <- degree(g, mode="in")
out_deg <- degree(g, mode="out")

# Plot degree distributions
par(mfrow=c(1,2))
hist(in_deg, main="In-Degree Distribution", 
     xlab="In-Degree", ylab="Frequency", col="lightblue")
hist(out_deg, main="Out-Degree Distribution", 
     xlab="Out-Degree", ylab="Frequency", col="lightgreen")

# Calculate centrality metrics
page_rank <- page_rank(g, weights=E(g)$weight, directed=TRUE)$vector
harmonic_close <- harmonic_centrality(g, weights=E(g)$weight, mode="out")
betweenness <- betweenness(g, weights=E(g)$weight, directed=TRUE, normalized=TRUE)


# Create a data frame with all centrality metrics
centrality_df <- data.frame(
  node = V(g),
  in_degree = in_deg,
  out_degree = out_deg,
  pagerank = page_rank,
  harmonic_closeness = harmonic_close,
  betweenness = betweenness
)

# Function to get top 10 nodes for each metric
get_top_10 <- function(metric, metric_name) {
  top_indices <- order(metric, decreasing=TRUE)[1:10]
  data.frame(
    Metric = rep(metric_name, 10),
    Node = top_indices,
    Value = round(metric[top_indices], 4)
  )
}

# Get top 10 for each metric
top_nodes <- rbind(
  get_top_10(in_deg, "In-Degree"),
  get_top_10(out_deg, "Out-Degree"),
  get_top_10(page_rank, "PageRank"),
  get_top_10(harmonic_close, "Harmonic Closeness"),
  get_top_10(betweenness, "Betweenness")
)

# Print results
print("Top 10 nodes by different centrality measures:")
print(top_nodes)

# Load necessary library for data manipulation 
library(dplyr)



# Load tidyr package
library(tidyr)


# Pivot table to get one row per unique node
node_table <- top_nodes %>%
  pivot_wider(
    id_cols = Node,
    names_from = Metric,
    values_from = Value
  ) %>%
  arrange(desc(`In-Degree`))  # Order rows by In-Degree in descending order

# Print the resulting table
print("Table of unique nodes with centrality measures (ordered by In-Degree):")
print(node_table)

# Optionally, save the table to a CSV file
write.csv(node_table, "node_table.csv", row.names = FALSE)


# Calculate overall influence score (normalized sum of rankings)
# for a data point x_i, scale(x_i)=(x_i-mean(x_1,..,x_n))/standard deviation
centrality_df$influence_score <- scale(centrality_df$in_degree)+
  scale(centrality_df$pagerank) +
  scale(centrality_df$harmonic_closeness) +
  scale(centrality_df$betweenness)

# Get top 10 most influential nodes overall
top_10_influential <- head(centrality_df[order(centrality_df$influence_score, 
                                              decreasing=TRUE), ], 10)

print("\nOverall top 10 most influential nodes:")
print(top_10_influential)


#####
# Additional summary statistics
print("\nNetwork Summary Statistics:")
print(paste("Number of nodes:", vcount(g)))
print(paste("Number of edges:", ecount(g)))
print(paste("Network density:", edge_density(g)))
print(paste("Global clustering coefficient:", transitivity(g)))









