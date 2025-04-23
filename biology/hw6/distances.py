import heapq
from collections import defaultdict

class Cluster:
    def __init__(self, name, members):
        self.name = name
        self.members = set(members)

def upgma(dist_matrix, sequence_names):
    n = len(sequence_names)
    clusters = {i: Cluster(sequence_names[i], [sequence_names[i]]) for i in range(n)}
    distances = {(i, j): dist_matrix[i][j] for i in range(n) for j in range(i) if i != j}
    heap = [(dist, i, j) for (i, j), dist in distances.items()]
    heapq.heapify(heap)

    next_cluster_id = n
    tree = defaultdict(list)

    while len(clusters) > 1:
        while True:
            dist, i, j = heapq.heappop(heap)
            if i in clusters and j in clusters:
                break

        new_name = f"node{next_cluster_id}"
        new_members = clusters[i].members.union(clusters[j].members)
        clusters[next_cluster_id] = Cluster(new_name, new_members)

        tree[new_name].append((clusters[i].name, dist / 2))
        tree[new_name].append((clusters[j].name, dist / 2))

        del clusters[i]
        del clusters[j]

        for k in clusters:
            if k == next_cluster_id:
                continue
            d1 = dist_matrix[i][k] if i < k else dist_matrix[k][i]
            d2 = dist_matrix[j][k] if j < k else dist_matrix[k][j]
            new_dist = (d1 + d2) / 2
            dist_matrix.append([0]*len(dist_matrix))  # Expand for new row
            for row in dist_matrix:
                row.append(0)
            dist_matrix[next_cluster_id][k] = new_dist
            dist_matrix[k][next_cluster_id] = new_dist
            heapq.heappush(heap, (new_dist, next_cluster_id, k))

        next_cluster_id += 1

    return tree

def dot_format(tree):
    dot = "graph UPGMA {\n"
    for parent, children in tree.items():
        for child, weight in children:
            dot += f'    "{parent}" -- "{child}" [weight={weight:.2f}, label="{weight:.2f}"];\n'
    dot += "}"
    return dot


def parse_pir_file(filename):
    sequences = []
    with open(filename, 'r') as file:
        current_seq = ''
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    sequences.append(current_seq)
                    current_seq = ''
            else:
                current_seq += line
        if current_seq:
            sequences.append(current_seq)
    return sequences

def count_differences(seq1, seq2):
    if len(seq1) != len(seq2):
        raise Exception("Sequences aren't same length :(")
    
    distance = 0
    for i in range(len(seq1)):
        if seq1[i] != seq2[i]:
            distance += 1
    
    return distance

def calculate_distance(aligned_sequences):
    num_sequences = len(aligned_sequences)
    distances = [[0 for _ in range(num_sequences)] for _ in range(num_sequences)]


    for i in range(num_sequences):
        for j in range(num_sequences):
            if i == j:
                continue
            seq1 = aligned_sequences[i]
            seq2 = aligned_sequences[j]

            distances[i][j] = count_differences(seq1, seq2)
    
    return distances
    

def main():
    filename = 'msa.pir'
    aligned_sequences = parse_pir_file(filename)
    sequence_names = [f"seq{i}" for i in range(len(aligned_sequences))]

    distances = calculate_distance(aligned_sequences)
    

    tree = upgma(distances, sequence_names)
    dot = dot_format(tree)

    with open("tree.dot", "w") as f:
        f.write(dot)
if __name__ == "__main__":
    main()