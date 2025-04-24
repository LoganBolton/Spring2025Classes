import heapq
from collections import defaultdict
from pathlib import Path


def parse_pir_file(filename: str):
    sequences = []
    current = []
    with open(filename, "r") as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current.clear()
            else:
                current.append(line)
        if current:
            sequences.append("".join(current))
    return sequences


def hamming(seq1, seq2):
    if len(seq1) != len(seq2):
        raise Exception("Sequences aren't same length :(")
    
    distance = 0
    for i in range(len(seq1)):
        if seq1[i] != seq2[i]:
            distance += 1
    
    return distance


def distance_matrix(seqs):
    n = len(seqs)
    mat = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i):
            d = hamming(seqs[i], seqs[j])
            mat[i][j] = mat[j][i] = d
    return mat


class Cluster:
    __slots__ = ("id", "name", "size", "height", "left", "right")

    def __init__(self, id_, name, size, height=0.0, left=None, right=None):
        self.id = id_
        self.name = name
        self.size = size
        self.height = height
        self.left = left
        self.right = right


def upgma(dist_mat, leaf_names):
    n = len(leaf_names)
    clusters = {i: Cluster(i, leaf_names[i], 1) for i in range(n)}
    pq = [(dist_mat[i][j], i, j) for i in range(n) for j in range(i)]
    heapq.heapify(pq)

    next_id = n
    edges = []
    dists = {(i, j): dist_mat[i][j] for i in range(n) for j in range(i)}

    while len(clusters) > 1:
        # find closest alive pair
        while True:
            dist, i, j = heapq.heappop(pq)
            if i in clusters and j in clusters:
                break

        ci, cj = clusters.pop(i), clusters.pop(j)
        new_h = dist / 2.0
        parent = Cluster(next_id, f"node{next_id}", ci.size + cj.size,
                         height=new_h, left=ci, right=cj)
        clusters[next_id] = parent

        edges.append((parent.name, ci.name, new_h - ci.height))
        edges.append((parent.name, cj.name, new_h - cj.height))

        for k, ck in clusters.items():
            if k == next_id:
                continue
            dik = dists[(max(i, k), min(i, k))]
            djk = dists[(max(j, k), min(j, k))]
            new_dist = (dik * ci.size + djk * cj.size) / (ci.size + cj.size)
            dists[(max(next_id, k), min(next_id, k))] = new_dist
            heapq.heappush(pq, (new_dist, min(next_id, k), max(next_id, k)))

        next_id += 1

    root = next(iter(clusters.values()))
    return edges


def print_full_distance_matrix_from_edges(edges, leaf_names):
    internals = {
        parent for parent,_,_ in edges
        if parent not in leaf_names
    }
    internals = sorted(internals, key=lambda x: int(x.replace("node","")))
    labels = leaf_names + internals
    
    adj = {lbl: [] for lbl in labels}
    for p, c, w in edges:
        adj[p].append((c, w))
        adj[c].append((p, w))
    
    full_dist = {lbl: {} for lbl in labels}
    for src in labels:
        dist = {src: 0.0}
        stack = [src]
        while stack:
            u = stack.pop()
            for v, w in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + w
                    stack.append(v)
        full_dist[src] = dist
    
    print("\t" + "\t".join(labels))
    for i, src in enumerate(labels):
        row = [f"{full_dist[src][dst]:.2f}" for dst in labels]
        print(f"{src}\t" + "\t".join(row))



def edges_to_dot(edges):
    lines = ["graph UPGMA {"]
    for parent, child, w in edges:
        lines.append(f'    "{parent}" -- "{child}" '
                     f'[weight={w:.5f}, label="{w:.2f}"];')
    lines.append("}")
    return "\n".join(lines)


def main():
    pir_file = "msa.pir"

    seqs = parse_pir_file(pir_file)
    names = [f"seq{i}" for i in range(len(seqs))]
    mat = distance_matrix(seqs)

    print("Original distance matrix:")
    for row in mat:
        print(row)
        
    edge_list = upgma(mat, names)

    print("\nFull UPGMA distance matrix (including internal nodes):")
    print_full_distance_matrix_from_edges(edge_list, names)


    with open("tree.dot", "w") as f:
        f.write(edges_to_dot(edge_list))

if __name__ == "__main__":
    main()