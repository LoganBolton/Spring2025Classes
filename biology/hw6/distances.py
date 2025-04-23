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

filename = 'msa.pir'
aligned_sequences = parse_pir_file(filename)

num_sequences = len(aligned_sequences)
distances = [[0 for _ in range(num_sequences)] for _ in range(num_sequences)]

for i in range(num_sequences):
    for j in range(num_sequences):
        if i == j:
            continue
        seq1 = aligned_sequences[i]
        seq2 = aligned_sequences[j]

        distances[i][j] = count_differences(seq1, seq2)

for row in distances:
    print(row)