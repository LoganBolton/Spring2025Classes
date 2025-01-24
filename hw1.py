import numpy as np
import pandas as pd
import scipy
from plotnine import *


'''
Homework 1 problem 8 -- global alignment
use the simple scoring method of +1 for match and -1 for mismatch and indel
print the global alignment with one line per string with '-' characters for indels
'''
def global_alignment(sequence1, sequence2):
    # init blank matrix
    matrix = np.zeros((len(sequence1)+1, len(sequence2)+1))
    for i in range(1,len(sequence1)+1):
        matrix[i,0] = matrix[i-1,0] - 1
    for i in range(1,len(sequence2)+1):
        matrix[0,i] = matrix[0,i-1] - 1
    
    # fill in matrix values
    for i in range (1,len(sequence1)+1):
        for j in range(1,len(sequence2)+1):
            top_left = matrix[i-1,j-1]
            if sequence2[j-1] == sequence1[i-1]:
                top_left += 1
            else:
                top_left -= 1
            
            top = matrix[i-1,j] - 1
            left = matrix[i,j-1] - 1
            
            matrix[i,j] = max(top_left, top, left)
    print(matrix)
    



'''
support code for creating random sequence, no need to edit
'''
def random_sequence(n):
    return("".join(np.random.choice(["A","C","G","T"], n)))

'''
support code for mutating a sequence, no need to edit
'''
def mutate(s, snp_rate, indel_rate):
    x = [c for c in s]
    i = 0
    while i < len(x):
        if np.random.random() < snp_rate:
            x[i] = random_sequence(1)
        if np.random.random() < indel_rate:
            length = np.random.geometric(0.5)
            if np.random.random() < 0.5: # insertion
                x[i] = x[i] + random_sequence(length)
            else:
                for j in range(i,i+length):
                    if j < len(x):
                        x[j] = ""
                    i += 1
        i += 1
    return("".join(x))

# creating related sequences
s1 = random_sequence(5)
s2 = mutate(s1, 0.1, 0.1)
print(len(s2))
for c in s1:
    print(" ", c, end="")
print()
for c in s2:
    print(c)

# running your alignment code
global_alignment(s1, s2)
