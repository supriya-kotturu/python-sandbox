# WEIRD ALGORITHM
#
# Consider an algorithm that takes as input a positive integer n. If n is even, the algorithm divides it by two,
# and if n is odd, the algorithm multiplies it by three and adds one. The algorithm repeats this, until n is one. 
# 
# For example, the sequence for n=3 is as follows:
# 3 -> 10 -> 5 -> 16 -> 8 -> 4 -> 2 -> 1
#
# Your task is to simulate the execution of the algorithm for a given value of n.
# Input
# The only input line contains an integer n.
# Output
# Print a line that contains all values of n during the algorithm.
# 
# INPUT: n int
# OUTPUT: int[]

def weirdAlgorithm(n):
    sequence = [n]

    last = sequence[-1]
    while(last != 1):
        if(last % 2 == 0):
            sequence.append(last//2)
        else:
            sequence.append((last*3)+1)
        last = sequence[-1]

    return sequence


num = int(input())
print(' '.join(map(str, weirdAlgorithm(num))))