import sys
from pyspark import SparkConf, SparkContext

DATA_FILE = 'data/graph-full.txt'
OUTFILE = 'results.txt'
N = 1000  # Nodes in the graph
K = 40   # Iterations
B = 0.8  # Teleport constant

# Calculates a row of matrix M.
def fill_in_row(pair):
    result = [0] * N 
    node_in_edges = pair[1]  # Nodes that have a directed edge to target node
    for node in node_in_edges:
        result[node - 1] = float(1 / node_degrees[node - 1][1]) 
    return (pair[0], result) 

# Calculates the next value of the given element of r.
def calc_r(row, r):
    new_r = 0
    for row_i, r_i in zip(row, r):
        new_r += row_i * r_i
    return new_r * B + float((1 - B) / N)

# Writes top 5 and bottom 5 ranks to file. 
def write_output(r):
    with open(OUTFILE, 'w') as out_file:
        out_file.write('Top 5 PageRank scores:\n')
        for i in range(0, 5):
            out_file.write(str(r[i][0]) + ': ' + str(r[i][1]) + '\n')
        out_file.write('\nBottom 5 PageRank scores:\n')
        for i in reversed(range(N - 5, N)):
            out_file.write(str(r[i][0]) + ': ' + str(r[i][1]) + '\n')
       
# --------------------- Spark PageRank --------------------- #

conf = SparkConf()
sc = SparkContext(conf=conf)
r = [float(1 / N)] * N

# List of sorted (node, degree) pairs
node_degrees = sc.textFile(DATA_FILE).map(lambda line: (int(line.split()[0]), int(line.split()[1]))).distinct()\
                                     .map(lambda pair: (pair[0], 1)).reduceByKey(lambda n1, n2: n1 + n2).sortByKey().collect()

# Each element corresponds to a row of the matrix M
M_rdd = sc.textFile(DATA_FILE).map(lambda line: (int(line.split()[1]), int(line.split()[0]))).distinct()\
                                          .groupByKey().sortByKey().map(lambda pair: fill_in_row(pair))

# Perform K iterations of PageRank computations
for _ in range(K):
    r = M_rdd.map(lambda row: calc_r(row[1], r)).collect()

# Sort in decreasing order of PageRank
r = sorted([(i + 1, r[i]) for i in range(N)], key=lambda rank: rank[1], reverse=True)
write_output(r)
sc.stop()