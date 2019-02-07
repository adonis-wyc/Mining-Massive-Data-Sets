import sys
import math
# import numpy as np
from pyspark import SparkConf, SparkContext

MAX_ITER = 20
DIMENSIONS = 58
DATA_FILE = 'data/data.txt'
C1_FILE = 'data/c1.txt'
C2_FILE = 'data/c2.txt'

#  Reads in the initial values of the 10 centroids.
#  Expects infile to have one centroid per line, with dimensions separated by whitespace.
def read_initial_centroids(infile):
    initial_centroids = []
    with open(infile) as f:
        initial_centroids = [[float(x) for x in line.split()] for line in f.readlines()]
    return initial_centroids

#  Returns the Euclidean distance between two points.
def euclidean_distance_cost(a, b):
    distance = 0
    for a_i, b_i in zip(a, b):
        distance += (a_i - b_i) ** 2
    return distance


#  Returns the Manhattan distance between two points.
def manhattan_distance(a, b):
    distance = 0
    for a_i, b_i in zip(a, b):
        distance += abs(a_i - b_i)
    return distance 


#  Calculates the nearest centroid to the given point.
#  Represents chosen centroid as an index into the centroid list
def nearest_centroid(point):
    closest = -1
    shortest_distance = float('inf')
    for i in range(len(centroids)):
        distance = euclidean_distance_cost(point, centroids[i]) 
        # distance = manhattan_distance(point, centroids[i])
        if distance < shortest_distance:
            closest = i
            shortest_distance = distance
    return (closest, (point, shortest_distance))


#  Computes sum of 2 DIMENSION-dimensional points
def add_points(total, curr):
    for i in range(DIMENSIONS):
        total[i] += curr[i] 
    return total


#  Computes the new centroids by taking mean of clusters
def compute_new_centroids(clusters, counts):
    for i in range(len(clusters)):
        clusters[i] = (clusters[i][0], list(map(lambda x: x / counts[clusters[i][0]], clusters[i][1])))
    return [x[1] for x in clusters]


# def compute_cost(data_rdd, cost_function, initial_centroid_file):
    # centroids = read_initial_centroids(initial_centroid_file)


#  procedure Spark Iterative k-Means Algorithm
#     Select k points as initial centroids of the k clusters
#     for iterations := 1 to 10 do
#         map each point p in the dataset to closest centroid -> (centroid, point)
#         groupByKey (to get each centroid and its cluster)
#         calculate cost for this iteration by summing distances 
#         recompute each cluster centroid by mapping centroids to the points in their cluster, averaging points together
#         assign new centroids to centroids
#     end for
#  end procedure
conf = SparkConf()
sc = SparkContext(conf=conf)
data_rdd = sc.textFile(DATA_FILE).map(lambda line: [float(x) for x in line.split()])  # Each element of data_rdd is a list of floats that represents the point 
centroids = read_initial_centroids(C1_FILE)
iteration_costs = []

for _ in range(MAX_ITER):
    clusters_rdd = data_rdd.map(lambda point: nearest_centroid(point))  # Each element of clusters_rdd is a tuple: (index of centroid, (point, cost))
    iteration_cost = clusters_rdd.map(lambda point: point[1][1]).reduce(lambda total, x: total + x) # Calculates the total iteration cost by summing individual costs
    iteration_costs.append(iteration_cost)
    summed_clusters_rdd = clusters_rdd.map(lambda point: (point[0], point[1][0]))\
                                    .reduceByKey(lambda total, curr: add_points(total, curr))  # Each element of summed_clusters_rdd is a tuple (index of centroid, sum of points in cluster)
    centroids = compute_new_centroids(summed_clusters_rdd.collect(), clusters_rdd.countByKey())  # Recompute centroids by calculating mean of each cluster
    
print(iteration_costs)
