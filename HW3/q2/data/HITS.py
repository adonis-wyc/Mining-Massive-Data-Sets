import sys
from pyspark import SparkConf, Spark Context

DATA_FILE = 'data/graph-small.txt'
LAMBDA = 1
MU = 1
K = 40


conf = SparkConf()
sc = SparkContext(conf=conf)

data_rdd = sc.textFile(DATA_FILE).map(lambda line: (line.split()[1], line.split()[0]))\
                                 .groupByKey()

print(data_rdd.take(5))