from collections import defaultdict
from itertools import combinations

# Counts number of occurences of each individual item, then
# prunes items that don't meet threshold support. 
def get_singles(infile, singles, support):
    with open(infile) as browsing_history:
        for line in browsing_history:
            for item in line.split():
                singles[item] += 1
        browsing_history.close()
    for k, v in singles.items():
        if v < support:
            del singles[k] 


# Counts number of occurences of each tuple made up of frequent singles,
# then prunes tuples that don't meet threshold support.
def get_tuples(infile, singles, tuples, support):
    with open(infile) as browsing_history:
        for line in browsing_history:
            for pair in combinations(sorted(line.split()), 2):
                if pair[0] in singles and pair[1] in singles and pair[0] != pair[1]:
                    tuples[pair] += 1
        browsing_history.close()
    for k, v in tuples.items():
        if v < support:
            del tuples[k]


# Given a set of frequent tuples, computes the confidence scores of the tuple:
# conf(X -> Y) and conf (Y -> X), where conf(X -> Y) = support(X intersect Y) / support(X) 
def compute_tuple_confidences(singles, tuples, tuple_confidences):
    for pair in tuples:
        tuple_confidences[pair] = float(tuples[pair]) / singles[pair[0]]
        tuple_confidences[(pair[1], pair[0])] = float(tuples[pair]) / singles[pair[1]]
        
    
# Main function. 
# 1) Count occurences of individual items, then remove those that don't meet support.
# 2) Count occurences of tuples where both elements are frequent items, then remove 
#    those that don't meet support.
# 3) Count occurences of triples where all elemenents are frequent items, and the 3 
#    possible pairs also are frequent, then remove those that don't meet support. 
def solution(infile, support):
    singles = defaultdict(int) 
    tuples = defaultdict(int)
    triples = defaultdict(int)
    tuple_confidences = defaultdict(int)
    triple_confidences = defaultdict(int)
    get_singles(infile, singles, support)         # singles now contains only frequent items
    get_tuples(infile, singles, tuples, support)  # tuples now contains only frequent pairs
    compute_tuple_confidences(singles, tuples, tuple_confidences)
    result = sorted(tuple_confidences, key=tuple_confidences.get, reverse=True)
    for i in range(5):
        print result[i], tuple_confidences[result[i]]
    


solution('data/browsing.txt', 100)