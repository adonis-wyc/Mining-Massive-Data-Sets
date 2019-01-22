from collections import defaultdict
from itertools import combinations

# Counts number of occurrences of each individual item, then
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


# Counts number of occurrences of each tuple made up of frequent singles,
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


# Given a set of frequent tuples, computes the confidence scores of each tuple:
# conf(X -> Y) and conf (Y -> X), where conf(X -> Y) = support(X intersect Y) / support(X) 
def compute_tuple_confidences(singles, tuples, tuple_confidences):
    for pair in tuples:
        tuple_confidences[pair] = float(tuples[pair]) / singles[pair[0]]
        tuple_confidences[(pair[1], pair[0])] = float(tuples[pair]) / singles[pair[1]]
        

# Counts the number of occurrences of each triple made up of frequent tuples,
# then prunes triples that don't meet threshold support.
def get_triples(infile, tuples, triples, support):
    with open(infile) as browsing_history:
        for line in browsing_history:
            for triple in combinations(sorted(line.split()), 3):
                if ((triple[0], triple[1]) in tuples and (triple[0], triple[2]) in tuples and 
                    (triple[1], triple[2]) in tuples):
                    triples[triple] += 1
        browsing_history.close()
    for k, v in triples.items():
        if v < support:
            del triples[k]


# Given a set of frequent triples, computes the confidence scores of each triple:
# conf((X, Y) -> Z), conf((X, Z) -> Y), and conf((Y, Z) -> X), where conf((X, Y) -> Z) = 
# support((X, Y) intersect Z) / support((X, Y))
def compute_triple_confidences(tuples, triples, triple_confidences):
    for triple in triples:
        triple_confidences[triple] = float(triples[triple]) / tuples[(triple[0], triple[1])]
        triple_confidences[(triple[0], triple[2], triple[1])] = float(triples[triple]) / tuples[(triple[0], triple[2])]
        triple_confidences[(triple[1], triple[2], triple[0])] = float(triples[triple]) / tuples[(triple[1], triple[2])]

    
# Main function. 
# 1) Count occurrences of individual items, then remove those that don't meet support.
# 2) Count occurrences of tuples where both elements are frequent items, then remove 
#    those that don't meet support.
# 3) Count occurrences of triples where all elemenents are frequent items, and the 3 
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
    top_tuple_confidences = sorted(tuple_confidences, key=tuple_confidences.get, reverse=True)[0:5]
    # for pair in top_tuple_confidences:
        # print pair, tuple_confidences[pair]
    get_triples(infile, tuples, triples, support) # triples now contains only frequent triples
    compute_triple_confidences(tuples, triples, triple_confidences)
    top_triple_confidences = sorted(triple_confidences, key=triple_confidences.get, reverse=True)[0:50]
    # for triple in top_triple_confidences:
        # print triple, triple_confidences[triple]


solution('data/browsing.txt', 100)