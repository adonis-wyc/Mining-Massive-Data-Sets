import sys
from itertools import combinations
from pyspark import SparkConf, SparkContext

# Mapping function used to create key-value pairs of the form ((user1, user2), 0 | 1),
# where a 0 signifies that the users are friends, and a 1 signifies that they have a 
# mutual friend.
def make_friend_pairs(user):
    curr_user = user[0]
    friend_list = user[1]
    friend_pairs = []
    for friend in friend_list:
        friend_pairs.append(((curr_user, friend), 0) if curr_user < friend else ((friend, curr_user), 0))
    for mutual_friends in combinations(friend_list, 2):
        friend_pairs.append((mutual_friends, 1) if mutual_friends[0] < mutual_friends[1] else ((mutual_friends[1], mutual_friends[0]), 1))
    return friend_pairs

# Print resulting recommendations
def print_results(user):
    if user[0] == '924' or user[0] == '8941' or user[0] == '8942' or user[0] == '9019' or user[0] == '9020' or user[0] == '9021'\
                        or user[0] == '9022' or user[0] == '9990' or user[0] == '9992' or user[0] == '9993':
        print(user[0], ':', [x[0] for x in user[1]])
    
conf = SparkConf()
sc = SparkContext(conf=conf)
num_recommendations = int(sys.argv[2])

# user_friend_list_rdd's elements are of the form: (userID, [friends]) 
user_friend_list_rdd = sc.textFile(sys.argv[1])\
                         .map(lambda line: line.split('\t'))\
                         .map(lambda pair: (pair[0], pair[1].split(',')))

# all_friend_pairs_rdd's elements are of the form: ((user1, user2), 0) if friends, ((user1, user2), 1) if mutual friends
all_friend_pairs_rdd = user_friend_list_rdd.flatMap(lambda user: make_friend_pairs(user))

# mutual_friends_rdd's elements are of the form: ((user1, user2), # of mutual friends)
mutual_friends_rdd = all_friend_pairs_rdd.groupByKey()\
                                         .filter(lambda pair: 0 not in pair[1])\
                                         .map(lambda pair: (pair[0], sum(pair[1])))\
                                         .sortBy(lambda pair: pair[1], ascending=False)

# recommendations_rdd's elements are of the form: ((user1, [top recommendations]))
recommendations_rdd = mutual_friends_rdd.flatMap(lambda recommendation:\
                                        [(recommendation[0][0], (recommendation[0][1], recommendation[1])), 
                                        (recommendation[0][1], (recommendation[0][0], recommendation[1]))])\
                                        .groupByKey().mapValues(lambda recommendation: sorted(recommendation, key=lambda x: x[1], reverse=True)[:num_recommendations])

# print results
recommendations_rdd.foreach(print_results)
