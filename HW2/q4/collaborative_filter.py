import numpy as np
import math

"""
    R <- m x n matrix filled by data from user-shows.txt
    P <- m x m diagonal matrix where P_ii is the sum of the elements in row i of R
    Q <- n x n diagonal matrix where Q_jj is the sum of the elements in column j of R
"""

def collaborative_filter(data, shows):
    R = []  # Original m x n data matrix
    with open(data) as f:
        for user in f:
            ratings = [int(r) for r in user.split()]
            R.append(ratings)
        R = np.array(R)

    movies = [] # List of movie names
    with open(shows) as f:
        for movie in f:
            movies.append(movie)

    m = len(R) 
    n = len(R[0])
    P = np.array([np.array([0 if j != i else (1 / math.sqrt(sum(R[i]))) for j in range(m)]) for i in range(m)]) # Diag matrix P^{-1/2} where P_ii = 1 / sqrt(sum of elements in row vector R_i)
    Q = np.array([[0 if j != i else (1 / math.sqrt(sum(R[:,i]))) for j in range(n)] for i in range(n)])         # Diag matrix Q^{-1/2} where Q_ii = 1 / sqrt(sum of elements in column vector R_i)

    T_user = ((P.dot(R)).dot(R.T.dot(P))).dot(R)    # T_user = PRR.TPR --> user-user recommendation matrix
    T_item= ((R.dot(Q)).dot(R.T.dot(R))).dot(Q)     # T_item = RQR.TRQ --> item-item recommendation matrix

    user_filter = {movies[index] : T_user[499][index] for index in range(100)}
    item_filter = {movies[index] : T_item[499][index] for index in range(100)}

    top_user_filter_recs = [name for name in sorted(user_filter, key=user_filter.get, reverse=True)][:5]
    top_item_filter_recs = [name for name in sorted(item_filter, key=item_filter.get, reverse=True)][:5]

    print(top_user_filter_recs)
    print(top_item_filter_recs)


collaborative_filter("./data/user-shows.txt", "./data/shows.txt")