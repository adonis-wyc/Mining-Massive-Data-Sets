from collections import defaultdict
import numpy as np
import random
import math

"""
    @param data: text file of user-movie ratings
    @param iterations: number of training iterations
    @param k: training features
    @param l: regularization constant
    @param nu: learning rate
"""
def sgd(data, iterations=40, k=20, l=0.1, nu=0.01):
    q = defaultdict(lambda: np.array([random.uniform(0, math.sqrt(float(5)/k)) for i in range(k)]))
    p = defaultdict(lambda: np.array([random.uniform(0, math.sqrt(float(5)/k)) for i in range(k)]))
    with open(data) as ratings:
        for _ in range(iterations):
            # Update q and p
            ratings.seek(0)
            for rating in ratings: 
                u, i, r_iu = rating.split()                     # u: user id; i: movie id; r_iu: rating of movie i by user u
                e_iu = 2 * (float(r_iu) - np.dot(q[i], p[u]))   # e_iu: derivative of error for rating of movie i by user u
                q_update = np.multiply(nu, np.add(np.multiply(e_iu, p[u]), np.multiply(-2 * l, q[i])))
                p_update = np.multiply(nu, np.add(np.multiply(e_iu, q[i]), np.multiply(-2 * l, p[u])))
                q[i] = np.add(q[i], q_update)
                p[u] = np.add(p[u], p_update)

            # Compute E, the error
            ratings.seek(0)
            E = 0.0
            length = 0.0
            for rating in ratings:
                u, i, r_iu = rating.split()                     
                E += (float(r_iu) - np.dot(q[i], p[u])) ** 2
            for q_i in q.values():
                length += np.linalg.norm(q_i) ** 2
            for p_u in p.values():
                length += np.linalg.norm(p_u) ** 2
            length *= l
            E += length
            print(E)

sgd("./data/ratings.train.txt")