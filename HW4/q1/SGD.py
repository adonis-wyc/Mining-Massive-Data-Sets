## Stochastic Gradient Descent
import random 
import numpy as np
import time

OUTPUT = './results_sgd.txt'
FEATURES = './data/features.txt'
TARGET = './data/target.txt'
ETA = 0.0001     # Learning constant
EPSILON = 0.001  # Convergence criteria
C = 100          # Cost constant

# Read training data
def read_data():
    x, y, data, targets = ([] for i in range(4))
    with open(TARGET, 'r') as f:
        targets = [float(target) for target in f.readlines()]
    with open(FEATURES, 'r') as f:
        for i in range(len(targets)):
            data.append(([int(feature) for feature in f.readline().split(',')], targets[i]))
    random.shuffle(data)
    x = [np.array(row[0]) for row in data]
    y = [row[1] for row in data]
    return x, y 

# Cost function
def compute_cost(w, x, y, b):
    empirical_loss = 0.0
    weight_sq_sum = 0.0
    for i in range(len(x)):
        empirical_loss += max(0, 1 - y[i] * (np.dot(x[i], w) + b))
    for j in range(len(w)):
        weight_sq_sum += w[j] ** 2
    return weight_sq_sum / 2 + C * empirical_loss

# Update function for w_j
def update_w_j(w, x, y, b, j, i):
    gradient_w_j = 0.0 if (y[i] * (np.dot(x[i], w) + b) >= 1) else -y[i] * x[i][j]
    gradient_w_j = w[j] + C * gradient_w_j
    return ETA * gradient_w_j

# Update function for b
def update_b(w, x, y, b, i):
    return 0.0 if (y[i] * (np.dot(x[i], w) + b) >= 1) else ETA * C * -y[i]

# Computes delta cost
def compute_dcost(cost_0, cost_1, dcost_prev):
    return 0.5 * dcost_prev + 0.5 * ((abs(cost_0 - cost_1) * 100) / cost_0)

# -------- Stochastic Gradient Descent -------- #
def main():
    x, y = read_data()
    k = 0
    i = 1
    d = len(x[0])
    w_curr = np.array([0.0] * d)
    w_next = np.array([0.0] * d)
    b_curr = 0.0
    b_next = 0.0
    dcost_prev = 0.0
    dcost_curr = 0.0
    costs = [compute_cost(w_curr, x, y, b_curr)]

    start = time.time()
    while True:
        for j in range(d):
            w_next[j] -= update_w_j(w_curr, x, y, b_curr, j, i)
        b_next -= update_b(w_curr, x, y, b_curr, i)
        costs.append(compute_cost(w_next, x, y, b_next))
        i = i % len(x) + 1
        k += 1
        w_curr = w_next
        b_curr = b_next
        dcost_curr = compute_dcost(costs[k - 1], costs[k], dcost_prev)
        if dcost_curr < EPSILON:
            break
        dcost_prev = dcost_curr
    time_elapsed = time.time() - start 

    with open(OUTPUT, 'w') as f:
        f.write('Time to convergence: ' + str(time_elapsed) + '\n') 
        for cost in costs:
            f.write(str(cost) + '\n')

##########################
if __name__ == '__main__':
    main()