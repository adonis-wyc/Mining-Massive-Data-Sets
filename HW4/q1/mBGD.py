## Mini Batch Gradient Descent
import numpy as np
import random
import time

OUTPUT = './results_mbgd.txt'
FEATURES = './data/features.txt'
TARGET = './data/target.txt'
ETA = 0.00001    # Learning constant
EPSILON = 0.01   # Convergence criteria
C = 100          # Cost constant
BATCH_SIZE = 20   

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

# Update function for w_j
def update_w_j(w, x, y, b, j, l):
    gradient_w_j = 0.0
    for i in range(l * BATCH_SIZE, min(len(x), (l + 1) * BATCH_SIZE)): ########## bounds??
        gradient_w_j += 0.0 if (y[i] * (np.dot(x[i], w) + b) >= 1) else -y[i] * x[i][j]
    gradient_w_j = w[j] + C * gradient_w_j
    return ETA * gradient_w_j

# Update function for b
def update_b(w, x, y, b, l):
    gradient_b = 0.0
    for i in range(l * BATCH_SIZE, min(len(x), (l + 1) * BATCH_SIZE)): ########## bounds??
        gradient_b += 0.0 if (y[i] * (np.dot(x[i], w) + b) >= 1) else -y[i]
    return ETA * C * gradient_b

# Cost function
def compute_cost(w, x, y, b):
    empirical_loss = 0.0
    weight_sq_sum = 0.0
    for i in range(len(x)):
        empirical_loss += max(0, 1 - y[i] * (np.dot(x[i], w) + b))
    for j in range(len(w)):
        weight_sq_sum += w[j] ** 2
    return weight_sq_sum / 2 + C * empirical_loss

# Computes delta cost
def compute_dcost(cost_0, cost_1, dcost_prev):
    return 0.5 * dcost_prev + 0.5 * ((abs(cost_0 - cost_1) * 100) / cost_0)

# -------- Mini Batch Gradient Descent -------- #
def main():
    x, y = read_data()
    k = 0
    l = 0
    d = len(x[0])
    w_curr= np.array([0.0] * d)
    w_next = np.array([0.0] * d)
    b_curr = 0.0
    b_next = 0.0
    dcost_prev = 0.0
    dcost_curr = 0.0
    costs = [compute_cost(w_curr, x, y, b_curr)]

    start = time.time()
    while True: 
        for j in range(d):
            w_next[j] -= update_w_j(w_curr, x, y, b_curr, j, l)
        b_next -= update_b(w_curr, x, y, b_curr, l)
        costs.append(compute_cost(w_next, x, y, b_next))
        l = (l + 1) % ((len(x) + BATCH_SIZE - 1) / BATCH_SIZE)
        k += 1
        w_curr = w_next
        b_curr = b_next
        dcost_curr = compute_dcost(costs[k - 1], costs[k], dcost_prev)
        if (dcost_curr < EPSILON):
            break
        dcost_prev = dcost_curr
    time_elapsed = time.time() - start

    with open(OUTPUT, 'w') as f:
        f.write('Time to convergence: ' + str(time_elapsed))
        for cost in costs:
            f.write(str(cost) + '\n')

##########################
if __name__ == '__main__':
    main()