## Batch Gradient Descent
import numpy as np
import time

OUTPUT = './results_bgd.txt'
FEATURES = './data/features.txt'
TARGET = './data/target.txt'
ETA = 0.0000003     # Learning rate
EPSILON = 0.25      # Convergence criteria
C = 100             # Cost constant

# Read in training data
def read_data():
    x = []
    y = []
    with open(FEATURES, 'r') as f:
        x = [np.array([int(feature) for feature in line.split(',')]) for line in f.readlines()]
    with open(TARGET, 'r') as f:
        y = [float(target) for target in f.readlines()]
    return x, y

# Update function for w_j
def update_w_j(w, x, y, b, j):
    gradient_w_j = 0.0 
    for i in range(len(x)):
        gradient_w_j += 0.0 if (y[i] * (np.dot(x[i], w) + b) >= 1) else -y[i] * x[i][j]
    gradient_w_j = w[j] + C * gradient_w_j
    return ETA * gradient_w_j

# Update function for b
def update_b(w, x, y, b):
    gradient_b = 0.0
    for i in range(len(x)):
        gradient_b += 0.0 if (y[i] * (np.dot(x[i], w) + b)) >= 1 else -y[i]
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

# Calculate if convergence criteria was reached
def converged(cost_0, cost_1):
    return (abs(cost_0 - cost_1) * 100) / cost_0 < EPSILON

# -------- Batch Gradient Descent -------- #
def main():
    x, y = read_data()
    k = 0
    d = len(x[0])
    w_curr = np.array([0.0] * d)
    w_next = np.array([0.0] * d)
    b_curr = 0.0 
    b_next = 0.0 
    costs = [compute_cost(w_curr, x, y, b_curr)]
    start = time.time()
    while True: 
        for j in range(d):
            w_next[j] -= update_w_j(w_curr, x, y, b_curr, j)
        b_next -= update_b(w_curr, x, y, b_curr)
        costs.append(compute_cost(w_next, x, y, b_next))
        k += 1
        w_curr = w_next
        b_curr = b_next
        if converged(costs[k - 1], costs[k]):
            break
    time_elapsed = time.time() - start
    with open(OUTPUT, 'w') as f:
        f.write('Time to convergence: ' + str(time_elapsed) + '\n')
        for cost in costs:
            f.write(str(cost) + '\n')

##########################
if __name__ == '__main__':
    main()