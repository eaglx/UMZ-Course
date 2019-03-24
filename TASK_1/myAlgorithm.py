#!/usr/bin/python3

import numpy as np
import sys

# Funkcja pobierająca dane.
def load_train_data(fileName):
    print("> LOAD TRAIN DATA")
    loadDataField = []
    loadDataPrice = []
    count = 0
    with open(fileName) as f:
        for line in f.readlines():
            if(count == 0):
                count += 1
                continue
            loadDataField.append(float(line.split()[1]))
            loadDataPrice.append(float(line.split()[0]))
    loadData = []
    loadData.append(loadDataField) # oś-X
    loadData.append(loadDataPrice) # oś-Y

    # print(loadDataField)
    # print(loadDataPrice)
    # exit()

    return loadData
# ********************************************

# Funkcja liniowa jednej zmiennej
def h(theta, x):
    return theta[0] + theta[1] * x
# ********************************************

# Funkcja licząca błąd średniokwadratowy
def J(h, theta, x, y):
    m = len(y)
    return 1.0 / (2 * m) * sum((h(theta, x[i]) - y[i])**2 for i in range(m))
# ********************************************

# Gradient Descent
def gd(h, costfun, theta, x, y, alpha, eps):
    current_cost = costfun(h, theta, x, y)
    log = [[current_cost, theta]]
    m = len(y)
    while True:
        new_theta = [
        theta[0] - alpha/float(m) * sum(h(theta, x[i]) - y[i] for i in range(m)),
        theta[1] - alpha/float(m) * sum((h(theta, x[i]) - y[i]) * x[i] for i in range(m))
        ]
        theta = new_theta
        try:
            current_cost, prev_cost = costfun(h, theta, x, y), current_cost
        except OverflowError:
            break

        if abs(prev_cost - current_cost) <= eps:
            break
        log.append([current_cost, theta])
        # print(log[-1])
    # print(theta)
    # print(log)
    return log[-1][0]
# ********************************************

# Funkcja sterójąca
def main():
    print("ALGORITHM-TASK-SOLVER")
    loadTrainDataToProcessing = load_train_data(sys.argv[1])
# ********************************************
    current_cost = 0.0
    prev_cost = gd(h, J, [0, 0], loadTrainDataToProcessing[0], loadTrainDataToProcessing[1], alpha=0.001, eps=0.01)
    eps = 0.01 / 2
    alpha = 0.001
    while True:
        current_cost = gd(h, J, [0, 0], loadTrainDataToProcessing[0], loadTrainDataToProcessing[1], alpha, eps)
        if current_cost < prev_cost:
            prev_cost = current_cost
            current_cost = 0.0
            eps = eps / 2
        elif prev_cost == -1.0:
            prev_cost = current_cost
            current_cost = 0.0
            eps -= 0.0001
        else:
            break
    print(prev_cost)

if __name__ == "__main__":
    main()
