#!/usr/bin/python3

# ./myAlgorithm.py train/train.tsv dev-0/in.tsv dev-0/out.tsv

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
            loadDataField.append(float(line.split("\t")[1]))
            loadDataPrice.append(float(line.split("\t")[0]))
    loadData = []
    loadData.append(loadDataField) # oś-X
    loadData.append(loadDataPrice) # oś-Y
    return loadData

def load_data(fileName):
    print("> LOAD DEV DATA")
    loadDataField = []
    with open(fileName) as f:
        for line in f.readlines():
            loadDataField.append(float(line.split("\t")[0]))
    return loadDataField
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
    max_itern = 1200000
    count_itern = 0
    while True:
        count_itern += 1
        new_theta = [
        theta[0] - alpha/float(m) * sum(h(theta, x[i]) - y[i] for i in range(m)),
        theta[1] - alpha/float(m) * sum((h(theta, x[i]) - y[i]) * x[i] for i in range(m))
        ]
        theta = new_theta
        try:
            current_cost, prev_cost = costfun(h, theta, x, y), current_cost
            # print("diff: ",prev_cost - current_cost)
        except OverflowError:
            break

        if abs(prev_cost - current_cost) <= eps:
            break
        log.append([current_cost, theta])

        if count_itern == max_itern:
            break
    #     print(current_cost, prev_cost)
    # print(log[-1])
    return log
# ********************************************

# Funkcja sterująca
def main():
    print("ALGORITHM-TASK-SOLVER")
    loadTrainDataToProcessing = load_train_data(sys.argv[1])
# **************** TRAIN ****************************
    print("> TRAIN ALGORITHM")
    prev_cost = gd(h, J, [0, 0], loadTrainDataToProcessing[0], loadTrainDataToProcessing[1], alpha=0.000001, eps=1000)
    bestTheta = prev_cost[-1][1]
# ***************************************************
    loadTrainDataToProcessing = load_data(sys.argv[2])
    print("> SAVE OUT")
    with open(sys.argv[3], "w") as f:
        for i in loadTrainDataToProcessing:
            f.write(str(int(h(bestTheta, i))) + "\n")

if __name__ == "__main__":
    main()
