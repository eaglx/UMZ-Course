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
            loadDataField.append(line.split()[1])
            loadDataPrice.append(line.split()[0])
    loadData = []
    loadData.append(loadDataField) # oś-X
    loadData.append(loadDataPrice) # oś-Y
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

# Oblicz wartość funkcji kosztu
def regline2(fig, fun, theta, xx, yy):
    ax = fig.axes[0]
    x0, x1 = min(xx), max(xx)
    X = [x0, x1]
    Y = [fun(theta, x) for x in X]
    cost = J(fun, theta, xx, yy)
    ax.plot(X, Y, linewidth='2',
            label=(r'$y={theta0}{op}{theta1}x, \; J(\theta)={cost:.3}$'.format(
                theta0=theta[0],
                theta1=(theta[1] if theta[1] >= 0 else -theta[1]),
                op='+' if theta[1] >= 0 else '-',
                cost=cost)))
# ********************************************

# Funkcja sterójąca
def main():
    print("ALGORITHM-TASK-SOLVER")
    loadTrainDataToProcessing = load_train_data(sys.argv[1])
# ********************************************

if __name__ == "__main__":
    main()
