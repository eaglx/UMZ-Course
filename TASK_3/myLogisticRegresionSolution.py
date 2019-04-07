import sys
import numpy
import pandas
from sklearn.linear_model import LogisticRegression

if len(sys.argv) < 4:
    print("Not enough args!")
    print("python3 myLogisticRegresionSolution.py train.tsv in.tsv out.tsv")
    exit()

# ***************** SETUP *****************
train_file = sys.argv[1]
in_file = sys.argv[2]
out_file = sys.argv[3]
# ***************** TRAIN *****************
train_data = pandas.read_csv(train_file, sep='\t', header=None)
for c in train_data:
    train_data[c] = train_data[c].map(ord)
x_train = train_data.drop(train_data.columns[[0]], axis=1)
y_train = train_data[0]

regresion = LogisticRegression()
regresion.fit(x_train, y_train)
# ***************** PREDICT *****************
test_data = pandas.read_csv(in_file, sep='\t', header=None)
x_test = test_data[[x for x in range(22)]]
for c in x_test:
    x_test[c] = x_test[c].map(ord)

y_test = regresion.predict(x_test)
# ***************** SAVE *****************
f = open(out_file, "w")
for l in y_test:
    f.write(chr(l) + '\n')
f.close()
