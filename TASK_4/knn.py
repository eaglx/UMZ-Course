import sys
import numpy
import pandas
from sklearn.neighbors import KNeighborsClassifier

if len(sys.argv) < 4:
    print("Not enough args!")
    print("python3 knn.py train.tsv in.tsv out.tsv")
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

knn_mod = KNeighborsClassifier(n_neighbors=3)
model = knn_mod.fit(x_train, y_train)

# ***************** PREDICT *****************
test_data = pandas.read_csv(in_file, sep='\t', header=None)
for c in test_data:
    test_data[c] = test_data[c].map(ord)

y_test = knn_mod.predict(test_data)
# ***************** SAVE *****************
f = open(out_file, "w")
for l in y_test:
    f.write(chr(l) + '\n')
f.close()
