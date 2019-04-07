import sys
import csv
import pandas
import numpy
from sklearn import linear_model


if len(sys.argv) < 4:
    print("Not enough args!")
    print("python3 myReadyAlgorithm.py train.tsv in.tsv out.tsv")
    exit()

train_file = sys.argv[1]
in_file = sys.argv[2]
out_file = sys.argv[3]
# ***************** SETUP *****************
fields = ['Powierzchnia w m2', 'Liczba pokoi']
# ***************** TRAIN *****************
data = pandas.read_csv(train_file, header=0, sep='\t')
columns = data.columns[1:]
data = data[fields + ['cena']]
y = pandas.DataFrame(data['cena'])
x = pandas.DataFrame(data[fields])
regr = linear_model.LinearRegression()
regr.fit(x, y)
# ***************** PREDICT *****************
data = pandas.read_csv(in_file, header=None, sep='\t', names=columns)
x = pandas.DataFrame(data[fields])
y = regr.predict(x)
# ***************** SAVE *****************
pandas.DataFrame(y).to_csv(out_file, index=None, header=None, sep='\t')
