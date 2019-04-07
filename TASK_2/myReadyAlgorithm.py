import sys
import csv
import pandas
import numpy
from sklearn import linear_model


if len(sys.argv) < 4:
    print("Not enough args!")
    print("python3 myReadyAlgorithm.py train.tsv in.tsv out.tsv")
    exit()

# ***************** SETUP *****************
train_file = sys.argv[1]
in_file = sys.argv[2]
out_file = sys.argv[3]
fields = ['Liczba pokoi', 'Liczba pięter w budynku', 'Rok budowy', 'Powierzchnia w m2']
# ***************** TRAIN *****************
data = pandas.read_csv(train_file, header=0, sep='\t')
columns = data.columns[1:]
data = data[fields + ['cena']]
data = data.applymap(numpy.nan_to_num)
x = pandas.DataFrame(data[fields])
y = pandas.DataFrame(data['cena'])
regr = linear_model.LinearRegression()
regr.fit(x, y)
# ***************** PREDICT *****************
data = pandas.read_csv(in_file, header=None, sep='\t', names=columns)
x = pandas.DataFrame(data[fields])
x = x.applymap(numpy.nan_to_num)
y = regr.predict(x)
# ***************** SAVE *****************
pandas.DataFrame(y).astype('int64').to_csv(out_file, index=None, header=None, sep='\t')
